"""Deployed inference; only cached feedback enters the actor."""
from __future__ import annotations

import importlib
import hashlib
import math
import sys
from pathlib import Path

from .common import encoded, local_windows_runtime, read, sha, write_new


def tree_hashes(root):
    root = Path(root).resolve()
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py'))}


def register_actor(checkpoint, source_root, output, edges, scales, *, transfer=False,
                   architecture='edgewise', log_std_min=-5., log_std_max=1.):
    local_windows_runtime()
    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source_edges = int(payload['n_edges'])
    if architecture == 'edgewise' and not payload.get('edgewise_price_actor'):
        raise ValueError('An actual edge-wise checkpoint is required')
    if architecture=='shared_gaussian' and any(payload.get(key) for key in
            ('edgewise_price_actor','flat_price_bias_actor','residual_actor','feedback_age_gate')):
        raise ValueError('Cannot relabel another trained actor architecture as NoQL')
    if source_edges != edges and (not transfer or architecture != 'edgewise'):
        raise ValueError('Register edge-wise actor-only transfer explicitly; flat actors cannot resize')
    if architecture == 'edgewise' and payload['actor_obs_dim'] != 3 * source_edges + 7:
        raise ValueError('Historical actor feature layout mismatch')
    state = payload['actor']
    hidden = int(state['user_net.0.weight' if architecture == 'edgewise' else 'net.0.weight'].shape[0])
    value = dict(schema='systemlab-frozen-actor-v1', checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha(checkpoint), source_root=str(Path(source_root).resolve()),
        source_sha256=tree_hashes(source_root), architecture=architecture, source_edges=source_edges,
        target_edges=edges, source_actor_obs_dim=int(payload['actor_obs_dim']), hidden=hidden,
        log_std_min=log_std_min, log_std_max=log_std_max,
        price_logit_bias=float(payload.get('lcc_price_logit_bias', .5)),
        transfer='registered_zero_shot_edge_count_and_system_feature_transfer' if source_edges != edges else 'same_edge_count_new_system_features',
        feature_contract=read(scales), feature_contract_sha256=sha(scales), weights_only=True,
        evaluation_only=True, torch_version=torch.__version__)
    write_new(output, value)
    return value


def features(tasks, states, cfg, now_ns):
    """Same observable cost components for all policies; no telemetry fetch."""
    scales = cfg['feature_scales']
    if scales['fitted_role'] != 'calibration':
        raise ValueError('Feature scales must be selected using calibration only')
    m = len(cfg['edges'])
    ordered = [states[node['id']] for node in cfg['edges']]
    backlog = [max(0., x['predicted_wait_s']) for x in ordered]
    service = [x['service_estimate_s'] for x in ordered]
    age = [max(0., (now_ns - x['source_epoch_ns']) / 1e9) for x in ordered]
    count = len(tasks)
    payload = sum(x['payload_bytes'] for x in tasks)
    # Colocated host: registered transfer prediction is calibrated loopback
    # overhead, never a 20 MHz radio formula or the future request duration.
    transfer = scales['transfer_seconds_per_server']
    costs = [q + s + t for q, s, t in zip(backlog, service, transfer)]
    reference = count * scales['reference_service_s']
    load = reference / (scales['request_scale'] * scales['reference_service_s'])
    pressure = [min(1., q / scales['queue_reference_s']) for q in backlog]
    prices = [load * q + .5 * c / scales['latency_scale_s'] for q, c in zip(pressure, costs)]
    span = max(max(prices) - min(prices), 1e-8)
    normalized = [(p - min(prices)) / span for p in prices]
    price_scale = min(10., math.log1p(sum(max(0., p) for p in prices) / m))
    obs = [0., 0.] + [0.] * m + [count / scales['request_scale'],
        payload / (scales['request_scale'] * scales['mean_payload_bytes']), load] + [
        1. / (1. + c / scales['latency_scale_s']) for c in costs] + [
        sum(age) / (m * scales['age_scale_s'])] + normalized + [price_scale]
    if len(obs) != 3 * m + 7 or not all(math.isfinite(x) for x in obs):
        raise ValueError('Nonfinite or malformed actor observation')
    return obs, costs, age


class Router:
    def __init__(self, method, config, bundle=None):
        from .baselines import METHODS, FrozenBaseline
        if method not in ('round_robin', 'score_softmax', 'stale_shortest', 'qla', 'qla_no_bias_probe', 'frozen_noql') and method not in METHODS:
            raise ValueError('Unknown router')
        self.method, self.config, self.actor = method, config, None
        self.baseline = None
        self._round_robin_index = 0
        if method in METHODS:
            if bundle is None:
                raise ValueError('Registered frozen baseline bundle required')
            self.baseline = FrozenBaseline(method, config, bundle)
            self.receipt = self.baseline.receipt
        elif method in ('qla', 'qla_no_bias_probe', 'frozen_noql'):
            if bundle is None:
                raise ValueError('Frozen registered actor bundle required')
            spec = read(bundle)
            if (spec['schema'] != 'systemlab-frozen-actor-v1' or not spec['evaluation_only'] or
                    sha(spec['checkpoint']) != spec['checkpoint_sha256'] or
                    tree_hashes(spec['source_root']) != spec['source_sha256'] or
                    spec['target_edges'] != len(config['edges']) or
                    encoded(spec['feature_contract']) != encoded(config['feature_scales'])):
                raise ValueError('Checkpoint/source/target/feature contract mismatch')
            local_windows_runtime()
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(config['seed'])
            source = Path(spec['source_root']).resolve()
            loaded = sys.modules.get('mappo_gp_sf_dsr')
            if loaded and Path(loaded.__file__).resolve().parent != source / 'mappo_gp_sf_dsr':
                raise RuntimeError('Wrong historical package already imported')
            sys.path.insert(0, str(source))
            module = importlib.import_module('mappo_gp_sf_dsr.models')
            if Path(module.__file__).resolve() != source / 'mappo_gp_sf_dsr/models.py':
                raise RuntimeError('Model import provenance mismatch')
            payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
            m = len(config['edges'])
            if spec['architecture'] == 'edgewise' and method != 'frozen_noql':
                bias = 0. if method == 'qla_no_bias_probe' else spec['price_logit_bias']
                self.actor = module.EdgeWisePriceGaussianActor(3*m+7, m, spec['hidden'],
                    spec['log_std_min'], spec['log_std_max'], bias)
            elif spec['architecture'] == 'shared_gaussian' and method == 'frozen_noql':
                if spec['source_edges'] != m or spec['source_actor_obs_dim'] != 2*m+6:
                    raise ValueError('NoQL requires a genuinely trained matching unpriced/age-aware actor')
                self.actor = module.SharedGaussianActor(2*m+6, m, spec['hidden'], spec['log_std_min'], spec['log_std_max'])
            else:
                raise ValueError('Method and trained architecture disagree')
            self.actor.load_state_dict(payload['actor'], strict=True)
            self.actor.eval().requires_grad_(False)
            self.torch, self.bundle = torch, spec
            self.initial_parameter_sha256 = self.parameter_sha256()
            self.receipt = dict(method=method, bundle_sha256=sha(bundle), transfer=spec['transfer'],
                torch=torch.__version__, device='cpu', deterministic_actor=True,
                checkpoint_sha256=spec['checkpoint_sha256'], source_sha256=spec['source_sha256'])
        else:
            self.receipt = dict(method=method, deterministic_actor=True)
            if method == 'round_robin':
                self.receipt['assignment_rule'] = 'global_sequential_decision_batch_rotation_v2'

    def parameter_sha256(self):
        fingerprint=hashlib.sha256()
        for key,tensor in sorted(self.actor.state_dict().items()):
            fingerprint.update(key.encode())
            fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return fingerprint.hexdigest()

    def verify_frozen(self):
        if self.baseline is not None:
            return self.baseline.verify_frozen()
        if self.actor is None:
            return dict(actor_executed=False)
        spec=self.bundle
        after=self.parameter_sha256()
        if (after!=self.initial_parameter_sha256 or sha(spec['checkpoint'])!=spec['checkpoint_sha256'] or
                tree_hashes(spec['source_root'])!=spec['source_sha256']):
            raise RuntimeError('Frozen actor/source/checkpoint changed during execution')
        return dict(actor_executed=True,before_parameter_sha256=self.initial_parameter_sha256,
                    after_parameter_sha256=after,parameters_unchanged=True)

    def portfolio(self, tasks, states, now_ns, user_id):
        obs, costs, ages = features(tasks, states, self.config, now_ns)
        m = len(costs)
        baseline_details = {}
        if self.baseline is not None:
            weights, baseline_details = self.baseline.portfolio(tasks, states, now_ns, user_id)
        elif self.method == 'round_robin':
            # A user-id mapping strands edges when users < edges. Calibration
            # rotates across decision batches, not per user or per task. The
            # replay loop calls portfolio serially; all tasks in a batch share
            # its destination. Historical source/results remain unchanged.
            target = self._round_robin_index % m
            self._round_robin_index += 1
            weights = [float(j == target) for j in range(m)]
        elif self.method == 'stale_shortest':
            target = min(range(m), key=lambda j: (costs[j], j))
            weights = [float(j == target) for j in range(m)]
        elif self.method == 'score_softmax':
            logits = [-x / self.config['score_temperature_s'] for x in costs]
            weights = [math.exp(x - max(logits)) for x in logits]
            total = sum(weights)
            weights = [x / total for x in weights]
        else:
            with self.torch.inference_mode():
                x = obs[:2*m+6] if self.method == 'frozen_noql' else obs
                tensor = self.torch.tensor([x], dtype=self.torch.float32)
                latent = self.actor.distribution(tensor).mean
                weights = self.actor.portfolio(latent, tensor)[0].tolist()
        if any(not math.isfinite(x) or x < 0 for x in weights) or abs(sum(weights)-1.) > 1e-5:
            raise RuntimeError('Invalid portfolio')
        return weights, dict(actor_observation=obs, expected_costs_s=costs, feedback_age_s=ages, **baseline_details)

    def observe_released(self, counts):
        if self.baseline is not None:
            self.baseline.observe_released(counts)

    def observe_outcome(self, value):
        if self.baseline is not None:
            self.baseline.observe_outcome(value)

    def observe_cache(self, cache, now_ns):
        if self.baseline is not None:
            self.baseline.observe_cache(cache, now_ns)


def sampled_index(weights, uniform):
    if not 0 <= uniform < 1:
        raise ValueError('Common assignment draw must be in [0,1)')
    cumulative = 0.
    for i, weight in enumerate(weights):
        cumulative += weight
        if uniform < cumulative:
            return i
    return len(weights) - 1
