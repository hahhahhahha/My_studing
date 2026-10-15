"""Frozen baseline inference adapters. No optimizer, training or simulator."""
from __future__ import annotations

import ast
import hashlib
import math
import threading
from collections import deque
from pathlib import Path, PurePosixPath, WindowsPath
from types import SimpleNamespace

import numpy as np

from .common import SERVER_RESPONSE_TIMEOUT, encoded, local_windows_runtime, read, sha, source_receipt, write_new

METHODS = {'ppo_mv': 'PPO-MV-A', 'ppo_dsr': 'PPO-DSR-A', 'mfc_di': 'MFC-DI-A',
           'qeco': 'QECO-A', 'jseq': 'JSEQ-A', 'lyfoto_lc': 'LyFOTO-LC'}


def asset_path(root, relative):
    rel = PurePosixPath(relative)
    path = (root / str(rel)).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
        raise ValueError('Unsafe baseline asset path')
    return path


def asset_identity(root):
    root = Path(root).resolve()
    manifest_path = root / 'ASSET_MANIFEST.json'
    manifest = read(manifest_path)
    paths, seen = {str(manifest_path): sha(manifest_path)}, set()
    for entry in manifest['files']:
        path = asset_path(root, entry['path'])
        if str(path) in seen or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Baseline asset membership/bytes/SHA mismatch')
        seen.add(str(path))
        paths[str(path)] = entry['sha256']
    catalog = read(root / 'ASSET_CATALOG.json')
    if (catalog['users'], catalog['edges']) != (30, 35) or len(catalog['methods']) != 6:
        raise ValueError('Unexpected baseline inventory')
    package = hashlib.sha256()
    for path in sorted((root / 'legacy_source/mappo_gp_sf_dsr').glob('*.py')):
        if str(path.resolve()) not in paths:
            raise ValueError('Extra executable producer source')
        package.update(path.name.encode())
        package.update(path.read_bytes())
    if package.hexdigest() != catalog['producer_package_sha256']:
        raise ValueError('Producer package lock mismatch')
    return catalog, paths


def register_baseline(method, config_path, asset_root, output):
    if method not in METHODS:
        raise ValueError('Unknown original baseline')
    config = read(config_path)
    if config['users'] != 30 or len(config['edges']) != 35:
        raise ValueError('Original checkpoint deployment requires exactly 30 users / 35 edges')
    root = Path(asset_root).resolve()
    catalog, files = asset_identity(root)
    matches = [x for x in catalog['methods'] if x['method'] == METHODS[method]]
    if len(matches) != 1:
        raise ValueError('Baseline method identity missing/duplicated')
    entry = matches[0]
    raw = read(asset_path(root, entry['config']))
    cli = raw.get('cli', raw)
    if (cli['users'], cli['n_edges']) != (30, 35):
        raise ValueError('Training configuration geometry mismatch')
    write_new(output, dict(schema='systemlab-frozen-baseline-v1', method=method,
        paper_method=METHODS[method], asset_root=str(root), asset_sha256=files,
        entry=entry, cli=cli, target_users=30, target_edges=35,
        feature_contract=config['feature_scales'], source_sha256=source_receipt(),
        adapter_protocol='real_cached_baselines_v1', evaluation_only=True,
        formal_entry_approved=False, training_calls=0))


def validate_bundle(method, config, bundle):
    spec = read(bundle)
    if (spec.get('schema') != 'systemlab-frozen-baseline-v1' or spec['method'] != method or
            not spec['evaluation_only'] or spec['adapter_protocol'] != 'real_cached_baselines_v1' or
            spec['target_users'] != config['users'] or spec['target_edges'] != len(config['edges']) or
            encoded(spec['feature_contract']) != encoded(config['feature_scales']) or
            spec['source_sha256'] != source_receipt()):
        raise ValueError('Baseline method/source/geometry/features changed; register a fresh bundle')
    catalog, files = asset_identity(spec['asset_root'])
    matches = [x for x in catalog['methods'] if x['method'] == METHODS[method]]
    raw = read(asset_path(Path(spec['asset_root']), spec['entry']['config']))
    if (files != spec['asset_sha256'] or matches != [spec['entry']] or
            encoded(raw.get('cli', raw)) != encoded(spec['cli'])):
        raise ValueError('Baseline asset/config identity changed')
    return spec


def norm_max(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return np.zeros_like(values) if maximum <= 1e-8 else values / (maximum + 1e-8)


def cached_features(tasks, states, config, cli):
    ordered = [states[x['id']] for x in config['edges']]
    wait = np.asarray([max(0., x['predicted_wait_s']) for x in ordered])
    service = np.asarray([x['service_estimate_s'] for x in ordered], dtype=np.float64)
    count = np.asarray([x['waiting_count'] + int(x['busy']) for x in ordered], dtype=np.float64)
    if np.any(service <= 0) or not np.all(np.isfinite(service)):
        raise ValueError('Invalid cached service estimates')
    capacity = 1. / service  # tasks/second; no simulated MB or radio bandwidth.
    slot = float(cli['slot_s'])
    available = np.maximum(0., slot - wait) * capacity
    task_norm, wait_norm = norm_max(count), norm_max(wait)
    score = (float(cli['score_resource_weight']) * norm_max(available) +
             float(cli['score_capacity_weight']) * norm_max(capacity) -
             float(cli['score_task_count_weight']) * task_norm -
             float(cli['score_wait_weight']) * wait_norm)
    max_requests = float(cli['max_requests_per_user'])
    request = [len(tasks) / max_requests,
               sum(x['payload_bytes'] for x in tasks) / (max_requests * config['feature_scales']['mean_payload_bytes']),
               len(tasks) / max_requests]
    # All endpoints are colocated: normalized location/distance are explicitly 0.
    observation = np.asarray([0., 0.] + [0.] * len(ordered) + request + score.tolist(), dtype=np.float32)
    if len(observation) != 75 or not np.all(np.isfinite(observation)):
        raise ValueError('Legacy unpriced observation must have 75 finite entries')
    return observation, wait, service, task_norm, wait_norm


def original_definitions(root, filename, names, torch):
    """Compile only original inference definitions, not env/training imports."""
    path = root / 'legacy_source/mappo_gp_sf_dsr' / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if any(name not in definitions for name in names):
        raise ValueError('Original inference definition missing')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + [definitions[name] for name in names], type_ignores=[]))
    namespace = {'torch': torch, 'nn': torch.nn if torch is not None else None, 'np': np, 'EPS': 1e-8}
    if torch is not None:
        namespace['Normal'] = torch.distributions.Normal
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


class FrozenBaseline:
    def __init__(self, method, config, bundle):
        self.spec = validate_bundle(method, config, bundle)  # BEFORE torch import/model construction.
        self.bundle_path, self.bundle_sha = Path(bundle), sha(bundle)
        self.method, self.config, self.cli = method, config, self.spec['cli']
        self.root = Path(self.spec['asset_root'])
        self.models, self.lock = [], threading.RLock()
        self.released_counts = {}
        self.arrivals = {edge['id']: [] for edge in config['edges']}
        self.histories, self.history_times = {}, {}
        self.max_history_gap_s = 0.
        self.rng = np.random.default_rng(int(self.cli['seed']) * 7919 + 17)
        self.definitions = {}
        if method in ('ppo_mv', 'ppo_dsr', 'mfc_di', 'qeco'):
            local_windows_runtime()
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(int(self.cli['seed']))
            self.torch = torch
            with torch.serialization.safe_globals([WindowsPath]):
                payload = torch.load(asset_path(self.root, self.spec['entry']['checkpoint']), map_location='cpu', weights_only=True)
            if method in ('ppo_mv', 'ppo_dsr'):
                definitions = original_definitions(self.root, 'paper_policy_baselines.py', ['mlp', 'SharedGaussianPPO'], torch)
                model = definitions['SharedGaussianPPO'](75, 35, log_std_min=float(self.cli['log_std_min']))
                model.load_state_dict(payload, strict=True)
                self.models = [model]
            elif method == 'mfc_di':
                definitions = original_definitions(self.root, 'delayed_information_baselines.py', ['MFCActor'], torch)
                d, bins = int(self.cli['mfc_d']), int(self.cli['mfc_queue_bins'])
                model = definitions['MFCActor'](bins + 2*d + 2, d, int(self.cli['mfc_hidden']))
                model.load_state_dict(payload['actor'], strict=True)
                self.models = [model]
            else:
                definitions = original_definitions(self.root, 'recent_mec_baselines.py', ['QECONetwork', '_qeco_action_portfolios'], torch)
                if len(payload['online_models']) != 30:
                    raise ValueError('Never replicate or drop independent QECO user models')
                for state in payload['online_models']:
                    model = definitions['QECONetwork'](75, 35, 35, int(self.cli['qeco_hidden']), int(self.cli['qeco_lstm_hidden']))
                    model.load_state_dict(state, strict=True)
                    self.models.append(model)
                self.definitions = definitions
            for model in self.models:
                model.eval().requires_grad_(False)
        elif method == 'lyfoto_lc':
            # Pure original routing functions; torch only satisfies annotations and
            # is not imported/used by the function bodies for this rule baseline.
            self.definitions = original_definitions(self.root, 'recent_mec_baselines.py',
                ['_softmax_negative', '_lyapunov_local_stale_portfolios'], None)
        self.initial_parameters = self.parameter_sha256()
        self.receipt = dict(method=method, paper_method=METHODS[method], bundle_sha256=sha(bundle),
            checkpoint_sha256=self.spec['entry'].get('checkpoint_sha256'),
            adapter_protocol=self.spec['adapter_protocol'], deterministic_actor=True, device='cpu',
            inference_only=True, optimizer_calls=0, distinct_qeco_user_models=len(self.models) if method == 'qeco' else None)

    def parameter_sha256(self):
        h = hashlib.sha256()
        for index, model in enumerate(self.models):
            for key, tensor in sorted(model.state_dict().items()):
                h.update(f'{index}:{key}'.encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def observe_released(self, counts):
        with self.lock:
            self.released_counts = dict(counts)

    def observe_outcome(self, value):
        if self.method != 'jseq' or value.get('status') not in ('completed', SERVER_RESPONSE_TIMEOUT):
            return
        # Accepted work becomes known ONLY when its own HTTP response arrives.
        # Send epoch is an explicitly approximate admission time, not private telemetry.
        with self.lock:
            self.arrivals[value['server_id']].append((value['send_epoch_ns'], value['predicted_service_s']))

    def observe_cache(self, cache, now_ns):
        if self.method != 'qeco':
            return
        with self.lock:
            interval = int(float(self.cli['slot_s']) * 1e9)
            for user in range(30):
                if user in self.history_times and now_ns - self.history_times[user][-1] < interval:
                    continue
                states = cache.observation(user)
                if len(states) != 35:
                    continue
                values = np.asarray([states[e['id']]['waiting_count'] + int(states[e['id']]['busy']) for e in self.config['edges']])
                workload = norm_max(values).astype(np.float32)
                if user not in self.histories:
                    self.histories[user] = deque([workload.copy() for _ in range(int(self.cli['history_steps']))], maxlen=int(self.cli['history_steps']))
                    self.history_times[user] = deque([now_ns] * int(self.cli['history_steps']), maxlen=int(self.cli['history_steps']))
                else:
                    self.max_history_gap_s = max(self.max_history_gap_s, (now_ns-self.history_times[user][-1])/1e9)
                    self.histories[user].append(workload)
                    self.history_times[user].append(now_ns)

    def portfolio(self, tasks, states, now_ns, user):
        if not 0 <= user < 30 or any(int(x['source_epoch_ns']) > now_ns or
                int(x.get('received_epoch_ns', now_ns)) > now_ns for x in states.values()):
            raise ValueError('Invalid user or future feedback')
        obs, wait, service, workload, wait_norm = cached_features(tasks, states, self.config, self.cli)
        transfer = np.asarray(self.config['feature_scales']['transfer_seconds_per_server'])
        costs = wait + service + transfer
        details = {'baseline_observation': obs.tolist(), 'baseline_adapter_protocol': 'real_cached_baselines_v1'}
        with self.lock:
            if self.method in ('ppo_mv', 'ppo_dsr'):
                with self.torch.inference_mode():
                    x = self.torch.tensor(obs[None], dtype=self.torch.float32)
                    weights = self.torch.softmax(self.models[0].distribution(x).mean, dim=-1)[0].tolist()
            elif self.method == 'mfc_di':
                d, bins = int(self.cli['mfc_d']), int(self.cli['mfc_queue_bins'])
                indices = np.minimum((np.clip(wait_norm, 0, 1) * bins).astype(int), bins-1)
                histogram = np.bincount(indices, minlength=bins).astype(np.float32) / 35.
                candidates = self.rng.choice(35, size=d, replace=False)
                released = sum(self.released_counts.values())
                features = np.concatenate([histogram, wait_norm[candidates], costs[candidates]/float(self.cli['mfc_latency_norm']),
                    [len(tasks)/float(self.cli['max_requests_per_user']), released/(30.*float(self.cli['max_requests_per_user']))]]).astype(np.float32)
                with self.torch.inference_mode():
                    local = self.torch.softmax(self.models[0].distribution(self.torch.tensor(features[None])).mean, dim=-1)[0].tolist()
                weights = [0.] * 35
                for edge, probability in zip(candidates, local):
                    weights[int(edge)] = probability
                details.update(mfc_features=features.tolist(), mfc_candidates=candidates.tolist(), released_joint_tasks=released)
            elif self.method == 'qeco':
                if user not in self.histories:
                    raise RuntimeError('QECO initial delivered-feedback history not ready')
                history = np.asarray(self.histories[user], dtype=np.float32)[None]
                proxy = SimpleNamespace(feedback_delay=0, feedback_buffer=SimpleNamespace(
                    get=lambda ignored: SimpleNamespace(wait_norm=wait_norm, queue_task_count_norm=workload)))
                _, portfolios = self.definitions['_qeco_action_portfolios'](
                    [SimpleNamespace(online=self.models[user])], obs[None], history, proxy, 1., self.rng,
                    self.torch.device('cpu'), True, int(self.cli['qeco_topk']), float(self.cli['qeco_topk_temperature']),
                    bool(self.cli['qeco_load_mask']), float(self.cli['qeco_mask_quantile']))
                weights = portfolios[0].tolist()
                details.update(qeco_user_model=user, qeco_history_sample_epoch_ns=list(self.history_times[user]))
            elif self.method == 'jseq':
                estimate = wait.copy()
                for j, edge in enumerate(self.config['edges']):
                    source = int(states[edge['id']]['source_epoch_ns'])
                    previous = source
                    for sent, work in sorted(self.arrivals[edge['id']]):
                        if source < sent <= now_ns:
                            estimate[j] = max(0., estimate[j] - (sent-previous)/1e9) + work
                            previous = sent
                    estimate[j] = max(0., estimate[j] - (now_ns-previous)/1e9)
                predicted = estimate + service + transfer
                scale = max(float(np.std(predicted)), 1e-3)
                logits = -predicted / (max(float(self.cli['jseq_temperature']), 1e-4) * scale)
                value = np.exp(np.clip(logits-np.max(logits), -60, 60))
                weights = (value/value.sum()).tolist()
                details['jseq_estimated_wait_s'] = estimate.tolist()
            else:
                proxy = SimpleNamespace(
                    feedback_delay=0, feedback_buffer=SimpleNamespace(get=lambda ignored: SimpleNamespace(wait_s=wait)),
                    cfg=SimpleNamespace(n_users=1, n_edges=35, slot_s=float(self.cli['slot_s']), task_compute_mb=float(self.cli['task_compute_mb'])),
                    edge_capacities_mb_s=float(self.cli['task_compute_mb'])/service,
                    traces=SimpleNamespace(edge_positions=np.zeros((35, 2))),
                    _locations=lambda: np.zeros((1, 2)), _transmission_delay_s=lambda loc, j: float(transfer[j]))
                weights = self.definitions['_lyapunov_local_stale_portfolios'](proxy,
                    float(self.cli['lyapunov_local_v']), float(self.cli['lyapunov_local_temperature']),
                    int(self.cli['lyapunov_local_candidates']))[0].tolist()
                details['colocated_candidate_ties'] = True
        if len(weights) != 35 or any(not math.isfinite(x) or x < 0 for x in weights) or abs(sum(weights)-1.) > 1e-5:
            raise RuntimeError('Invalid frozen baseline portfolio')
        return weights, details

    def verify_frozen(self):
        after = self.parameter_sha256()
        _, files = asset_identity(self.root)
        if (after != self.initial_parameters or files != self.spec['asset_sha256'] or
                sha(self.bundle_path) != self.bundle_sha or self.spec['source_sha256'] != source_receipt()):
            raise RuntimeError('Frozen baseline parameters/assets changed')
        return dict(actor_executed=bool(self.models), parameters_unchanged=True,
                    before_parameter_sha256=self.initial_parameters, after_parameter_sha256=after,
                    optimizer_calls=0, max_qeco_history_sample_gap_s=self.max_history_gap_s)
