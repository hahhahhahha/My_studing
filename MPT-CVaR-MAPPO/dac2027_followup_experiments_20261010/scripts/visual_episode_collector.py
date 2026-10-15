"""Prepared neural episode collector, with separately testable pure assembly.

No model factory or training launcher is called here. The real collector
requires the actual CUDA controller; artificial samples are accepted only by
the explicitly named lower-level engineering assembly surface, never as a
substitute controller or formal policy result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

from visual_learning_buffer import VisualActionSample, VisualEpisodeBuffer, PreparedVisualEpisode
from visual_mean_variance_reward import finalize_mean_variance_rewards
from visual_method_features import PackedVisualFeatures, VisualFeatureScales, pack_features
from visual_neural_controller import NeuralVisualController
from visual_receipt_environment import ControllerDecisionObservation, VisualReceiptEnvironment
from visual_shared_reward import VisualSharedReward
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger, finalize_training_rewards

BASE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EpisodeClosureCandidate:
    fixed_drain_seconds: float = 30.
    gamma_per_second: float = .99
    lambda_per_second: float = .95
    reward_v: float = .5
    migration_weight: float = .05
    concentration_factor: float = 2.
    drop_penalty_seconds: float = 1.
    unfinished_penalty_seconds: float = 1.
    terminal_weight: float = 1.
    mv_mu_seconds: float = 1.5

    def __post_init__(self):
        positive = (self.fixed_drain_seconds, self.drop_penalty_seconds, self.unfinished_penalty_seconds)
        nonnegative = (self.reward_v, self.migration_weight, self.concentration_factor, self.terminal_weight)
        if (not all(math.isfinite(x) and x > 0 for x in positive)
                or not all(math.isfinite(x) and x >= 0 for x in nonnegative)
                or not all(math.isfinite(x) and 0 < x <= 1 for x in
                           (self.gamma_per_second, self.lambda_per_second))
                or not math.isfinite(self.mv_mu_seconds)):
            raise ValueError('Explicit finite candidate closure/discount settings required')


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_evidence(path, value):
    # Generated execution evidence, not editing old source/measurement files.
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class EpisodeAssembly:
    """No controller/model: joins validated samples to actual FIFO receipts.

    Passing literal samples here is an engineering fixture, NOT an actual
    neural behavior claim. A real model call is enforced by the separate
    RealLearningEpisodeCollector entry below.
    """
    def __init__(self, method, users, servers, decisions, scales, settings):
        if (method not in ('QLA-MAPPO', 'PPO-MV-A') or not isinstance(scales, VisualFeatureScales)
                or not isinstance(settings, EpisodeClosureCandidate) or servers != len(scales.profiles)
                or settings.fixed_drain_seconds < scales.decision_seconds):
            raise ValueError('Declared method/scales and fixed drain covering final interval required')
        semantics = 'QLA-shared-receipt' if method == 'QLA-MAPPO' else 'PPO-MV-A'
        self.method, self.scales, self.settings = method, scales, settings
        self.users, self.servers, self.decisions = users, servers, decisions
        actor_dim = 3 * servers + 7
        critic_dim = users * (actor_dim + servers) + 3 * servers + 3
        self.buffer = VisualEpisodeBuffer(users, servers, actor_dim, critic_dim, decisions, semantics)
        self.rewarder = VisualSharedReward(scales, settings.reward_v, settings.migration_weight,
                                          settings.concentration_factor, settings.drop_penalty_seconds)
        self.observations, self.transitions, self.identities = [], [], []
        self._environment = None
        self.closed, self.failed = False, False

    def features_for(self, pre):
        if self.failed or self.closed:
            raise RuntimeError('Failed/closed assembly cannot continue')
        return pack_features(pre, self.scales, prior_h_queue=self.rewarder.h_queue, v_weight=self.settings.reward_v)

    def execute_validated_sample(self, environment, pre, packed, sample):
        if self.failed or self.closed:
            raise RuntimeError('Failed/closed assembly cannot append or retry')
        try:
            if (type(environment) is not VisualReceiptEnvironment or type(pre) is not ControllerDecisionObservation
                    or type(packed) is not PackedVisualFeatures or not isinstance(sample, VisualActionSample)
                    or environment._pending is not pre or pre.index != len(self.transitions) + 1
                    or len(pre.users) != self.users or environment._schedule.decisions != self.decisions
                    or environment._schedule.decision_seconds != self.scales.decision_seconds
                    or environment._profiles != self.scales.profiles
                    or environment._edges != self.scales.edge_positions
                    or (self._environment is not None and environment is not self._environment)):
                raise ValueError('The current untouched pre-action environment observation is required')
            expected = self.features_for(pre)
            if any(not np.array_equal(getattr(packed, name), getattr(expected, name)) for name in
                   ('actor', 'critic', 'q_pressure', 'expected_latency_seconds',
                    'reference_demand_seconds', 'previous_portfolios')):
                raise ValueError('Features must be packed from this pre-action observation/prior state')
            self._environment = environment
            critic_sha = hashlib.sha256(packed.critic.tobytes()).hexdigest()
            end = float((pre.index + 1) * Fraction(str(self.scales.decision_seconds)))
            # Retain sampled float32 behavior BEFORE execution; do not replace
            # it with a post-action recomputation or discard earlier samples.
            self.buffer.add_action(pre.index, pre.physical_seconds, end, packed, sample)
            self.observations.append(asdict(pre))
            self.identities.extend(IssuedIdentity(request.task_id, user.user, request.arrival_seconds)
                                   for user in pre.users for request in user.requests)
            # A numerical representation bridge only: unit-sum float64 for
            # the original strict execution adapter. Record BOTH forms and
            # require the latent/portfolio agreement via the buffer guard.
            executed = sample.portfolios.astype(np.float64)
            executed /= executed.sum(axis=1, keepdims=True)
            environment.apply(executed)
            outcomes = environment.settle_action()
            if not math.isclose(outcomes.end_seconds, end, rel_tol=0., abs_tol=1e-12):
                raise RuntimeError('Actual settlement does not match the stored behavior interval')
            reward = self.rewarder.compute(pre, executed, outcomes)
            if critic_sha != hashlib.sha256(packed.critic.tobytes()).hexdigest():
                raise RuntimeError('Current pre-action critic changed after executing the action')
            transition = {'index': pre.index, 'time_seconds': pre.physical_seconds,
                'end_seconds': outcomes.end_seconds, 'pre_critic_sha256': critic_sha,
                'sampled_float32_portfolios': sample.portfolios.tolist(), 'portfolios': executed.tolist(),
                'outcome_batch': asdict(outcomes), 'reward': jsonable(reward)}
            self.transitions.append(transition)
            return transition
        except Exception:
            self.failed = True
            raise

    def close_after_fixed_drain(self, environment):
        if self.failed or self.closed or len(self.transitions) != self.decisions:
            raise RuntimeError('Exactly one full episode may be closed; no premature/repeated drain')
        if environment is not self._environment:
            raise ValueError('The same episode environment must supply its actual fixed-drain outcomes')
        self.closed = True  # even a partial close must never auto-retry/extend
        try:
            final = jsonable(environment.finish(self.settings.fixed_drain_seconds))
            end = environment._schedule.horizon_seconds + self.settings.fixed_drain_seconds
            ledger = build_closed_ledger(final, self.identities, end, self.users,
                self.settings.drop_penalty_seconds, self.settings.unfinished_penalty_seconds)
            if self.method == 'QLA-MAPPO':
                closed = finalize_training_rewards(self.transitions, ledger, self.scales.latency_scale_seconds,
                    self.settings.reward_v, self.settings.migration_weight, self.settings.terminal_weight)
            else:
                closed = finalize_mean_variance_rewards(self.transitions, ledger, self.identities, self.users,
                    self.settings.mv_mu_seconds, self.settings.drop_penalty_seconds)
            prepared = self.buffer.prepare_closed_episode(closed, self.settings.gamma_per_second,
                                                           self.settings.lambda_per_second)
            return {'final': final, 'ledger': ledger, 'ordinary_transitions': self.transitions,
                    'closed_transitions': closed, 'observations': self.observations,
                    'prepared_episode': prepared, 'candidate_settings': asdict(self.settings),
                    'MV_statistic_reset_rule': 'Fresh statistic per closed episode; provisional visual adaptation, not frozen training design'}
        except Exception:
            self.failed = True
            raise


def require_actual_cuda_controller(controller):
    if (type(controller) is not NeuralVisualController or getattr(controller, '_failed', True)
            or getattr(controller, '_torch', None) is None
            or controller._torch is not sys.modules.get('torch')
            or getattr(controller, 'method', None) not in ('QLA-MAPPO', 'PPO-MV-A')):
        raise TypeError('Initialized actual historical CUDA controller required; fixtures/stubs/CPU refused')


class RealLearningEpisodeCollector:
    """SOURCE PREPARED ONLY: actual constructor/collect never run in CPU checks.

    Collecting does NOT update parameters. The future reviewed trainer must
    update only the returned complete prepared episode, then use a new
    environment/collector for the next budget segment. No run/resume entry is
    provided until the pilot, actual model smoke and independent review pass.
    """
    def __init__(self, controller, scales, settings=None):
        require_actual_cuda_controller(controller)
        if not isinstance(scales, VisualFeatureScales):
            raise TypeError('Frozen calibration-derived scales required')
        self.controller, self.scales = controller, scales
        self.settings = settings or EpisodeClosureCandidate()
        if not isinstance(self.settings, EpisodeClosureCandidate):
            raise TypeError('Explicit supported episode closure settings required')
        self.started = False

    def collect_training_episode(self, environment, new_attempt_directory):
        if self.started:
            raise RuntimeError('Collector is one-shot; no append/retry')
        require_actual_cuda_controller(self.controller)
        if (type(environment) is not VisualReceiptEnvironment or environment._catalog.role != 'calibration'
                or environment._index != 0 or environment._pending is not None
                or environment._failed or environment._finished):
            raise ValueError('New calibration-role episode required; validation/test cannot train')
        path = Path(new_attempt_directory).resolve()
        root = (BASE / 'visual_calibrated_policy').resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError('New evidence directory must be inside the visual experiment output')
        path.mkdir(parents=True, exist_ok=False)  # partial evidence cannot be reused
        self.started = True
        assembly = None
        write_evidence(path / 'STARTED.json', {'at': datetime.now().astimezone().isoformat(),
            'method': self.controller.method, 'schedule_sha256': environment._schedule.fingerprint(),
            'role': 'calibration', 'settings': asdict(self.settings), 'optimizer_update_called': False})
        try:
            assembly = EpisodeAssembly(self.controller.method, environment._schedule.users,
                len(environment._profiles), environment._schedule.decisions, self.scales, self.settings)
            for _ in range(environment._schedule.decisions):
                pre = environment.observe()
                packed = assembly.features_for(pre)
                # The actual controller receives ONLY packed pre-action features.
                sample = self.controller.act(packed, deterministic=False)
                assembly.execute_validated_sample(environment, pre, packed, sample)
                write_evidence(path / 'STATE.json', {'phase': 'collecting',
                    'decisions_collected': len(assembly.transitions), 'optimizer_update_called': False})
            result = assembly.close_after_fixed_drain(environment)
            prepared = result['prepared_episode']
            reporting = {key: value for key, value in result.items() if key != 'prepared_episode'}
            write_evidence(path / 'CLOSED_EPISODE.json', reporting)
            np.savez_compressed(path / 'BEHAVIOR_AND_GAE.npz', **prepared.time_user_fields,
                advantages=prepared.advantages, returns=prepared.returns, physical_durations=prepared.physical_durations)
            write_evidence(path / 'STATE.json', {'phase': 'collected_closed_not_updated',
                'decisions_collected': len(assembly.transitions), 'episode_token': prepared.episode_token,
                'reward_semantics': prepared.reward_semantics, 'optimizer_update_called': False})
            return result
        except Exception as error:
            write_evidence(path / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(),
                'error': repr(error), 'observations': assembly.observations if assembly else [],
                'ordinary_transitions': assembly.transitions if assembly else [],
                'retained_behavior_rows': assembly.buffer._rows if assembly else [],
                'optimizer_update_called': False, 'automatic_retry_allowed': False})
            raise
