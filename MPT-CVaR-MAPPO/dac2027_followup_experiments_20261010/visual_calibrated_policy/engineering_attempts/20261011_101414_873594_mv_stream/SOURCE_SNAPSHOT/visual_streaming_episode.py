"""Additive persistent-MV episode assembly and prepared actual collector.

Previous per-episode-reset source/evidence stay unchanged. Real constructor,
collect, model and optimizer are not executed by the arithmetic checks.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np

from visual_episode_collector import (BASE, EpisodeAssembly, RealLearningEpisodeCollector,
    require_actual_cuda_controller, jsonable, write_evidence)
from visual_mv_stream import PersistentMVStream
from visual_receipt_environment import VisualReceiptEnvironment
from visual_terminal_accounting import build_closed_ledger


class StreamingEpisodeAssembly(EpisodeAssembly):
    def __init__(self, method, users, servers, decisions, scales, settings,
                 mv_stream=None, episode_index=None, episode_key=None):
        super().__init__(method, users, servers, decisions, scales, settings)
        if method == 'PPO-MV-A':
            if (type(mv_stream) is not PersistentMVStream or mv_stream.users != users
                    or mv_stream.failed or mv_stream.state().mu_seconds != settings.mv_mu_seconds
                    or type(episode_index) is not int or episode_index != mv_stream.next_episode_index
                    or not isinstance(episode_key, str) or not episode_key.strip()):
                raise ValueError('Explicit matching ordered persistent MV stream required')
        elif mv_stream is not None:
            raise ValueError('QLA shared objective cannot consume an MV statistic')
        self.mv_stream, self.episode_index, self.episode_key = mv_stream, episode_index, episode_key

    def close_after_fixed_drain(self, environment):
        if self.method == 'QLA-MAPPO':
            return super().close_after_fixed_drain(environment)
        if self.failed or self.closed or len(self.transitions) != self.decisions:
            raise RuntimeError('One full ordered episode required before persistent MV closure')
        if environment is not self._environment:
            raise ValueError('The same episode environment must supply its past outcomes')
        expected_role = 'calibration' if self.mv_stream.purpose == 'training' else 'validation'
        if environment._catalog.role != expected_role:
            raise ValueError('Engineering validation statistic cannot enter real training, or vice versa')
        self.closed = True
        try:
            final = jsonable(environment.finish(self.settings.fixed_drain_seconds))
            end = environment._schedule.horizon_seconds + self.settings.fixed_drain_seconds
            ledger = build_closed_ledger(final, self.identities, end, self.users,
                self.settings.drop_penalty_seconds, self.settings.unfinished_penalty_seconds)
            result = self.mv_stream.close_episode(self.episode_index, self.episode_key, expected_role,
                self.transitions, ledger, self.identities, self.settings.drop_penalty_seconds)
            closed = result['closed_transitions']
            prepared = self.buffer.prepare_closed_episode(closed, self.settings.gamma_per_second,
                                                           self.settings.lambda_per_second)
            return {'final': final, 'ledger': ledger, 'ordinary_transitions': self.transitions,
                    'closed_transitions': closed, 'observations': self.observations,
                    'prepared_episode': prepared, 'candidate_settings': asdict(self.settings),
                    'MV_stream': {key: value for key, value in result.items() if key != 'closed_transitions'},
                    'MV_statistic_reset_rule': 'No reset across ordered episodes; shared ascending-user historical statistic',
                    'past_receipt_and_terminal_input_adaptation_is_not_exact_old_slot_input': True}
        except Exception as error:
            self.failed = True
            self.mv_stream.mark_failed(error)
            raise


class RealStreamingEpisodeCollector(RealLearningEpisodeCollector):
    """Source prepared only; constructor/collect have never been run on models.

    One persistent training stream is supplied by the future trainer to each
    PPO episode. The old collector never silently changes its reset rule.
    """
    def __init__(self, controller, scales, settings=None, mv_stream=None,
                 episode_index=None, episode_key=None):
        super().__init__(controller, scales, settings)
        if controller.method == 'PPO-MV-A':
            if type(mv_stream) is not PersistentMVStream or mv_stream.purpose != 'training' or mv_stream.failed:
                raise ValueError('Actual PPO training needs its own training-purpose persistent statistic')
            if mv_stream.state().mu_seconds != self.settings.mv_mu_seconds:
                raise ValueError('Fixed historical MV mu mismatch')
        elif mv_stream is not None:
            raise ValueError('QLA cannot consume a PPO MV stream')
        self.mv_stream, self.episode_index, self.episode_key = mv_stream, episode_index, episode_key

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
            raise ValueError('New episode evidence must stay inside the visual output directory')
        path.mkdir(parents=True, exist_ok=False)
        self.started = True
        assembly = None
        write_evidence(path / 'STARTED.json', {'at': datetime.now().astimezone().isoformat(),
            'method': self.controller.method, 'schedule_sha256': environment._schedule.fingerprint(),
            'role': 'calibration', 'settings': asdict(self.settings), 'optimizer_update_called': False,
            'MV_state_before': self.mv_stream.state() if self.mv_stream is not None else None})
        try:
            assembly = StreamingEpisodeAssembly(self.controller.method, environment._schedule.users,
                len(environment._profiles), environment._schedule.decisions, self.scales, self.settings,
                self.mv_stream, self.episode_index, self.episode_key)
            for _ in range(environment._schedule.decisions):
                pre = environment.observe()
                packed = assembly.features_for(pre)
                sample = self.controller.act(packed, deterministic=False)
                assembly.execute_validated_sample(environment, pre, packed, sample)
                write_evidence(path / 'STATE.json', {'phase': 'collecting',
                    'decisions_collected': len(assembly.transitions), 'optimizer_update_called': False})
            result = assembly.close_after_fixed_drain(environment)
            prepared = result['prepared_episode']
            write_evidence(path / 'CLOSED_EPISODE.json', {key: value for key, value in result.items()
                                                        if key != 'prepared_episode'})
            np.savez_compressed(path / 'BEHAVIOR_AND_GAE.npz', **prepared.time_user_fields,
                advantages=prepared.advantages, returns=prepared.returns, physical_durations=prepared.physical_durations)
            write_evidence(path / 'STATE.json', {'phase': 'collected_closed_not_updated',
                'decisions_collected': len(assembly.transitions), 'episode_token': prepared.episode_token,
                'reward_semantics': prepared.reward_semantics, 'optimizer_update_called': False,
                'MV_state_after': self.mv_stream.state() if self.mv_stream is not None else None})
            return result
        except Exception as error:
            if self.mv_stream is not None:
                self.mv_stream.mark_failed(error)
            write_evidence(path / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(),
                'error': repr(error), 'ordinary_transitions': assembly.transitions if assembly else [],
                'observations': assembly.observations if assembly else [],
                'retained_behavior_rows': assembly.buffer._rows if assembly else [],
                'MV_failure_evidence': self.mv_stream.failure_evidence if self.mv_stream is not None else None,
                'optimizer_update_called': False, 'automatic_retry_allowed': False})
            raise
