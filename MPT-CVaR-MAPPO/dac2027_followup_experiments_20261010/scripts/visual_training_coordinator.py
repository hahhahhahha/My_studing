"""Prepared actual train/validation orchestration, no launch or CPU fallback.

Constructors require external frozen pilot, actual model-smoke and fresh
complete Type-A entry artifacts. Those artifacts do not currently exist.
Engineering checks execute only low-level assembly and rejection surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from visual_episode_collector import (BASE, EpisodeClosureCandidate, require_actual_cuda_controller,
    jsonable, write_evidence)
from visual_learning_buffer import PreparedVisualEpisode
from visual_method_features import VisualFeatureScales, pack_features
from visual_mv_stream import PersistentMVStream
from visual_receipt_environment import VisualReceiptEnvironment
from visual_shared_reward import VisualSharedReward
from visual_streaming_episode import StreamingEpisodeAssembly
from visual_training_transactions import (TrainingTransactionLedger, CommonBudgetCandidate,
    ClosedEpisodeCounts, digest, file_sha, require_sha, write_actual_controller_checkpoint,
    validation_checkpoint_reference)


@dataclass(frozen=True)
class FrozenRuntimeArtifactBindings:
    protocol_path: str
    protocol_sha256: str
    entry_review_path: str
    entry_review_sha256: str


def _read_bound(path, sha, root):
    path = Path(path)
    if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Registered local artifact inside its intended root required')
    if file_sha(path) != require_sha(sha):
        raise RuntimeError('Registered artifact bytes changed')
    return json.loads(path.read_text(encoding='utf-8'))


def require_reviewed_runtime(bindings):
    if type(bindings) is not FrozenRuntimeArtifactBindings:
        raise TypeError('External frozen pilot and fresh complete-entry artifact bindings required')
    protocol = _read_bound(bindings.protocol_path, bindings.protocol_sha256, BASE/'protocol')
    review = _read_bound(bindings.entry_review_path, bindings.entry_review_sha256, BASE.parent/'.aris/traces')
    if (protocol.get('status') != 'frozen_after_cal_validation_only_pilot'
            or review.get('review_type') != 'Type-A' or review.get('model') != 'gpt-6-astra'
            or review.get('reasoning_effort') != 'xhigh'
            or review.get('scope') != 'complete_visual_entry_after_actual_model_smoke'
            or review.get('blockers') != [] or review.get('status') not in ('PASS', 'WARN')
            or review.get('protocol_sha256') != bindings.protocol_sha256):
        raise RuntimeError('Foundation/dry-entry review or uncompleted pilot is not formal authority')
    sources = protocol.get('source_files_sha256', {})
    required = {str(BASE/'scripts'/name) for name in ('visual_training_coordinator.py',
        'visual_training_transactions.py', 'visual_neural_controller.py', 'visual_streaming_episode.py',
        'visual_episode_collector.py', 'visual_learning_buffer.py', 'visual_temporal_gae.py',
        'visual_mean_variance_reward.py', 'visual_mv_stream.py', 'visual_terminal_accounting.py',
        'visual_method_features.py', 'visual_shared_reward.py', 'visual_receipt_environment.py',
        'visual_environment_adapter.py', 'visual_controller_receipts.py', 'predicted_fifo_feedback.py',
        'fifo_execution.py', 'visual_upload_link.py', 'visual_request_schedule.py',
        'execution_service_catalog.py', 'physical_mobility.py', 'visual_learned_bindings.py',
        'metadata_service_predictor.py', 'windows_runtime.py', 'qla_environment.py')}
    if (not required.issubset(sources) or review.get('source_files_sha256') != sources
            or not all(file_sha(Path(path)) == require_sha(value) for path, value in sources.items())):
        raise RuntimeError('Fresh exact integrated source lock required; never reuse earlier fixture certification')
    from visual_learned_bindings import source_bindings
    historical = source_bindings()
    if any(sources.get(path) != value for path, value in historical.items()):
        raise RuntimeError('Historical architecture/config/model-source bindings must also be reviewed and unchanged')
    smoke = _read_bound(protocol['actual_model_smoke_path'], protocol['actual_model_smoke_sha256'],
                        BASE/'visual_calibrated_policy')
    if (smoke.get('status') != 'actual_cuda_two_arm_smoke_pass'
            or smoke.get('methods') != ['QLA-MAPPO', 'PPO-MV-A']
            or smoke.get('source_files_sha256') != sources
            or smoke.get('actual_model_forward_and_optimizer_executed') is not True):
        raise RuntimeError('Actual reviewed two-method CUDA smoke required, not a synthetic fixture')
    service = json.loads((BASE.parent/'dac_visual_calibration_20261010_v2/protocol/RETAINED_TEST_STATE.json').read_text(encoding='utf-8'))
    if service['status'] != 'complete_audited':
        raise RuntimeError('Completed independently audited service evidence required')
    if protocol.get('virtual_H_episode_rule') not in ('carry_across_training_episodes', 'reset_each_independent_episode'):
        raise ValueError('An explicit pilot-frozen virtual-H boundary rule required')
    return protocol


def closed_episode_counts(prepared, users, decisions, decision_seconds):
    if type(prepared) is not PreparedVisualEpisode:
        raise TypeError('Own complete collected and closed episode required')
    data = prepared.time_user_fields
    expected = {'actor_obs', 'critic_obs', 'latent_action', 'log_prob', 'value', 'reward', 'user_id', 'slot', 'done'}
    if (set(data) != expected or prepared.advantages.shape != (decisions, users)
            or prepared.returns.shape != (decisions, users)
            or prepared.physical_durations.shape != (decisions,)
            or not np.allclose(prepared.physical_durations, decision_seconds, atol=1e-12, rtol=0)
            or not all(np.asarray(value).shape[:2] == (decisions, users) for value in data.values())
            or not all(np.isfinite(value).all() for value in (*data.values(), prepared.advantages, prepared.returns))
            or not np.array_equal(data['user_id'], np.tile(np.arange(users), (decisions, 1)))
            or not np.array_equal(data['slot'], np.tile(np.arange(1, decisions+1)[:, None], (1, users)))
            or not np.asarray(data['done'][-1]).all() or np.asarray(data['done'][:-1]).any()):
        raise ValueError('Exactly complete T×N behavior and separate-user physical GAE required')
    return ClosedEpisodeCounts(prepared.episode_token, decisions, users, decisions*users,
        float(np.sum(prepared.physical_durations)), prepared.reward_semantics)


class CompactCoordinatorAssembly(StreamingEpisodeAssembly):
    """Low-level engineering surface, never a substitute for a real actor."""
    def __init__(self, *args, initial_H=0., **kwargs):
        super().__init__(*args, **kwargs)
        if type(initial_H) not in (int, float) or not math.isfinite(initial_H) or initial_H < 0:
            raise ValueError('Finite nonnegative past boundary H required')
        self.rewarder.h_queue = float(initial_H)
        self.initial_H = float(initial_H)
        self._ack_archive, self._ack_seen = [], set()

    def execute_validated_sample(self, environment, pre, packed, sample):
        super().execute_validated_sample(environment, pre, packed, sample)
        # Preserve every admission exactly once; reconstruct each historical
        # observation using its append-only archive prefix. Policy inputs are
        # unchanged. Parent still temporarily creates one full observation.
        row = self.observations[-1]
        history = list(row.pop('admission_acknowledgments'))
        if history[:len(self._ack_archive)] != self._ack_archive:
            self.failed = True
            raise RuntimeError('Controller admission history changed; no lossy deduplication')
        fresh = history[len(self._ack_archive):]
        if (len({item['task_id'] for item in fresh}) != len(fresh)
                or any(item['task_id'] in self._ack_seen for item in fresh)):
            self.failed = True
            raise RuntimeError('Duplicate admission acknowledgment')
        self._ack_archive.extend(fresh)
        self._ack_seen.update(item['task_id'] for item in fresh)
        row['admission_history_archive_prefix_length'] = len(history)
        row['admission_history_storage'] = 'Exact append-only ACK_ARCHIVE prefix, not a changed method input'

    def close_after_fixed_drain(self, environment):
        result = super().close_after_fixed_drain(environment)
        result.update(ACK_ARCHIVE=self._ack_archive, virtual_H_before=self.initial_H,
            virtual_H_after=self.rewarder.h_queue,
            environment_queue_upload_previous_portfolio_reset_each_episode=True,
            full_cumulative_admission_lists_not_repeated_in_each_saved_observation=True)
        return result


def actual_parameter_sha256(controller):
    require_actual_cuda_controller(controller)
    model = controller.model
    if controller.method == 'PPO-MV-A':
        rows = list(model.named_parameters())
    else:
        rows = [('actor.'+name, value) for name, value in model.actor.named_parameters()]
        for i, critic in enumerate(model.critics):
            rows.extend((f'critic{i}.'+name, value) for name, value in critic.named_parameters())
        rows.append(('log_portfolio_entropy_coef', model.log_portfolio_entropy_coef))
    h = hashlib.sha256()
    for name, value in rows:
        array = value.detach().cpu().contiguous()
        h.update(name.encode('utf-8'))
        h.update(str((tuple(array.shape), str(array.dtype))).encode('utf-8'))
        h.update(array.reshape(-1).view(controller._torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _new_environment(environment, role, scales, decisions):
    if (type(environment) is not VisualReceiptEnvironment or environment._catalog.role != role
            or environment._schedule.role != role or environment._index != 0
            or environment._pending is not None or environment._failed or environment._finished
            or environment._schedule.decisions != decisions
            or environment._schedule.decision_seconds != scales.decision_seconds
            or environment._profiles != scales.profiles or environment._edges != scales.edge_positions
            or environment._link != scales.upload_link):
        raise ValueError('New role-matched predeclared environment and same calibrated geometry required')


def environment_binding(environment):
    """Privileged preflight identity, never passed to actor/QL/critic."""
    if type(environment) is not VisualReceiptEnvironment:
        raise TypeError('Own declared environment required')
    return dict(schedule_sha256=environment._schedule.fingerprint(), role=environment._catalog.role,
        profiles=environment._profiles, edge_positions=environment._edges,
        mobility_start_seconds=environment._mobility_start,
        broadcast_seconds=float(environment._feedback._period), nominal_delay_seconds=float(environment._feedback._delay),
        waiting_capacity=environment._feedback._execution.waiting_capacity, upload_link=asdict(environment._link),
        catalog_source_sha256=environment._catalog.source_sha256, mobility_source_sha256=environment._mobility.source_hashes)


class RealMultiepisodeTrainingCoordinator:
    """SOURCE ONLY; no actual constructor, collect, update or export run."""
    def __init__(self, controller, scales, settings, ledger, bindings):
        self.protocol = require_reviewed_runtime(bindings)
        require_actual_cuda_controller(controller)
        if (type(scales) is not VisualFeatureScales or type(settings) is not EpisodeClosureCandidate
                or type(ledger) is not TrainingTransactionLedger or ledger.method != controller.method
                or ledger.phase != 'idle' or controller._started_episode_tokens
                or digest(jsonable(asdict(ledger.budget))) != digest(self.protocol['common_budget'])
                or digest(jsonable(asdict(settings))) != digest(self.protocol['closure_settings'])
                or digest(jsonable(asdict(scales))) != digest(self.protocol['feature_scales'])
                or ledger.paired_design_sha256 != self.protocol['paired_design_by_seed'][str(ledger.seed)]):
            raise ValueError('Untouched actual controller and exact common pilot-frozen design required')
        self.controller, self.scales, self.settings, self.ledger, self.bindings = controller, scales, settings, ledger, bindings
        self.mv_stream = (PersistentMVStream(ledger.budget.users, settings.mv_mu_seconds, 'training')
                          if controller.method == 'PPO-MV-A' else None)
        self._started, self._failed, self._H = False, False, 0.

    def run(self, new_training_environments, new_directory):
        if self._started or self._failed:
            raise RuntimeError('Coordinator is one-shot; no retry/append/resume')
        require_reviewed_runtime(self.bindings)  # recheck artifacts at actual entry
        environments = tuple(new_training_environments)
        if len(environments) != len(self.ledger.budget.episode_decisions):
            raise ValueError('Exactly one environment for each predeclared budget segment')
        for i, env in enumerate(environments):
            _new_environment(env, 'calibration', self.scales, self.ledger.budget.episode_decisions[i])
            if env._schedule.fingerprint() != self.ledger._schedules[i]:
                raise ValueError('Every schedule must match the declared pair before any model collection')
            expected = self.protocol['training_environment_sha256_by_seed'][str(self.ledger.seed)][i]
            if digest(jsonable(environment_binding(env))) != expected:
                raise ValueError('Link, delay, capacity, physical mobility and raw data must match the common registered environment')
        if len({id(env) for env in environments}) != len(environments):
            raise ValueError('An episode cannot reuse a consumed environment')
        path = Path(new_directory).resolve()
        if path == (BASE/'visual_calibrated_policy').resolve() or not path.is_relative_to((BASE/'visual_calibrated_policy').resolve()):
            raise ValueError('New training evidence must stay inside its declared output')
        path.mkdir(parents=True, exist_ok=False)
        self._started = True
        write_evidence(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(),
            method=self.controller.method, seed=self.ledger.seed, bindings=asdict(self.bindings),
            budget=self.ledger.budget.accounting(), automatic_retry_allowed=False))
        result, assembly = None, None
        try:
            records = []
            for episode, environment in enumerate(environments, 1):
                result, assembly = None, None
                self.ledger.begin_episode(episode, environment._schedule.fingerprint(), 'calibration')
                episode_path = path/f'episode_{episode:02d}'
                episode_path.mkdir(exist_ok=False)
                before_parameters = actual_parameter_sha256(self.controller)
                initial_H = self._H if self.protocol['virtual_H_episode_rule'] == 'carry_across_training_episodes' else 0.
                assembly = CompactCoordinatorAssembly(self.controller.method, self.ledger.budget.users,
                    len(self.scales.profiles), environment._schedule.decisions, self.scales, self.settings,
                    self.mv_stream, episode, f'{self.ledger.seed}:{episode}', initial_H=initial_H)
                for _ in range(environment._schedule.decisions):
                    pre = environment.observe()
                    packed = assembly.features_for(pre)
                    sample = self.controller.act(packed, deterministic=False)
                    assembly.execute_validated_sample(environment, pre, packed, sample)
                result = assembly.close_after_fixed_drain(environment)
                prepared = result['prepared_episode']
                self._H = result['virtual_H_after']
                if actual_parameter_sha256(self.controller) != before_parameters:
                    raise RuntimeError('Parameters changed during behavior collection before optimizer transaction')
                write_evidence(episode_path/'CLOSED_EPISODE.json', {key: value for key, value in result.items()
                                                                  if key != 'prepared_episode'})
                np.savez_compressed(episode_path/'BEHAVIOR_AND_GAE.npz', **prepared.time_user_fields,
                    advantages=prepared.advantages, returns=prepared.returns, physical_durations=prepared.physical_durations)
                self.ledger.close_collection(closed_episode_counts(prepared, self.ledger.budget.users,
                    environment._schedule.decisions, self.scales.decision_seconds))
                self.ledger.begin_update(prepared.episode_token)  # burn BEFORE actual optimizer
                stats = self.controller.update_closed_episode(prepared)
                self.ledger.commit_update(prepared.episode_token, stats)
                after_parameters = actual_parameter_sha256(self.controller)
                descriptor = self.ledger.checkpoint_descriptor()
                archive = write_actual_controller_checkpoint(self.controller, self.ledger,
                    episode_path/'checkpoint', self.mv_stream)
                record = dict(episode=episode, descriptor=descriptor, checkpoint=archive,
                    actual_parameter_sha256=after_parameters, before_update_parameter_sha256=before_parameters,
                    virtual_H_before=initial_H, virtual_H_after=self._H,
                    MV_state=self.mv_stream.state() if self.mv_stream else None, update_statistics=stats,
                    source_and_raw_schedule_sha256=environment._schedule.fingerprint(),
                    catalog_source_sha256=environment._catalog.source_sha256,
                    closure_settings=asdict(self.settings), feature_scales=asdict(self.scales),
                    automatic_resume_supported=False)
                write_evidence(episode_path/'EPISODE_COMPLETE.json', record)
                records.append(record)
                write_evidence(path/'STATE.json', dict(phase=self.ledger.phase, ledger=self.ledger.snapshot(),
                    completed_episodes=records, actual_training_completed_pending_validation=False))
            if self.ledger.phase != 'budget_complete_pending_validation':
                raise RuntimeError('Cannot finish on a partial environment budget')
            write_evidence(path/'BUDGET_COMPLETE_PENDING_VALIDATION.json', dict(ledger=self.ledger.snapshot(),
                episodes=records, checkpoint_selection_done=False, independent_final_result_audit_done=False))
            return records
        except Exception as error:
            self._failed = True
            self.ledger.mark_failed(error)
            self.controller._failed = True
            if self.mv_stream is not None:
                self.mv_stream.mark_failed(error)
            write_evidence(path/'FAILURE.json', dict(error=repr(error), ledger=self.ledger.snapshot(),
                virtual_H_after_partial_execution=self._H, automatic_retry_allowed=False,
                partial_behavior_rows=assembly.buffer._rows if assembly else [],
                partial_closed_transitions=result['closed_transitions'] if result else [],
                completed_archives_preserved=True))
            raise


class RealReadonlyValidationEvaluator:
    """Prepared fresh loaded-weight controller; never constructs an optimizer."""
    def __init__(self, fresh_loaded_controller, scales, settings, bindings,
                 checkpoint_path, checkpoint_descriptor, checkpoint_manifest_sha256, expected_parameter_sha256):
        self.protocol = require_reviewed_runtime(bindings)
        require_actual_cuda_controller(fresh_loaded_controller)
        if (fresh_loaded_controller._started_episode_tokens
                or fresh_loaded_controller.method != checkpoint_descriptor['method']
                or actual_parameter_sha256(fresh_loaded_controller) != require_sha(expected_parameter_sha256)
                or type(scales) is not VisualFeatureScales or type(settings) is not EpisodeClosureCandidate
                or digest(jsonable(asdict(scales))) != digest(self.protocol['feature_scales'])
                or digest(jsonable(asdict(settings))) != digest(self.protocol['closure_settings'])):
            raise ValueError('Fresh exactly loaded bound weights and calibration-frozen scales required')
        self.reference = validation_checkpoint_reference(checkpoint_path, checkpoint_descriptor,
                                                        checkpoint_manifest_sha256, 'validation')
        self._checkpoint = checkpoint_path, checkpoint_descriptor.copy(), checkpoint_manifest_sha256
        self.controller, self.scales, self.settings, self.bindings = fresh_loaded_controller, scales, settings, bindings
        self._started = False
        self.expected_parameter_sha256 = expected_parameter_sha256

    def evaluate(self, new_validation_environment, new_directory):
        if self._started:
            raise RuntimeError('Validation entry is one-shot, no result-dependent retries')
        require_reviewed_runtime(self.bindings)
        validation_checkpoint_reference(*self._checkpoint, 'validation')
        if actual_parameter_sha256(self.controller) != self.expected_parameter_sha256:
            raise RuntimeError('Loaded weights changed between validation construction and entry')
        env = new_validation_environment
        _new_environment(env, 'validation', self.scales, env._schedule.decisions)
        if env._schedule.fingerprint() not in self.protocol['validation_schedule_sha256']:
            raise ValueError('Only predeclared common validation schedule, no test or outcome-driven choice')
        if digest(jsonable(environment_binding(env))) != self.protocol['validation_environment_sha256_by_schedule'][env._schedule.fingerprint()]:
            raise ValueError('Validation link/delay/capacity/mobility/raw source differs from its registered common condition')
        path = Path(new_directory).resolve()
        if path == (BASE/'visual_calibrated_policy').resolve() or not path.is_relative_to((BASE/'visual_calibrated_policy').resolve()):
            raise ValueError('New validation evidence must remain in its intended output')
        path.mkdir(parents=True, exist_ok=False)
        self._started = True
        rewarder = VisualSharedReward(self.scales, self.settings.reward_v, self.settings.migration_weight,
                                    self.settings.concentration_factor, self.settings.drop_penalty_seconds)
        records = []
        try:
            torch = self.controller._torch
            rng_before = [torch.get_rng_state().clone(), *[row.clone() for row in torch.cuda.get_rng_state_all()]]
            for _ in range(env._schedule.decisions):
                pre = env.observe()
                packed = pack_features(pre, self.scales, prior_h_queue=rewarder.h_queue, v_weight=self.settings.reward_v)
                sample = self.controller.act(packed, deterministic=True)
                weights = sample.portfolios.astype(np.float64)
                weights /= weights.sum(axis=1, keepdims=True)
                env.apply(weights)
                reward = rewarder.compute(pre, weights, env.settle_action())
                records.append(dict(index=pre.index, physical_seconds=pre.physical_seconds,
                    portfolios=weights, pre_critic_sha256=hashlib.sha256(packed.critic.tobytes()).hexdigest(),
                    online_H_before=reward['h_before'], online_H_after=reward['h_after']))
            final = env.finish(self.settings.fixed_drain_seconds)
            if actual_parameter_sha256(self.controller) != self.expected_parameter_sha256:
                raise RuntimeError('Loaded parameters changed during readonly validation')
            validation_checkpoint_reference(*self._checkpoint, 'validation')
            rng_after = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
            if len(rng_before) != len(rng_after) or not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
                raise RuntimeError('Deterministic readonly policy altered Torch RNG state')
            write_evidence(path/'VALIDATION_COMPLETE.json', dict(reference=self.reference, final=final,
                records=records, parameter_sha256_before_and_after=self.expected_parameter_sha256,
                schedule_sha256=env._schedule.fingerprint(), initial_online_H=0.,
                training_H_or_MV_statistics_restored=False, optimizer_update_called=False,
                torch_rng_changed=False, checkpoint_selection_done=False, independent_final_result_audit_done=False))
            return final
        except Exception as error:
            self.controller._failed = True
            write_evidence(path/'FAILURE.json', dict(error=repr(error), records=records,
                optimizer_update_called=False, automatic_retry_allowed=False))
            raise
