"""SOURCE ONLY: four-method fixed-checkpoint readonly retained evaluation.

No launcher, model factory, actual test execution or training in this module.
Fresh integrated review, actual smoke, pilot and frozen weights remain absent.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np

from retained_test_schedule import RegisteredRetainedRequestSchedule
from retained_test_service_catalog import RetainedExecutionServiceCatalog
from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable, require_actual_cuda_controller, write_evidence
from visual_method_features import VisualFeatureScales, pack_features
from visual_receipt_environment import VisualReceiptEnvironment
from visual_rule_policies import ScoreSoftmaxPolicy, JSEQVisualPolicy
from visual_shared_reward import VisualSharedReward
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger
from visual_test_contract import FrozenTestExecutionContract, TEST_METHODS
from visual_training_coordinator import require_reviewed_runtime, actual_parameter_sha256, environment_binding
from visual_training_transactions import digest, file_sha, inspect_checkpoint_archive, require_sha
from visual_upload_link import UploadLinkConfig


def _environment_key(cell):
    return cell['load_label'], str(cell['seed']), str(cell['nominal_delay_seconds'])


def _expected_environment_sha(protocol, cell):
    load, seed, delay = _environment_key(cell)
    return require_sha(protocol['test_environment_binding_sha256_by_load_seed_delay'][load][seed][delay])


def _scientific_cell_identity(protocol, cell, weight_entries):
    """Semantic lock independent of output name/source revision/JSON encoding.

    Cosmetic review paths and byte encodings do not unlock a measured cell.
    Scientific conditions and immutable own weights remain part of identity.
    Any recovery must be a separately reviewed additive attempt, not reuse.
    """
    weights = sorted((dict(method=row['descriptor']['method'], seed=row['descriptor']['seed'],
        final_descriptor=row['descriptor'], manifest_sha256=row['checkpoint_manifest_sha256'],
        parameter_sha256=row['actual_parameter_sha256']) for row in weight_entries),
        key=lambda row: (row['method'], row['seed']))
    return digest(dict(cell=cell, feature_scales=protocol['feature_scales'],
        closure_settings=protocol['closure_settings'], rule_policy_settings=protocol['test_rule_policy_settings'],
        environment_sha256=_expected_environment_sha(protocol, cell), frozen_weights=weights))


def _write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _claim_once(lock_root, identity, record):
    """Low-level exclusive file fixture is not runtime/experiment authority."""
    require_sha(identity)
    root = Path(lock_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root/(identity+'.json')
    _write_once(path, dict(semantic_cell_identity_sha256=identity, **record))
    return path


def _check_reporting_final(final, generated, decisions, cell):
    """Pure reliability/mass checks, never filter adverse outcomes."""
    tasks = final['tasks']
    names = ('generated', 'accepted', 'completed', 'dropped', 'unfinished', 'inflight_upload')
    if (any(type(tasks[name]) is not int or tasks[name] < 0 for name in names)
            or tasks['generated'] != generated or decisions != cell['generation_decisions']
            or tasks['completed']+tasks['dropped']+tasks['unfinished'] != generated
            or tasks['accepted']+tasks['dropped']+tasks['inflight_upload'] != generated
            or tasks['accepted'] < tasks['completed']
            or tasks['unfinished'] != tasks['inflight_upload']+tasks['accepted']-tasks['completed']):
        raise ValueError('Complete generated/accepted/completed/drop/unfinished/upload counts required')
    latencies = final['uncapped_latency_seconds']
    if (len(latencies) != tasks['completed']
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= cell['fixed_end_seconds'] for x in latencies)
            or sorted(latencies) != sorted(tasks['latency_seconds'])):
        raise ValueError('All uncapped true-arrival completed latencies required, not capped/imputed samples')
    accounting = final['controller_accounting']
    if (accounting['admission_acknowledgments'] != tasks['accepted']
            or accounting['settled_completed']+accounting['drain_only_completed'] != tasks['completed']
            or accounting['settled_dropped']+accounting['drain_only_dropped'] != tasks['dropped']
            or not math.isclose(accounting['last_action_settlement_seconds'],
                cell['generation_seconds']+cell['decision_seconds'], rel_tol=0., abs_tol=1e-10)):
        raise ValueError('Complete interval and fixed-drain controller receipts required')
    if len(final['work']) != len(cell['profiles']):
        raise ValueError('All virtual-server conservation rows required')
    for row in final['work']:
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in
                ('offered', 'inflight', 'dropped', 'admitted', 'executed', 'admitted_remaining')):
            raise ValueError('Finite nonnegative measured service-second accounting')
        tolerance = 1e-10*max(1., row['offered'])
        if (abs(row['offered']-row['inflight']-row['dropped']-row['admitted']) > tolerance
                or abs(row['admitted']-row['executed']-row['admitted_remaining']) > tolerance):
            raise ValueError('No service mass loss or fabricated replacement')
    n = len(latencies)
    ordered = sorted(latencies, reverse=True)
    m = max(1, n-math.ceil(Fraction(99, 100)*n)) if n else 0
    return dict(completed_samples=n, tail_samples=m, mean_seconds=math.fsum(latencies)/n if n else None,
        p99_seconds=sorted(latencies)[math.ceil(Fraction(99, 100)*n)-1] if n else None,
        empirical_worst_approximately_one_percent_mean_seconds=math.fsum(ordered[:m])/m if m else None,
        no_completed_samples=n == 0, drops_and_unfinished_retained=True,
        raw_latency_clipping_or_completion_imputation=False, independent_result_audit_done=False)


class RealReadonlyRetainedEvaluator:
    """SOURCE PREPARED ONLY; constructor/evaluate never actually executed."""
    def __init__(self, contract, cell, scales, settings, fresh_loaded_controller=None):
        if type(contract) is not FrozenTestExecutionContract:
            raise TypeError('Completed actual frozen-weight common test contract required')
        rows = contract.cell_descriptors()
        protocol = require_reviewed_runtime(contract._bindings)
        if (sum(digest(row) == digest(cell) for row in rows) != 1 or cell['method'] not in TEST_METHODS
                or protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__))
                or type(scales) is not VisualFeatureScales or type(settings) is not EpisodeClosureCandidate
                or digest(jsonable(asdict(scales))) != digest(protocol['feature_scales'])
                or digest(jsonable(asdict(settings))) != digest(protocol['closure_settings'])
                or settings.fixed_drain_seconds != cell['fixed_drain_seconds']):
            raise ValueError('Exact current reviewed common cell/scales/fixed closure required')
        self.contract, self.cell, self.scales, self.settings = contract, dict(cell), scales, settings
        self.protocol, self.controller = protocol, fresh_loaded_controller
        self._started, self._finished, self._failed = False, False, False
        self._checkpoint, self._parameter_sha, self.rule = None, None, None
        if cell['method'] in ('QLA-MAPPO', 'PPO-MV-A'):
            require_actual_cuda_controller(self.controller)
            entries = [row for row in contract._weights['entries']
                if row['descriptor']['method'] == cell['method'] and row['descriptor']['seed'] == cell['seed']]
            if (len(entries) != 1 or self.controller.method != cell['method']
                    or self.controller._started_episode_tokens):
                raise ValueError('Own fresh correctly loaded final weights, no training history/update tokens')
            self._checkpoint = entries[0]
            self._parameter_sha = require_sha(entries[0]['actual_parameter_sha256'])
            self._check_weights()
        else:
            if fresh_loaded_controller is not None:
                raise ValueError('Rule cells cannot substitute a neural/fake controller')
            config = protocol['test_rule_policy_settings'][cell['method']]
            self.rule = (ScoreSoftmaxPolicy(**config) if cell['method'] == 'Score-softmax'
                         else JSEQVisualPolicy(**config))
        self.semantic_identity_sha256 = _scientific_cell_identity(protocol, cell, contract._weights['entries'])

    def _check_weights(self):
        if self.controller is None:
            return
        require_actual_cuda_controller(self.controller)
        row = self._checkpoint
        inspect_checkpoint_archive(row['checkpoint_path'], row['descriptor'], row['checkpoint_manifest_sha256'])
        if actual_parameter_sha256(self.controller) != self._parameter_sha or self.controller._started_episode_tokens:
            raise RuntimeError('Actual loaded weights changed or optimizer transaction occurred')

    def _check_environment(self, env):
        if (type(env) is not VisualReceiptEnvironment
                or type(env._catalog) is not RetainedExecutionServiceCatalog
                or type(env._schedule) is not RegisteredRetainedRequestSchedule
                or env._catalog._contract is not self.contract or env._schedule._contract is not self.contract
                or env._index != 0 or env._pending is not None or env._unsettled is not None
                or env._failed or env._finished or env._schedule.seed != self.cell['seed']
                or env._schedule.load_label != self.cell['load_label']
                or env._schedule.fingerprint() != self.cell['common_exogenous_schedule_sha256']
                or env._profiles != self.scales.profiles or env._edges != self.scales.edge_positions
                or env._schedule.decisions != self.cell['generation_decisions']
                or env._schedule.decision_seconds != self.scales.decision_seconds):
            raise ValueError('Fresh exact registered test environment, never fixture/partial/resumed state')
        # Existing no-request feature rows still use the default simulated link.
        # A separately authorized higher-bandwidth pilot needs an additive
        # reviewed feature adapter before this evaluator may accept that link.
        if asdict(env._link) != asdict(UploadLinkConfig()):
            raise ValueError('Nondefault link unsupported by current no-request feature adapter; no silent mismatch')
        binding = environment_binding(env)
        binding['catalog_source_sha256'] = dict(binding['catalog_source_sha256'])
        if digest(jsonable(binding)) != _expected_environment_sha(self.protocol, self.cell):
            raise RuntimeError('Original mobility/raw/link/delay/capacity/profile/pool schedule binding changed')
        env._schedule.verify_boundary()
        env._catalog.verify_boundary()
        for name, sha in env._mobility.source_hashes.items():
            if file_sha(Path(name)) != sha:
                raise RuntimeError('Registered physical mobility bytes changed')
        return jsonable(binding)

    def evaluate(self, environment, new_directory):
        if self._started or self._failed or self._finished:
            raise RuntimeError('Readonly evaluator is one-shot; no retry, append or outcome extension')
        self.contract.cell_descriptors()
        binding = self._check_environment(environment)
        self._check_weights()
        path = Path(new_directory)
        root = (BASE/'visual_calibrated_policy/test_cells').resolve()
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root) or path.exists():
            raise ValueError('New cell output inside intended root; never overwrite old/partial cells')
        # Identity excludes output path/cosmetic source-review encoding, so
        # choosing another directory cannot reuse a completed/failed cell.
        lock = _claim_once(BASE/'visual_calibrated_policy/test_cell_locks', self.semantic_identity_sha256,
            dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), cell=self.cell,
                 output_path=str(path), policy_not_yet_executed_at_lock_creation=True, automatic_retry_allowed=False))
        self._started = True
        records, observations, ack_archive, owned = [], [], [], False
        torch = self.controller._torch if self.controller is not None else None
        rng_before = None
        try:
            path.mkdir(parents=True, exist_ok=False)
            owned = True
            _write_once(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                cell=self.cell, semantic_identity_sha256=self.semantic_identity_sha256, lock=str(lock),
                environment_binding=binding, optimizer_update_allowed=False, training_state_restore_allowed=False))
            if torch is not None:
                rng_before = [torch.get_rng_state().clone(), *[row.clone() for row in torch.cuda.get_rng_state_all()]]
            diagnostic = VisualSharedReward(self.scales, v=self.settings.reward_v,
                migration_weight=self.settings.migration_weight, concentration_factor=self.settings.concentration_factor,
                drop_penalty_seconds=self.settings.drop_penalty_seconds)
            for _ in range(environment._schedule.decisions):
                pre = environment.observe()
                compact = asdict(pre)
                history = compact.pop('admission_acknowledgments')
                if history[:len(ack_archive)] != ack_archive:
                    raise RuntimeError('Past shared admission history changed')
                fresh = history[len(ack_archive):]
                if len({row['task_id'] for row in history}) != len(history):
                    raise RuntimeError('Duplicate admission history, no filtering')
                ack_archive.extend(fresh)
                compact['admission_history_prefix_count'] = len(ack_archive)
                observations.append(compact)
                packed = pack_features(pre, self.scales, prior_h_queue=diagnostic.h_queue,
                                       v_weight=self.settings.reward_v)
                actor_sha = hashlib.sha256(packed.actor.tobytes()).hexdigest()
                critic_sha = hashlib.sha256(packed.critic.tobytes()).hexdigest()
                if torch is not None:
                    torch.cuda.synchronize()
                start = perf_counter()
                raw = (self.controller.act(packed, deterministic=True).portfolios if self.controller is not None
                       else self.rule.portfolios(pre, packed))
                if torch is not None:
                    torch.cuda.synchronize()
                policy_seconds = perf_counter()-start
                weights = np.asarray(raw, dtype=np.float64).copy()
                if (weights.shape != (self.cell['users'], len(self.cell['profiles']))
                        or not np.isfinite(weights).all() or np.any(weights < 0)
                        or np.any(weights.sum(axis=1) <= 0)):
                    raise ValueError('Finite valid actual policy portfolios required')
                weights /= weights.sum(axis=1, keepdims=True)
                environment.apply(weights)
                outcome = environment.settle_action()
                reward = diagnostic.compute(pre, weights, outcome)
                if (actor_sha != hashlib.sha256(packed.actor.tobytes()).hexdigest()
                        or critic_sha != hashlib.sha256(packed.critic.tobytes()).hexdigest()):
                    raise RuntimeError('Current pre-action input mutated by action/execution/receipts')
                records.append(dict(index=pre.index, physical_seconds=pre.physical_seconds,
                    actor_sha256=actor_sha, current_action_free_critic_sha256=critic_sha,
                    portfolios=weights, policy_compute_wall_seconds=policy_seconds,
                    policy_compute_is_not_simulated_latency_or_vision_service=True,
                    past_outcome_batch=asdict(outcome), shared_QL_diagnostic_not_PPO_training_reward=reward))
                if pre.index % 100 == 0:
                    write_evidence(path/'STATE.json', dict(phase='actual_fixed_policy_test', completed_decisions=len(records),
                        total_decisions=self.cell['generation_decisions'], cell=self.cell, pid=os.getpid(),
                        at=datetime.now().astimezone().isoformat(), optimizer_updates=0, automatic_retry_allowed=False))
            final = environment.finish(self.cell['fixed_drain_seconds'])
            if not math.isclose(environment._feedback.time, self.cell['fixed_end_seconds'], rel_tol=0., abs_tol=1e-10):
                raise RuntimeError('Fixed common drain end changed')
            final = jsonable(final)
            # Only validate observed task/event closure. Candidate training
            # penalties returned by this shared helper are discarded; they
            # are never imputed into completed latency or used for updates.
            build_closed_ledger(final, tuple(IssuedIdentity(row.task_id, row.user, row.arrival_seconds)
                for row in environment._schedule.requests), self.cell['fixed_end_seconds'], self.cell['users'],
                self.settings.drop_penalty_seconds, self.settings.unfinished_penalty_seconds)
            summary = _check_reporting_final(final, len(environment._schedule.requests), len(records), self.cell)
            environment._schedule.verify_boundary()
            environment._catalog.verify_boundary()
            for name, sha in environment._mobility.source_hashes.items():
                if file_sha(Path(name)) != sha:
                    raise RuntimeError('Mobility changed during cell')
            self._check_weights()
            if torch is not None:
                rng_after = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
                if len(rng_before) != len(rng_after) or not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
                    raise RuntimeError('Deterministic fixed policy changed Torch RNG state')
            # Privileged complete execution record, NEVER fed to any policy.
            _write_once(path/'EXECUTION_REPORTING_ONLY.json', dict(final=final,
                actual_request_records=environment._feedback._execution.records,
                broadcast_prediction_history=environment._feedback._history,
                service_true_future_values_allowed_only_in_this_reporting_export=True))
            _write_once(path/'OBSERVATIONS_AND_DECISIONS.json', dict(observations=observations,
                common_available_admission_history_archive=ack_archive, records=records,
                initial_online_H=0., training_H_MV_or_RNG_restored=False, optimizer_update_called=False))
            write_evidence(path/'STATE.json', dict(phase='actual_cell_complete_pending_independent_audit',
                completed_decisions=len(records), cell=self.cell, at=datetime.now().astimezone().isoformat(), optimizer_updates=0))
            outputs = {name: file_sha(path/name) for name in ('EXECUTION_REPORTING_ONLY.json',
                'OBSERVATIONS_AND_DECISIONS.json', 'STARTED.json', 'STATE.json')}
            _write_once(path/'TEST_CELL_COMPLETE_PENDING_AUDIT.json', dict(status='actual_cell_complete_pending_independent_audit',
                cell=self.cell, semantic_identity_sha256=self.semantic_identity_sha256, outputs_sha256=outputs,
                final_summary=summary, actual_parameter_sha256_before_and_after=self._parameter_sha,
                deterministic_Torch_RNG_changed=False if torch is not None else None,
                fixed_horizon_and_drain=True, optimizer_updates=0, source_protocol_sha256=self.contract._bindings.protocol_sha256,
                complete_120_grid_or_final_audit_claim=False))
            self._finished = True
            return final
        except Exception as error:
            self._failed = True
            environment._failed = True
            if self.controller is not None:
                self.controller._failed = True
            failure = dict(error=repr(error), cell=self.cell, semantic_identity_sha256=self.semantic_identity_sha256,
                records=records, observations=observations, admission_archive=ack_archive,
                reporting_only_partial_execution_records=environment._feedback._execution.records,
                reporting_only_partial_execution_time=environment._feedback.time,
                automatic_append_resume_or_retry_allowed=False, partial_evidence_preserved=True)
            if owned:
                write_evidence(path/'STATE.json', dict(phase='failed_preserved_no_automatic_retry', cell=self.cell,
                    completed_decisions=len(records), at=datetime.now().astimezone().isoformat(), optimizer_updates=0))
                _write_once(path/'FAILURE.json', failure)
            _write_once(lock.with_name(lock.stem+'_FAILURE.json'), failure)
            raise
