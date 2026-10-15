"""SOURCE ONLY: whole paired readonly engineering evidence, no pilot freeze.

Receives the SAME successful actual driver; byte/closure/GAE/FIFO checks do
not recreate neural actions or independently certify leakage/model behavior.
Actual constructor/register/role parsing/NPZ/replay/output calls remain zero.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from fifo_execution import FIFOExecution, Request
from retained_test_evaluator import _check_reporting_final, _write_once
from retained_test_grid_registration import _safe_path, _read_actual, _claim_grid_once
from visual_episode_collector import BASE, jsonable, write_evidence
from visual_pilot_driver import RealPairedReadonlyPilotDriver, DRIVER_SCOPE, ORDER_RULE, validate_pilot_driver_job_coverage
from visual_pilot_episode import SCOPE, _registered_condition, _actual_smoke_weight_bindings, pilot_episode_scientific_identity
from visual_protocol_assembly import _read_engineering_schedule, merge_frozen_file_bindings
from visual_temporal_gae import per_user_temporal_gae
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger
from visual_training_transactions import digest, file_sha, require_sha

FILES = frozenset(('STARTED.json', 'CLOSED_ENGINEERING_EPISODE.json', 'EXECUTION_RECORDS_REPORTING_ONLY.json',
    'ACTUAL_BEHAVIOR_AND_GAE.npz', 'ACTUAL_READONLY_RNG.npz'))
BEHAVIOR = frozenset(('actor_obs', 'critic_obs', 'latent_action', 'log_prob', 'value', 'reward', 'user_id', 'slot', 'done'))
REGISTRATION_SCOPE = 'all_paired_readonly_engineering_files_registered_pending_independent_pilot_sufficiency_check'


def pilot_registration_identity(driver_scientific_identity):
    return digest(dict(scope=REGISTRATION_SCOPE, actual_driver_scientific_identity=require_sha(driver_scientific_identity)))


def check_pilot_behavior_arrays(arrays, users, decisions, servers, decision_seconds, gamma, trace, method):
    """Pure numeric closure check; artificial arrays are NOT model evidence."""
    if (type(arrays) is not dict or set(arrays) != BEHAVIOR | {'advantages', 'returns', 'physical_durations'}
            or any(type(x) is not int or x < 1 for x in (users, decisions, servers))
            or method not in ('QLA-MAPPO', 'PPO-MV-A')):
        raise ValueError('Whole explicit per-user episode fields and native dimensions required')
    actor, critic = 3*servers+7, users*(3*servers+7+servers)+3*servers+3
    shapes = dict(actor_obs=(decisions, users, actor), critic_obs=(decisions, users, critic),
        latent_action=(decisions, users, servers), physical_durations=(decisions,))
    shapes.update({name: (decisions, users) for name in set(arrays)-set(shapes)})
    if (any(type(value) is not np.ndarray or value.shape != shapes[name]
            or value.dtype.kind not in 'biuf' or not np.isfinite(value).all() for name, value in arrays.items())
            or arrays['done'].dtype != np.bool_ or arrays['user_id'].dtype.kind not in 'iu' or arrays['slot'].dtype.kind not in 'iu'
            or not np.array_equal(arrays['user_id'], np.tile(np.arange(users), (decisions, 1)))
            or not np.array_equal(arrays['slot'], np.tile(np.arange(1, decisions+1)[:, None], (1, users)))
            or not arrays['done'][-1].all() or arrays['done'][:-1].any()
            or not np.allclose(arrays['physical_durations'], decision_seconds, atol=1e-12, rtol=0)):
        raise ValueError('Exact full time/user layouts, finite arrays, fixed physical intervals and terminal mask required')
    reward = arrays['reward']
    if ((method == 'QLA-MAPPO' and (np.any(reward < -1) or np.any(reward > 0)
            or not np.array_equal(reward, np.repeat(reward[:, :1], users, axis=1))))
            or (method == 'PPO-MV-A' and not np.all((reward == 0) | (reward == 1)))):
        raise ValueError('Original own method reward semantics, no clipping/relabeling')
    next_values = np.zeros_like(arrays['value'])
    next_values[:-1] = arrays['value'][1:]
    advantages, returns = per_user_temporal_gae(reward, arrays['value'], next_values, arrays['done'],
        arrays['physical_durations'], gamma, trace)
    if not np.array_equal(advantages, arrays['advantages']) or not np.array_equal(returns, arrays['returns']):
        raise ValueError('Saved GAE must recompute on each separate user/physical time, not flattened or clipped')
    return dict(decisions=decisions, users=users, behavior_samples=decisions*users,
        action_interval_seconds=float(np.sum(arrays['physical_durations'])), separate_user_GAE_recomputed=True,
        actual_neural_behavior_proven_by_pure_arrays=False)


def _load_npz(path, sha):
    path = _safe_path(path, BASE/'visual_calibrated_policy')
    if file_sha(path) != require_sha(sha):
        raise RuntimeError('Exact original immutable actual episode archive required')
    with np.load(path, allow_pickle=False) as payload:
        if len(payload.files) != len(set(payload.files)):
            raise ValueError('No duplicate array names in a whole original archive')
        arrays = {name: payload[name].copy() for name in payload.files}
    if file_sha(path) != sha:
        raise RuntimeError('Actual archive changed during readonly decoding')
    return arrays


def _check_actual_execution_records(records, schedule, catalog, profiles, capacity, end, final):
    """Actual saved dispatch replay with original pairs; NOT independent FIFO certification."""
    if type(catalog) is not ExecutionServiceCatalog or catalog.role != schedule.role:
        raise TypeError('Actual whole original cal/validation catalog required')
    scheduled = {row.task_id: row for row in schedule.requests}
    if type(records) is not dict or set(records) != set(scheduled):
        raise ValueError('Every generated task must retain its actual execution record, including drops/unfinished')
    oracle = FIFOExecution(len(profiles), capacity)
    for task in sorted(records):
        saved, original = records[task], scheduled[task]
        request = Request(**saved['request'])
        if (request.task_id != task or request.image_id != original.image_id
                or type(request.server) is not int or not 0 <= request.server < len(profiles)
                or request.arrival != schedule.decision_time(schedule.decision_index(original))
                or request.payload_bytes != catalog.observable_metadata(original.image_id).payload_bytes
                or request.service != catalog.measured_pair_seconds(original.image_id, original.measured_session)[profiles[request.server]]):
            raise ValueError('Actual dispatch time/JPEG bytes/original per-image-session-config measured seconds required')
        oracle.submit(request)
    oracle.advance(end)  # same declared H+drain, never wait until all tasks finish
    if digest(jsonable(oracle.records)) != digest(records):
        raise ValueError('Whole original actual execution statuses/times do not match fixed-drain FIFO replay')
    got = oracle.results()
    names = ('generated', 'accepted', 'completed', 'dropped', 'unfinished', 'inflight_upload')
    if any(got[name] != final['tasks'][name] for name in names):
        raise ValueError('No task loss or partial favorable execution subset')
    latencies = sorted(row['completed']-scheduled[task].arrival_seconds for task, row in oracle.records.items()
        if row['status'] == 'completed')
    want = sorted(final['uncapped_latency_seconds'])
    if len(latencies) != len(want) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-10) for a, b in zip(latencies, want)):
        raise ValueError('Whole uncapped true-arrival latencies, not issuance/capped/imputed labels')
    work = oracle.work_accounting()
    if len(work) != len(final['work']) or any(abs(row[key]-saved[key]) > 1e-10*max(1., row['offered'])
            for row, saved in zip(work, final['work']) for key in row):
        raise ValueError('Original full per-server measured service-second accounting changed')
    return dict(saved_actual_dispatch_records=len(records), complete_fixed_drain_FIFO_replay_checked=True,
        original_measured_pair_seconds_and_JPEG_bytes_checked=True,
        dispatch_portfolio_assignment_and_upload_feature_reconstruction_not_independently_certified=True,
        same_foundation_FIFO_replay_not_independent_analytic_or_heldout_entry_certification=True)


class RealPairedPilotEvidenceRegistrar:
    """SOURCE PREPARED ONLY; actual constructor/check/register calls zero."""
    def __init__(self, driver, externally_bound_driver_marker_sha256):
        if type(driver) is not RealPairedReadonlyPilotDriver:
            raise TypeError('SAME real successful live paired driver required, not an author marker/fixture')
        if driver._finished is not True or driver._failed or driver._marker is None:
            raise RuntimeError('Every registered paired episode must actually close before registration')
        driver._boundary()
        self.driver, self.marker_sha = driver, require_sha(externally_bound_driver_marker_sha256)
        if file_sha(driver._marker) != self.marker_sha or driver._completed_files.get(str(driver._marker)) != self.marker_sha:
            raise RuntimeError('Original actual whole-driver marker external bytes required')
        self.identity = pilot_registration_identity(driver.identity)
        self.inputs = merge_frozen_file_bindings(driver.inputs, driver._completed_files)
        self._started, self._failed, self._finished = False, False, False
        self._current, self._checked_summaries = None, []

    def _boundary(self):
        if self._failed:
            raise RuntimeError('Failed registration cannot retry/append/select partial records')
        self.driver._boundary()
        if (self.driver._finished is not True or self.driver._failed
                or self.inputs != merge_frozen_file_bindings(self.driver.inputs, self.driver._completed_files)
                or pilot_registration_identity(self.driver.identity) != self.identity):
            raise RuntimeError('Original successful live driver and all exact immutable files must remain unchanged')

    def _check_all_actual_evidence(self):
        self._boundary()
        driver, root = self.driver, BASE/'visual_calibrated_policy'
        marker = _read_actual(driver._marker, self.marker_sha, root)
        flags = dict(all_registered_conditions_both_methods_closed=True, unused_reference_identity_unchanged=True,
            diagnostic_H_MV_reset_not_formal_training_carry_test=True, full_pilot_completed=False, final_pilot_frozen=False,
            independent_entry_or_result_audit=False, metadata_schedule_generation_permission=False, formal_training_or_test_permission=False)
        if (marker['status'] != 'actual_paired_readonly_closed_episodes_pending_independent_engineering_check'
                or marker['scope'] != DRIVER_SCOPE or marker['order_rule'] != ORDER_RULE
                or marker['scientific_driver_identity_sha256'] != driver.identity
                or marker['completed'] != driver._completed or marker['original_inputs_sha256'] != driver.inputs
                or marker['actual_parameter_sha256_by_method'] != driver.parameter_hashes
                or marker['immutable_closed_files_sha256'] != {name: sha for name, sha in driver._completed_files.items() if name != str(driver._marker)}
                or any(marker.get(name) is not value for name, value in flags.items())
                or any(type(marker[name]) is not int or marker[name] != expected for name, expected in
                    (('original_factory_invocations', 1), ('optimizer_updates', 0), ('formal_cells', 0), ('actual_own_MODEL_loads_and_episodes', len(driver.jobs))))):
            raise ValueError('Exact actual paired readonly driver evidence, no freeze/permission/partial-arm substitutes')
        validate_pilot_driver_job_coverage([row['descriptor'] for row in marker['completed']], driver.candidate)
        catalogs = {role: ExecutionServiceCatalog(role) for role in ('calibration', 'validation')}
        summaries = []
        for index, row in enumerate(marker['completed'], 1):
            self._boundary()
            job = driver.jobs[index-1]
            self._current = dict(index=index, descriptor=deepcopy(job), phase='original_marker_load_and_whole_output_bytes')
            condition = _registered_condition(driver.candidate, job)
            folder = root/'pilot_episodes'/f'{driver._path.name}_E{index:03d}'
            closed_marker = folder/'CLOSED_EPISODE_PENDING_REVIEW.json'
            load_path = driver._path/f'L{index:03d}.json'
            if (row['index'] != index or row['episode_path'] != str(folder) or row['closed_marker_path'] != str(closed_marker)
                    or row['own_MODEL_load_witness_path'] != str(load_path)):
                raise ValueError('All actual original ordered driver-owned episode and load paths required')
            episode = _read_actual(closed_marker, row['closed_marker_sha256'], root)
            witness = _read_actual(load_path, row['own_MODEL_load_witness_sha256'], root)
            report, weights = _actual_smoke_weight_bindings(driver.preparation, job['method'])
            model_path = Path(driver.bindings.actual_smoke_path).parent/job['method']/'checkpoint/MODEL_STATE.pt'
            expected_id = pilot_episode_scientific_identity(driver.candidate, job, report['after_parameter_sha256'])
            if (episode['status'] != 'actual_closed_no_optimizer_pilot_episode_pending_independent_check'
                    or episode['scope'] != SCOPE or episode['descriptor'] != job or episode['scientific_identity_sha256'] != expected_id
                    or row['scientific_episode_identity_sha256'] != expected_id or set(episode['files_sha256']) != FILES
                    or episode['original_preparation_bindings'] != asdict(driver.bindings)
                    or episode['inputs_sha256'] != merge_frozen_file_bindings(driver.preparation._inputs, weights)
                    or episode['actual_own_functional_smoke_parameter_sha256_before_and_after'] != report['after_parameter_sha256']
                    or episode['actual_model_actions_and_values_collected'] is not True or episode['deterministic_Torch_RNG_unchanged'] is not True
                    or episode['independent_episode_diagnostic_H_MV_reset'] is not True or episode['formal_H_carry_rule_exercised'] is not False
                    or type(episode['optimizer_updates']) is not int or episode['optimizer_updates'] != 0
                    or type(episode['actual_formal_training_or_test_cells']) is not int or episode['actual_formal_training_or_test_cells'] != 0
                    or any(episode[name] is not False for name in ('smoke_training_H_MV_optimizer_or_RNG_restored',
                        'complete_candidate_conditions_registered', 'full_pilot_completed', 'final_pilot_frozen', 'independent_entry_or_result_audit_done'))):
                raise ValueError('Exact actual own-weight whole readonly episode, not formal learning or completed pilot')
            if (witness['scope'] != SCOPE or witness['method'] != job['method'] or witness['initialization_seed'] != job['seed']
                    or witness['own_actual_smoke_path'] != driver.bindings.actual_smoke_path or witness['own_actual_smoke_sha256'] != driver.bindings.actual_smoke_sha256
                    or witness['original_own_smoke_files_sha256'] != weights or witness['actual_loaded_parameter_sha256'] != report['after_parameter_sha256']
                    or witness['model_payload_path'] != str(model_path) or witness['model_payload_sha256'] != weights[str(model_path)]
                    or witness['actual_loaded_structure_sha256'] != episode['own_smoke_structure_identity_sha256_before_and_after']
                    or witness['unused_optimizer_identity_sha256'] != episode['unused_optimizer_identity_sha256_before_and_after']
                    or witness['actual_MODEL_parameter_load_performed'] is not True or witness['readonly_cal_validation_diagnostic_only'] is not True
                    or any(witness[name] is not False for name in ('optimizer_training_state_restored', 'H_MV_or_training_RNG_state_restored', 'test_or_formal_training_permission'))):
                raise ValueError('Exact OWN current model-load witness and original archive, no training-state restoration')
            for name, sha in episode['files_sha256'].items():
                if self.inputs.get(str(folder/name)) != sha or file_sha(_safe_path(folder/name, root)) != sha:
                    raise RuntimeError('Every actual episode file is part of the same whole driver byte map')
            body = _read_actual(folder/'CLOSED_ENGINEERING_EPISODE.json', episode['files_sha256']['CLOSED_ENGINEERING_EPISODE.json'], root)
            settings = driver.candidate['closure_settings']
            if (body['candidate_settings'] != settings or body['virtual_H_before'] != 0
                    or any(len(body[name]) != job['decisions'] for name in ('observations', 'ordinary_transitions', 'closed_transitions'))):
                raise ValueError('Whole shared settings, diagnostic H reset and every pre-action/ordinary/closed interval required')
            schedule = _read_engineering_schedule(condition['schedule_artifact'], job['role'], job['decisions'], .05)
            end = schedule.horizon_seconds+settings['fixed_drain_seconds']
            if (episode['fixed_generation_seconds'] != schedule.horizon_seconds
                    or not math.isclose(episode['fixed_drain_end_seconds'], end, rel_tol=0, abs_tol=1e-10)):
                raise ValueError('Original common generation and fixed drain, no outcome-based extensions')
            self._current['phase'] = 'whole_ledger_reliability_behavior_GAE_and_original_readonly_RNG'
            ledger = build_closed_ledger(body['final'], [IssuedIdentity(req.task_id, req.user, req.arrival_seconds) for req in schedule.requests],
                end, 30, settings['drop_penalty_seconds'], settings['unfinished_penalty_seconds'])
            if digest(jsonable(ledger)) != digest(body['ledger']):
                raise ValueError('Every original task outcome must rebuild the unmodified closed ledger')
            reliability = _check_reporting_final(body['final'], len(schedule.requests), job['decisions'], dict(generation_decisions=job['decisions'],
                generation_seconds=schedule.horizon_seconds, decision_seconds=.05, fixed_end_seconds=end, profiles=driver.candidate['profiles']))
            if digest(reliability) != digest(episode['uncapped_reliability_diagnostics']) or episode['whole_generated_requests'] != len(schedule.requests):
                raise ValueError('Original adverse/zero/drop/unfinished and uncapped diagnostics must remain complete')
            arrays = _load_npz(folder/'ACTUAL_BEHAVIOR_AND_GAE.npz', episode['files_sha256']['ACTUAL_BEHAVIOR_AND_GAE.npz'])
            counts = check_pilot_behavior_arrays(arrays, 30, job['decisions'], 4, .05,
                settings['gamma_per_second'], settings['lambda_per_second'], job['method'])
            declared = episode['full_behavior_counts']
            if any(declared[name] != counts[name] for name in ('decisions', 'users', 'behavior_samples', 'action_interval_seconds')):
                raise ValueError('Whole saved behavior counts, no row/user/interval truncation')
            for i, transition in enumerate(body['closed_transitions']):
                if (transition['index'] != i+1 or hashlib.sha256(arrays['critic_obs'][i].tobytes()).hexdigest() != transition['pre_critic_sha256']
                        or not np.array_equal(arrays['reward'][i], np.asarray(transition['reward']['per_user_rewards']))
                        or body['ordinary_transitions'][i]['pre_critic_sha256'] != transition['pre_critic_sha256']):
                    raise ValueError('Saved reward and unchanged pre-action critic bytes must match every closed transition')
            rng = _load_npz(folder/'ACTUAL_READONLY_RNG.npz', episode['files_sha256']['ACTUAL_READONLY_RNG.npz'])
            n = len(rng)//2
            if (n < 2 or set(rng) != {f'{prefix}_{i}' for prefix in ('before', 'after') for i in range(n)}
                    or any(rng[f'before_{i}'].dtype != np.uint8 or rng[f'before_{i}'].ndim != 1
                        or not np.array_equal(rng[f'before_{i}'], rng[f'after_{i}']) for i in range(n))):
                raise ValueError('Whole original readonly CPU/CUDA RNG states must remain unchanged')
            catalog = catalogs[job['role']]
            if catalog.source_sha256 != condition['environment_descriptor']['catalog_source_sha256']:
                raise RuntimeError('Exactly the whole original paired measured role pool, no fit/test/filter')
            records = _read_actual(folder/'EXECUTION_RECORDS_REPORTING_ONLY.json', episode['files_sha256']['EXECUTION_RECORDS_REPORTING_ONLY.json'], root)
            self._current['phase'] = 'original_measured_pair_bytes_and_full_reporting_only_FIFO_replay'
            replay = _check_actual_execution_records(records, schedule, catalog, tuple(driver.candidate['profiles']), 32, end, body['final'])
            summaries.append(dict(index=index, descriptor=job, scientific_episode_identity_sha256=expected_id,
                original_closed_marker_path=str(closed_marker), original_closed_marker_sha256=row['closed_marker_sha256'],
                behavior_array_checks=counts, reliability_diagnostics=reliability, reporting_only_FIFO_replay=replay,
                parameter_load_witness_verified_not_model_rerun=True, independent_leakage_or_learning_benefit_certified=False))
            self._checked_summaries = deepcopy(summaries)
            self._boundary()
        return summaries

    def register_once(self, new_directory):
        if self._started or self._failed or self._finished:
            raise RuntimeError('Whole registration one-shot, no append/resume/retry or partial favorable subset')
        self._started = True
        path, lock, owned, summaries = None, None, False, []
        try:
            self._boundary()
            root = BASE/'visual_calibrated_policy/pilot_evidence_registry'
            path = _safe_path(new_directory, root)
            if (path.parent.resolve() != root.resolve() or path.exists() or not 1 <= len(path.name) <= 24
                    or any(not (x.isascii() and (x.isalnum() or x in '-_')) for x in path.name)):
                raise ValueError('New short direct-child whole paired registration directory only')
            lock = _claim_grid_once(BASE/'visual_calibrated_policy/pilot_registration_locks', self.identity,
                dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), output_path=str(path), automatic_retry_allowed=False))
            path.mkdir(parents=True, exist_ok=False)
            owned = True
            _write_once(path/'STARTED.json', dict(scope=REGISTRATION_SCOPE, driver_marker_path=str(self.driver._marker),
                driver_marker_sha256=self.marker_sha, original_immutable_files_sha256=self.inputs))
            summaries = self._check_all_actual_evidence()
            self._boundary()
            _write_once(path/'ALL_PAIRED_ENGINEERING_EVIDENCE_PENDING_CHECK.json', dict(scope=REGISTRATION_SCOPE,
                status='all_actual_paired_readonly_engineering_evidence_registered_pending_independent_check',
                scientific_registration_identity_sha256=self.identity, driver_scientific_identity_sha256=self.driver.identity,
                original_immutable_files_sha256=self.inputs, complete_ordered_episode_summaries=summaries,
                no_adverse_or_zero_completion_outcomes_filtered=True, optimizer_updates=0, formal_cells=0,
                independent_pilot_sufficiency_no_leakage_analytic_heldout_check_completed=False,
                final_H_MV_boundary_rule_budget_or_normalization_frozen=False, full_pilot_completed=False,
                metadata_generation_training_test_or_figure_permission=False))
            self._boundary()
            self._finished = True
            return path/'ALL_PAIRED_ENGINEERING_EVIDENCE_PENDING_CHECK.json'
        except Exception as error:
            self._failed = True
            failure = dict(error=repr(error), pid=os.getpid(), scope=REGISTRATION_SCOPE,
                original_driver_and_all_child_evidence_preserved=True, original_immutable_files_sha256=self.inputs,
                completed_summaries_if_all_checked=summaries, automatic_append_resume_retry_or_subset_merge_allowed=False,
                checked_summaries_before_failure=self._checked_summaries, current_descriptor_and_stage=self._current,
                full_pilot_completed_or_frozen=False, formal_cells=0)
            if owned:
                _write_once(path/'FAILURE.json', failure)
                write_evidence(path/'STATE.json', dict(status='failed_preserved_not_pilot_complete', **failure))
            if lock is not None:
                _write_once(lock.with_name(lock.stem+'.FAILURE.json'), failure)
            raise
