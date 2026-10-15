"""SOURCE ONLY: actual readonly, closed cal/validation engineering episode.

No factory/launcher/optimizer/fitting/selection/pilot freeze/test execution.
Receives a real current preparation and actual freshly loaded OWN functional
smoke parameters. A separate reviewed resource/load driver is still required.
No actual constructor or collect call has run in the engineering checks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import os
from pathlib import Path
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility
from retained_test_evaluator import _check_reporting_final, _claim_once, _write_once
from retained_test_grid_registration import _read_actual, _safe_path
from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable, require_actual_cuda_controller, write_evidence
from visual_method_features import VisualFeatureScales
from visual_learned_bindings import VisualPolicyDimensions
from visual_mv_stream import PersistentMVStream
from visual_neural_controller import VisualOptimizerSettings
from visual_pilot_preparation import RealCalValidationPilotPreparation, validate_cal_validation_pilot_candidate, pilot_role_physical_windows
from visual_protocol_assembly import _complete_dataclass, _read_engineering_schedule, merge_frozen_file_bindings
from visual_receipt_environment import VisualReceiptEnvironment
from visual_training_coordinator import CompactCoordinatorAssembly, actual_parameter_sha256, closed_episode_counts, environment_binding
from visual_training_transactions import METHODS, digest, file_sha, inspect_checkpoint_archive, require_sha

SCOPE = 'actual_own_functional_smoke_weights_closed_no_optimizer_cal_validation_engineering_only'


def _registered_condition(candidate, descriptor):
    checked = validate_cal_validation_pilot_candidate(candidate)
    if (type(descriptor) is not dict
            or sum(digest(row) == digest(descriptor) for row in checked['paired_engineering_descriptors']) != 1):
        raise ValueError('Exact common registered paired descriptor; booleans/extra fields/method-specific conditions refused')
    return deepcopy(next(row for row in candidate['engineering_conditions'] if row['condition_id'] == descriptor['condition_id']))


def pilot_episode_scientific_identity(candidate, descriptor, parameter_sha256):
    """Pure semantic lock, not proof that an actual episode/model exists.

    Cosmetic condition names, schedule-file paths, review paths, output names
    or JSON encoding do not permit a second execution of the same condition.
    This phase resets diagnostic H/MV, never claims to test training carry.
    """
    condition = _registered_condition(candidate, descriptor)
    return digest(dict(scope=SCOPE, method=descriptor['method'], role=condition['role'],
        seed=condition['seed'], decisions=condition['decisions'], users=candidate['users'],
        decision_seconds=candidate['decision_seconds'], common_environment=condition['environment_descriptor'],
        common_schedule_semantic_sha256=condition['schedule_artifact']['semantic_sha256'],
        feature_scales=candidate['feature_scales'], closure_settings=candidate['closure_settings'],
        placement_calibration_window_seconds=candidate['placement_calibration_window_seconds'],
        placement_seed=candidate['placement_seed'], mobility_role_gap_seconds=candidate['mobility_role_gap_seconds'],
        actual_own_smoke_parameter_sha256=require_sha(parameter_sha256),
        diagnostic_H_and_MV_reset_for_this_independent_episode=True, optimizer_updates_allowed=False))


def _actual_smoke_weight_bindings(preparation, method):
    preparation.verify_boundary()
    b = preparation.bindings
    root = BASE/'visual_calibrated_policy'
    smoke = _read_actual(b.actual_smoke_path, b.actual_smoke_sha256, root)
    reports = smoke['method_reports']
    if (type(reports) is not list or len(reports) != 2 or tuple(row['method'] for row in reports) != METHODS):
        raise RuntimeError('Both actual original method smoke reports in original order required')
    report = next(row for row in reports if row['method'] == method)
    child = _safe_path(Path(b.actual_smoke_path).parent/method, root)
    marker = _safe_path(child/'METHOD_SMOKE_PASS.json', root)
    marker_sha = file_sha(marker)
    if _read_actual(marker, marker_sha, root) != report:
        raise RuntimeError('Actual child smoke bytes must retain the externally bound parent report values')
    if (report['status'] != 'actual_cuda_method_smoke_pass'
            or report['actual_model_forward_and_optimizer_executed'] is not True
            or report['actual_roundtrip_performed'] is not True
            or report['exact_loaded_parameters_and_deterministic_outputs'] is not True
            or report['deterministic_Torch_RNG_unchanged'] is not True
            or report['source_files_sha256'] != smoke['source_files_sha256']
            or report['protected_input_and_source_sha256'] != smoke['protected_input_and_source_sha256']
            or report['plan_sha256'] != smoke['plan_sha256']
            or report['semantic_identity_sha256'] != smoke['semantic_identity_sha256']
            or report['input_condition_sha256'] != smoke['input_condition_sha256']):
        raise RuntimeError('Current actual forward/update/load/roundtrip witness, never an artificial model')
    checkpoint = _safe_path(child/'checkpoint', root)
    archive = report['checkpoint_archive']
    inspect_checkpoint_archive(checkpoint, archive['descriptor'], archive['artifact_sha256'])
    complete = _read_actual(checkpoint/'CHECKPOINT_COMPLETE.json', archive['artifact_sha256'], root)
    files = {str(marker): marker_sha, str(checkpoint/'CHECKPOINT_COMPLETE.json'): archive['artifact_sha256']}
    witnesses = report['roundtrip_witness_files_sha256']
    if set(witnesses) != {'ACTUAL_ROUNDTRIP_INPUT_OUTPUT.npz', 'ACTUAL_READONLY_TORCH_RNG.npz'}:
        raise ValueError('Whole original actual smoke roundtrip and RNG witness files required')
    for name, sha in witnesses.items():
        witness = _safe_path(child/name, root)
        if not witness.is_file() or file_sha(witness) != require_sha(sha):
            raise RuntimeError('Original actual model roundtrip/RNG evidence changed; never synthesize replacement witnesses')
        files[str(witness)] = sha
    for name, sha in complete['files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Exact own direct checkpoint payload, no hidden path')
        path = _safe_path(checkpoint/name, root)
        if file_sha(path) != require_sha(sha):
            raise RuntimeError('Original actual smoke payload changed')
        files[str(path)] = sha
    return deepcopy(report), files


def _unused_optimizer_identity(controller):
    require_actual_cuda_controller(controller)
    state = controller.optimizer.state_dict()
    if controller._started_episode_tokens or state['state'] != {}:
        raise RuntimeError('Fresh own MODEL-only loaded controller, no optimizer transaction/state restoration')
    return digest(state)


def _own_smoke_structure_identity(controller):
    """Actual structure/configuration matters in addition to parameter bytes."""
    require_actual_cuda_controller(controller)
    if controller.settings != VisualOptimizerSettings():
        raise ValueError('Original own functional-smoke optimizer metadata, though unused, must remain unchanged')
    model = controller.model
    if controller.method == 'QLA-MAPPO':
        expected = type(model.cfg)(edgewise_price_actor=True, lcc_price_logit_bias=.5)
        dimensions = VisualPolicyDimensions(30, 4, 19, 705, 30)
        if asdict(model.cfg) != asdict(expected) or asdict(model.sys_cfg) != asdict(dimensions):
            raise ValueError('Own smoke original Full configuration and30/4/19/705/.5 structure, not parameter-equal altered behavior')
        structure = dict(dimensions=asdict(dimensions), configuration=asdict(model.cfg))
    else:
        if (model.actor[0].in_features != 19 or model.critic[0].in_features != 19
                or model.log_std.numel() != 4 or model.log_std_min != -1.5 or model.log_std_max != 1.):
            raise ValueError('Original own smoke PPO19/4/default log-std limits required')
        structure = dict(actor_obs_dim=19, n_edges=4, log_std_min=model.log_std_min, log_std_max=model.log_std_max)
    return digest(dict(method=controller.method, structure=structure, unused_optimizer_settings=asdict(controller.settings)))


class RealReadonlyPilotEpisodeCollector:
    """SOURCE PREPARED ONLY. Actual constructor/collection/evidence count zero.

    Actual model loading/resource factory belongs to a separate reviewed
    driver. Matching real CUDA parameters is mandatory here; initial random
    weights or author-made action/value fixtures can never substitute for it.
    """
    def __init__(self, preparation, descriptor, fresh_loaded_controller):
        if type(preparation) is not RealCalValidationPilotPreparation:
            raise TypeError('Actual current live pre-pilot preparation required, no completed-pilot circular prerequisite')
        preparation.verify_boundary()
        self.preparation, self.descriptor = preparation, deepcopy(descriptor)
        self.condition = _registered_condition(preparation._candidate, descriptor)
        require_actual_cuda_controller(fresh_loaded_controller)
        if fresh_loaded_controller.method != descriptor['method']:
            raise ValueError('Actual own method, not another arm or a rule/fake controller')
        self.controller = fresh_loaded_controller
        report, weight_files = _actual_smoke_weight_bindings(preparation, descriptor['method'])
        self.parameter_sha = require_sha(report['after_parameter_sha256'])
        if actual_parameter_sha256(self.controller) != self.parameter_sha:
            raise RuntimeError('Actual loaded parameters must equal this OWN functional smoke checkpoint, not initial weights')
        self.optimizer_identity = _unused_optimizer_identity(self.controller)
        self.structure_identity = _own_smoke_structure_identity(self.controller)
        candidate = preparation._candidate
        self.scales = _complete_dataclass(VisualFeatureScales, candidate['feature_scales'],
            ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
        self.settings = _complete_dataclass(EpisodeClosureCandidate, candidate['closure_settings'])
        self.identity = pilot_episode_scientific_identity(candidate, descriptor, self.parameter_sha)
        self.inputs = merge_frozen_file_bindings(preparation._inputs, weight_files)
        self._started, self._finished, self._failed = False, False, False
        self._assembly, self._path, self._complete = None, None, None

    def _verify_boundary(self):
        if self._failed:
            raise RuntimeError('Failed episode cannot append/retry/re-register')
        self.preparation.verify_boundary()
        current = _registered_condition(self.preparation._candidate, self.descriptor)
        if (current != self.condition
                or pilot_episode_scientific_identity(self.preparation._candidate, self.descriptor, self.parameter_sha) != self.identity
                or actual_parameter_sha256(self.controller) != self.parameter_sha
                or _unused_optimizer_identity(self.controller) != self.optimizer_identity
                or _own_smoke_structure_identity(self.controller) != self.structure_identity):
            raise RuntimeError('Actual condition/weights/optimizer state changed; no outcome-selected replacement')
        for name, sha in self.inputs.items():
            path = _safe_path(name, BASE.parent)
            if not path.is_file() or file_sha(path) != sha:
                raise RuntimeError('Original protected/input/smoke payload bytes changed')

    def _require_fresh_environment(self, environment):
        if (type(environment) is not VisualReceiptEnvironment
                or type(environment._catalog) is not ExecutionServiceCatalog
                or type(environment._mobility) is not PhysicalMobility
                or environment._index != 0 or environment._pending is not None
                or environment._unsettled is not None or environment._failed or environment._finished
                or environment._issued or environment._admissions or environment._settled_receipts):
            raise ValueError('Fresh actual original role queue/upload/controller state required, no test or fixture')
        row = self.condition
        schedule = _read_engineering_schedule(row['schedule_artifact'], row['role'], row['decisions'], self.scales.decision_seconds)
        if (type(schedule.seed) is not int or schedule.seed != row['seed']
                or digest(environment._schedule.to_dict()) != digest(schedule.to_dict())
                or digest(jsonable(environment_binding(environment))) != digest(row['environment_descriptor'])):
            raise RuntimeError('Exact full registered schedule/role/seed/environment, no omitted rows or changed windows')
        pilot_role_physical_windows(self.preparation._candidate,
            (environment._mobility.common_start_seconds, environment._mobility.common_end_seconds))
        end = float(Fraction(str(environment._mobility_start))
            +schedule.decisions*Fraction(str(schedule.decision_seconds))+Fraction(str(self.settings.fixed_drain_seconds)))
        environment._mobility.positions(environment._mobility_start)
        environment._mobility.positions(end)
        return schedule

    def collect_once(self, environment, new_directory):
        if self._started or self._failed or self._finished:
            raise RuntimeError('One-shot complete episode only; no append/retry/partial-arm merge')
        self._verify_boundary()
        schedule = self._require_fresh_environment(environment)
        root = BASE/'visual_calibrated_policy/pilot_episodes'
        path = _safe_path(new_directory, root)
        if (path.parent.resolve() != root.resolve() or path.exists() or not 1 <= len(path.name) <= 32
                or any(not (char.isascii() and (char.isalnum() or char in '-_')) for char in path.name)):
            raise ValueError('New short direct child evidence directory, never overwrite/long-path global changes')
        self._started = True
        lock = _claim_once(BASE/'visual_calibrated_policy/pilot_episode_locks', self.identity,
            dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), output_path=str(path), scope=SCOPE,
                automatic_append_retry_or_recovery_allowed=False))
        owned, stream, result = False, None, None
        try:
            path.mkdir(parents=True, exist_ok=False)
            owned, self._path = True, path
            _write_once(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
                scope=SCOPE, descriptor=self.descriptor, scientific_identity_sha256=self.identity,
                original_preparation_bindings=asdict(self.preparation.bindings), inputs_sha256=self.inputs,
                own_functional_smoke_parameter_sha256=self.parameter_sha,
                optimizer_updates_allowed=False, full_pilot_completed=False, formal_training_or_test_cell=False))
            torch = self.controller._torch
            rng_before = [torch.get_rng_state().clone(), *[x.clone() for x in torch.cuda.get_rng_state_all()]]
            # These are independent diagnostic episodes, not formal training
            # carry/reset tests. Do not restore smoke H/MV/optimizer/RNG state.
            if self.descriptor['method'] == 'PPO-MV-A':
                purpose = 'training' if self.condition['role'] == 'calibration' else 'engineering_validation_receipts'
                stream = PersistentMVStream(30, self.settings.mv_mu_seconds, purpose)
            assembly = CompactCoordinatorAssembly(self.descriptor['method'], 30, 4, schedule.decisions,
                self.scales, self.settings, stream, 1, self.identity, initial_H=0.)
            self._assembly = assembly
            for _ in range(schedule.decisions):
                pre = environment.observe()
                packed = assembly.features_for(pre)
                sample = self.controller.act(packed, deterministic=True)
                assembly.execute_validated_sample(environment, pre, packed, sample)
                write_evidence(path/'STATE.json', dict(phase='readonly_collecting_full_engineering_episode',
                    pid=os.getpid(), decisions_collected=len(assembly.transitions), optimizer_updates=0))
            result = assembly.close_after_fixed_drain(environment)
            prepared = result['prepared_episode']
            counts = asdict(closed_episode_counts(prepared, 30, schedule.decisions, self.scales.decision_seconds))
            generation = schedule.horizon_seconds
            end = float(schedule.decisions*Fraction(str(schedule.decision_seconds))+Fraction(str(self.settings.fixed_drain_seconds)))
            reporting = _check_reporting_final(result['final'], len(schedule.requests), len(assembly.transitions),
                dict(generation_decisions=schedule.decisions, generation_seconds=generation,
                    decision_seconds=schedule.decision_seconds, fixed_end_seconds=end, profiles=list(self.scales.profiles)))
            rng_after = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
            if len(rng_before) != len(rng_after) or not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
                raise RuntimeError('Readonly deterministic engineering actions changed Torch RNG')
            self._verify_boundary()
            _write_once(path/'CLOSED_ENGINEERING_EPISODE.json', {name: value for name, value in result.items() if name != 'prepared_episode'})
            _write_once(path/'EXECUTION_RECORDS_REPORTING_ONLY.json',
                jsonable(deepcopy(environment._feedback._execution.records)))
            np.savez_compressed(path/'ACTUAL_BEHAVIOR_AND_GAE.npz', **prepared.time_user_fields,
                advantages=prepared.advantages, returns=prepared.returns, physical_durations=prepared.physical_durations)
            np.savez_compressed(path/'ACTUAL_READONLY_RNG.npz',
                **{f'before_{i}': value.cpu().numpy() for i, value in enumerate(rng_before)},
                **{f'after_{i}': value.cpu().numpy() for i, value in enumerate(rng_after)})
            names = ('STARTED.json', 'CLOSED_ENGINEERING_EPISODE.json', 'EXECUTION_RECORDS_REPORTING_ONLY.json',
                'ACTUAL_BEHAVIOR_AND_GAE.npz', 'ACTUAL_READONLY_RNG.npz')
            files = {name: file_sha(path/name) for name in names}
            self._verify_boundary()
            complete = dict(status='actual_closed_no_optimizer_pilot_episode_pending_independent_check',
                scope=SCOPE, descriptor=self.descriptor, scientific_identity_sha256=self.identity, files_sha256=files,
                original_preparation_bindings=asdict(self.preparation.bindings), inputs_sha256=self.inputs,
                full_behavior_counts=counts, uncapped_reliability_diagnostics=reporting,
                whole_generated_requests=len(schedule.requests), fixed_generation_seconds=generation, fixed_drain_end_seconds=end,
                actual_own_functional_smoke_parameter_sha256_before_and_after=self.parameter_sha,
                unused_optimizer_identity_sha256_before_and_after=self.optimizer_identity,
                own_smoke_structure_identity_sha256_before_and_after=self.structure_identity,
                actual_model_actions_and_values_collected=True, deterministic_Torch_RNG_unchanged=True,
                optimizer_updates=0, smoke_training_H_MV_optimizer_or_RNG_restored=False,
                independent_episode_diagnostic_H_MV_reset=True, formal_H_carry_rule_exercised=False,
                data_are_engineering_diagnostics_not_formal_learning_results=True,
                complete_candidate_conditions_registered=False, full_pilot_completed=False, final_pilot_frozen=False,
                actual_formal_training_or_test_cells=0, independent_entry_or_result_audit_done=False)
            _write_once(path/'CLOSED_EPISODE_PENDING_REVIEW.json', complete)
            self._complete = complete
            write_evidence(path/'STATE.json', dict(phase='complete_closed_engineering_pending_independent_check', pid=os.getpid(), optimizer_updates=0))
            self._finished = True
            return deepcopy(complete)
        except Exception as error:
            self._failed = True
            self.controller._failed = True
            if stream is not None:
                stream.mark_failed(error)
            # Privileged actual execution truth is saved ONLY as diagnostics;
            # never fed back into current policy features or replacement rows.
            try:
                actual_execution_snapshot = jsonable(deepcopy(environment._feedback._execution.records))
            except Exception as snapshot_error:
                actual_execution_snapshot = dict(snapshot_error=repr(snapshot_error))
            failure = dict(at=datetime.now().astimezone().isoformat(), error=repr(error), scope=SCOPE,
                partial_evidence_preserved=True, optimizer_updates=0, automatic_append_retry_or_recovery_allowed=False,
                observations=self._assembly.observations if self._assembly else [],
                ordinary_transitions=self._assembly.transitions if self._assembly else [],
                partial_behavior_rows=self._assembly.buffer._rows if self._assembly else [],
                partial_actual_execution_records_reporting_only=actual_execution_snapshot,
                actual_closed_reporting_result_if_available={name: value for name, value in result.items()
                    if name != 'prepared_episode'} if result is not None else None,
                MV_failure_evidence=stream.failure_evidence if stream is not None else None,
                full_pilot_completed=False, actual_formal_training_or_test_cells=0)
            if owned:
                write_evidence(path/'FAILURE.json', failure)
                write_evidence(path/'STATE.json', dict(phase='failed', pid=os.getpid(), optimizer_updates=0))
            _write_once(lock.with_suffix('.FAILURE.json'), dict(error=repr(error), owned_output_directory=owned))
            raise
