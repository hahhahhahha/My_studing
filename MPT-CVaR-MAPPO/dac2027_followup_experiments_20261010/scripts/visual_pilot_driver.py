"""SOURCE ONLY: one verified CUDA factory, OWN smoke MODEL loads, paired pilot.

Readonly cal/validation diagnostics, not formal training, test or pilot freeze.
Actual preparation, resource, model/load/collector and driver calls are zero.
No command-line launcher, automatic retry or import-time Torch/CUDA work.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import gc
import importlib
import os
from pathlib import Path

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility
from retained_test_evaluator import _write_once
from retained_test_grid_registration import _safe_path, _claim_grid_once
from visual_episode_collector import BASE, jsonable, require_actual_cuda_controller, write_evidence
from visual_learned_bindings import VisualPolicyDimensions, build_cuda_model, source_bindings
from visual_neural_controller import NeuralVisualController, VisualOptimizerSettings
from visual_pilot_episode import (SCOPE, RealReadonlyPilotEpisodeCollector, _registered_condition,
    _actual_smoke_weight_bindings, _unused_optimizer_identity, _own_smoke_structure_identity,
    pilot_episode_scientific_identity)
from visual_pilot_preparation import (PilotPreparationArtifactBindings, RealCalValidationPilotPreparation,
    validate_cal_validation_pilot_candidate, pilot_role_physical_windows)
from visual_protocol_assembly import _read_engineering_schedule, merge_frozen_file_bindings
from visual_receipt_environment import VisualReceiptEnvironment
from visual_training_coordinator import actual_parameter_sha256, environment_binding
from visual_training_transactions import METHODS, digest, file_sha, require_sha
from visual_upload_link import UploadLinkConfig

ORDER_RULE = 'calibration_then_validation_semantic_condition_then_QLA_PPO_v1'
DRIVER_SCOPE = 'paired_own_smoke_readonly_closed_cal_validation_engineering_pending_check'


def pilot_driver_job_descriptors(candidate):
    """Pure full paired order; artificial descriptors never prove execution."""
    rows = validate_cal_validation_pilot_candidate(candidate)['paired_engineering_descriptors']
    keys = [pilot_episode_scientific_identity(candidate, row, '0'*64) for row in rows]
    if len(set(keys)) != len(rows):
        raise ValueError('No renamed duplicate scientific episode in a paired engineering candidate')
    common_keys = {}
    for row in rows:
        reference = dict(row, method='QLA-MAPPO')
        common_keys[row['condition_id']] = pilot_episode_scientific_identity(candidate, reference, '0'*64)
    role_order = {'calibration': 0, 'validation': 1}
    return sorted(deepcopy(rows), key=lambda row: (role_order[row['role']], common_keys[row['condition_id']],
        METHODS.index(row['method'])))


def validate_pilot_driver_job_coverage(rows, candidate):
    expected = pilot_driver_job_descriptors(candidate)
    if (type(rows) is not list or any(type(row) is not dict for row in rows)
            or [digest(row) for row in rows] != [digest(row) for row in expected]):
        raise ValueError('Every common condition, both methods, whole declared N and exact semantic order required')
    return dict(common_conditions=len(expected)//2, paired_episode_descriptors=len(expected),
        actual_closed_episodes_proven=0, descriptor_check_not_completed_pilot=True)


def pilot_driver_scientific_identity(candidate, actual_parameter_sha256_by_method):
    """Semantic one-shot key, independent of labels/output/JSON/review paths."""
    hashes = actual_parameter_sha256_by_method
    if type(hashes) is not dict or set(hashes) != set(METHODS):
        raise ValueError('Exactly both OWN smoke parameter identities required')
    rows = pilot_driver_job_descriptors(candidate)
    episodes = [pilot_episode_scientific_identity(candidate, row, require_sha(hashes[row['method']])) for row in rows]
    return digest(dict(scope=DRIVER_SCOPE, order_rule=ORDER_RULE, ordered_scientific_episodes=episodes,
        diagnostic_H_MV_reset=True, optimizer_updates_allowed=False, final_pilot_freeze_allowed=False))


def _reference_identity(reference):
    require_actual_cuda_controller(reference)
    if reference.method != 'QLA-MAPPO':
        raise ValueError('Unused actual original one-factory QLA reference required')
    return dict(actual_parameters=actual_parameter_sha256(reference),
        original_structure=_own_smoke_structure_identity(reference), unused_optimizer=_unused_optimizer_identity(reference))


def load_own_smoke_model_for_readonly_pilot(reference, preparation, method, initialization_seed):
    """SOURCE ONLY dedicated pre-pilot MODEL loader, NOT a resume/val/test API.

    Its real reference comes from this driver's ONE original resource/kernel
    factory. No second GPU-free check against our own model allocation. Only
    actual own hash-bound smoke files can load; no optimizer/H/MV/RNG restore.
    Fresh constructors may advance/reseed RNG before readonly episode capture.
    """
    if type(preparation) is not RealCalValidationPilotPreparation:
        raise TypeError('Actual current pre-pilot preparation, no stub or final-pilot circular gate')
    if method not in METHODS or type(initialization_seed) is not int or initialization_seed < 0:
        raise ValueError('Actual own method and native registered fresh constructor seed required')
    preparation.verify_boundary()
    before = _reference_identity(reference)
    source_bindings()
    report, files = _actual_smoke_weight_bindings(preparation, method)
    model_path = _safe_path(Path(preparation.bindings.actual_smoke_path).parent/method/'checkpoint/MODEL_STATE.pt',
        BASE/'visual_calibrated_policy')
    if str(model_path) not in files or file_sha(model_path) != files[str(model_path)]:
        raise RuntimeError('Only the original actual OWN smoke MODEL payload, no initial/random/foreign weights')
    torch = reference._torch
    if method == 'QLA-MAPPO':
        module = importlib.import_module('mappo_gp_sf_dsr.models')
        model = module.MAPPOPolicy(deepcopy(reference.model.sys_cfg), deepcopy(reference.model.cfg),
            seed=initialization_seed, device='cuda')
        model.load_checkpoint(model_path, actor_only=False)
    else:
        module = importlib.import_module('mappo_gp_sf_dsr.paper_policy_baselines')
        torch.manual_seed(initialization_seed)
        model = module.SharedGaussianPPO(19, 4, -1.5, 1.).to('cuda')
        payload = torch.load(model_path, map_location='cuda', weights_only=True)
        if type(payload) is not dict or set(payload) != {'method', 'state_dict'} or payload['method'] != method:
            raise ValueError('Exact original own PPO MODEL-only schema required')
        model.load_state_dict(payload['state_dict'], strict=True)
    loaded = NeuralVisualController(method, model, VisualOptimizerSettings())
    expected_sha = require_sha(report['after_parameter_sha256'])
    structure, optimizer = _own_smoke_structure_identity(loaded), _unused_optimizer_identity(loaded)
    if actual_parameter_sha256(loaded) != expected_sha:
        raise RuntimeError('Actual loaded actor/critic parameters must equal the OWN actual smoke checkpoint')
    current_report, current_files = _actual_smoke_weight_bindings(preparation, method)
    if current_report != report or current_files != files or _reference_identity(reference) != before:
        raise RuntimeError('Own original checkpoint/input bytes or unused factory reference changed during loading')
    return loaded, dict(scope=SCOPE, method=method, own_actual_smoke_path=preparation.bindings.actual_smoke_path,
        own_actual_smoke_sha256=preparation.bindings.actual_smoke_sha256, model_payload_path=str(model_path),
        model_payload_sha256=files[str(model_path)], original_own_smoke_files_sha256=files,
        actual_loaded_parameter_sha256=expected_sha, actual_loaded_structure_sha256=structure,
        unused_optimizer_identity_sha256=optimizer, initialization_seed=initialization_seed,
        actual_MODEL_parameter_load_performed=True, optimizer_training_state_restored=False,
        H_MV_or_training_RNG_state_restored=False, fresh_constructor_may_advance_or_reseed_global_RNG=True,
        readonly_cal_validation_diagnostic_only=True, test_or_formal_training_permission=False,
        load_witness_is_not_independent_audit_or_model_output_roundtrip=True)


class RealPairedReadonlyPilotDriver:
    """SOURCE PREPARED ONLY. Actual constructor/run/model/load/episodes zero."""
    def __init__(self, bindings):
        if type(bindings) is not PilotPreparationArtifactBindings:
            raise TypeError('Actual current smoke/repair/entry/human-direction/candidate bindings required')
        self.preparation = RealCalValidationPilotPreparation(bindings)
        self.bindings = bindings
        self.candidate = deepcopy(self.preparation._candidate)
        self.jobs = pilot_driver_job_descriptors(self.candidate)
        self.reports, self.parameter_hashes = {}, {}
        self.inputs = dict(self.preparation._inputs)
        for method in METHODS:
            report, files = _actual_smoke_weight_bindings(self.preparation, method)
            self.reports[method] = report
            self.parameter_hashes[method] = require_sha(report['after_parameter_sha256'])
            self.inputs = merge_frozen_file_bindings(self.inputs, files)
        self.identity = pilot_driver_scientific_identity(self.candidate, self.parameter_hashes)
        self._completed_files, self._completed = {}, []
        self._started, self._failed, self._finished = False, False, False
        self._path, self._marker = None, None
        self._boundary()

    def _boundary(self):
        if self._failed:
            raise RuntimeError('Failed/partial paired pilot driver cannot append/retry/re-register')
        self.preparation.verify_boundary()
        if (self.candidate != self.preparation._candidate
                or validate_pilot_driver_job_coverage(self.jobs, self.candidate)['paired_episode_descriptors'] != len(self.jobs)
                or pilot_driver_scientific_identity(self.candidate, self.parameter_hashes) != self.identity):
            raise RuntimeError('No post-review candidate, method, schedule, scale, weight or job-order substitution')
        for method in METHODS:
            report, files = _actual_smoke_weight_bindings(self.preparation, method)
            if (report != self.reports[method] or report['after_parameter_sha256'] != self.parameter_hashes[method]
                    or any(self.inputs.get(name) != sha for name, sha in files.items())):
                raise RuntimeError('Both original own actual smoke reports/payloads/witnesses remain bound')
        for name, sha in merge_frozen_file_bindings(self.inputs, self._completed_files).items():
            path = _safe_path(name, BASE.parent)
            if not path.is_file() or file_sha(path) != sha:
                raise RuntimeError('Original protected inputs/source or already closed actual evidence changed')

    def _fresh_environment(self, job, catalog, mobility):
        if type(catalog) is not ExecutionServiceCatalog or type(mobility) is not PhysicalMobility:
            raise TypeError('Actual full role service pairs and true physical mobility only')
        row = _registered_condition(self.candidate, job)
        env = row['environment_descriptor']
        schedule = _read_engineering_schedule(row['schedule_artifact'], row['role'], row['decisions'], .05)
        if (catalog.role != row['role'] or catalog.source_sha256 != env['catalog_source_sha256']
                or mobility.source_hashes != env['mobility_source_sha256']
                or type(schedule.seed) is not int or schedule.seed != row['seed']):
            raise RuntimeError('Exactly the full original common role inputs and declared seed required')
        pilot_role_physical_windows(self.candidate, (mobility.common_start_seconds, mobility.common_end_seconds))
        end = float(Fraction(str(env['mobility_start_seconds']))
            +schedule.decisions*Fraction(str(schedule.decision_seconds))
            +Fraction(str(self.candidate['closure_settings']['fixed_drain_seconds'])))
        mobility.positions(env['mobility_start_seconds'])
        mobility.positions(end)
        actual = VisualReceiptEnvironment(schedule, catalog, mobility,
            tuple(tuple(x) for x in env['edge_positions']), tuple(env['profiles']), env['mobility_start_seconds'],
            env['broadcast_seconds'], env['nominal_delay_seconds'], env['waiting_capacity'], UploadLinkConfig(**env['upload_link']))
        if digest(jsonable(environment_binding(actual))) != digest(env):
            raise RuntimeError('Exact whole paired execution inputs/time/drain/link/placement, no hidden changes')
        return actual

    def run_once(self, new_directory):
        """Future actual run only. Nothing in the author check calls this API."""
        if self._started or self._failed or self._finished:
            raise RuntimeError('Permanent one-shot driver; never append/resume/retry partial paired episodes')
        self._started = True
        path, lock, owned, current, reference = None, None, False, None, None
        try:
            self._boundary()
            root = BASE/'visual_calibrated_policy/pilot_drivers'
            path = _safe_path(new_directory, root)
            if (path.parent.resolve() != root.resolve() or path.exists() or not 1 <= len(path.name) <= 24
                    or any(not (x.isascii() and (x.isalnum() or x in '-_')) for x in path.name)):
                raise ValueError('New short direct-child paired evidence directory, no overwrite or path changes')
            lock = _claim_grid_once(BASE/'visual_calibrated_policy/pilot_driver_locks', self.identity,
                dict(at=datetime.now().astimezone().isoformat(), output_path=str(path), pid=os.getpid(),
                    scope=DRIVER_SCOPE, automatic_append_resume_retry_allowed=False))
            path.mkdir(parents=True, exist_ok=False)
            owned, self._path = True, path
            started = path/'STARTED.json'
            _write_once(started, dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), scope=DRIVER_SCOPE,
                bindings=asdict(self.bindings), scientific_driver_identity_sha256=self.identity, order_rule=ORDER_RULE,
                jobs=self.jobs, original_inputs_sha256=self.inputs, actual_parameter_sha256_by_method=self.parameter_hashes,
                original_resource_and_seeded_kernel_gate_required_before_first_CUDA=True,
                initialized_reference_never_used_for_policy_actions=True, optimizer_updates_allowed=False,
                full_pilot_completed_or_frozen=False, formal_training_or_test_permission=False))
            self._completed_files[str(started)] = file_sha(started)
            self._boundary()
            # ONE original factory: resource <500MiB + separate seeded CUDA
            # witness before first CUDA. Never test free-memory against self.
            reference = NeuralVisualController('QLA-MAPPO',
                build_cuda_model('QLA-MAPPO', VisualPolicyDimensions(30, 4, 19, 705, 30), {}, 1), VisualOptimizerSettings())
            reference_identity = _reference_identity(reference)
            catalogs = {role: ExecutionServiceCatalog(role) for role in ('calibration', 'validation')}
            mobility = PhysicalMobility(users=30)
            for index, job in enumerate(self.jobs, 1):
                self._boundary()
                current = dict(index=index, descriptor=deepcopy(job), phase='own_actual_smoke_MODEL_only_loading')
                loaded, witness = load_own_smoke_model_for_readonly_pilot(reference, self.preparation, job['method'], job['seed'])
                witness_path = path/f'L{index:03d}.json'
                _write_once(witness_path, witness)
                self._completed_files[str(witness_path)] = file_sha(witness_path)
                self._boundary()
                current['phase'] = 'full_closed_readonly_engineering_episode'
                environment = self._fresh_environment(job, catalogs[job['role']], mobility)
                collector = RealReadonlyPilotEpisodeCollector(self.preparation, job, loaded)
                episode_path = BASE/'visual_calibrated_policy/pilot_episodes'/f'{path.name}_E{index:03d}'
                complete = collector.collect_once(environment, episode_path)
                collector._verify_boundary()
                if (collector._finished is not True or collector._failed or collector._complete != complete
                        or collector._path != episode_path or complete['descriptor'] != job
                        or complete['scientific_identity_sha256'] != pilot_episode_scientific_identity(self.candidate, job, self.parameter_hashes[job['method']])
                        or _reference_identity(reference) != reference_identity):
                    raise RuntimeError('Actual complete own-weight collector and untouched unused reference required')
                marker = episode_path/'CLOSED_EPISODE_PENDING_REVIEW.json'
                files = {str(marker): file_sha(marker)}
                for name, sha in complete['files_sha256'].items():
                    child = _safe_path(episode_path/name, BASE/'visual_calibrated_policy')
                    if Path(name).name != name or not child.is_file() or file_sha(child) != sha:
                        raise RuntimeError('All original closed episode files, no missing/partial/changed export')
                    files[str(child)] = sha
                self._completed_files = merge_frozen_file_bindings(self._completed_files, files)
                self._completed.append(dict(index=index, descriptor=deepcopy(job), episode_path=str(episode_path),
                    closed_marker_path=str(marker), closed_marker_sha256=files[str(marker)],
                    own_MODEL_load_witness_path=str(witness_path), own_MODEL_load_witness_sha256=file_sha(witness_path),
                    scientific_episode_identity_sha256=collector.identity))
                self._boundary()
                write_evidence(path/'STATE.json', dict(status='partial_paired_readonly_episodes_not_pilot_complete',
                    pid=os.getpid(), completed=self._completed, expected_episodes=len(self.jobs),
                    optimizer_updates=0, formal_cells=0, automatic_append_resume_retry_allowed=False))
                del environment, collector, loaded
                gc.collect()
                reference._torch.cuda.empty_cache()  # release only our unused cache, never a driver/system reset
            validate_pilot_driver_job_coverage([row['descriptor'] for row in self._completed], self.candidate)
            self._boundary()
            if _reference_identity(reference) != reference_identity:
                raise RuntimeError('Unused factory model/optimizer/configuration changed')
            inventory = path/'CLOSED_DRIVER_INPUTS.json'
            _write_once(inventory, dict(scope=DRIVER_SCOPE, completed=self._completed, original_inputs_sha256=self.inputs,
                immutable_actual_episode_and_load_files_sha256=self._completed_files,
                mutable_grid_STATE_not_result_evidence=True, optimizer_updates=0, formal_cells=0))
            self._completed_files[str(inventory)] = file_sha(inventory)
            self._boundary()
            marker = path/'ACTUAL_PAIRED_EPISODES_PENDING_REVIEW.json'
            _write_once(marker, dict(status='actual_paired_readonly_closed_episodes_pending_independent_engineering_check',
                scope=DRIVER_SCOPE, scientific_driver_identity_sha256=self.identity, order_rule=ORDER_RULE,
                completed=self._completed, actual_parameter_sha256_by_method=self.parameter_hashes,
                original_inputs_sha256=self.inputs, immutable_closed_files_sha256=self._completed_files,
                all_registered_conditions_both_methods_closed=True, actual_own_MODEL_loads_and_episodes=len(self.jobs),
                original_factory_invocations=1, unused_reference_identity_unchanged=True,
                optimizer_updates=0, diagnostic_H_MV_reset_not_formal_training_carry_test=True,
                full_pilot_completed=False, final_pilot_frozen=False, independent_entry_or_result_audit=False,
                metadata_schedule_generation_permission=False, formal_training_or_test_permission=False, formal_cells=0))
            self._completed_files[str(marker)] = file_sha(marker)
            self._boundary()
            write_evidence(path/'STATE.json', dict(status='paired_episodes_closed_pending_independent_engineering_check',
                pid=os.getpid(), completed=self._completed, actual_closed_episodes=len(self.jobs),
                full_pilot_completed_or_frozen=False, optimizer_updates=0, formal_cells=0))
            self._marker, self._finished = marker, True
            return marker, file_sha(marker)
        except Exception as error:
            self._failed = True
            failure = dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), scope=DRIVER_SCOPE,
                error=repr(error), completed_episode_markers_preserved=self._completed,
                immutable_completed_files_preserved=self._completed_files, current_descriptor_and_phase=current,
                partial_current_collector_evidence_retained_in_its_own_attempt=True,
                automatic_append_resume_retry_or_partial_arm_merge_allowed=False,
                original_raw_or_checkpoint_modified=False, full_pilot_completed_or_frozen=False, formal_cells=0)
            if owned:
                _write_once(path/'FAILURE.json', failure)
                write_evidence(path/'STATE.json', dict(status='failed_preserved_not_pilot_complete', **failure))
            if lock is not None:
                _write_once(lock.with_name(lock.stem+'.FAILURE.json'), failure)
            raise
