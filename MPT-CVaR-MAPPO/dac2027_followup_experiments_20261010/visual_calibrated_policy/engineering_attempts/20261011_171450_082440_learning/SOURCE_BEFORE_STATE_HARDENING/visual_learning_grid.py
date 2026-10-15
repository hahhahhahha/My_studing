"""SOURCE ONLY: ten paired train/final-checkpoint/readonly-validation jobs.

Real pilot/full protocol/smoke/Type-A/resource gates precede any model work.
No import-time model, automatic launch/retry, test evaluation or CPU fallback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import importlib
from pathlib import Path

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility
from retained_test_grid_execution import policy_dimensions_from_frozen_scales
from retained_test_grid_registration import _safe_path, _read_actual, _claim_grid_once
from retained_test_evaluator import _write_once
from visual_checkpoint_load import load_own_snapshot_in_verified_cuda_process
from visual_checkpoint_registration import FinalCheckpointRuleCandidate, RealFinalCheckpointRegistrar
from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable, require_actual_cuda_controller, write_evidence
from visual_learned_bindings import build_cuda_model, source_bindings
from visual_method_features import VisualFeatureScales
from visual_neural_controller import NeuralVisualController, VisualOptimizerSettings
from visual_protocol_assembly import (_complete_dataclass, _read_engineering_schedule,
    merge_frozen_file_bindings, validate_environment_descriptor)
from visual_receipt_environment import VisualReceiptEnvironment
from visual_training_coordinator import (FrozenRuntimeArtifactBindings, require_reviewed_runtime,
    RealMultiepisodeTrainingCoordinator, RealReadonlyValidationEvaluator, actual_parameter_sha256, environment_binding)
from visual_training_transactions import TrainingTransactionLedger, CommonBudgetCandidate, digest, file_sha, require_sha
from visual_upload_link import UploadLinkConfig

METHODS = ('QLA-MAPPO', 'PPO-MV-A')
ORDER_RULE = 'seed1to5_then_QLA_PPO_train_final_all_registered_validations_v1'


def common_learning_job_descriptors(rule):
    """Pure declared identities/budgets ONLY, not actual training or weights."""
    if type(rule) is not FinalCheckpointRuleCandidate:
        raise TypeError('Complete declared final-common-budget checkpoint rule required')
    return [dict(method=method, seed=seed, common_budget=jsonable(asdict(rule.budget)),
        training_role='calibration', validation_role='validation', test_access=False,
        checkpoint_rule=rule.definition()['strategy'], performance_filter_allowed=False)
        for seed in rule.seeds for method in METHODS]


def validate_learning_job_coverage(rows, rule):
    expected = common_learning_job_descriptors(rule)
    if (type(rows) is not list or len(rows) != 10 or any(type(row) is not dict for row in rows)
            or any(type(row.get('seed')) is not int for row in rows)
            or sorted(digest(row) for row in rows) != sorted(digest(row) for row in expected)):
        raise ValueError('Exact two methods/five native seeds/common budget, no missing/duplicate/ranked/test job')
    return dict(descriptor_jobs=10, paired_seeds=5, actual_training_or_validation_proven=0)


def learning_scientific_identity(protocol, rule):
    """Pure design identity independent of output/review/JSON cosmetic names."""
    if type(protocol) is not dict:
        raise TypeError('Explicit common scientific design required')
    return digest(dict(rule=rule.definition(), paired_design=protocol['paired_design_by_seed'],
        feature_scales=protocol['feature_scales'], closure_settings=protocol['closure_settings'],
        optimizer_settings=protocol['optimizer_settings'], learner_kwargs=protocol['learner_kwargs_by_method'],
        H_rule=protocol['virtual_H_episode_rule'], training_schedules=protocol['training_schedule_sha256_by_seed'],
        validation_schedules=protocol['validation_schedule_sha256_by_seed'],
        training_environments=protocol['training_environment_sha256_by_seed'],
        validation_environments=protocol['validation_environment_sha256_by_schedule']))


def _require_training_runtime(bindings):
    # Must fail before actual model/Torch/path output if real authority absent.
    protocol = require_reviewed_runtime(bindings)
    if protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__)):
        raise RuntimeError('This exact complete train/validation/final-registration entry needs current Type-A source review')
    inputs = {}
    for mapping in (protocol['source_files_sha256'], protocol['protected_input_and_source_sha256'],
                    protocol['input_files_sha256']):
        inputs = merge_frozen_file_bindings(inputs, mapping)
    inputs = merge_frozen_file_bindings(inputs, {
        bindings.protocol_path: bindings.protocol_sha256,
        bindings.entry_review_path: bindings.entry_review_sha256,
        protocol['actual_model_smoke_path']: protocol['actual_model_smoke_sha256']})
    for name, sha in inputs.items():
        path = _safe_path(name, BASE.parent)
        if not path.is_file() or file_sha(path) != require_sha(sha):
            raise RuntimeError('Protected raw/source/pilot/protocol/common schedule evidence bytes changed')
    return protocol, inputs


def _new_actual_training_controller(reference, method, seed, dimensions, settings):
    """Fresh INITIAL model for actual training, never random test substitution.

    The runner's unused reference comes from the ONE original CUDA factory
    gate in this process. No second free-memory gate against own allocation.
    This function is not a resume/checkpoint/training-state restoration API.
    """
    require_actual_cuda_controller(reference)
    if (reference.method != 'QLA-MAPPO' or reference._started_episode_tokens
            or method not in METHODS or type(seed) is not int or seed not in (1, 2, 3, 4, 5)
            or asdict(reference.model.sys_cfg) != asdict(dimensions) or reference.settings != settings):
        raise ValueError('Unused actual verified architecture and own native training seed required')
    source_bindings()
    if method == 'QLA-MAPPO':
        module = importlib.import_module('mappo_gp_sf_dsr.models')
        model = module.MAPPOPolicy(deepcopy(reference.model.sys_cfg), deepcopy(reference.model.cfg), seed=seed, device='cuda')
    else:
        module = importlib.import_module('mappo_gp_sf_dsr.paper_policy_baselines')
        reference._torch.manual_seed(seed)
        model = module.SharedGaussianPPO(dimensions.actor_obs_dim, dimensions.n_edges).to('cuda')
    controller = NeuralVisualController(method, model, settings)
    if controller._started_episode_tokens:
        raise RuntimeError('No reused controller/optimizer transactions between independent training jobs')
    return controller


class RealTenLearningValidationRunner:
    """SOURCE PREPARED ONLY. Actual constructor/run/training/validation zero."""
    def __init__(self, bindings):
        if type(bindings) is not FrozenRuntimeArtifactBindings:
            raise TypeError('Actual completed immutable pilot/smoke/full-entry bindings required')
        self.protocol, self.inputs = _require_training_runtime(bindings)
        self.bindings = bindings
        p = self.protocol
        self.budget = _complete_dataclass(CommonBudgetCandidate, p['common_budget'], ('episode_decisions',))
        self.rule = FinalCheckpointRuleCandidate(self.budget)
        if p['checkpoint_selection_rule'] != self.rule.definition() or p['checkpoint_rule_registered_before_training'] is not True:
            raise RuntimeError('Final common budget rule must be frozen before any training, no outcome ranking')
        self.scales = _complete_dataclass(VisualFeatureScales, p['feature_scales'],
            ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
        self.closure = _complete_dataclass(EpisodeClosureCandidate, p['closure_settings'])
        self.optimizer = _complete_dataclass(VisualOptimizerSettings, p['optimizer_settings'])
        self.dimensions = policy_dimensions_from_frozen_scales(self.scales, self.budget.users)
        if self.budget.fixed_drain_seconds != self.closure.fixed_drain_seconds or self.budget.decision_seconds != self.scales.decision_seconds:
            raise ValueError('Same fixed physical budget/closure/scales for both methods')
        self.jobs = common_learning_job_descriptors(self.rule)
        self.identity = learning_scientific_identity(p, self.rule)
        self._started, self._failed, self._finished = False, False, False

    def _boundary(self):
        protocol, inputs = _require_training_runtime(self.bindings)
        if digest(protocol) != digest(self.protocol) or inputs != self.inputs:
            raise RuntimeError('No post-entry protocol/raw/source rebinding')
        if self.jobs != common_learning_job_descriptors(self.rule) or learning_scientific_identity(protocol, self.rule) != self.identity:
            raise RuntimeError('No outcome-based job removal/reordering/reconfiguration')

    def _environment(self, role, seed, index, catalog, mobility):
        if role not in ('calibration', 'validation') or catalog.role != role:
            raise ValueError('Training/readonly validation role only, no retained test activation')
        prefix = 'training' if role == 'calibration' else 'validation'
        reg = self.protocol[prefix+'_schedule_artifacts_by_seed'][str(seed)][index]
        raw = _read_actual(reg['path'], reg['file_sha256'], BASE/'visual_calibrated_policy')
        decisions = self.budget.episode_decisions[index] if role == 'calibration' else raw['decisions']
        schedule = _read_engineering_schedule(reg, role, decisions, self.budget.decision_seconds)
        expected_schedule = self.protocol[prefix+'_schedule_sha256_by_seed'][str(seed)][index]
        if schedule.fingerprint() != expected_schedule:
            raise RuntimeError('Exact full predeclared ordered common role schedule required')
        binding = self.protocol['cal_validation_frozen_components'][prefix+'_environment_descriptors_by_seed'][str(seed)][index]
        binding = validate_environment_descriptor(binding, role, expected_schedule, self.protocol['test_matrix_definition']['broadcast_seconds'])
        expected_env = (self.protocol['training_environment_sha256_by_seed'][str(seed)][index]
            if role == 'calibration' else self.protocol['validation_environment_sha256_by_schedule'][expected_schedule])
        if (binding['catalog_source_sha256'] != catalog.source_sha256 or binding['mobility_source_sha256'] != mobility.source_hashes
                or binding['edge_positions'] != jsonable(self.scales.edge_positions) or digest(binding) != expected_env):
            raise RuntimeError('Common original execution pairs/physical mobility/default link/geometry changed')
        mobility.positions(binding['mobility_start_seconds']+schedule.horizon_seconds+self.closure.fixed_drain_seconds)
        env = VisualReceiptEnvironment(schedule, catalog, mobility, self.scales.edge_positions, self.scales.profiles,
            binding['mobility_start_seconds'], binding['broadcast_seconds'], binding['nominal_delay_seconds'],
            binding['waiting_capacity'], UploadLinkConfig(**binding['upload_link']))
        if digest(jsonable(environment_binding(env))) != expected_env:
            raise RuntimeError('Constructed role environment must match whole registered physical window/drain')
        return env

    def _add_completed_files(self, registrar, folder):
        files = {}
        for path in sorted(Path(folder).rglob('*')):
            if path.is_symlink() or path.is_junction():
                raise RuntimeError('No completed output/parent path substitution')
            if path.is_file():
                if path.name == 'FAILURE.json':
                    raise RuntimeError('Never merge partial/failed training or validation evidence')
                path = _safe_path(path, BASE/'visual_calibrated_policy')
                files[str(path)] = file_sha(path)
        registrar._inputs = merge_frozen_file_bindings(registrar._inputs, files)

    def run_once(self, new_directory):
        """Future actual ten jobs only; no heartbeat invokes this while blocked."""
        if self._started or self._failed or self._finished:
            raise RuntimeError('Permanent one-shot grid, never auto append/resume/retry a partial arm')
        self._started = True
        path, lock, owned, reference, registrar = None, None, False, None, None
        completed, current = [], None
        try:
            self._boundary()
            root = BASE/'visual_calibrated_policy/learning_grids'
            path = _safe_path(new_directory, root)
            if path.parent.resolve() != root.resolve() or len(path.name) > 32 or path.exists():
                raise ValueError('New short direct-child grid directory only, no overwrite/deep Windows path')
            lock = _claim_grid_once(BASE/'visual_calibrated_policy/learning_grid_locks', self.identity,
                dict(output_path=str(path), order_rule=ORDER_RULE, automatic_append_resume_retry_allowed=False))
            path.mkdir(parents=True, exist_ok=False)
            owned = True
            _write_once(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(), bindings=asdict(self.bindings),
                scientific_learning_grid_identity=self.identity, jobs=self.jobs,
                initialized_reference_never_used_for_policy_actions=True, test_evaluation_or_selection_allowed=False))
            registrar = RealFinalCheckpointRegistrar(self.rule, self.bindings)
            registrar._inputs = merge_frozen_file_bindings(registrar._inputs, self.inputs)
            self._boundary()
            reference = NeuralVisualController('QLA-MAPPO', build_cuda_model('QLA-MAPPO', self.dimensions,
                self.protocol['learner_kwargs_by_method']['QLA-MAPPO'], 1), self.optimizer)
            reference_sha = actual_parameter_sha256(reference)
            catalogs = {role: ExecutionServiceCatalog(role) for role in ('calibration', 'validation')}
            expected_catalogs = self.protocol['cal_validation_frozen_components']['catalog_input_files_sha256_by_role']
            if any(catalogs[role].source_sha256 != expected_catalogs[role] for role in catalogs):
                raise RuntimeError('Exactly the entire original calibration/validation measured pools required')
            mobility = PhysicalMobility(users=self.budget.users)
            for index, job in enumerate(self.jobs, 1):
                self._boundary()
                method, seed = job['method'], job['seed']
                current = dict(index=index, method=method, seed=seed, phase='fresh_common_budget_training')
                job_root = path/f'J{index:02d}'
                job_root.mkdir(exist_ok=False)
                controller = _new_actual_training_controller(reference, method, seed, self.dimensions, self.optimizer)
                schedules = tuple(self.protocol['training_schedule_sha256_by_seed'][str(seed)])
                ledger = TrainingTransactionLedger(method, seed, self.budget, schedules,
                    self.protocol['training_common_condition_sha256_by_seed'][str(seed)])
                training = RealMultiepisodeTrainingCoordinator(controller, self.scales, self.closure, ledger, self.bindings)
                envs = tuple(self._environment('calibration', seed, i, catalogs['calibration'], mobility)
                    for i in range(len(self.budget.episode_decisions)))
                records = training.run(envs, job_root/'T')
                self._boundary()
                training_marker = job_root/'T/BUDGET_COMPLETE_PENDING_VALIDATION.json'
                final = records[-1]
                archive = job_root/f'T/episode_{len(records):02d}/checkpoint'
                validations = []
                registered_val = self.protocol['validation_schedule_sha256_by_seed'][str(seed)]
                if (not registered_val or len(registered_val) != len(self.protocol['validation_schedule_artifacts_by_seed'][str(seed)])
                        or len(set(registered_val)) != len(registered_val)):
                    raise ValueError('Every ordered registered validation required, not selected by outcome')
                for vi, schedule_sha in enumerate(registered_val):
                    current['phase'] = f'readonly_own_final_checkpoint_validation_{vi+1}'
                    self._boundary()
                    loaded, witness = load_own_snapshot_in_verified_cuda_process(controller, archive,
                        final['descriptor'], final['checkpoint']['artifact_sha256'], final['actual_parameter_sha256'], seed)
                    evaluator = RealReadonlyValidationEvaluator(loaded, self.scales, self.closure, self.bindings,
                        archive, final['descriptor'], final['checkpoint']['artifact_sha256'], final['actual_parameter_sha256'])
                    env = self._environment('validation', seed, vi, catalogs['validation'], mobility)
                    if env._schedule.fingerprint() != schedule_sha:
                        raise RuntimeError('Ordered paired validation identity changed')
                    validation_path = job_root/f'V{vi+1:02d}'
                    evaluator.evaluate(env, validation_path)
                    _write_once(validation_path/'OWN_FINAL_WEIGHT_LOAD.json', witness)
                    marker = validation_path/'VALIDATION_COMPLETE.json'
                    validations.append((evaluator, marker, file_sha(marker)))
                    self._boundary()
                    del env, loaded
                registrar.register(training_marker, file_sha(training_marker), validations)
                self._add_completed_files(registrar, job_root)
                self._boundary()
                if actual_parameter_sha256(reference) != reference_sha or reference._started_episode_tokens:
                    raise RuntimeError('Unused reference model changed or executed an optimizer transaction')
                completed.append(dict(index=index, method=method, seed=seed,
                    training_complete_path=str(training_marker), training_complete_sha256=file_sha(training_marker),
                    ordered_validation_count=len(validations), final_checkpoint_is_common_budget_not_ranked=True))
                write_evidence(path/'STATE.json', dict(status='partial_ten_jobs_not_all_weights_registered', completed_jobs=completed,
                    expected_jobs=10, automatic_append_resume_retry_allowed=False, test_cells=0,
                    at=datetime.now().astimezone().isoformat()))
                del controller, training, ledger, envs, records, validations, evaluator
                gc.collect()
                reference._torch.cuda.empty_cache()  # own unused allocation only, never a driver/network reset
            validate_learning_job_coverage(self.jobs, self.rule)
            self._boundary()
            self._add_completed_files(registrar, path)
            registrar.seal_all(path/'W')
            marker = path/'W/ALL_FINAL_CHECKPOINTS_REGISTERED_PENDING_TEST_ENTRY.json'
            _write_once(path/'TEN_JOBS_COMPLETE_PENDING_TEST_ENTRY.json', dict(status='actual_ten_training_validation_jobs_complete_pending_test_entry',
                completed_jobs=completed, weight_registration_path=str(marker), weight_registration_sha256=file_sha(marker),
                formal_test_launch_done=False, independent_final_result_audit_done=False))
            self._finished = True
            return marker
        except Exception as error:
            self._failed = True
            if reference is not None:
                reference._failed = True
            if registrar is not None:
                registrar._failed = True
            failure = dict(error=repr(error), completed_jobs=completed, last_job=current,
                partial_evidence_preserved=True, automatic_append_resume_retry_allowed=False,
                test_evaluation_or_selection_performed=False, at=datetime.now().astimezone().isoformat())
            if owned and not (path/'FAILURE.json').exists():
                _write_once(path/'FAILURE.json', failure)
            if lock is not None and not lock.with_name(lock.stem+'_FAILURE.json').exists():
                _write_once(lock.with_name(lock.stem+'_FAILURE.json'), failure)
            raise
