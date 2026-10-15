"""SOURCE ONLY: actual own-final-weight loading and common 120-cell execution.

No import-time Torch/model/queue, no automatic launch, no CPU neural fallback.
Construction and execution require the real immutable protocol, complete
Type-A entry, actual two-arm smoke and all ten actual registered checkpoints.
Author checks exercise descriptors/early rejection ONLY, never this runtime.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import importlib
from pathlib import Path

from physical_mobility import PhysicalMobility
from retained_test_evaluator import RealReadonlyRetainedEvaluator, _expected_environment_sha, _write_once
from retained_test_grid_registration import RealRetainedGridRegistrar, check_complete_descriptor_coverage, _safe_path
from retained_test_schedule import RegisteredRetainedRequestSchedule
from retained_test_service_catalog import RetainedExecutionServiceCatalog
from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable, require_actual_cuda_controller, write_evidence
from visual_learned_bindings import VisualPolicyDimensions, build_cuda_model, source_bindings
from visual_method_features import VisualFeatureScales
from visual_neural_controller import NeuralVisualController, VisualOptimizerSettings
from visual_protocol_assembly import _complete_dataclass, merge_frozen_file_bindings, validate_environment_descriptor
from visual_receipt_environment import VisualReceiptEnvironment
from visual_test_contract import FixedTestMatrixCandidate, FrozenTestExecutionContract, TEST_METHODS
from visual_training_coordinator import actual_parameter_sha256, environment_binding, require_reviewed_runtime
from visual_training_transactions import digest, file_sha, inspect_checkpoint_archive, require_sha
from visual_upload_link import UploadLinkConfig

ORDER_RULE = 'declared_load_then_seed_then_physical_delay_then_QLA_PPO_JSEQ_Score_v1'
LEARNED_METHODS = ('QLA-MAPPO', 'PPO-MV-A')


def fixed_cell_execution_order(rows, candidate):
    """Pure descriptor ordering, NOT successful tests/weights or launch authority."""
    check_complete_descriptor_coverage(rows, candidate)
    methods = ('QLA-MAPPO', 'PPO-MV-A', 'JSEQ-A', 'Score-softmax')
    ordered = sorted(deepcopy(rows), key=lambda row: (
        candidate.load_labels.index(row['load_label']), candidate.seeds.index(row['seed']),
        row['nominal_delay_seconds'], methods.index(row['method'])))
    return ordered


def required_own_weight_key(cell):
    """Pure routing identity only; never invent a completed checkpoint entry."""
    if (type(cell) is not dict or cell.get('method') not in TEST_METHODS
            or type(cell.get('seed')) is not int or cell['seed'] not in (1, 2, 3, 4, 5)):
        raise ValueError('Exactly a declared method/native paired seed required')
    return (cell['method'], cell['seed']) if cell['method'] in LEARNED_METHODS else None


def require_matching_weight_descriptor(cell, descriptor):
    key = required_own_weight_key(cell)
    if (key is None or type(descriptor) is not dict or descriptor.get('method') != key[0]
            or type(descriptor.get('seed')) is not int or descriptor['seed'] != key[1]):
        raise ValueError('Own method and own seed final checkpoint, not a rule/random/foreign model')
    return key


def policy_dimensions_from_frozen_scales(scales, users):
    if type(scales) is not VisualFeatureScales or type(users) is not int or users != 30 or len(scales.profiles) != 4:
        raise TypeError('Original thirty users and four action-aligned measured profiles required')
    edges = len(scales.profiles)
    actor = 3*edges+7
    # Same current-action-free global feature layout as pack_features;
    # no test observation, label or policy output needed for construction.
    critic = users*actor+users*edges+3*edges+3
    return VisualPolicyDimensions(users, edges, actor, critic, users, True, scales.age_scale_seconds)


def _require_this_runtime(contract):
    if type(contract) is not FrozenTestExecutionContract:
        raise TypeError('Actual reviewed frozen-weight test contract required before model/timing access')
    contract.cell_descriptors()
    protocol = require_reviewed_runtime(contract._bindings)
    if protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__)):
        raise RuntimeError('Own test loader and complete grid entry must be in current complete source review')
    # Runtime protected raw continuity, not only author-check continuity.
    maps = (protocol['source_files_sha256'], protocol['protected_input_and_source_sha256'],
            protocol['input_files_sha256'], contract._weights['input_files_sha256'])
    inputs = {}
    for mapping in maps:
        inputs = merge_frozen_file_bindings(inputs, mapping)
    inputs = merge_frozen_file_bindings(inputs, {
        str(contract._registration_path): contract._registration_sha,
        contract._bindings.protocol_path: contract._bindings.protocol_sha256,
        contract._bindings.entry_review_path: contract._bindings.entry_review_sha256,
        protocol['actual_model_smoke_path']: protocol['actual_model_smoke_sha256']})
    for name, sha in inputs.items():
        path = _safe_path(name, BASE.parent)
        if not path.is_file() or file_sha(path) != require_sha(sha):
            raise RuntimeError('Registered raw/source/protocol/smoke/weight evidence bytes changed')
    return protocol, inputs


def load_registered_own_test_controller(reference_controller, contract, cell):
    """Actual fresh clone + weights ONLY, not validation relabeling/resume.

    reference_controller is one unused actual QLA architecture constructed by
    build_cuda_model AFTER its original free-GPU/kernel gate in THIS process.
    It never supplies test actions. Both method constructors reuse that own
    verified CUDA context, avoiding a second occupancy gate on our allocation.
    No OPTIMIZER_AND_RNG_STATE.pt or TRAINING_STATE.json is deserialized.
    """
    protocol, _ = _require_this_runtime(contract)  # before actual class/Torch access
    if sum(digest(row) == digest(cell) for row in contract.cell_descriptors()) != 1:
        raise ValueError('Exact unchanged pre-registered cell required')
    key = required_own_weight_key(cell)
    if key is None:
        raise ValueError('Rule baseline has no neural checkpoint to load')
    require_actual_cuda_controller(reference_controller)
    if (reference_controller.method != 'QLA-MAPPO' or reference_controller._started_episode_tokens
            or reference_controller._failed):
        raise ValueError('Unused actual verified QLA architecture reference, never a trained/failed controller')
    source_bindings()
    entries = [row for row in contract._weights['entries']
               if (row['descriptor']['method'], row['descriptor']['seed']) == key]
    if len(entries) != 1:
        raise ValueError('One actual registered final checkpoint per own method/seed')
    row = entries[0]
    require_matching_weight_descriptor(cell, row['descriptor'])
    path = _safe_path(row['checkpoint_path'], BASE/'visual_calibrated_policy')
    manifest_sha, parameter_sha = require_sha(row['checkpoint_manifest_sha256']), require_sha(row['actual_parameter_sha256'])
    inspect_checkpoint_archive(path, row['descriptor'], manifest_sha)
    scales = _complete_dataclass(VisualFeatureScales, protocol['feature_scales'],
        ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
    dimensions = policy_dimensions_from_frozen_scales(scales, cell['users'])
    if asdict(reference_controller.model.sys_cfg) != asdict(dimensions):
        raise RuntimeError('Reference architecture differs from the pilot-frozen matched feature dimensions')
    settings = _complete_dataclass(VisualOptimizerSettings, protocol['optimizer_settings'])
    if reference_controller.settings != settings:
        raise RuntimeError('Shared frozen controller settings required')
    torch = reference_controller._torch
    loaded = None
    try:
        if key[0] == 'QLA-MAPPO':
            module = importlib.import_module('mappo_gp_sf_dsr.models')
            model = module.MAPPOPolicy(deepcopy(reference_controller.model.sys_cfg),
                deepcopy(reference_controller.model.cfg), seed=key[1], device='cuda')
            # Original unchanged loader restores actor/critics/entropy parameter
            # only; it does not read the separate optimizer/training files.
            model.load_checkpoint(path/'MODEL_STATE.pt', actor_only=False)
        else:
            module = importlib.import_module('mappo_gp_sf_dsr.paper_policy_baselines')
            torch.manual_seed(key[1])
            model = module.SharedGaussianPPO(dimensions.actor_obs_dim, dimensions.n_edges).to('cuda')
            payload = torch.load(path/'MODEL_STATE.pt', map_location='cuda', weights_only=True)
            if type(payload) is not dict or set(payload) != {'method', 'state_dict'} or payload['method'] != key[0]:
                raise ValueError('Exact own PPO model-only checkpoint schema required')
            model.load_state_dict(payload['state_dict'], strict=True)
        loaded = NeuralVisualController(key[0], model, settings)
        if actual_parameter_sha256(loaded) != parameter_sha or loaded._started_episode_tokens or loaded._failed:
            raise RuntimeError('Fresh actual loaded parameters differ from registered own final model')
        inspect_checkpoint_archive(path, row['descriptor'], manifest_sha)
        _require_this_runtime(contract)
        return loaded, dict(status='actual_own_final_test_weights_loaded_pending_cell_execution',
            method=key[0], seed=key[1], descriptor=row['descriptor'], checkpoint_path=str(path),
            checkpoint_manifest_sha256=manifest_sha, actual_loaded_parameter_sha256=parameter_sha,
            model_state_only_deserialized=True, optimizer_or_H_MV_training_RNG_restored=False,
            constructor_optimizer_created_but_not_stepped=True,
            fresh_constructor_may_reseed_global_Torch_RNG=True,
            data_role='test', initialized_reference_used_for_policy_actions=False)
    except Exception:
        if loaded is not None:
            loaded._failed = True
        raise


class RealFrozenWeightTestGridRunner:
    """SOURCE PREPARED ONLY. Actual constructor/run/model/cell counts zero."""
    def __init__(self, contract):
        self.protocol, self.inputs = _require_this_runtime(contract)
        self.contract = contract
        self.rows = fixed_cell_execution_order(contract.cell_descriptors(), contract._candidate)
        self.scales = _complete_dataclass(VisualFeatureScales, self.protocol['feature_scales'],
            ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
        self.settings = _complete_dataclass(EpisodeClosureCandidate, self.protocol['closure_settings'])
        self.optimizer = _complete_dataclass(VisualOptimizerSettings, self.protocol['optimizer_settings'])
        self.dimensions = policy_dimensions_from_frozen_scales(self.scales, contract._candidate.users)
        self._started, self._failed, self._finished = False, False, False

    def _boundary(self):
        protocol, inputs = _require_this_runtime(self.contract)
        if digest(protocol) != digest(self.protocol) or inputs != self.inputs:
            raise RuntimeError('Immutable full protocol/input identities changed; no post-weight rebinding')
        if self.rows != fixed_cell_execution_order(self.contract.cell_descriptors(), self.contract._candidate):
            raise RuntimeError('No outcome-adaptive ordering, cell removal or replacement')

    def _new_environment(self, cell, catalog, mobility):
        registration = self.protocol['test_schedule_artifacts_by_load_seed'][cell['load_label']][str(cell['seed'])]
        schedule = RegisteredRetainedRequestSchedule(self.contract, cell['load_label'], cell['seed'],
            registration['path'], registration['file_sha256'])
        binding = self.protocol['test_environment_descriptors_by_load_seed_delay'][cell['load_label']][str(cell['seed'])][str(cell['nominal_delay_seconds'])]
        binding = validate_environment_descriptor(binding, 'test', cell['common_exogenous_schedule_sha256'], cell['broadcast_seconds'])
        if (binding['mobility_source_sha256'] != mobility.source_hashes
                or binding['catalog_source_sha256'] != dict(catalog.source_sha256)
                or binding['edge_positions'] != jsonable(self.scales.edge_positions)
                or digest(binding) != _expected_environment_sha(self.protocol, cell)):
            raise RuntimeError('Exact common measured service/physical mobility/action order required')
        mobility.positions(binding['mobility_start_seconds']+cell['fixed_end_seconds'])
        env = VisualReceiptEnvironment(schedule, catalog, mobility, self.scales.edge_positions, self.scales.profiles,
            binding['mobility_start_seconds'], binding['broadcast_seconds'], binding['nominal_delay_seconds'],
            binding['waiting_capacity'], UploadLinkConfig(**binding['upload_link']))
        actual = environment_binding(env)
        actual['catalog_source_sha256'] = dict(actual['catalog_source_sha256'])
        if digest(jsonable(actual)) != digest(binding):
            raise RuntimeError('Constructed fresh environment differs from registered common condition')
        return env

    def run_once(self, new_directory):
        """Future real run only, one verified CUDA context, no retry/resume.

        This is intentionally not called by a heartbeat or author check.
        Grid/cell semantic locks survive failure and changing output names.
        """
        if self._started or self._failed or self._finished:
            raise RuntimeError('One-shot entire grid, never retry/append/remeasure a partial arm')
        self._started = True
        registrar, reference, completed, current = None, None, [], None
        try:
            self._boundary()
            registrar = RealRetainedGridRegistrar(self.contract, new_directory)
            registrar._merge_inputs(self.inputs)
            registrar._verify_inputs()
            _write_once(registrar.path/'GRID_EXECUTION_ENTRY.json', dict(order_rule=ORDER_RULE,
                ordered_cell_descriptor_sha256=[digest(row) for row in self.rows],
                dimensions=asdict(self.dimensions), protocol_sha256=self.contract._bindings.protocol_sha256,
                own_final_weight_registration_sha256=self.contract._registration_sha,
                reference_random_architecture_never_used_for_test_actions=True,
                one_initial_original_free_GPU_kernel_gate_before_any_model=True,
                no_CPU_fallback_or_competing_process_changes=True, independent_final_result_audit_done=False))
            # ONE original main-process gate + seeded separate-process kernel
            # in the existing factory, before our first CUDA allocation.
            self._boundary()
            reference = NeuralVisualController('QLA-MAPPO', build_cuda_model('QLA-MAPPO', self.dimensions,
                self.protocol['learner_kwargs_by_method']['QLA-MAPPO'], 1), self.optimizer)
            reference_sha = actual_parameter_sha256(reference)
            catalog = RetainedExecutionServiceCatalog(self.contract)
            mobility = PhysicalMobility(users=self.contract._candidate.users)
            for index, cell in enumerate(self.rows, 1):
                current = dict(index=index, cell=cell, phase='before_fresh_own_weight_load')
                self._boundary()
                controller = None
                if required_own_weight_key(cell) is not None:
                    controller, witness = load_registered_own_test_controller(reference, self.contract, cell)
                    witness_path = registrar.path/f'CELL_{index:03d}_OWN_WEIGHT_LOAD.json'
                    _write_once(witness_path, witness)
                    registrar._merge_inputs({str(witness_path): file_sha(witness_path)})
                env = self._new_environment(cell, catalog, mobility)
                evaluator = RealReadonlyRetainedEvaluator(self.contract, cell, self.scales, self.settings, controller)
                cell_path = BASE/'visual_calibrated_policy/test_cells'/evaluator.semantic_identity_sha256
                current['phase'] = 'actual_readonly_fixed_policy_cell'
                evaluator.evaluate(env, cell_path)
                marker = cell_path/'TEST_CELL_COMPLETE_PENDING_AUDIT.json'
                registrar.register(evaluator, marker, file_sha(marker))
                self._boundary()
                if actual_parameter_sha256(reference) != reference_sha or reference._started_episode_tokens:
                    raise RuntimeError('Unused reference architecture changed; no hidden policy training')
                completed.append(dict(index=index, semantic_cell_identity_sha256=evaluator.semantic_identity_sha256,
                    complete_path=str(marker), complete_sha256=file_sha(marker)))
                write_evidence(registrar.path/'GRID_EXECUTION_STATE.json', dict(status='partial_actual_cells_not_final_audited',
                    completed_cells=len(completed), expected_cells=120, last_cell=current,
                    optimizer_updates=0, automatic_append_resume_retry_allowed=False,
                    at=datetime.now().astimezone().isoformat()))
                del env, evaluator, controller
                gc.collect()
                reference._torch.cuda.empty_cache()  # release only our own unused cached allocations
            self._boundary()
            registrar.seal_all()
            self._finished = True
            return registrar.path/'ALL_120_ACTUAL_CELLS_PENDING_FINAL_AUDIT.json'
        except Exception as error:
            self._failed = True
            if reference is not None:
                reference._failed = True
            if registrar is not None:
                registrar._fail(error)
                failure = registrar.path/'GRID_EXECUTION_FAILURE.json'
                if not failure.exists():
                    _write_once(failure, dict(error=repr(error), last_cell=current, completed_cells=completed,
                        partial_evidence_preserved=True, automatic_append_resume_retry_allowed=False,
                        independent_final_result_audit_done=False, at=datetime.now().astimezone().isoformat()))
            raise

