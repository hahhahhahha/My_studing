"""SOURCE ONLY: complete-pilot sufficiency evidence BEFORE a freeze export.

Readonly contract, not a reviewer, pilot selector, freeze writer or launcher.
First-stage paired registration alone is insufficient. Actual construction,
artifact reading and independent judgement paths have NOT been exercised.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from retained_test_grid_registration import _safe_path, _read_actual
from retained_test_schedule_generation import RetainedGenerationDesign
from visual_episode_collector import BASE, EpisodeClosureCandidate
from visual_method_features import VisualFeatureScales
from visual_neural_controller import VisualOptimizerSettings
from visual_pilot_registration import RealPairedPilotEvidenceRegistrar, REGISTRATION_SCOPE
from visual_protocol_assembly import (COMPONENT_FIELDS, PILOT_TEST_TEMPLATE_FIELDS, SEEDS,
    _complete_dataclass, validate_environment_descriptor, merge_frozen_file_bindings)
from visual_training_transactions import CommonBudgetCandidate, digest, file_sha, require_sha

REVIEW_SCOPE = 'complete_cal_validation_pilot_sufficiency_and_freeze_decisions_after_actual_smoke_and_verified_user_direction'
SELECTION_STATUS = 'cal_validation_pilot_selection_candidate_not_frozen'
REQUIREMENTS = (
    'independent_analytic_FIFO_upload_admission_and_fixed_drain',
    'independent_heldout_validation_replay_task_and_service_mass',
    'metadata_only_actor_QL_stale_prediction_current_action_free_critic',
    'equal_baseline_input_access_and_original_paired_exogenous_draws',
    'selected_common_H_MV_episode_terminal_and_optimizer_input_rules',
    'calibration_only_scales_geometry_discount_load_budget_and_role_windows')


def complete_pilot_review_requirements():
    """Pure checklist only. Never emit PASS/authority or numeric thresholds."""
    return dict(scope=REVIEW_SCOPE, requirement_names=list(REQUIREMENTS),
        first_stage_readonly_registration_alone_sufficient=False,
        same_foundation_FIFO_replay_alone_independent_certification=False,
        actual_fresh_gpt_6_astra_xhigh_Type_A_required=True,
        actual_evidence_not_source_or_artificial_fixtures_required=True,
        numerical_thresholds_created_by_this_contract=False,
        test_labels_or_outcomes_for_selection_allowed=False,
        pilot_freeze_metadata_generation_or_formal_launch_permission=False)


def validate_pilot_selection_schema(selection):
    """Pure completeness, not proof selections came from an actual pilot."""
    if (type(selection) is not dict or set(selection) != {'status', 'generation_design_fields', 'formal_protocol_components'}
            or selection['status'] != SELECTION_STATUS):
        raise ValueError('Complete explicit unfrozen selection schema, no authority/outcome/default fields')
    design = _complete_dataclass(RetainedGenerationDesign, selection['generation_design_fields'],
        ('load_labels', 'total_rates_per_second', 'seeds'))
    components = selection['formal_protocol_components']
    if type(components) is not dict or set(components) != COMPONENT_FIELDS:
        raise ValueError('All sixteen cal-validation choices required, not author-generated missing defaults')
    scales = _complete_dataclass(VisualFeatureScales, components['feature_scales'],
        ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
    closure = _complete_dataclass(EpisodeClosureCandidate, components['closure_settings'])
    budget = _complete_dataclass(CommonBudgetCandidate, components['common_budget'], ('episode_decisions',))
    _complete_dataclass(VisualOptimizerSettings, components['optimizer_settings'])
    if (components['virtual_H_episode_rule'] not in ('carry_across_training_episodes', 'reset_each_independent_episode')
            or budget.users != 30 or design.users != 30 or design.decision_seconds != .05 or design.broadcast_seconds != .1
            or scales.decision_seconds != design.decision_seconds or budget.decision_seconds != design.decision_seconds
            or scales.profiles != tuple(design.definition()['profiles']) or closure.fixed_drain_seconds != design.fixed_drain_seconds
            or budget.fixed_drain_seconds != closure.fixed_drain_seconds
            or type(components['learner_kwargs_by_method']) is not dict or set(components['learner_kwargs_by_method']) != {'QLA-MAPPO', 'PPO-MV-A'}
            or type(components['test_rule_policy_settings']) is not dict or set(components['test_rule_policy_settings']) != {'JSEQ-A', 'Score-softmax'}):
        raise ValueError('Explicit SAME common users/physical times/closure/budget/methods and episode rule required')
    table_keys = ('training_environment_descriptors_by_seed', 'validation_environment_descriptors_by_seed',
        'training_schedule_artifacts_by_seed', 'validation_schedule_artifacts_by_seed')
    if any(type(components[key]) is not dict or set(components[key]) != SEEDS for key in table_keys):
        raise ValueError('All five common seeds in every explicit cal/validation environment/schedule table')
    for role, prefix in (('calibration', 'training'), ('validation', 'validation')):
        for seed in SEEDS:
            envs, schedules = (components[prefix+suffix][seed] for suffix in
                ('_environment_descriptors_by_seed', '_schedule_artifacts_by_seed'))
            if (type(envs) is not list or type(schedules) is not list or not schedules or len(envs) != len(schedules)
                    or (role == 'calibration' and len(schedules) != len(budget.episode_decisions))):
                raise ValueError('Whole common role schedule/environment lists, not empty or truncated budgets')
            for environment, schedule in zip(envs, schedules):
                if type(schedule) is not dict or set(schedule) != {'path', 'file_sha256', 'semantic_sha256'} or not schedule['path']:
                    raise ValueError('Complete explicit registered common schedule bytes and semantics required')
                require_sha(schedule['file_sha256'])
                validate_environment_descriptor(environment, role, schedule['semantic_sha256'], design.broadcast_seconds)
    # Deeper original schedule/window/geometry/learner equivalence remains the
    # existing actual assembler's responsibility after metadata generation.
    templates = components['test_environment_templates_by_load_seed']
    if type(templates) is not dict or set(templates) != set(design.load_labels):
        raise ValueError('Exact two cal-validation selected loads in metadata-only condition templates')
    for table in templates.values():
        if type(table) is not dict or set(table) != SEEDS:
            raise ValueError('All five paired seeds required, no future test schedule identities')
        for row in table.values():
            if type(row) is not dict or set(row) != PILOT_TEST_TEMPLATE_FIELDS:
                raise ValueError('No future retained schedule SHA before actual metadata generation')
    return dict(generation_design_definition=design.definition(),
        formal_components_identity_sha256=digest(components), selection_identity_sha256=digest(selection),
        all_actual_pilot_evidence_and_full_formal_protocol_proven=False)


@dataclass(frozen=True)
class CompletePilotAssessmentBindings:
    registration_marker_path: str
    registration_marker_sha256: str
    selection_path: str
    selection_sha256: str
    independent_assessment_path: str
    independent_assessment_sha256: str


class RealCompletePilotFreezeEvidence:
    """SOURCE PREPARED ONLY, actual successful constructor/read count zero.

    A later narrowly scoped exporter must consume THIS same successful live
    object. This class does not write a completed pilot or grant generation,
    training/test permission; existing fresh downstream gates remain separate.
    """
    def __init__(self, same_actual_registrar, bindings):
        if type(same_actual_registrar) is not RealPairedPilotEvidenceRegistrar or type(bindings) is not CompletePilotAssessmentBindings:
            raise TypeError('SAME real successful registrar plus external complete assessment/selection bindings required')
        if same_actual_registrar._finished is not True or same_actual_registrar._failed:
            raise RuntimeError('Actual entire paired registration required, not unfinished/artificial source records')
        same_actual_registrar._boundary()
        root, traces = BASE/'visual_calibrated_policy', BASE.parent/'.aris/traces'
        registered = _read_actual(bindings.registration_marker_path, bindings.registration_marker_sha256,
            root/'pilot_evidence_registry')
        if (registered['status'] != 'all_actual_paired_readonly_engineering_evidence_registered_pending_independent_check'
                or registered['scope'] != REGISTRATION_SCOPE or registered['scientific_registration_identity_sha256'] != same_actual_registrar.identity
                or registered['original_immutable_files_sha256'] != same_actual_registrar.inputs
                or registered['complete_ordered_episode_summaries'] != same_actual_registrar._checked_summaries
                or registered['independent_pilot_sufficiency_no_leakage_analytic_heldout_check_completed'] is not False):
            raise ValueError('Exact actual first-stage pending evidence, not a completed-pilot/permission substitute')
        selection = _read_actual(bindings.selection_path, bindings.selection_sha256, BASE/'protocol')
        schema = validate_pilot_selection_schema(selection)
        review = _read_actual(bindings.independent_assessment_path, bindings.independent_assessment_sha256, traces)
        driver = same_actual_registrar.driver
        smoke = driver.reports['QLA-MAPPO']
        sources, protected = smoke['source_files_sha256'], smoke['protected_input_and_source_sha256']
        if (review.get('review_type') != 'Type-A' or review.get('model') != 'gpt-6-astra' or review.get('reasoning_effort') != 'xhigh'
                or review.get('scope') != REVIEW_SCOPE or review.get('status') not in ('PASS', 'WARN') or review.get('blockers') != []
                or review.get('source_files_sha256') != sources or review.get('protected_input_and_source_sha256') != protected
                or review.get('registration_marker_sha256') != bindings.registration_marker_sha256
                or review.get('registration_scientific_identity_sha256') != same_actual_registrar.identity
                or review.get('selection_sha256') != bindings.selection_sha256
                or review.get('selection_identity_sha256') != schema['selection_identity_sha256']
                or review.get('formal_components_identity_sha256') != schema['formal_components_identity_sha256']
                or review.get('generation_design_definition') != schema['generation_design_definition']
                or review.get('actual_model_smoke_sha256') != driver.bindings.actual_smoke_sha256
                or review.get('source_repair_review_sha256') != driver.bindings.source_repair_review_sha256
                or review.get('human_direction_record_sha256') != driver.bindings.human_direction_record_sha256):
            raise RuntimeError('Fresh exact independent actual evidence/selection assessment, not old FAIL or author certification')
        required_true = ('actual_user_direction_verified_against_trusted_message',
            'actual_cal_validation_pilot_evidence_sufficient_for_declared_selections',
            'first_stage_readonly_registration_alone_not_assumed_sufficient',
            'additional_independent_checks_performed_not_same_foundation_replay_only')
        required_false = ('test_labels_or_outcomes_used_for_selection', 'formal_training_or_test_launch_authorized',
            'metadata_request_generation_authorized', 'hidden_numeric_gates_or_negative_result_filters_introduced')
        if (any(review.get(name) is not True for name in required_true)
                or any(review.get(name) is not False for name in required_false)
                or review.get('permitted_actions') != ['export_verified_cal_validation_pilot_decisions_only']):
            raise RuntimeError('Actual sufficiency judged explicitly; no automatic pass or broadened launch authority')
        assessments = review.get('requirement_assessments')
        if type(assessments) is not dict or set(assessments) != set(REQUIREMENTS):
            raise ValueError('All six independent engineering/causal/common-design requirements, no missing/favorable subset')
        inputs = dict(same_actual_registrar.inputs)
        additional = {}
        for name in REQUIREMENTS:
            item = assessments[name]
            if (type(item) is not dict or set(item) != {'checked_on_actual_cal_validation_evidence', 'conclusion', 'evidence_files_sha256'}
                    or item['checked_on_actual_cal_validation_evidence'] is not True or item['conclusion'] != 'sufficient'
                    or type(item['evidence_files_sha256']) is not dict or not item['evidence_files_sha256']):
                raise RuntimeError('Independent actual evidence and explicit sufficiency required, no source-only checklists')
            for path, sha in item['evidence_files_sha256'].items():
                actual = _safe_path(path, root)
                if actual.suffix.lower() in ('.py', '.pyc') or not actual.is_file() or file_sha(actual) != require_sha(sha):
                    raise RuntimeError('Original actual evidence bytes, not Python source or manufactured replacement')
            additional = merge_frozen_file_bindings(additional or item['evidence_files_sha256'], item['evidence_files_sha256'])
        if not set(additional)-set(same_actual_registrar.inputs):
            raise RuntimeError('Paired registration alone is insufficient: additional independently produced evidence required')
        components, candidate = selection['formal_protocol_components'], driver.candidate
        if (components['feature_scales'] != candidate['feature_scales'] or components['closure_settings'] != candidate['closure_settings']
                or components['placement_training_window_seconds'] != candidate['placement_calibration_window_seconds']
                or components['placement_seed'] != candidate['placement_seed'] or components['mobility_role_gap_seconds'] != candidate['mobility_role_gap_seconds']
                or components['virtual_H_episode_rule'] != candidate['virtual_H_episode_rule']):
            raise ValueError('Selections must retain actual reviewed pilot settings; changed settings require new additive pilot evidence')
        inputs = merge_frozen_file_bindings(inputs, additional)
        inputs = merge_frozen_file_bindings(inputs, {str(Path(bindings.registration_marker_path)): bindings.registration_marker_sha256,
            str(Path(bindings.selection_path)): bindings.selection_sha256,
            str(Path(bindings.independent_assessment_path)): bindings.independent_assessment_sha256})
        self.registrar, self.bindings, self._inputs = same_actual_registrar, bindings, inputs
        self._selection, self._schema, self._review = deepcopy(selection), schema, deepcopy(review)
        self._failed = False
        self.verify_boundary()

    def verify_boundary(self):
        if self._failed:
            raise RuntimeError('Failed evidence contract cannot be reused/auto repaired')
        try:
            self.registrar._boundary()
            for name, sha in self._inputs.items():
                path = _safe_path(name, BASE.parent)
                if not path.is_file() or file_sha(path) != sha:
                    raise RuntimeError('Original source/raw/actual reports/independent evidence/selections changed')
            if (_read_actual(self.bindings.selection_path, self.bindings.selection_sha256, BASE/'protocol') != self._selection
                    or validate_pilot_selection_schema(self._selection) != self._schema):
                raise RuntimeError('No post-assessment tuning or semantic/byte rebind')
        except Exception:
            self._failed = True
            raise

    def reviewed_inputs_for_separate_export(self):
        self.verify_boundary()
        return dict(selection=deepcopy(self._selection), external_bindings=asdict(self.bindings),
            complete_input_files_sha256=dict(self._inputs), completed_pilot_export_created=False,
            metadata_request_generation_permission=False, formal_training_or_test_permission=False)
