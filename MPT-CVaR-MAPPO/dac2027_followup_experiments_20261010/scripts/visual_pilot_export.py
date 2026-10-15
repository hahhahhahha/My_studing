"""SOURCE ONLY: export genuine reviewed pilot decisions, never run a pilot.

Actual constructor/export/reader count zero. A completed export is not request
generation or formal launch permission. Do not manufacture completed fixtures.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from retained_test_evaluator import _write_once
from retained_test_grid_registration import _safe_path, _read_actual, _claim_grid_once
from visual_episode_collector import BASE
from visual_pilot_freeze_contract import (RealCompletePilotFreezeEvidence,
    CompletePilotAssessmentBindings, REVIEW_SCOPE, REQUIREMENTS,
    validate_pilot_selection_schema)
from visual_pilot_registration import REGISTRATION_SCOPE
from visual_protocol_assembly import merge_frozen_file_bindings
from visual_training_transactions import digest, file_sha, require_sha

EXPORT_SCOPE = 'completed_cal_validation_pilot_decision_export_only_not_metadata_or_formal_permission'
EXPORT_STATUS = 'cal_validation_generation_design_frozen_not_formal_entry'
EXPORT_FIELDS = frozenset(('status', 'scope', 'at', 'generation_design', 'formal_protocol_components',
    'selection_identity_sha256', 'registration_scientific_identity_sha256', 'export_scientific_identity_sha256',
    'complete_assessment_bindings', 'source_files_sha256', 'protected_input_and_source_sha256',
    'actual_model_smoke_path', 'actual_model_smoke_sha256', 'source_repair_review_path', 'source_repair_review_sha256',
    'pilot_entry_review_path', 'pilot_entry_review_sha256', 'human_direction_record_path', 'human_direction_record_sha256',
    'cal_validation_pilot_evidence_sha256', 'complete_input_files_sha256',
    'cal_validation_engineering_pilot_completed', 'independent_complete_pilot_sufficiency_assessment_completed',
    'first_stage_readonly_registration_alone_sufficient', 'test_labels_or_outcomes_used_for_design',
    'metadata_request_generation_authorized', 'formal_training_or_test_launch_authorized',
    'immutable_formal_protocol_frozen', 'ten_learned_training_or_120_test_completed'))


def pilot_export_scientific_identity(registration_identity, selection_identity):
    """Pure one-shot identity; path/encoding/review delivery cannot reset it."""
    return digest(dict(registration_identity_sha256=require_sha(registration_identity),
        selection_identity_sha256=require_sha(selection_identity), scope=EXPORT_SCOPE))


def merge_complete_export_bindings(*maps):
    """Pure non-rebinding merge; never a completed-pilot or permission check."""
    if not maps:
        raise ValueError('Explicit nonempty complete byte maps required')
    merged = deepcopy(maps[0])
    for mapping in maps:
        merged = merge_frozen_file_bindings(merged, mapping)
    return merged


def selected_pilot_reference_bindings(selection):
    """Pure referenced-file completeness, not IO or actual design evidence."""
    validate_pilot_selection_schema(selection)
    c = selection['formal_protocol_components']
    maps = list(c['catalog_input_files_sha256_by_role'].values())
    for prefix in ('training', 'validation'):
        for rows in c[prefix+'_schedule_artifacts_by_seed'].values():
            for row in rows:
                maps.append({row['path']: row['file_sha256']})
        for rows in c[prefix+'_environment_descriptors_by_seed'].values():
            for row in rows:
                maps.extend((row['catalog_source_sha256'], row['mobility_source_sha256']))
    for table in c['test_environment_templates_by_load_seed'].values():
        for row in table.values():
            maps.extend((row['catalog_source_sha256'], row['mobility_source_sha256']))
    return merge_complete_export_bindings(*maps)


def require_completed_pilot_export_inputs(pilot):
    """Readonly actual reader, called ONLY by the metadata preparation gate.

Reconstruct the complete original input set, do not trust a subset manifest.
The caller must additionally validate current smoke, source/raw and a separate
fresh metadata-only review. No model/pilot/selection/generation API here.
"""
    if (type(pilot) is not dict or set(pilot) != EXPORT_FIELDS or pilot['status'] != EXPORT_STATUS
            or pilot['scope'] != EXPORT_SCOPE
            or pilot['cal_validation_engineering_pilot_completed'] is not True
            or pilot['independent_complete_pilot_sufficiency_assessment_completed'] is not True
            or any(pilot[key] is not False for key in ('first_stage_readonly_registration_alone_sufficient',
                'test_labels_or_outcomes_used_for_design', 'metadata_request_generation_authorized',
                'formal_training_or_test_launch_authorized', 'immutable_formal_protocol_frozen',
                'ten_learned_training_or_120_test_completed'))):
        raise RuntimeError('Exact completed decision export, not first-stage registration or expanded authority')
    raw_bindings = pilot['complete_assessment_bindings']
    if type(raw_bindings) is not dict or set(raw_bindings) != set(CompletePilotAssessmentBindings.__dataclass_fields__):
        raise ValueError('All original external complete assessment bindings required')
    b = CompletePilotAssessmentBindings(**raw_bindings)
    selection = _read_actual(b.selection_path, b.selection_sha256, BASE/'protocol')
    schema = validate_pilot_selection_schema(selection)
    registered = _read_actual(b.registration_marker_path, b.registration_marker_sha256,
        BASE/'visual_calibrated_policy/pilot_evidence_registry')
    review = _read_actual(b.independent_assessment_path, b.independent_assessment_sha256, BASE.parent/'.aris/traces')
    reg_identity = registered['scientific_registration_identity_sha256']
    if (registered['status'] != 'all_actual_paired_readonly_engineering_evidence_registered_pending_independent_check'
            or registered['scope'] != REGISTRATION_SCOPE
            or registered['independent_pilot_sufficiency_no_leakage_analytic_heldout_check_completed'] is not False
            or pilot['registration_scientific_identity_sha256'] != reg_identity
            or pilot['selection_identity_sha256'] != schema['selection_identity_sha256']
            or pilot['formal_protocol_components'] != selection['formal_protocol_components']
            or pilot['generation_design'] != schema['generation_design_definition']
            or pilot['export_scientific_identity_sha256'] != pilot_export_scientific_identity(reg_identity, schema['selection_identity_sha256'])):
        raise RuntimeError('Original assessed selections/registration semantics and export identity unchanged')
    expected_equal = dict(scope=REVIEW_SCOPE, review_type='Type-A', model='gpt-6-astra', reasoning_effort='xhigh',
        blockers=[], source_files_sha256=pilot['source_files_sha256'],
        protected_input_and_source_sha256=pilot['protected_input_and_source_sha256'],
        registration_marker_sha256=b.registration_marker_sha256,
        registration_scientific_identity_sha256=reg_identity, selection_sha256=b.selection_sha256,
        selection_identity_sha256=schema['selection_identity_sha256'],
        formal_components_identity_sha256=schema['formal_components_identity_sha256'],
        generation_design_definition=schema['generation_design_definition'],
        actual_model_smoke_sha256=pilot['actual_model_smoke_sha256'],
        source_repair_review_sha256=pilot['source_repair_review_sha256'],
        human_direction_record_sha256=pilot['human_direction_record_sha256'],
        permitted_actions=['export_verified_cal_validation_pilot_decisions_only'])
    true_flags = ('actual_user_direction_verified_against_trusted_message',
        'actual_cal_validation_pilot_evidence_sufficient_for_declared_selections',
        'first_stage_readonly_registration_alone_not_assumed_sufficient',
        'additional_independent_checks_performed_not_same_foundation_replay_only')
    false_flags = ('test_labels_or_outcomes_used_for_selection', 'formal_training_or_test_launch_authorized',
        'metadata_request_generation_authorized', 'hidden_numeric_gates_or_negative_result_filters_introduced')
    if (review.get('status') not in ('PASS', 'WARN') or any(review.get(k) != v for k, v in expected_equal.items())
            or any(review.get(k) is not True for k in true_flags) or any(review.get(k) is not False for k in false_flags)):
        raise RuntimeError('Original genuine fresh independent sufficiency/selection assessment must continue downstream')
    assessments = review.get('requirement_assessments')
    if type(assessments) is not dict or set(assessments) != set(REQUIREMENTS):
        raise RuntimeError('All six independent actual requirements remain required')
    additional = {}
    for item in assessments.values():
        if (type(item) is not dict or set(item) != {'checked_on_actual_cal_validation_evidence', 'conclusion', 'evidence_files_sha256'}
                or item['checked_on_actual_cal_validation_evidence'] is not True or item['conclusion'] != 'sufficient'):
            raise RuntimeError('Actual independent evidence and judgement, no author/source-only substitution')
        mapping = item['evidence_files_sha256']
        for name in mapping:
            path = _safe_path(name, BASE/'visual_calibrated_policy')
            if path.suffix.lower() in ('.py', '.pyc'):
                raise RuntimeError('Independent actual evidence cannot be source-only fixtures')
        additional = merge_complete_export_bindings(additional or mapping, mapping)
    original = registered['original_immutable_files_sha256']
    if not set(additional)-set(original):
        raise RuntimeError('Extra independent actual evidence required, not same foundation replay only')
    expected = merge_complete_export_bindings(original, additional, {
        str(Path(b.registration_marker_path)): b.registration_marker_sha256,
        str(Path(b.selection_path)): b.selection_sha256,
        str(Path(b.independent_assessment_path)): b.independent_assessment_sha256})
    if pilot['complete_input_files_sha256'] != expected:
        raise RuntimeError('No omitted original source/raw/smoke/driver/independent/human/selection input bytes')
    if any(expected.get(name) != sha for name, sha in selected_pilot_reference_bindings(selection).items()):
        raise RuntimeError('Every selected common schedule/catalog/mobility must be in the actual assessed input set')
    evidence = {}
    for name, sha in expected.items():
        path = _safe_path(name, BASE.parent)
        if not path.is_file() or file_sha(path) != require_sha(sha):
            raise RuntimeError('Completed pilot full original input bytes changed; stop, never replace/refit')
        if path.resolve().is_relative_to((BASE/'visual_calibrated_policy').resolve()):
            evidence[name] = sha
    if not evidence or pilot['cal_validation_pilot_evidence_sha256'] != evidence:
        raise RuntimeError('Complete visual evidence subset, no favorable/partial filtering')
    required = merge_complete_export_bindings(pilot['source_files_sha256'], pilot['protected_input_and_source_sha256'], {
        pilot[key+'_path']: require_sha(pilot[key+'_sha256']) for key in
        ('actual_model_smoke', 'source_repair_review', 'pilot_entry_review', 'human_direction_record')})
    if any(expected.get(name) != sha for name, sha in required.items()):
        raise RuntimeError('All original actual source/raw/smoke/repair/entry/human anchors must be inherited')
    return dict(expected)


class RealCompletedPilotExporter:
    """SOURCE PREPARED ONLY. Actual constructor/export/write count zero."""
    def __init__(self, same_actual_complete_evidence):
        if type(same_actual_complete_evidence) is not RealCompletePilotFreezeEvidence:
            raise TypeError('SAME genuine successful complete pilot contract, not descriptors/author flags')
        same_actual_complete_evidence.verify_boundary()
        self.contract = same_actual_complete_evidence
        self._started, self._failed, self._finished = False, False, False

    def export_once(self, new_directory):
        if self._started or self._failed or self._finished:
            raise RuntimeError('One-shot export only, no append/retry/partial merge/reassessment')
        self._started = True
        path = _safe_path(new_directory, BASE/'protocol/pilot_exports')
        if path.exists() or path.parent.resolve() != (BASE/'protocol/pilot_exports').resolve() or len(path.name) > 24:
            raise ValueError('New short direct owned export directory only')
        bundle = self.contract.reviewed_inputs_for_separate_export()
        selection, schema = bundle['selection'], self.contract._schema
        registrar, driver = self.contract.registrar, self.contract.registrar.driver
        identity = pilot_export_scientific_identity(registrar.identity, schema['selection_identity_sha256'])
        inputs = bundle['complete_input_files_sha256']
        lock, owned = None, False
        try:
            lock = _claim_grid_once(BASE/'visual_calibrated_policy/pilot_export_locks', identity,
                dict(at=datetime.now().astimezone().isoformat(), output_directory=str(path),
                    automatic_retry_append_or_partial_merge_allowed=False))
            path.mkdir(parents=True, exist_ok=False)
            owned = True
            _write_once(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(),
                scope=EXPORT_SCOPE, export_scientific_identity_sha256=identity,
                complete_input_files_sha256=inputs, actual_training_and_test_cells=0))
            self.contract.verify_boundary()
            report = driver.reports['QLA-MAPPO']
            evidence = {name: sha for name, sha in inputs.items()
                if Path(name).resolve().is_relative_to((BASE/'visual_calibrated_policy').resolve())}
            record = dict(status=EXPORT_STATUS, scope=EXPORT_SCOPE, at=datetime.now().astimezone().isoformat(),
                generation_design=schema['generation_design_definition'], formal_protocol_components=selection['formal_protocol_components'],
                selection_identity_sha256=schema['selection_identity_sha256'], registration_scientific_identity_sha256=registrar.identity,
                export_scientific_identity_sha256=identity, complete_assessment_bindings=bundle['external_bindings'],
                source_files_sha256=report['source_files_sha256'], protected_input_and_source_sha256=report['protected_input_and_source_sha256'],
                actual_model_smoke_path=driver.bindings.actual_smoke_path,
                actual_model_smoke_sha256=driver.bindings.actual_smoke_sha256,
                **{key+'_'+suffix: getattr(driver.bindings, key+'_'+suffix)
                    for key in ('source_repair_review', 'pilot_entry_review', 'human_direction_record')
                    for suffix in ('path', 'sha256')},
                cal_validation_pilot_evidence_sha256=evidence, complete_input_files_sha256=inputs,
                cal_validation_engineering_pilot_completed=True, independent_complete_pilot_sufficiency_assessment_completed=True,
                first_stage_readonly_registration_alone_sufficient=False, test_labels_or_outcomes_used_for_design=False,
                metadata_request_generation_authorized=False, formal_training_or_test_launch_authorized=False,
                immutable_formal_protocol_frozen=False, ten_learned_training_or_120_test_completed=False)
            require_completed_pilot_export_inputs(record)
            self.contract.verify_boundary()
            published = path/'COMPLETED_CAL_VALIDATION_PILOT.json'
            _write_once(published, record)
            published_sha = file_sha(published)
            self.contract.verify_boundary()
            if _read_actual(published, published_sha, BASE/'protocol/pilot_exports') != record:
                raise RuntimeError('Written pilot decisions changed at export boundary')
            self._finished = True
            self._published, self._published_sha = published, published_sha
            return deepcopy(record)
        except Exception as error:
            self._failed = True
            failure = dict(at=datetime.now().astimezone().isoformat(), error=repr(error),
                export_scientific_identity_sha256=identity, complete_input_files_sha256=inputs,
                partial_evidence_preserved=True, automatic_retry_append_or_partial_merge_allowed=False,
                metadata_or_formal_launch_authorized=False, actual_training_and_test_cells=0)
            if owned and not (path/'FAILURE.json').exists():
                _write_once(path/'FAILURE.json', failure)
            if lock is not None:
                failed_lock = lock.with_name(lock.stem+'_FAILURE.json')
                if not failed_lock.exists():
                    _write_once(failed_lock, failure)
            raise
