"""Pure artificial schema/checklist and AST only; no assessment fixtures."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from check_visual_pilot_episode import fixture, refused
from visual_episode_collector import BASE, write_evidence
from visual_neural_controller import VisualOptimizerSettings
from visual_pilot_freeze_contract import (SELECTION_STATUS, REQUIREMENTS, complete_pilot_review_requirements,
    validate_pilot_selection_schema, RealCompletePilotFreezeEvidence)
from visual_protocol_assembly import COMPONENT_FIELDS
from retained_test_schedule_generation import RetainedGenerationDesign
from visual_training_transactions import CommonBudgetCandidate, file_sha

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_202234_125464_pilot_registry/RESULT.json'
PRIOR_SHA = 'ad9328551f87f983c7bed94961840e656810dcaa0c25484a8eadea1a24a63b82'


def artificial_selection():
    candidate = fixture()  # Explicit ARTIFICIAL paths/hashes/scales, not actual selections.
    design = RetainedGenerationDesign(('ART_LOW', 'ART_HIGH'), (1., 2.), 16, 30., '0'*64, .1, 2.)
    role_envs, role_artifacts = {}, {}
    for row in candidate['engineering_conditions']:
        role_envs[row['role']] = {str(seed): [deepcopy(row['environment_descriptor'])] for seed in range(1, 6)}
        role_artifacts[row['role']] = {str(seed): [deepcopy(row['schedule_artifact'])] for seed in range(1, 6)}
    template = deepcopy(candidate['engineering_conditions'][1]['environment_descriptor'])
    template.pop('schedule_sha256')
    template['role'] = 'test'
    template['catalog_source_sha256'] = {'ART_TEST_METADATA_MAP_NOT_TIMING_OR_FITTING': '0'*64}
    c = dict(feature_scales=deepcopy(candidate['feature_scales']), closure_settings=deepcopy(candidate['closure_settings']),
        common_budget=asdict(CommonBudgetCandidate((16,))), optimizer_settings=asdict(VisualOptimizerSettings()),
        learner_kwargs_by_method={'QLA-MAPPO': {}, 'PPO-MV-A': {}}, virtual_H_episode_rule=candidate['virtual_H_episode_rule'],
        test_rule_policy_settings={'JSEQ-A': {'temperature': .25}, 'Score-softmax': {'temperature': .25}},
        training_environment_descriptors_by_seed=role_envs['calibration'], validation_environment_descriptors_by_seed=role_envs['validation'],
        training_schedule_artifacts_by_seed=role_artifacts['calibration'], validation_schedule_artifacts_by_seed=role_artifacts['validation'],
        test_environment_templates_by_load_seed={label: {str(seed): deepcopy(template) for seed in range(1, 6)} for label in design.load_labels},
        catalog_input_files_sha256_by_role={role: deepcopy(row['environment_descriptor']['catalog_source_sha256'])
            for role, row in zip(('calibration', 'validation'), candidate['engineering_conditions'])},
        placement_training_window_seconds=[0., 500.], placement_seed=123, mobility_role_gap_seconds=20.)
    c['catalog_input_files_sha256_by_role']['test'] = {'ART_TEST_METADATA_MAP_NOT_TIMING_OR_FITTING': '0'*64}
    assert set(c) == COMPONENT_FIELDS
    return dict(status=SELECTION_STATUS, generation_design_fields=json.loads(json.dumps(asdict(design))), formal_protocol_components=c)


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_freeze_contract')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No test timing in artificial freeze evidence schema check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            selection = artificial_selection()
            before = deepcopy(selection)
            checked = validate_pilot_selection_schema(selection)
            assert checked['all_actual_pilot_evidence_and_full_formal_protocol_proven'] is False and selection == before
            assert validate_pilot_selection_schema(json.loads(json.dumps(selection, indent=7))) == checked
            rules = complete_pilot_review_requirements()
            assert len(REQUIREMENTS) == 6 and rules['first_stage_readonly_registration_alone_sufficient'] is False
            assert rules['numerical_thresholds_created_by_this_contract'] is False and rules['pilot_freeze_metadata_generation_or_formal_launch_permission'] is False
            bad = []
            for field in ('status', 'generation_design_fields', 'formal_protocol_components'):
                changed = deepcopy(selection); changed.pop(field); bad.append(changed)
            for field, value in (('origin_user', True), ('status', 'cal_validation_generation_design_frozen_not_formal_entry')):
                changed = deepcopy(selection); changed[field] = value; bad.append(changed)
            changed = deepcopy(selection); changed['generation_design_fields'].pop('seeds'); bad.append(changed)
            changed = deepcopy(selection); changed['generation_design_fields']['users'] = True; bad.append(changed)
            for field in ('feature_scales', 'common_budget', 'placement_seed'):
                changed = deepcopy(selection); changed['formal_protocol_components'].pop(field); bad.append(changed)
            changed = deepcopy(selection); changed['formal_protocol_components']['common_budget']['episode_decisions'] = [True]; bad.append(changed)
            for field in ('training_environment_descriptors_by_seed', 'validation_schedule_artifacts_by_seed'):
                changed = deepcopy(selection); changed['formal_protocol_components'][field].pop('5'); bad.append(changed)
            changed = deepcopy(selection); changed['formal_protocol_components']['training_schedule_artifacts_by_seed']['1'] = []; bad.append(changed)
            changed = deepcopy(selection); changed['formal_protocol_components']['test_environment_templates_by_load_seed']['ART_LOW']['1']['schedule_sha256'] = '0'*64; bad.append(changed)
            changed = deepcopy(selection); changed['formal_protocol_components']['test_environment_templates_by_load_seed'].pop('ART_HIGH'); bad.append(changed)
            for changed in bad:
                refused(lambda: validate_pilot_selection_schema(changed))
            refused(lambda: RealCompletePilotFreezeEvidence(None, None))
            source = (BASE/'scripts/visual_pilot_freeze_contract.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else '' for node in calls}
            assert not names & {'build_cuda_model','witness','act','collect_once','register_once','freeze_once','_write_once','write_evidence',
                'train','step','load_checkpoint','generate_once','require_reviewed_runtime','seal_all','assemble_once','fit'}
            for text in ('same_actual_registrar._finished is not True', 'requirement_assessments', 'additional independently produced evidence required',
                    "'.py', '.pyc'", "'gpt-6-astra'", "'xhigh'", "'Type-A'", 'source_repair_review_sha256',
                    'actual_user_direction_verified_against_trusted_message', 'human_direction_record_sha256',
                    'actual_cal_validation_pilot_evidence_sufficient_for_declared_selections',
                    'first_stage_readonly_registration_alone_not_assumed_sufficient',
                    'hidden_numeric_gates_or_negative_result_filters_introduced', 'metadata_request_generation_permission=False',
                    'formal_training_or_test_permission=False', 'completed_pilot_export_created=False'):
                assert text in source
            registry_tree = ast.parse((BASE/'scripts/visual_pilot_registration.py').read_text(encoding='utf-8'))
            produced = [node for node in ast.walk(registry_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'dict' and any(kw.arg == 'status' and isinstance(kw.value, ast.Constant)
                    and kw.value.value == 'all_actual_paired_readonly_engineering_evidence_registered_pending_independent_check' for kw in node.keywords)]
            assert len(produced) == 1
            keys = {kw.arg for kw in produced[0].keywords}
            consumed = {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == 'registered' and isinstance(node.slice, ast.Constant)}
            assert consumed <= keys and 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_freeze_contract.py',
                BASE/'README_VISUAL_PILOT_FREEZE_CONTRACT_中文.md')}
            write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(), artificial_selection_schema_and_static_freeze_contract_checks_passed=True,
                artificial_complete_unfrozen_schema_cases=1, artificial_invalid_or_missing_schema_cases_rejected=len(bad),
                requirement_checklist_categories=6, artificial_schema_not_actual_pilot_choices_or_evidence=True,
                static_same_live_actual_registrar_fresh_assessment_additional_evidence_source_raw_user_selection_bindings_checked=True,
                static_original_registry_producer_consumer_fields_match=True,
                static_no_model_optimizer_freeze_export_selection_or_generation_API=True,
                absent_actual_registrar_early_type_rejection_exercised=True,
                actual_complete_pilot_contract_constructor_artifact_read_permission_or_source_raw_guards_exercised=False,
                actual_independent_assessment_user_pilot_smoke_MODEL_completed_marker_or_PASS_fixtures_fabricated=False,
                actual_pilot_freeze_metadata_generation_training_or_test_calls=0, actual_test_timing_CSV_opens=timing_opens,
                torch_imported=False, complete_pilot_or_formal_entry_or_environment_ready=False, independent_audit=False,
                original_protected_files_sha256=protected, new_implementation_sha256=sources))
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_SELECTION_SCHEMA_STATIC_COMPLETE_PILOT_CONTRACT_ONLY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_pilot_freeze_model_review_or_permission_calls=0,
            partial_evidence_preserved=True, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
