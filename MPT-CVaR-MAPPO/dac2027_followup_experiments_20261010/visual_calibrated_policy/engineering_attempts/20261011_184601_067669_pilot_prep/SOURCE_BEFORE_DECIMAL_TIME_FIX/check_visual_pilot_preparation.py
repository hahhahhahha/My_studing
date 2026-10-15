"""Artificial descriptors only. No fabricated actual smoke/user/PASS fixtures."""
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

from visual_episode_collector import BASE, EpisodeClosureCandidate, write_evidence
from visual_method_features import VisualFeatureScales
from visual_pilot_preparation import (STATUS, RealCalValidationPilotPreparation,
    validate_cal_validation_pilot_candidate, pilot_role_physical_windows,
    _require_actual_calibration_geometry_and_scales)
from visual_test_contract import PROFILES
from visual_training_transactions import digest, file_sha
from visual_upload_link import UploadLinkConfig

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_174358_952614_pilot_order/RESULT.json'
PRIOR_SHA = '20ad9962698fc46cfa6e68f12e72473363919c09375a4fb5f5c1d019306f781a'
FIRST = BASE/'visual_calibrated_policy/engineering_attempts/20261011_181937_587393_pilot_prep/RESULT.json'
FIRST_SHA = '9427ef9e216f7e03fe0b047329652c111455f8be5d1f67f068f99c13a809c9a0'
BEFORE = FIRST.parent/'SOURCE_BEFORE_SEED_FAIL_CONTINUITY_HARDENING'
SECOND = BASE/'visual_calibrated_policy/engineering_attempts/20261011_182140_771916_pilot_prep/RESULT.json'
SECOND_SHA = 'a4bb2b746a017a0f72f6258b76f58486902e1a6072e9433b725790dbd76f904d'
GEOMETRY_BEFORE = BASE/'visual_calibrated_policy/engineering_attempts/20261011_184630_pilot_geometry_repair/SOURCE_BEFORE_REPAIR'


def refused(call):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid artificial candidate or absent actual artifacts accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_prep')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        assert file_sha(FIRST) == FIRST_SHA
        first = json.loads(FIRST.read_text(encoding='utf-8'))
        protected.update({str(FIRST): FIRST_SHA})
        protected.update({str(BEFORE/Path(name).name): sha for name, sha in first['new_implementation_sha256'].items()})
        assert file_sha(SECOND) == SECOND_SHA
        second = json.loads(SECOND.read_text(encoding='utf-8'))
        protected.update({str(SECOND): SECOND_SHA})
        protected.update({str(GEOMETRY_BEFORE/Path(name).name): sha for name, sha in second['new_implementation_sha256'].items()})
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No retained timing CSV in pure pre-pilot candidate check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            # ALL values below are artificial, not fitted scales/frozen settings.
            edges = ((0., 0.), (1., 0.), (0., 1.), (1., 1.))
            scales = VisualFeatureScales((0., 0.), (1., 1.), edges, PROFILES,
                100., (.03, .03, .01, .01), .03)
            candidate = dict(status=STATUS, users=30, profiles=list(PROFILES), decision_seconds=.05,
                broadcast_seconds=.1, waiting_capacity=32, upload_link=asdict(UploadLinkConfig()),
                virtual_H_episode_rule='reset_each_independent_episode', feature_scales=json.loads(json.dumps(asdict(scales))),
                closure_settings=asdict(EpisodeClosureCandidate()), engineering_conditions=[],
                placement_calibration_window_seconds=[0., 500.], placement_seed=123, mobility_role_gap_seconds=20.)
            for i, role in enumerate(('calibration', 'validation')):
                sha = f'{i+1:064x}'
                candidate['engineering_conditions'].append(dict(condition_id='ARTIFICIAL_'+role, role=role,
                    seed=123, decisions=16, schedule_artifact=dict(path='EXPLICIT_ARTIFICIAL_NO_SCHEDULE_'+role,
                        file_sha256=sha, semantic_sha256=sha),
                    environment_descriptor=dict(schedule_sha256=sha, role=role, profiles=list(PROFILES),
                        edge_positions=[list(point) for point in edges], mobility_start_seconds=1000.*i,
                        broadcast_seconds=.1, nominal_delay_seconds=.1, waiting_capacity=32,
                        upload_link=asdict(UploadLinkConfig()),
                        catalog_source_sha256={'EXPLICIT_ARTIFICIAL_NO_CATALOG_'+role: '0'*64},
                        mobility_source_sha256={f'EXPLICIT_ARTIFICIAL_NO_MOBILITY_{u}': '0'*64 for u in range(30)})))
            before = deepcopy(candidate)
            summary = validate_cal_validation_pilot_candidate(candidate)
            assert summary['common_conditions'] == 2 and summary['actual_pilot_or_model_execution_proven'] == 0
            rows = summary['paired_engineering_descriptors']
            assert len(rows) == 4 and [x['method'] for x in rows] == ['QLA-MAPPO', 'PPO-MV-A']*2
            for i in (0, 2):
                assert rows[i]['common_condition_sha256'] == rows[i+1]['common_condition_sha256']
                assert rows[i]['schedule_sha256'] == rows[i+1]['schedule_sha256']
                assert not rows[i]['optimizer_updates_allowed'] and not rows[i]['formal_training_or_test_cell']
            assert candidate == before and validate_cal_validation_pilot_candidate(json.loads(json.dumps(candidate, indent=7))) == summary
            windows = pilot_role_physical_windows(candidate, (0., 15150.))  # EXPLICIT ARTIFICIAL physical support
            assert len(windows) == 3 and windows[1]['end_seconds'] == 30.8 and windows[2]['end_seconds'] == 1030.8
            assert summary['declared_role_physical_windows'] == windows
            rows[0]['condition_id'] = 'CHANGED_ARTIFICIAL_RETURN'
            assert candidate == before
            bad = []
            for field, value in (('users', True), ('decision_seconds', 1.), ('waiting_capacity', True),
                    ('future_test_schedule_sha256', '0'*64), ('origin', 'user'), ('actual_pilot_completed', True)):
                changed = deepcopy(candidate)
                changed[field] = value
                bad.append(changed)
            for field, value in (('role', 'test'), ('seed', True), ('decisions', True),
                    ('method', 'QLA-MAPPO'), ('future_service_seconds', .03)):
                changed = deepcopy(candidate)
                changed['engineering_conditions'][0][field] = value
                bad.append(changed)
            for change in ('missing_role', 'duplicate_condition', 'missing_explicit_closure', 'bool_closure',
                    'higher_bandwidth', 'validation_fitted_scales', 'changed_schedule', 'changed_environment_geometry'):
                changed = deepcopy(candidate)
                if change == 'missing_role':
                    changed['engineering_conditions'].pop()
                elif change == 'duplicate_condition':
                    changed['engineering_conditions'][1]['condition_id'] = changed['engineering_conditions'][0]['condition_id']
                elif change == 'missing_explicit_closure':
                    changed['closure_settings'].pop('gamma_per_second')
                elif change == 'bool_closure':
                    changed['closure_settings']['gamma_per_second'] = True
                elif change == 'higher_bandwidth':
                    changed['upload_link']['bandwidth_hz'] = 20e6
                elif change == 'validation_fitted_scales':
                    changed['feature_scales']['calibration_role'] = 'validation'
                elif change == 'changed_schedule':
                    changed['engineering_conditions'][0]['schedule_artifact']['semantic_sha256'] = 'f'*64
                else:
                    changed['engineering_conditions'][0]['environment_descriptor']['edge_positions'][0][0] = 999.
                bad.append(changed)
            for changed in bad:
                refused(lambda: validate_cal_validation_pilot_candidate(changed))
            geometry_bad = []
            for field, value in (('placement_calibration_window_seconds', [0., 1100.]),
                    ('placement_calibration_window_seconds', [True, 500.]), ('placement_seed', True),
                    ('mobility_role_gap_seconds', True), ('mobility_role_gap_seconds', -1.)):
                changed = deepcopy(candidate)
                changed[field] = value
                geometry_bad.append(changed)
            changed = deepcopy(candidate)
            changed.pop('placement_seed')
            geometry_bad.append(changed)
            changed = deepcopy(candidate)
            changed['engineering_conditions'][1]['environment_descriptor']['mobility_start_seconds'] = 10.
            geometry_bad.append(changed)
            changed = deepcopy(candidate)
            changed['engineering_conditions'][1]['environment_descriptor']['mobility_start_seconds'] = 519.
            geometry_bad.append(changed)  # 500s geometry fit +20s gap, not merely execution .8s horizon
            changed = deepcopy(candidate)
            changed['closure_settings']['fixed_drain_seconds'] = 1000.
            geometry_bad.append(changed)  # full drain overlaps held-out window
            for changed in geometry_bad:
                refused(lambda: validate_cal_validation_pilot_candidate(changed))
            for support in ((0., 1020.), (1., 15150.), (True, 15150.), (0., float('inf'))):
                refused(lambda: pilot_role_physical_windows(candidate, support))
            good_boundary = deepcopy(candidate)
            good_boundary['engineering_conditions'][1]['environment_descriptor']['mobility_start_seconds'] = 520.
            assert pilot_role_physical_windows(good_boundary, (0., 15150.))[-1]['start_seconds'] == 520.
            # No fake catalog/mobility to certify actual calibration provenance.
            refused(lambda: _require_actual_calibration_geometry_and_scales(candidate, None, None))
            changed = deepcopy(candidate)
            changed['closure_settings']['gamma_per_second'] = .98
            assert validate_cal_validation_pilot_candidate(changed)['candidate_scientific_identity_sha256'] != summary['candidate_scientific_identity_sha256']
            for field, value in (('placement_seed', 124), ('mobility_role_gap_seconds', 21.),
                    ('placement_calibration_window_seconds', [0., 400.])):
                changed = deepcopy(candidate)
                changed[field] = value
                assert validate_cal_validation_pilot_candidate(changed)['candidate_scientific_identity_sha256'] != summary['candidate_scientific_identity_sha256']
            refused(lambda: RealCalValidationPilotPreparation(None))
            tree = ast.parse((BASE/'scripts/visual_pilot_preparation.py').read_text(encoding='utf-8'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RealCalValidationPilotPreparation')
            assert {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} == {'__init__', 'verify_boundary', 'paired_condition_descriptors'}
            text = (BASE/'scripts/visual_pilot_preparation.py').read_text(encoding='utf-8')
            for name in ('actual_cuda_two_arm_smoke_pass', 'actual_roundtrip_performed',
                    'preserved_failed_review_sha256', 'human_direction_verified_against_actual_user_message',
                    'engineering_input_files_sha256', 'metadata_schedule_generation_authorized',
                    'formal_training_or_test_launch_authorized', '_read_engineering_schedule',
                    'type(schedule.seed) is not int', 'schedule.seed != row',
                    'file_sha(_safe_path(FAILED_REVIEW, output)) != FAILED_REVIEW_SHA',
                    '_require_actual_calibration_geometry_and_scales(candidate, calibration, mobility)',
                    "ExecutionServiceCatalog('calibration')"):
                assert name in text
            calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ''
                for n in ast.walk(tree) if isinstance(n, ast.Call)}
            forbidden = {'require_reviewed_runtime', 'FrozenRuntimeArtifactBindings', 'build_cuda_model', 'witness',
                'generate_schedule', 'act', 'collect_training_episode', 'evaluate', 'update_episode', 'assemble_once'}
            assert not calls & forbidden
            assert 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_preparation.py',
                BASE/'README_VISUAL_PILOT_PREPARATION_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(), pure_explicit_artificial_pilot_candidate_checks_passed=True,
                artificial_common_conditions=2, artificial_paired_method_descriptors=4, invalid_artificial_cases_rejected=len(bad),
                explicit_artificial_geometry_or_role_window_cases_rejected=len(geometry_bad),
                explicit_artificial_support_overflow_or_invalid_cases_rejected=4,
                geometry_fit_and_complete_generation_drain_role_windows_checked_in_artificial_fixture=True,
                exact_declared_artificial_gap_boundary_accepted_without_window_shift=True,
                declared_geometry_window_seed_gap_bound_into_candidate_scientific_identity=True,
                artificial_descriptor_encoding_independent_and_defensive_copy=True,
                static_distinct_pre_pilot_authority_raw_continuity_and_no_launch_surface_checked=True,
                actual_successful_preparation_or_permission_path_exercised=False,
                actual_input_reading_physical_window_serialization_or_execution_guards_exercised=False,
                missing_actual_bindings_early_type_rejection_exercised=True,
                actual_user_reply_smoke_completed_pilot_review_or_PASS_fixtures_fabricated=False,
                actual_test_timing_CSV_opens=timing_opens, torch_imported=False,
                actual_model_optimizer_pilot_freeze_training_validation_or_test_calls=0,
                final_pilot_or_protocol_frozen=False, independent_entry_review=False,
                source_only_not_launch_authority=True,
                first_artificial_success_and_three_pre_hardening_sources_preserved=True,
                prior_182140_success_and_three_pre_geometry_repair_sources_preserved=True,
                static_whole_calibration_only_derived_scale_and_placement_verification_connected=True,
                actual_calibration_fitting_raw_catalog_or_geometry_provenance_path_exercised=False,
                static_actual_schedule_seed_and_original_FAIL_byte_continuity_only=True,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_PILOT_PREPARATION_ONLY_NOT_ACTUAL_AUTHORITY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_model_or_pilot_calls=0, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
