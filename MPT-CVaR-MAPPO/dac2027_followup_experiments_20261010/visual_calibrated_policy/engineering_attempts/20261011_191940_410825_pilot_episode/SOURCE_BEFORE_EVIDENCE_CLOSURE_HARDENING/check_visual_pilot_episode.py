"""Artificial identity/static checks only; no real actor/permission fixtures."""
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
from visual_pilot_preparation import STATUS, validate_cal_validation_pilot_candidate
from visual_pilot_episode import pilot_episode_scientific_identity, RealReadonlyPilotEpisodeCollector
from visual_test_contract import PROFILES
from visual_training_transactions import file_sha
from visual_upload_link import UploadLinkConfig

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_184921_477702_pilot_prep/RESULT.json'
PRIOR_SHA = 'deff3d303dd80547170f01d2e21f08368289c552cabb267549d3a05eefe55385'


def refused(call):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid artificial descriptor or absent actual preparation accepted')


def fixture():
    edges = ((0., 0.), (1., 0.), (0., 1.), (1., 1.))
    scales = VisualFeatureScales((0., 0.), (1., 1.), edges, PROFILES, 100., (.03, .03, .01, .01), .03)
    candidate = dict(status=STATUS, users=30, profiles=list(PROFILES), decision_seconds=.05,
        broadcast_seconds=.1, waiting_capacity=32, upload_link=asdict(UploadLinkConfig()),
        virtual_H_episode_rule='reset_each_independent_episode', feature_scales=json.loads(json.dumps(asdict(scales))),
        closure_settings=asdict(EpisodeClosureCandidate()), placement_calibration_window_seconds=[0., 500.],
        placement_seed=123, mobility_role_gap_seconds=20., engineering_conditions=[])
    for i, role in enumerate(('calibration', 'validation')):
        sha = f'{i+1:064x}'
        candidate['engineering_conditions'].append(dict(condition_id='ARTIFICIAL_'+role, role=role, seed=1,
            decisions=16, schedule_artifact=dict(path='EXPLICIT_ARTIFICIAL_NO_SCHEDULE_'+role,
                file_sha256=sha, semantic_sha256=sha),
            environment_descriptor=dict(schedule_sha256=sha, role=role, profiles=list(PROFILES),
                edge_positions=[list(point) for point in edges], mobility_start_seconds=1000.*i,
                broadcast_seconds=.1, nominal_delay_seconds=.1, waiting_capacity=32, upload_link=asdict(UploadLinkConfig()),
                catalog_source_sha256={'EXPLICIT_ARTIFICIAL_NO_CATALOG_'+role: '0'*64},
                mobility_source_sha256={f'EXPLICIT_ARTIFICIAL_NO_MOBILITY_{u}': '0'*64 for u in range(30)})))
    return candidate


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_episode')
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
                raise AssertionError('No test timing parsing in artificial pilot collector source check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            candidate = fixture()  # EXPLICIT ARTIFICIAL paths/hashes/numbers, not fitted/frozen data.
            before = deepcopy(candidate)
            descriptors = validate_cal_validation_pilot_candidate(candidate)['paired_engineering_descriptors']
            identities = [pilot_episode_scientific_identity(candidate, row, '0'*64) for row in descriptors]
            assert len(descriptors) == 4 and len(set(identities)) == 4 and candidate == before
            assert [row['role'] for row in descriptors] == ['calibration']*2+['validation']*2
            assert [row['method'] for row in descriptors] == ['QLA-MAPPO', 'PPO-MV-A']*2
            bad = []
            for field, value in (('method', 'Score-softmax'), ('role', 'test'), ('seed', True),
                    ('decisions', True), ('optimizer_updates_allowed', True), ('formal_training_or_test_cell', True),
                    ('extra_future_service', .03), ('schedule_sha256', 'f'*64)):
                changed = deepcopy(descriptors[0])
                changed[field] = value
                bad.append(changed)
            changed = deepcopy(descriptors[0])
            changed.pop('role')
            bad.append(changed)
            for row in bad:
                refused(lambda: pilot_episode_scientific_identity(candidate, row, '0'*64))
            renamed = deepcopy(candidate)
            renamed['engineering_conditions'][0]['condition_id'] = 'RENAMED_ARTIFICIAL_OUTPUT_LABEL'
            renamed['engineering_conditions'][0]['schedule_artifact']['path'] = 'ANOTHER_ARTIFICIAL_FILE_ENCODING_LOCATION'
            renamed['engineering_conditions'][0]['schedule_artifact']['file_sha256'] = 'f'*64
            renamed_descriptor = validate_cal_validation_pilot_candidate(renamed)['paired_engineering_descriptors'][0]
            assert pilot_episode_scientific_identity(renamed, renamed_descriptor, '0'*64) == identities[0]
            assert pilot_episode_scientific_identity(json.loads(json.dumps(candidate, indent=7)), descriptors[0], '0'*64) == identities[0]
            changed_ids = [pilot_episode_scientific_identity(candidate, descriptors[0], 'f'*64)]
            for change in ('seed', 'decisions', 'delay', 'calibration_window', 'placement_seed', 'role_gap', 'scale', 'closure'):
                changed = deepcopy(candidate)
                if change in ('seed', 'decisions'):
                    changed['engineering_conditions'][0][change] += 1
                elif change == 'delay':
                    changed['engineering_conditions'][0]['environment_descriptor']['nominal_delay_seconds'] = 0.
                elif change == 'calibration_window':
                    changed['placement_calibration_window_seconds'] = [0., 400.]
                elif change == 'placement_seed':
                    changed['placement_seed'] = 124
                elif change == 'role_gap':
                    changed['mobility_role_gap_seconds'] = 21.
                elif change == 'scale':
                    changed['feature_scales']['latency_scale_seconds'] = 2.
                else:
                    changed['closure_settings']['gamma_per_second'] = .98
                row = validate_cal_validation_pilot_candidate(changed)['paired_engineering_descriptors'][0]
                changed_ids.append(pilot_episode_scientific_identity(changed, row, '0'*64))
            assert all(identity != identities[0] for identity in changed_ids)
            refused(lambda: RealReadonlyPilotEpisodeCollector(None, descriptors[0], None))
            source = (BASE/'scripts/visual_pilot_episode.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else '' for node in calls}
            forbidden = {'build_cuda_model', 'witness', 'update_closed_episode', 'step', 'train', 'generate_schedule',
                'manual_seed', 'set_rng_state', 'load_checkpoint', 'load_state_dict', 'save_checkpoint',
                'require_reviewed_runtime', 'FrozenRuntimeArtifactBindings', 'assemble_once'}
            assert not names & forbidden
            actor_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'act']
            assert len(actor_calls) == 1 and any(k.arg == 'deterministic' and isinstance(k.value, ast.Constant) and k.value.value is True for k in actor_calls[0].keywords)
            for text in ('inspect_checkpoint_archive', 'actual_parameter_sha256', '_unused_optimizer_identity',
                    '_own_smoke_structure_identity', '_read_engineering_schedule', 'environment_binding',
                    'pilot_role_physical_windows', 'close_after_fixed_drain', 'closed_episode_counts',
                    '_check_reporting_final', 'partial_actual_execution_records_reporting_only',
                    'actual_closed_reporting_result_if_available', 'ACTUAL_BEHAVIOR_AND_GAE.npz',
                    'ACTUAL_READONLY_RNG.npz', "full_pilot_completed=False", 'optimizer_updates=0'):
                assert text in source
            assert names & {'observe', 'features_for', 'execute_validated_sample', 'close_after_fixed_drain'} == {'observe', 'features_for', 'execute_validated_sample', 'close_after_fixed_drain'}
            assert 'torch' not in sys.modules and not timing_opens and candidate == before
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_episode.py',
                BASE/'README_VISUAL_PILOT_EPISODE_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(), explicit_artificial_pilot_episode_identity_checks_passed=True,
                artificial_paired_descriptors=4, artificial_invalid_descriptor_cases_rejected=len(bad),
                artificial_scientific_condition_or_weight_changes_alter_identity=len(changed_ids),
                artificial_name_path_byte_encoding_changes_do_not_unlock_same_scientific_episode=True,
                static_actual_own_smoke_parameters_structure_unused_optimizer_full_episode_closure_rng_evidence_pipeline_checked=True,
                static_no_model_factory_optimizer_training_rng_restore_or_formal_entry_calls=True,
                actual_current_preparation_constructor_model_weights_load_environment_collector_or_serialization_calls=0,
                actual_resource_permission_model_path_raw_window_or_evidence_guards_exercised=False,
                actual_user_reply_smoke_checkpoint_complete_model_or_PASS_fixtures_fabricated=False,
                missing_actual_preparation_early_type_rejection_exercised=True,
                actual_test_timing_CSV_opens=timing_opens, torch_imported=False,
                full_pilot_frozen_or_completed=False, independent_entry_or_result_audit=False,
                source_only_not_launch_authority=True, original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_PILOT_EPISODE_IDENTITY_STATIC_ONLY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_actor_model_optimizer_or_pilot_calls=0, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
