"""Pure artificial order/identity and AST checks; no actual pilot fixtures."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from check_visual_pilot_episode import fixture, refused
from visual_episode_collector import BASE, write_evidence
from visual_pilot_driver import (pilot_driver_job_descriptors, validate_pilot_driver_job_coverage,
    pilot_driver_scientific_identity, RealPairedReadonlyPilotDriver, load_own_smoke_model_for_readonly_pilot)
from visual_training_transactions import METHODS, file_sha

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_192159_383900_pilot_episode/RESULT.json'
PRIOR_SHA = '1c571dcbaa192c58468f086145ec58e0737620c7f5ef0c557b4c4cf28eec9c02'


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_driver')
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
                raise AssertionError('No test timing parsing in artificial pilot driver check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            candidate = fixture()  # EXPLICIT ARTIFICIAL paths/hashes/scales, never actual candidate/authority.
            before = deepcopy(candidate)
            hashes = dict(zip(METHODS, ('1'*64, '2'*64)))  # Artificial identity strings, NOT saved model parameters.
            jobs = pilot_driver_job_descriptors(candidate)
            checked = validate_pilot_driver_job_coverage(jobs, candidate)
            assert checked['paired_episode_descriptors'] == 4 and checked['actual_closed_episodes_proven'] == 0
            assert [row['role'] for row in jobs] == ['calibration']*2+['validation']*2
            assert [row['method'] for row in jobs] == list(METHODS)*2 and candidate == before
            invalid = [jobs[:-1], jobs+[jobs[0]], list(reversed(jobs)), jobs[1:]+jobs[:1]]
            for field, value in (('method', 'JSEQ'), ('role', 'test'), ('seed', True), ('decisions', True),
                    ('optimizer_updates_allowed', True), ('formal_training_or_test_cell', True), ('future_truth', .01)):
                rows = deepcopy(jobs)
                rows[0][field] = value
                invalid.append(rows)
            for rows in invalid:
                refused(lambda: validate_pilot_driver_job_coverage(rows, candidate))
            identity = pilot_driver_scientific_identity(candidate, hashes)
            renamed = deepcopy(candidate)
            renamed['engineering_conditions'].reverse()
            for row in renamed['engineering_conditions']:
                row['condition_id'] = 'RENAMED_'+row['condition_id']
                row['schedule_artifact']['path'] = 'ANOTHER_ARTIFICIAL_ENCODING_'+row['role']
                row['schedule_artifact']['file_sha256'] = 'f'*64
            assert pilot_driver_scientific_identity(renamed, hashes) == identity
            assert pilot_driver_scientific_identity(json.loads(json.dumps(candidate, indent=7)), hashes) == identity
            duplicate = deepcopy(candidate)
            another = deepcopy(duplicate['engineering_conditions'][0])
            another['condition_id'] = 'RENAMED_DUPLICATE_ARTIFICIAL_SCIENCE'
            another['schedule_artifact']['path'] = 'OTHER_ARTIFICIAL_BYTES_SAME_SCHEDULE'
            another['schedule_artifact']['file_sha256'] = 'f'*64
            duplicate['engineering_conditions'].append(another)
            refused(lambda: pilot_driver_job_descriptors(duplicate))
            hash_rejections = [None, {}, {METHODS[0]: '1'*64}, dict(hashes, foreign='3'*64),
                {METHODS[0]: 'NOT_MODEL_OR_HASH', METHODS[1]: '2'*64}]
            for value in hash_rejections:
                refused(lambda: pilot_driver_scientific_identity(candidate, value))
            changes = []
            for method in METHODS:
                changed = dict(hashes)
                changed[method] = 'f'*64
                changes.append(pilot_driver_scientific_identity(candidate, changed))
            for field in ('seed', 'decisions', 'delay', 'scale', 'closure', 'placement_seed'):
                changed = deepcopy(candidate)
                if field in ('seed', 'decisions'):
                    changed['engineering_conditions'][0][field] += 1
                elif field == 'delay':
                    changed['engineering_conditions'][0]['environment_descriptor']['nominal_delay_seconds'] = 0.
                elif field == 'scale':
                    changed['feature_scales']['latency_scale_seconds'] = 2.
                elif field == 'closure':
                    changed['closure_settings']['gamma_per_second'] = .98
                else:
                    changed['placement_seed'] += 1
                changes.append(pilot_driver_scientific_identity(changed, hashes))
            assert all(value != identity for value in changes)
            refused(lambda: RealPairedReadonlyPilotDriver(None))
            refused(lambda: load_own_smoke_model_for_readonly_pilot(None, None, METHODS[0], 1))
            source = (BASE/'scripts/visual_pilot_driver.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            def call_name(node):
                return node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
            names = [call_name(node) for node in calls]
            assert names.count('build_cuda_model') == 1 and 'witness' not in names
            assert names.count('load_own_smoke_model_for_readonly_pilot') == 1 and names.count('collect_once') == 1
            forbidden = {'require_reviewed_runtime', 'FrozenRuntimeArtifactBindings', 'validation_checkpoint_reference',
                'load_own_snapshot_in_verified_cuda_process', 'update_closed_episode', 'step', 'train',
                'generate_schedule', 'set_rng_state', 'set_rng_state_all', 'load_optimizer_state_dict', 'seal_all', 'assemble_once'}
            assert not set(names) & forbidden and 'act' not in names
            assert source.index('self._boundary()\n            # ONE original factory') < source.index('build_cuda_model(\'QLA-MAPPO\'')
            assert source.index("current = dict(index=index") < source.index('loaded, witness = load_own_smoke_model_for_readonly_pilot')
            assert source.index('loaded, witness = load_own_smoke_model_for_readonly_pilot') < source.index('collector = RealReadonlyPilotEpisodeCollector') < source.index('complete = collector.collect_once')
            for text in ('_actual_smoke_weight_bindings', '_own_smoke_structure_identity', '_unused_optimizer_identity',
                    '_reference_identity(reference) != before', "map_location='cuda', weights_only=True", 'actor_only=False',
                    'pilot_role_physical_windows', '_read_engineering_schedule', 'environment_binding', 'Fraction(str(',
                    'CLOSED_DRIVER_INPUTS.json', 'ACTUAL_PAIRED_EPISODES_PENDING_REVIEW.json',
                    'full_pilot_completed=False', 'optimizer_updates=0', 'formal_cells=0',
                    'completed_episode_markers_preserved', 'automatic_append_resume_retry_or_partial_arm_merge_allowed=False'):
                assert text in source
            assert not any(isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names) for node in ast.walk(tree))
            assert 'torch' not in sys.modules and not timing_opens and candidate == before
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_driver.py',
                BASE/'README_VISUAL_PILOT_DRIVER_中文.md')}
            write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(),
                explicit_artificial_paired_driver_order_identity_and_AST_checks_passed=True,
                artificial_condition_count=2, artificial_paired_episode_descriptors=4,
                artificial_bad_job_coverage_cases_rejected=len(invalid), artificial_bad_parameter_binding_cases_rejected=len(hash_rejections),
                artificial_scientific_or_parameter_changes_alter_identity=len(changes),
                artificial_label_path_byte_encoding_condition_order_does_not_unlock_same_driver=True,
                artificial_renamed_duplicate_scientific_episode_refused=True,
                static_one_original_factory_before_MODEL_only_own_load_then_full_collector_pipeline_checked=True,
                static_no_second_witness_formal_pilot_circular_gate_optimizer_resume_test_or_training_calls=True,
                absent_actual_preparation_and_binding_early_type_rejections_exercised=True,
                actual_preparation_resource_kernel_model_load_environment_collect_serialization_or_driver_calls=0,
                actual_resource_permission_model_path_raw_window_or_evidence_guards_exercised=False,
                actual_user_reply_smoke_checkpoint_MODEL_or_PASS_fixtures_fabricated=False,
                actual_test_timing_CSV_opens=timing_opens, torch_imported=False,
                completed_pilot_or_final_frozen_protocol=False, independent_audit=False, source_only_not_launch_authority=True,
                original_protected_files_sha256=protected, new_implementation_sha256=sources))
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_PAIRED_PILOT_DRIVER_IDENTITY_AST_ONLY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_resource_model_load_collector_driver_or_optimizer_calls=0, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
