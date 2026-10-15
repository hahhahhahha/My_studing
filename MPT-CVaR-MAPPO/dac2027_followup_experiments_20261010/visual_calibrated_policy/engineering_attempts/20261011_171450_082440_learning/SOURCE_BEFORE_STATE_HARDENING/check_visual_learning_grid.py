"""Pure explicit artificial job identities and static wiring, no training."""
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

from visual_episode_collector import BASE, write_evidence
from visual_checkpoint_registration import FinalCheckpointRuleCandidate
from visual_training_transactions import CommonBudgetCandidate, file_sha
from visual_learning_grid import (common_learning_job_descriptors, validate_learning_job_coverage,
    learning_scientific_identity, RealTenLearningValidationRunner, _new_actual_training_controller)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_165034_526108_exec/RESULT.json'
PRIOR_SHA = '53ccc7afde59269a5354b4cd6a832d157b78ad91342ebc5d39507a64781d0af0'


def refused(call):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Changed artificial job or missing actual controller/authority accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_learning')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        raw_test = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        original_open = Path.open
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw_test.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No actual retained test timing CSV in artificial learning-entry check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            rule = FinalCheckpointRuleCandidate(CommonBudgetCandidate((4, 4)))
            rows = common_learning_job_descriptors(rule)
            assert [(row['seed'], row['method']) for row in rows] == [(seed, method)
                for seed in (1, 2, 3, 4, 5) for method in ('QLA-MAPPO', 'PPO-MV-A')]
            assert validate_learning_job_coverage(rows, rule) == validate_learning_job_coverage(list(reversed(rows)), rule)
            assert all(row['common_budget']['episode_decisions'] == [4, 4] for row in rows)
            damaged = [rows[:-1], rows+[deepcopy(rows[0])]]
            for field, value in (('seed', True), ('method', 'ARTIFICIAL_OTHER_METHOD'), ('training_role', 'test'),
                    ('validation_role', 'test'), ('test_access', True), ('performance_filter_allowed', True),
                    ('checkpoint_rule', 'ARTIFICIAL_BEST_OUTCOME'), ('EXTRA_FUTURE_VALUE', 1.)):
                changed = deepcopy(rows)
                changed[0][field] = value
                damaged.append(changed)
            changed = deepcopy(rows)
            changed[0]['common_budget']['episode_decisions'] = [4, 5]
            damaged.append(changed)
            for changed in damaged:
                refused(lambda: validate_learning_job_coverage(changed, rule))
            artificial = dict(EXPLICIT_ARTIFICIAL_NO_PILOT_OR_COMPLETED_MARKER=True,
                paired_design_by_seed={str(seed): f'{seed:064x}' for seed in range(1, 6)},
                feature_scales={'ARTIFICIAL_ONLY': 1.}, closure_settings={'ARTIFICIAL_DRAIN': 1.},
                optimizer_settings={'ARTIFICIAL_ONLY': .001}, learner_kwargs_by_method={'ARTIFICIAL_ONLY': True},
                virtual_H_episode_rule='reset_each_independent_episode',
                training_schedule_sha256_by_seed={'ARTIFICIAL_ONLY': ['0'*64]},
                validation_schedule_sha256_by_seed={'ARTIFICIAL_ONLY': ['1'*64]},
                training_environment_sha256_by_seed={'ARTIFICIAL_ONLY': ['2'*64]},
                validation_environment_sha256_by_schedule={'ARTIFICIAL_ONLY': '3'*64})
            identity = learning_scientific_identity(artificial, rule)
            assert learning_scientific_identity(json.loads(json.dumps(artificial, indent=7)), rule) == identity
            assert learning_scientific_identity(dict(artificial, output_path='ARTIFICIAL-B', review_bytes='ARTIFICIAL'), rule) == identity
            changed_identity_fields = ('feature_scales', 'closure_settings', 'optimizer_settings',
                'learner_kwargs_by_method', 'training_schedule_sha256_by_seed', 'validation_schedule_sha256_by_seed',
                'training_environment_sha256_by_seed', 'validation_environment_sha256_by_schedule', 'paired_design_by_seed')
            for field in changed_identity_fields:
                changed = deepcopy(artificial)
                changed[field] = {'EXPLICIT_ARTIFICIAL_CHANGED_CONDITION': 'f'*64}
                assert learning_scientific_identity(changed, rule) != identity
            refused(lambda: RealTenLearningValidationRunner(None))
            refused(lambda: _new_actual_training_controller(None, 'QLA-MAPPO', 1, None, None))
            source = BASE/'scripts/visual_learning_grid.py'
            tree = ast.parse(source.read_text(encoding='utf-8'))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RealTenLearningValidationRunner')
            run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'run_once')
            calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
            named = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
            attr = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
            assert named.count('build_cuda_model') == 1
            assert all(name in named for name in ('_new_actual_training_controller', 'RealMultiepisodeTrainingCoordinator',
                'load_own_snapshot_in_verified_cuda_process', 'RealReadonlyValidationEvaluator', 'RealFinalCheckpointRegistrar'))
            assert all(name in attr for name in ('run', 'evaluate', 'register', 'seal_all', '_boundary'))
            assert not {'RetainedExecutionServiceCatalog', 'RealReadonlyRetainedEvaluator',
                'RealFrozenWeightTestGridRunner', 'generate_schedule', 'fitting_rows'} & set(named+attr)
            assert 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), source, BASE/'README_VISUAL_LEARNING_GRID_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(),
                pure_artificial_common_job_identity_budget_and_static_wiring_checks_passed=True,
                explicit_artificial_training_job_descriptors=10, actual_training_proven_by_descriptors=0,
                artificial_missing_duplicate_foreign_seed_role_budget_ranking_cases_rejected=len(damaged),
                order_independent_coverage_and_fixed_seed_method_execution_order=True,
                scientific_identity_ignores_output_review_and_JSON_cosmetics=True,
                artificial_scientific_condition_changes_alter_identity=len(changed_identity_fields),
                missing_real_authority_and_missing_actual_controller_rejected_before_model_or_output=True,
                static_one_initial_gated_factory_train_final_all_validation_register_seal_wiring=True,
                real_constructor_run_factory_model_load_train_validation_register_or_seal_calls=0,
                actual_calibration_validation_catalog_environment_or_generator_calls=0,
                actual_original_test_timing_CSV_opens=timing_opens, torch_imported=False,
                actual_runtime_path_resource_model_or_protected_input_guards_exercised=False,
                actual_model_weights_complete_training_validation_or_PASS_fixtures_fabricated=False,
                actual_formal_training_validation_or_test_cells=0, independent_entry_review=False,
                formal_pilot_or_protocol_frozen=False, source_only_not_launch_authority=True,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_JOBS_AND_STATIC_WIRING_ONLY_NOT_ACTUAL_LEARNING', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_model_or_training_validation_calls=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
