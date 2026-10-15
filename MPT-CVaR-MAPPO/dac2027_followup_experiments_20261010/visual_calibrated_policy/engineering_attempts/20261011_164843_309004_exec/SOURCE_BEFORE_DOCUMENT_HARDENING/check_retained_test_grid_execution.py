"""Explicit artificial ordering/routing and static wiring ONLY, no models."""
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

from visual_episode_collector import BASE, write_evidence
from visual_method_features import VisualFeatureScales
from visual_test_contract import FixedTestMatrixCandidate, PROFILES
from visual_training_transactions import file_sha, digest
from retained_test_grid_execution import (fixed_cell_execution_order, required_own_weight_key,
    require_matching_weight_descriptor, policy_dimensions_from_frozen_scales,
    load_registered_own_test_controller, RealFrozenWeightTestGridRunner)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_162102_429274_proto/RESULT.json'
PRIOR_SHA = 'fdc401dacf3e0903c5a2e75afb121e1c76eb431acfa2b3fa30a55f07f5bdbf4f'


def refused(call):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid artificial identity/coverage or missing real authority accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_exec')
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
                raise AssertionError('No actual retained timing CSV in artificial execution-entry check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            common = tuple(tuple(f'{li*5+seed:064x}' for seed in range(1, 6)) for li in range(2))
            candidate = FixedTestMatrixCandidate(('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH'), (30., 60.), 8, 1., common, '0'*64)
            rows = candidate.cells()
            ordered = fixed_cell_execution_order(list(reversed(rows)), candidate)
            assert ordered == fixed_cell_execution_order(rows, candidate)
            assert ordered[:4] == [next(row for row in rows if row['load_label'] == 'ARTIFICIAL_LOW'
                and row['seed'] == 1 and row['nominal_delay_seconds'] == 0. and row['method'] == method)
                for method in ('QLA-MAPPO', 'PPO-MV-A', 'JSEQ-A', 'Score-softmax')]
            assert len({digest(row) for row in ordered}) == 120
            counts = {}
            for cell in ordered:
                key = required_own_weight_key(cell)
                if key is not None:
                    counts[key] = counts.get(key, 0)+1
                    # This is a two-field ARTIFICIAL identity ONLY. No path,
                    # model bytes, actual parameter SHA or completed marker.
                    assert require_matching_weight_descriptor(cell, dict(method=key[0], seed=key[1])) == key
                    refused(lambda: require_matching_weight_descriptor(cell, dict(method=key[0], seed=True)))
                    refused(lambda: require_matching_weight_descriptor(cell, dict(method=key[0], seed=6-key[1] if key[1] != 3 else 1)))
                    refused(lambda: require_matching_weight_descriptor(cell, dict(method='ARTIFICIAL_FOREIGN_METHOD', seed=key[1])))
                else:
                    refused(lambda: require_matching_weight_descriptor(cell, dict(method=cell['method'], seed=cell['seed'])))
            assert len(counts) == 10 and set(counts.values()) == {6}
            assert sum(counts.values()) == 60 and sum(required_own_weight_key(c) is None for c in ordered) == 60
            damaged = [rows[:-1], rows+[deepcopy(rows[0])]]
            for field, value in (('stop_on_completed_task_count', True), ('seed', True), ('method', 'ARTIFICIAL_OTHER'),
                    ('common_exogenous_schedule_sha256', 'f'*64), ('nominal_delay_seconds', .2)):
                changed = deepcopy(rows)
                changed[0][field] = value
                damaged.append(changed)
            for changed in damaged:
                refused(lambda: fixed_cell_execution_order(changed, candidate))
            saved = deepcopy(rows)
            ordered[0]['generation_decisions'] = 999
            assert rows == saved
            scales = VisualFeatureScales((0., 0.), (1., 1.), ((0., 0.), (1., 0.), (0., 1.), (1., 1.)),
                PROFILES, 100., (.03, .03, .01, .01), .03)
            dimensions = policy_dimensions_from_frozen_scales(scales, 30)
            assert asdict(dimensions) == dict(n_users=30, n_edges=4, actor_obs_dim=19, critic_obs_dim=705,
                max_supported_users=30, include_feedback_age=True, feedback_age_norm=.5)
            refused(lambda: policy_dimensions_from_frozen_scales(scales, True))
            refused(lambda: RealFrozenWeightTestGridRunner(None))
            refused(lambda: load_registered_own_test_controller(None, None, rows[0]))
            source = BASE/'scripts/retained_test_grid_execution.py'
            tree = ast.parse(source.read_text(encoding='utf-8'))
            definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            loader = definitions['load_registered_own_test_controller']
            loader_calls = [node for node in ast.walk(loader) if isinstance(node, ast.Call)]
            assert not any(isinstance(node.func, ast.Name) and node.func.id == 'build_cuda_model' for node in loader_calls)
            assert any(isinstance(node.func, ast.Name) and node.func.id == 'actual_parameter_sha256' for node in loader_calls)
            assert not any(isinstance(node.func, ast.Name) and node.func.id == 'validation_checkpoint_reference' for node in loader_calls)
            torch_loads = [node for node in loader_calls if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load']
            assert len(torch_loads) == 1 and {kw.arg for kw in torch_loads[0].keywords} == {'map_location', 'weights_only'}
            assert any(kw.arg == 'weights_only' and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in torch_loads[0].keywords)
            run = next(node for node in definitions['RealFrozenWeightTestGridRunner'].body
                if isinstance(node, ast.FunctionDef) and node.name == 'run_once')
            calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
            named = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
            attr = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
            assert named.count('build_cuda_model') == 1 and named.count('load_registered_own_test_controller') == 1
            assert all(name in attr for name in ('evaluate', 'register', 'seal_all', '_boundary'))
            assert not {'step', 'backward', 'update_closed_episode', 'generate_schedule', 'fitting_rows'} & set(attr+named)
            assert not timing_opens and 'torch' not in sys.modules
            sources = {str(path): file_sha(path) for path in (Path(__file__), source, BASE/'README_RETAINED_TEST_EXECUTION_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(),
                pure_artificial_ordering_weight_identity_and_static_wiring_checks_passed=True,
                explicit_artificial_cell_descriptors=120, learned_weight_identity_keys=10,
                neural_descriptors=60, rule_descriptors=60, descriptors_per_own_method_seed=6,
                order_independent_then_fixed_execution_order=True, input_descriptors_not_mutated=True,
                artificial_wrong_method_seed_bool_or_rule_identity_cases_rejected=240,
                artificial_invalid_missing_duplicate_early_stop_schedule_delay_cases_rejected=len(damaged),
                pure_actor_critic_dimension_formula=[30, 4, 19, 705],
                missing_actual_contract_rejected_before_Torch_model_or_output=True,
                static_one_initial_factory_and_per_cell_own_loader_evaluator_register_seal_wiring=True,
                real_grid_constructor_run_or_cell_calls=0, actual_model_constructor_load_or_update_calls=0,
                actual_original_test_timing_CSV_opens=timing_opens, actual_metadata_generator_or_environment_calls=0,
                torch_imported=False, actual_runtime_source_or_path_or_model_guards_exercised=False,
                actual_completed_weight_smoke_review_or_cell_fixtures_fabricated=False,
                actual_training_validation_or_test_cells=0, independent_entry_review=False,
                formal_pilot_or_protocol_frozen=False, source_only_not_launch_permission=True,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_ORDERING_AND_STATIC_ENTRY_ONLY_NOT_ACTUAL_EXECUTION', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_model_or_grid_or_cell_calls=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
