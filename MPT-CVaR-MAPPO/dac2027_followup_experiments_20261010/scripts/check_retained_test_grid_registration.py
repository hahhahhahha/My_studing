"""Pure artificial descriptor coverage/lock guards, NOT actual results."""
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
from visual_training_transactions import file_sha, digest
from visual_test_contract import FixedTestMatrixCandidate
from retained_test_grid_registration import (check_complete_descriptor_coverage,
    _claim_grid_once, RealRetainedGridRegistrar)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_145043_217789_retained_evaluator/RESULT.json'
PRIOR_SHA = '49272a215b213a05e553fc778be94f4ef28941a1164bc9eed9e51e6a5edf844f'
MATRIX = BASE/'visual_calibrated_policy/engineering_attempts/20261011_131601_349980_test_contract/EXPLICIT_ARTIFICIAL_120_DESCRIPTOR_CONTRACT.json'
MATRIX_SHA = '5ed5b3e11b47955b9374cae64d472d46f95998641f27774b75b7f253cbad2e3e'
PRE_HARDENING = BASE/'visual_calibrated_policy/engineering_attempts/20261011_151614_872600_test_grid'
PRE_HARDENING_BINDINGS = {
    str(PRE_HARDENING/'RESULT.json'): '4993e791899c5eae261c8cb7283d7cf4e48c76e495b18e62253f00330eb2fb2b',
    str(PRE_HARDENING/'SOURCE_BEFORE_COUNT_HARDENING/check_retained_test_grid_registration.py'): 'dbe4bd99cd4d16288626e9a4aff540bee5034c4163cd9e608431275d947a09a1',
    str(PRE_HARDENING/'SOURCE_BEFORE_COUNT_HARDENING/retained_test_grid_registration.py'): '02da24a217e26cc039db215da863ec9c2d15fb9f62728dea8d074304e9e672b6',
    str(PRE_HARDENING/'SOURCE_BEFORE_COUNT_HARDENING/README_RETAINED_TEST_GRID_中文.md'): '3fe3038b445d67d6367b38174795ddb9c3f1acdd49c77ac20828205b011696c1'}


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, KeyError, FileExistsError):
        return
    raise AssertionError('Invalid artificial coverage/lock/missing authority accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_test_grid')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA and file_sha(MATRIX) == MATRIX_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = (old['original_protected_files_sha256'] | old['new_implementation_sha256']
            | {str(PRIOR): PRIOR_SHA} | PRE_HARDENING_BINDINGS)
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No actual retained timing CSV in pure descriptor check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            matrix = json.loads(MATRIX.read_text(encoding='utf-8'))
            definition = dict(matrix['definition'])
            for name in ('load_labels', 'total_rates_per_second', 'profiles', 'seeds'):
                definition[name] = tuple(definition[name])
            definition['common_schedule_sha256_by_load_seed'] = tuple(tuple(row)
                for row in definition['common_schedule_sha256_by_load_seed'])
            candidate = FixedTestMatrixCandidate(**definition)
            rows = matrix['cells']
            coverage = check_complete_descriptor_coverage(rows, candidate)
            assert coverage['actual_result_cells_proven'] == 0
            assert check_complete_descriptor_coverage(list(reversed(rows)), candidate) == coverage
            corruptions = [rows[:-1], rows+[deepcopy(rows[0])], [deepcopy(rows[0]) for _ in rows]]
            for key, value in (('method', 'ARTIFICIAL_CHANGED_METHOD'), ('seed', True),
                    ('load_label', 'ARTIFICIAL_OTHER_LOAD'), ('nominal_delay_seconds', .2),
                    ('common_exogenous_schedule_sha256', 'f'*64), ('total_requests_per_second', 1.),
                    ('generation_decisions', 1), ('fixed_drain_seconds', 1.),
                    ('stop_on_completed_task_count', True), ('role', 'calibration'),
                    ('retained_pool_identity_sha256', 'f'*64), ('EXTRA_UNREGISTERED_FIELD', True)):
                altered = deepcopy(rows)
                altered[0][key] = value
                corruptions.append(altered)
            for altered in corruptions:
                refused(lambda: check_complete_descriptor_coverage(altered, candidate))
            refused(lambda: RealRetainedGridRegistrar(None, attempt/'NO_ACTUAL_GRID'))
            assert not (attempt/'NO_ACTUAL_GRID').exists()
            identity = digest(dict(EXPLICIT_ARTIFICIAL_DESCRIPTOR_ONLY=candidate.definition()))
            lock = _claim_grid_once(attempt/'ART_LOCK', identity,
                dict(EXPLICIT_ARTIFICIAL_NOT_ACTUAL_RESULTS=True, output_path='ARTIFICIAL-A'))
            sha = file_sha(lock)
            refused(lambda: _claim_grid_once(attempt/'ART_LOCK', identity,
                dict(EXPLICIT_ARTIFICIAL_NOT_ACTUAL_RESULTS=True, output_path='ARTIFICIAL-B')))
            assert file_sha(lock) == sha
            tree = ast.parse((BASE/'scripts/retained_test_grid_registration.py').read_text(encoding='utf-8'))
            called = {node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            assert not {'evaluate', 'act', 'portfolios', 'step', 'backward', 'update_closed_episode', 'load_checkpoint'} & called
            assert not timing_opens and 'torch' not in sys.modules
            sources = {str(path): file_sha(path) for path in (Path(__file__),
                BASE/'scripts/retained_test_grid_registration.py', BASE/'README_RETAINED_TEST_GRID_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(), pure_descriptor_coverage_and_file_lock_guards_passed=True,
                explicit_artificial_descriptor_cells=120, coverage=coverage,
                changed_missing_duplicate_early_stop_or_pairing_descriptor_cases_rejected=len(corruptions),
                order_independent_exact_matrix=True, exclusive_artificial_grid_lock_rejects_output_path_change=True,
                real_missing_authority_constructor_rejected_before_output_creation=True,
                actual_grid_registrar_constructor_succeeded=False, actual_register_or_seal_calls=0,
                actual_evaluator_or_queue_policy_calls=0, actual_test_cells=0,
                actual_original_test_timing_CSV_opens=timing_opens, torch_imported=False,
                original_protected_files_sha256=protected, new_implementation_sha256=sources,
                actual_result_or_completed_marker_fixtures_fabricated=False,
                independent_entry_review=False, formal_pilot_or_protocol_frozen=False,
                whole_grid_execution_launcher_prepared=False, source_registration_only_not_launcher=True,
                pre_hardening_success_and_sources_preserved=str(PRE_HARDENING),
                generated_count_reconstructed_from_registered_common_requests_source_only=True,
                actual_completed_result_marker_or_real_registrar_guards_exercised=False)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_DESCRIPTOR_COVERAGE_ONLY_NOT_ACTUAL_120_RESULTS', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_grid_constructor_register_or_seal_calls=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
