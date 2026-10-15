"""Pure artificial reporting/identity/file guards; never run evaluator."""
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
from visual_training_transactions import digest, file_sha
from retained_test_evaluator import (_scientific_cell_identity, _claim_once,
    _check_reporting_final, RealReadonlyRetainedEvaluator)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_141321_058783_retained_schedule/RESULT.json'
PRIOR_SHA = 'bcf54fbbfd1ef378fcffffef3d0a281c29bb109980b02fce7ad317b2d9211769'
MATRIX = BASE/'visual_calibrated_policy/engineering_attempts/20261011_131601_349980_test_contract/EXPLICIT_ARTIFICIAL_120_DESCRIPTOR_CONTRACT.json'
MATRIX_SHA = '5ed5b3e11b47955b9374cae64d472d46f95998641f27774b75b7f253cbad2e3e'
FAILED = BASE/'visual_calibrated_policy/engineering_attempts/20261011_144904_961163_retained_evaluator'
FAILED_BINDINGS = {
    str(FAILED/'FAILURE.json'): '3f7e9875114dbc83af2f335abd13e1601b8887c4720d1a6c1ed050ca54110031',
    str(FAILED/'SOURCE_BEFORE_PATH_FIX/check_retained_test_evaluator.py'): '8127019edf24c722a433cf1b0d49a74c03132382c344e3be9fe5d1a642c0f1c1',
    str(FAILED/'SOURCE_BEFORE_PATH_FIX/retained_test_evaluator.py'): '7386050b8874f9b28d08dfd98d2b695dff6f57885417ad323138c2531d6c976e',
    str(FAILED/'SOURCE_BEFORE_PATH_FIX/README_RETAINED_TEST_EVALUATOR_中文.md'): '3380b401c95524d65d208e045246ba904e480358b2c737655e149a26417461de'}


def refused(call):
    try:
        call()
    except (ValueError, RuntimeError, TypeError, KeyError, FileExistsError):
        return
    raise AssertionError('Artificial invalid reporting/lock/source access accepted')


def artificial_final(cell, zero_completed=False, empty=False):
    # Literal counts/mass/latencies: NOT outcomes from a queue or policy.
    n = 0 if empty or zero_completed else 2
    generated, accepted, dropped, upload = (0, 0, 0, 0) if empty else (5, 3, 1, 1)
    latencies = [] if n == 0 else [2., 20.]
    work = [dict(offered=1., inflight=.1, dropped=.1, admitted=.8,
                 executed=.6, admitted_remaining=.2)] if not empty else [dict(offered=0., inflight=0.,
                 dropped=0., admitted=0., executed=0., admitted_remaining=0.)]
    work.extend(dict(offered=0., inflight=0., dropped=0., admitted=0., executed=0., admitted_remaining=0.)
                for _ in range(3))
    return dict(tasks=dict(generated=generated, accepted=accepted, completed=n, dropped=dropped,
        unfinished=generated-n-dropped, inflight_upload=upload, latency_seconds=latencies),
        uncapped_latency_seconds=latencies, work=work,
        controller_accounting=dict(admission_acknowledgments=accepted, settled_completed=n,
            drain_only_completed=0, settled_dropped=dropped, drain_only_dropped=0,
            last_action_settlement_seconds=cell['generation_seconds']+cell['decision_seconds']))


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_retained_evaluator')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA and file_sha(MATRIX) == MATRIX_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = (old['original_protected_files_sha256'] | old['new_implementation_sha256']
            | {str(PRIOR): PRIOR_SHA, str(MATRIX): MATRIX_SHA} | FAILED_BINDINGS)
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No actual retained timing CSV in this pure check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            rows = json.loads(MATRIX.read_text(encoding='utf-8'))['cells']
            cell = rows[0]
            protocol = dict(feature_scales={'EXPLICIT_ARTIFICIAL': True}, closure_settings={'EXPLICIT_ARTIFICIAL': True},
                test_rule_policy_settings={'EXPLICIT_ARTIFICIAL_NOT_A_REAL_POLICY': True},
                test_environment_binding_sha256_by_load_seed_delay={label: {str(seed): {str(delay): '0'*64
                    for delay in (0., .1, .4)} for seed in range(1, 6)} for label in ('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH')})
            weights = [dict(descriptor={'method': 'EXPLICIT_ARTIFICIAL_NOT_MODEL', 'seed': 1},
                checkpoint_manifest_sha256='EXPLICIT_ARTIFICIAL_NOT_A_CHECKPOINT',
                actual_parameter_sha256='EXPLICIT_ARTIFICIAL_NOT_PARAMETERS')]
            identity = _scientific_cell_identity(protocol, cell, weights)
            cosmetic = dict(protocol, source_files_sha256={'ARTIFICIAL_SOURCE_REVISION': 'changed'},
                review_path='ARTIFICIAL_OTHER_REVIEW_PATH', output_path='ARTIFICIAL_OTHER_OUTPUT')
            assert _scientific_cell_identity(cosmetic, deepcopy(cell), deepcopy(weights)) == identity
            assert _scientific_cell_identity(json.loads(json.dumps(protocol, indent=3)), cell, weights) == identity
            other = next(row for row in rows if row['method'] != cell['method'])
            assert _scientific_cell_identity(protocol, other, weights) != identity
            changed_weight = deepcopy(weights)
            changed_weight[0]['actual_parameter_sha256'] = 'EXPLICIT_ARTIFICIAL_DIFFERENT_PARAMETERS'
            assert _scientific_cell_identity(protocol, cell, changed_weight) != identity
            changed_design = deepcopy(protocol)
            changed_design['closure_settings']['ARTIFICIAL_DIFFERENT_DRAIN_RULE'] = True
            assert _scientific_cell_identity(changed_design, cell, weights) != identity
            # Short artificial-only fixture name avoids this workspace's
            # Windows long-path boundary. Formal lock root/identity unchanged.
            fixture_root = attempt/'ART_LOCK'
            lock = _claim_once(fixture_root, identity, dict(EXPLICIT_ARTIFICIAL_FILE_IO=True, output_path='ARTIFICIAL-A'))
            lock_sha = file_sha(lock)
            refused(lambda: _claim_once(fixture_root, identity, dict(EXPLICIT_ARTIFICIAL_FILE_IO=True, output_path='ARTIFICIAL-B')))
            assert file_sha(lock) == lock_sha
            normal = artificial_final(cell)
            summary = _check_reporting_final(normal, 5, cell['generation_decisions'], cell)
            assert summary['completed_samples'] == 2 and summary['mean_seconds'] == 11.
            assert summary['p99_seconds'] == 20. and summary['empirical_worst_approximately_one_percent_mean_seconds'] == 20.
            zero = _check_reporting_final(artificial_final(cell, zero_completed=True), 5, cell['generation_decisions'], cell)
            empty = _check_reporting_final(artificial_final(cell, empty=True), 0, cell['generation_decisions'], cell)
            assert zero['mean_seconds'] is None and zero['tail_samples'] == 0 and zero['drops_and_unfinished_retained']
            assert empty['mean_seconds'] is None and empty['completed_samples'] == 0
            corruptions = []
            def bad(edit):
                f = deepcopy(normal)
                edit(f)
                corruptions.append(f)
            bad(lambda f: f['tasks'].update(generated=4))
            bad(lambda f: f['tasks'].update(accepted=True))
            bad(lambda f: f['tasks'].update(unfinished=0))
            bad(lambda f: f.update(uncapped_latency_seconds=[1.]))
            bad(lambda f: f.update(uncapped_latency_seconds=[2., float('nan')]))
            bad(lambda f: f.update(uncapped_latency_seconds=[2., 1000000.]))
            bad(lambda f: f['tasks'].update(latency_seconds=[2., 19.]))
            bad(lambda f: f['controller_accounting'].update(drain_only_completed=1))
            bad(lambda f: f['controller_accounting'].update(last_action_settlement_seconds=3.))
            bad(lambda f: f['work'][0].update(admitted_remaining=.1))
            bad(lambda f: f['work'].pop())
            for f in corruptions:
                refused(lambda: _check_reporting_final(f, 5, cell['generation_decisions'], cell))
            refused(lambda: _check_reporting_final(normal, 5, cell['generation_decisions']-1, cell))
            refused(lambda: RealReadonlyRetainedEvaluator(None, cell, None, None))
            tree = ast.parse((BASE/'scripts/retained_test_evaluator.py').read_text(encoding='utf-8'))
            called_attributes = {node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            assert not {'update_closed_episode', 'step', 'backward', 'load_checkpoint'} & called_attributes
            assert not timing_opens and 'torch' not in sys.modules
            sources = {str(path): file_sha(path) for path in (Path(__file__),
                BASE/'scripts/retained_test_evaluator.py', BASE/'README_RETAINED_TEST_EVALUATOR_中文.md')}
            write_evidence(attempt/'EXPLICIT_ARTIFICIAL_REPORTING_FIXTURES_NOT_RESULTS.json',
                dict(normal=normal, zero_completed=artificial_final(cell, zero_completed=True),
                    empty=artificial_final(cell, empty=True), summaries=[summary, zero, empty],
                    literal_not_queue_or_policy_outcomes=True))
            report = dict(at=datetime.now().astimezone().isoformat(), pure_reporting_and_missing_authority_guards_passed=True,
                artificial_reporting_cases_not_execution=3, altered_reporting_cases_rejected=len(corruptions)+1,
                zero_completed_drop_unfinished_and_empty_arrival_cases_retained=True,
                exclusive_artificial_file_lock_rejects_output_path_change=True,
                semantic_cell_identity_does_not_change_with_cosmetic_source_review_encoding_or_output_name=True,
                cell_identity_changes_with_scientific_method_weights_or_closure=True,
                no_optimizer_training_or_model_loader_calls_in_evaluator_source=True,
                actual_evaluator_constructor_succeeded=False, actual_evaluator_or_queue_policy_calls=0,
                actual_test_cells=0, actual_original_test_timing_CSV_opens=timing_opens,
                torch_imported=False, original_protected_files_sha256=protected, new_implementation_sha256=sources,
                complete_grid_runner_prepared=False, independent_entry_review=False, formal_pilot_or_protocol_frozen=False)
            report.update(prior_artificial_fixture_path_failure_preserved=str(FAILED),
                only_artificial_file_fixture_directory_shortened=True, formal_evaluator_lock_logic_changed_for_path_repair=False,
                global_Windows_long_path_setting_changed=False)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_REPORTING_LOCK_SOURCE_ONLY_NOT_RETAINED_POLICY_EXECUTION', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_evaluator_or_policy_calls=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
