"""Original metadata identity + artificial 120-row contract, not evaluations."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

from visual_episode_collector import BASE, write_evidence
from visual_training_transactions import digest, file_sha
from visual_training_coordinator import FrozenRuntimeArtifactBindings
from visual_test_contract import (ORIGINAL, OriginalRetainedMetadataPool,
    FixedTestMatrixCandidate, verify_cell_matrix, fixed_generation_decisions_candidate,
    FrozenTestExecutionContract, TEST_METHODS)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_124513_352423_checkpoint_rule/RESULT.json'
PRIOR_SHA = 'f0cde8c6df710b946af2010c16fa9c879872eca14c2654aa414cd3b8ecd0b3aa'


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Invalid artificial test contract accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_test_contract')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | ORIGINAL | {str(PRIOR): PRIOR_SHA}
        assert all(file_sha(Path(path)) == sha for path, sha in protected.items())
        pool = OriginalRetainedMetadataPool()
        assert len(pool.image_ids()) == 1000
        metadata = pool.observable_metadata(pool.image_ids()[0])
        assert set(metadata.__dataclass_fields__) == {'width', 'height', 'payload_bytes'}
        refused(lambda: pool.measured_pair_seconds(pool.image_ids()[0], 1))
        refused(lambda: pool.fitting_rows())
        # SHA strings are explicit artificial schedule descriptors. No actual
        # retained-test arrival/session/uniform/request schedule is generated.
        artificial_schedules = tuple(tuple(digest(dict(EXPLICIT_ARTIFICIAL_SCHEDULE=True, load=i, seed=seed))
            for seed in (1, 2, 3, 4, 5)) for i in (0, 1))
        candidate = FixedTestMatrixCandidate(('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH'), (2., 3.),
            60, 30., artificial_schedules, pool.identity_sha256)
        rows = candidate.cells()
        result = verify_cell_matrix(rows, candidate)
        assert result == dict(descriptor_cells=120, paired_seeds=5, common_exogenous_bindings=10,
                              actual_test_execution_cells=0, descriptor_check_not_results=True)
        assert all(row['generation_seconds'] == 3. and row['fixed_end_seconds'] == 33.
            and row['stop_on_completed_task_count'] is False for row in rows)
        for label in candidate.load_labels:
            for seed in candidate.seeds:
                group = [row for row in rows if row['load_label'] == label and row['seed'] == seed]
                assert len(group) == 12 and len({row['common_exogenous_schedule_sha256'] for row in group}) == 1
                assert {row['method'] for row in group} == set(TEST_METHODS)
                assert {row['nominal_delay_seconds'] for row in group} == {0., .1, .4}
        verify_cell_matrix(list(reversed(rows)), candidate)
        refused(lambda: verify_cell_matrix(rows[:-1], candidate))
        refused(lambda: verify_cell_matrix(rows[:-1]+[rows[0]], candidate))
        for field, value in (('generation_decisions', 61), ('fixed_drain_seconds', 31.),
                ('fixed_end_seconds', 34.), ('stop_on_completed_task_count', True),
                ('common_exogenous_schedule_sha256', '0'*64), ('role', 'calibration'),
                ('retained_pool_identity_sha256', '0'*64), ('nominal_delay_seconds', .2),
                ('total_requests_per_second', 999.)):
            bad = deepcopy(rows)
            bad[0][field] = value
            refused(lambda bad=bad: verify_cell_matrix(bad, candidate))
        bad = deepcopy(rows)
        bad[0]['ignored_extra_field'] = 'Not a new test'
        refused(lambda: verify_cell_matrix(bad, candidate))
        refused(lambda: replace(candidate, seeds=(True, 2, 3, 4, 5)))
        refused(lambda: replace(candidate, profiles=('CPU-4T',)*4))
        refused(lambda: replace(candidate, total_rates_per_second=(3., 2.)))
        refused(lambda: replace(candidate, generation_decisions=True))
        assert fixed_generation_decisions_candidate(200., .05) == 2400
        assert fixed_generation_decisions_candidate(2., .05) == 100000
        # Non-integral physical rounding: 10000/100.01 versus .07 step.
        n = fixed_generation_decisions_candidate(100.01, .07, 1.)
        assert n*.07 >= 10000/100.01 and (n-1)*.07 < 10000/100.01
        refused(lambda: fixed_generation_decisions_candidate(2., .05, target_generated=True))
        bindings = FrozenRuntimeArtifactBindings(str(attempt/'MISSING_NONCERTIFICATE_PROTOCOL.json'), '0'*64,
            str(attempt/'MISSING_NONCERTIFICATE_REVIEW.json'), '0'*64)
        refused(lambda: FrozenTestExecutionContract(candidate, bindings,
            attempt/'MISSING_NONWEIGHT_REGISTRATION.json', '0'*64))
        assert 'torch' not in sys.modules
        write_evidence(attempt/'EXPLICIT_ARTIFICIAL_120_DESCRIPTOR_CONTRACT.json', candidate.prepared_record())
        sources = {str(path): file_sha(path) for path in (Path(__file__),
            BASE/'scripts/visual_test_contract.py', BASE/'README_VISUAL_TEST_CONTRACT_中文.md')}
        report = dict(at=datetime.now().astimezone().isoformat(), source_and_descriptor_guards_passed=True,
            original_retained_metadata_images=1000, original_retained_metadata_identity_sha256=pool.identity_sha256,
            original_test_image_ids_or_splits_changed=False,
            retained_timing_CSV_rows_loaded_or_test_outcomes_used_for_selection=0,
            descriptor_matrix_uses_explicit_artificial_schedule_SHA_and_load_values=True,
            descriptor_matrix_not_120_evaluations=result, model_or_queue_or_policy_evaluation_calls=0,
            actual_test_request_schedules_generated=False, actual_test_execution_cells=0,
            source_test_contract_constructor_succeeded=False, test_launch_authorized=False,
            rule_network_load_horizon_or_pilot_frozen=False, torch_imported=False,
            original_protected_files_sha256=protected, new_implementation_sha256=sources,
            independent_entry_review=False, policy_training_cells=0, formal_policy_test_cells=0)
        write_evidence(attempt/'RESULT.json', report)
        assert all(file_sha(Path(path)) == sha for path, sha in (protected | sources).items())
        print('TEST_METADATA_AND_ARTIFICIAL_CONTRACT_CHECK_ONLY_NOT_120_RESULTS', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_test_execution_cells=0,
            model_or_queue_or_policy_evaluation_calls=0))
        raise


if __name__ == '__main__':
    main()
