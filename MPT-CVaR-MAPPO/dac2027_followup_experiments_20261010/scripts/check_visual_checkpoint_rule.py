"""Pure artificial descriptor/record fixtures, never models or real weights."""
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

import numpy as np
from visual_episode_collector import BASE, jsonable, write_evidence
from visual_training_coordinator import FrozenRuntimeArtifactBindings
from visual_training_transactions import CommonBudgetCandidate, digest, file_sha, METHODS
from visual_checkpoint_registration import (FinalCheckpointRuleCandidate, final_descriptor_from_closed_budget,
    validate_validation_record, check_complete_paired_grid, RealFinalCheckpointRegistrar)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_121100_687420_model_smoke_entry/RESULT.json'
PRIOR_SHA = '257c8275280b61bb656ca9a56e48cb0262e93b105660a10e52e82b5c80e90cfa'
PLAN = BASE.parent/'paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md'


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Candidate guard accepted an invalid artificial description')


def fixture(method, seed, rule):
    # Every token/hash/statistic below is a literal artificial description.
    # No checkpoint file or optimizer state is created, loaded or implied.
    rows = []
    for index, count in enumerate(rule.budget.episode_decisions, 1):
        rows.append(dict(episode_index=index, data_role='calibration', decisions=count,
            users=rule.budget.users, behavior_samples=count*rule.budget.users,
            action_interval_seconds=count*rule.budget.decision_seconds,
            reward_semantics='QLA-shared-receipt' if method == METHODS[0] else 'PPO-MV-A',
            episode_token=f'EXPLICIT_ARTIFICIAL:{method}:{seed}:{index}',
            schedule_sha256=digest(dict(artificial_seed=seed, segment=index)),
            update_statistics_sha256=digest(dict(EXPLICIT_NONMODEL_STATISTIC=1)),
            checkpoint_sha256=digest(dict(EXPLICIT_NONMODEL_HASH=method, seed=seed, segment=index))))
    return dict(method=method, seed=seed, candidate_budget=jsonable(asdict(rule.budget)),
        phase='budget_complete_pending_validation', failure=None, current_episode=None,
        paired_design_sha256=digest(dict(EXPLICIT_ARTIFICIAL_COMMON_DESIGN=seed)), committed_episodes=rows)


def validation_fixture(descriptor, schedule, manifest, parameter):
    # Adverse zero-completion fixture: keep all seven uploading unfinished
    # plus three dropped. It must not be filtered as a bad checkpoint.
    counts = dict(generated=10, accepted=0, completed=0, dropped=3, unfinished=7,
        inflight_upload=7, latency_seconds=[])
    return dict(reference=dict(data_role='validation', descriptor=descriptor, artifact_sha256=manifest),
        parameter_sha256_before_and_after=parameter, schedule_sha256=schedule, initial_online_H=0.,
        optimizer_update_called=False, training_H_or_MV_statistics_restored=False,
        torch_rng_changed=False, checkpoint_selection_done=False,
        final=dict(schedule_sha256=schedule, tasks=counts, uncapped_latency_seconds=[]))


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_checkpoint_rule')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = prior['protected_input_and_source_sha256'] | prior['source_files_sha256'] | {str(PRIOR): PRIOR_SHA, str(PLAN): file_sha(PLAN)}
        assert all(file_sha(Path(path)) == sha for path, sha in protected.items())
        budget = CommonBudgetCandidate((1000,)*10)
        rule = FinalCheckpointRuleCandidate(budget)
        assert rule.prepared_record()['actual_checkpoint_selection_done'] is False
        assert rule.definition()['checkpoint_episode_index'] == 10
        refused(lambda: FinalCheckpointRuleCandidate(budget, (True, 2, 3, 4, 5)))
        grid, descriptions = [], []
        for method in METHODS:
            for seed in rule.seeds:
                raw = fixture(method, seed, rule)
                descriptor = final_descriptor_from_closed_budget(raw, rule)
                assert descriptor['committed_environment_decisions'] == 10000
                assert descriptor['committed_behavior_samples'] == 300000
                schedule = digest(dict(EXPLICIT_ARTIFICIAL_VALIDATION_SCHEDULE=seed))
                manifest = raw['committed_episodes'][-1]['checkpoint_sha256']
                parameter = digest(dict(EXPLICIT_NONMODEL_PARAMETER_HASH=method, seed=seed))
                record = validation_fixture(descriptor, schedule, manifest, parameter)
                result = validate_validation_record(record, descriptor, manifest, parameter, schedule)
                assert result['completed'] == 0 and result['dropped'] == 3 and result['unfinished'] == 7
                assert result['performance_filter_applied'] is False
                long = deepcopy(record)
                long['final']['tasks'].update(accepted=10, completed=10, dropped=0, unfinished=0,
                    inflight_upload=0, latency_seconds=[1e9]*10)
                long['final']['uncapped_latency_seconds'] = [1e9]*10
                assert validate_validation_record(long, descriptor, manifest, parameter, schedule)['uncapped_latency_seconds'] == [1e9]*10
                for field, value in (('phase', 'failed'), ('failure', {'EXPLICIT_SYNTHETIC': 'partial'}),
                                     ('seed', True), ('current_episode', {'EXPLICIT_SYNTHETIC': 'open'})):
                    bad = deepcopy(raw)
                    bad[field] = value
                    refused(lambda bad=bad: final_descriptor_from_closed_budget(bad, rule))
                for mutate in ('partial_budget', 'reordered', 'wrong_role', 'truncated', 'duplicate_token', 'missing_archive'):
                    bad = deepcopy(raw)
                    if mutate == 'partial_budget': bad['committed_episodes'].pop()
                    elif mutate == 'reordered': bad['committed_episodes'][0]['episode_index'] = 2
                    elif mutate == 'wrong_role': bad['committed_episodes'][0]['data_role'] = 'test'
                    elif mutate == 'truncated': bad['committed_episodes'][0]['behavior_samples'] -= 1
                    elif mutate == 'duplicate_token': bad['committed_episodes'][1]['episode_token'] = bad['committed_episodes'][0]['episode_token']
                    else: bad['committed_episodes'][0]['checkpoint_sha256'] = 'not a hash'
                    refused(lambda bad=bad: final_descriptor_from_closed_budget(bad, rule))
                for key, value in (('optimizer_update_called', True), ('torch_rng_changed', True),
                                   ('parameter_sha256_before_and_after', '0'*64), ('schedule_sha256', '0'*64)):
                    bad = deepcopy(record)
                    bad[key] = value
                    refused(lambda bad=bad: validate_validation_record(bad, descriptor, manifest, parameter, schedule))
                bad = deepcopy(record)
                bad['reference']['data_role'] = 'test'
                refused(lambda: validate_validation_record(bad, descriptor, manifest, parameter, schedule))
                bad = deepcopy(record)
                bad['final']['tasks']['unfinished'] -= 1
                refused(lambda: validate_validation_record(bad, descriptor, manifest, parameter, schedule))
                grid.append(dict(descriptor=descriptor, validation_schedule_sha256=[schedule]))
                descriptions.append(dict(EXPLICIT_ARTIFICIAL_NOT_WEIGHTS=True, method=method, seed=seed,
                    final_episode=10, environment_decisions=10000, user_behavior_samples=300000,
                    adverse_zero_completed_validation=result, long_latency_not_filtered_seconds=1e9))
        shape = check_complete_paired_grid(grid, rule)
        assert shape['descriptor_rows'] == 10 and shape['actual_weights_or_model_execution_proven'] is False
        refused(lambda: check_complete_paired_grid(grid[:-1], rule))
        refused(lambda: check_complete_paired_grid(grid[:-1]+[grid[0]], rule))
        for key in ('paired_design_sha256', 'committed_environment_decisions'):
            bad = deepcopy(grid)
            bad[-1]['descriptor'][key] = '0'*64 if key.endswith('sha256') else 9999
            refused(lambda bad=bad: check_complete_paired_grid(bad, rule))
        bad = deepcopy(grid)
        bad[-1]['validation_schedule_sha256'] = ['0'*64]
        refused(lambda: check_complete_paired_grid(bad, rule))
        missing = FrozenRuntimeArtifactBindings(str(attempt/'MISSING_NONCERTIFICATE_PROTOCOL.json'), '0'*64,
            str(attempt/'MISSING_NONCERTIFICATE_REVIEW.json'), '0'*64)
        refused(lambda: RealFinalCheckpointRegistrar(rule, missing))
        source = BASE/'scripts/visual_checkpoint_registration.py'
        ast.parse(source.read_text(encoding='utf-8'))
        assert 'torch' not in sys.modules
        write_evidence(attempt/'PREPARED_RULE_NOT_FROZEN.json', rule.prepared_record())
        write_evidence(attempt/'EXPLICIT_ARTIFICIAL_DESCRIPTOR_FIXTURES.json', descriptions)
        new_sources = {str(path): file_sha(path) for path in
            (source, Path(__file__), BASE/'README_VISUAL_CHECKPOINT_RULE_中文.md')}
        write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(),
            source_prepared_and_pure_descriptor_guards_passed=True,
            final_checkpoint_rule_pilot_frozen=False, actual_checkpoint_selection_done=False,
            real_registrar_constructor_succeeded_or_register_or_seal_executed=False,
            actual_model_or_optimizer_or_weight_load_or_validation_calls=0, torch_imported=False,
            artificial_descriptor_grid_rows=10, common_final_budget_selected_by_description_only=True,
            adverse_zero_complete_and_long_latency_descriptions_retained=True,
            partial_failed_reordered_role_truncation_pairing_duplicate_and_changed_identity_guards_passed=True,
            original_protected_files_sha256=protected, new_implementation_sha256=new_sources,
            independent_rereview=False, policy_training_cells=0, formal_policy_test_cells=0))
        assert all(file_sha(Path(path)) == sha for path, sha in (protected | new_sources).items())
        print('CHECKPOINT_RULE_PREPARED_PURE_FIXTURES_ONLY_NOT_WEIGHT_SELECTION', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), artificial_fixture_only=True,
            actual_model_or_optimizer_or_weight_load_or_validation_calls=0))
        raise


if __name__ == '__main__':
    main()
