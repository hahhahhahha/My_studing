"""Candidate terminal closure and GAE self-check; neural values are NOT used."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

from visual_temporal_gae import per_user_temporal_gae, flatten_after_gae
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger, finalize_training_rewards

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_071415_395503_controller_loop'
EXPECTED_REPORT = '0d8b37f4157b90a1d133f0f5e10f90028a19bc1da4e845f937569aed38066c08'
EXPECTED_SCHEDULE = '19d10b656932f2c7ee51f077df978a55273e3f38589d9ae76220ccd0c172ca25'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Temporal or terminal accounting boundary was not enforced')


def gae_fixtures():
    rewards = np.array([[1., 10.], [2., 20.], [3., 30.]])
    zero = np.zeros_like(rewards)
    done = np.zeros_like(rewards, dtype=bool)
    done[-1] = True
    advantage, returns = per_user_temporal_gae(rewards, zero, zero, done, np.ones(3), 1., 1.)
    assert np.array_equal(advantage, [[6., 60.], [5., 50.], [3., 30.]])
    altered = rewards.copy()
    altered[:, 1] *= 1000
    changed, _ = per_user_temporal_gae(altered, zero, zero, done, np.ones(3), 1., 1.)
    assert np.array_equal(changed[:, 0], advantage[:, 0])  # independent users
    inner_done = done.copy()
    inner_done[1, 0] = True
    reset, _ = per_user_temporal_gae(rewards, zero, zero, inner_done, np.ones(3), 1., 1.)
    assert np.array_equal(reset[:, 0], [3., 2., 3.]) and np.array_equal(reset[:, 1], advantage[:, 1])
    following = zero.copy()
    following[-1] = 1000.
    terminated, _ = per_user_temporal_gae(rewards, zero, following, done, np.ones(3), 1., 1.)
    assert np.array_equal(terminated, advantage)
    truncated, _ = per_user_temporal_gae(rewards, zero, following, np.zeros_like(done), np.ones(3), 1., 1.)
    assert np.array_equal(truncated[-1], rewards[-1] + 1000.)
    r, v, nxt = np.array([[1.], [2.]]), np.array([[.5], [.25]]), np.array([[.25], [999.]])
    physical, _ = per_user_temporal_gae(r, v, nxt, np.array([[False], [True]]), [.5, 1.5], .9, .8)
    expected = 1 - .5 + .9**.5 * .25 + .9**.5 * .8**.5 * (2 - .25)
    assert math.isclose(physical[0, 0], expected, rel_tol=0., abs_tol=1e-12)
    assert physical[1, 0] == 1.75
    refused(lambda: per_user_temporal_gae(rewards.reshape(-1), zero, zero, done, np.ones(3)))
    refused(lambda: per_user_temporal_gae(rewards, zero, zero, done.astype(int), np.ones(3)))
    refused(lambda: per_user_temporal_gae(rewards, zero, zero, done, [1., 0., 1.]))
    refused(lambda: per_user_temporal_gae(rewards * np.nan, zero, zero, done, np.ones(3)))
    refused(lambda: per_user_temporal_gae(rewards, zero, zero, done, np.ones(3), 0., 1.))
    fields = {'actor': np.arange(12.).reshape(3, 2, 2)}
    flat = flatten_after_gae(fields, advantage, returns)
    assert flat['actor'].shape == (6, 2) and np.array_equal(flat['advantages'], advantage.reshape(-1))
    refused(lambda: flatten_after_gae({'actor': fields['actor'].reshape(6, 2)}, advantage, returns))
    return {'groups': 7, 'explicit_synthetic_values': True, 'cross_user_isolation': True,
            'episode_reset_and_truncation': True, 'physical_discount_formula': True,
            'flatten_only_after_gae': True, 'invalid_inputs_rejected': True}


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_terminal_gae')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert sha(SOURCE / 'RESULT.json') == EXPECTED_REPORT
        assert sha(SOURCE / 'SHARED_ENGINEERING_SCHEDULE.json') == EXPECTED_SCHEDULE
        source_report = json.loads((SOURCE / 'RESULT.json').read_text(encoding='utf-8'))
        schedule = json.loads((SOURCE / 'SHARED_ENGINEERING_SCHEDULE.json').read_text(encoding='utf-8'))
        assert source_report['passed'] and source_report['data_role'] == schedule['role'] == 'validation'
        inputs = {str(SOURCE / 'RESULT.json'): EXPECTED_REPORT,
                  str(SOURCE / 'SHARED_ENGINEERING_SCHEDULE.json'): EXPECTED_SCHEDULE}
        # The existing source-bound receipt engineering entry has not changed.
        for path, expected in source_report['implementation_sha256'].items():
            assert sha(Path(path)) == expected
        synthetic = gae_fixtures()
        identities = tuple(IssuedIdentity(row['task_id'], row['user'], row['arrival_seconds']) for row in schedule['requests'])
        cases = []
        for case in source_report['cases']:
            path = SOURCE / (case['case'] + '_ENGINEERING_TRACE.json')
            assert sha(path) == case['trace_sha256']
            inputs[str(path)] = sha(path)
            episode = json.loads(path.read_text(encoding='utf-8'))
            before = json.dumps(episode, sort_keys=True)
            end = schedule['decisions'] * schedule['decision_seconds'] + case['fixed_drain_seconds']
            ledger = build_closed_ledger(episode['final'], identities, end, schedule['users'])
            closed = finalize_training_rewards(episode['transitions'], ledger)
            assert closed[:-1] == episode['transitions'][:-1]
            assert closed[-1]['pre_critic_sha256'] == episode['transitions'][-1]['pre_critic_sha256']
            assert closed[-1]['reward']['h_after'] == episode['transitions'][-1]['reward']['h_after']
            assert closed[-1]['reward']['team_reward'] <= closed[-1]['unmodified_action_team_reward']
            assert all(-1 <= row['reward']['team_reward'] <= 0 for row in closed)
            assert all(row.completed_latency_seconds is None and row.elapsed_lower_bound_seconds is not None
                       for row in ledger.tail_outcomes if row.kind == 'unfinished')
            assert ledger.unfinished == case['unfinished'] and ledger.dropped == case['dropped']
            assert math.isclose(ledger.all_completed_latency_sum_seconds,
                    math.fsum(episode['final']['uncapped_latency_seconds']), abs_tol=1e-10, rel_tol=0.)
            refused(lambda: finalize_training_rewards(closed, ledger))
            future = deepcopy(episode['final'])
            future['controller_events_reporting_only'][-1]['physical_seconds'] = end + 1.
            refused(lambda: build_closed_ledger(future, identities, end, schedule['users']))
            duplicate = deepcopy(episode['final'])
            duplicate['controller_events_reporting_only'].append(duplicate['controller_events_reporting_only'][-1])
            refused(lambda: build_closed_ledger(duplicate, identities, end, schedule['users']))
            # This is an interface check using explicitly synthetic all-zero
            # value estimates, NOT a neural critic result or trained rollout.
            rewards = np.asarray([row['reward']['per_user_rewards'] for row in closed])
            zero = np.zeros_like(rewards)
            done = np.zeros_like(rewards, dtype=bool)
            done[-1] = True
            duration = [row['end_seconds'] - row['time_seconds'] for row in closed]
            advantages, returns = per_user_temporal_gae(rewards, zero, zero, done, duration)
            assert rewards.shape == advantages.shape == returns.shape == (80, 30)
            assert np.all(advantages == advantages[:, :1])  # shared rewards and zero value fixtures only
            assert json.dumps(episode, sort_keys=True) == before
            output = {'ledger': asdict(ledger), 'closed_transitions': closed,
                'synthetic_zero_value_GAE_fixture': {'advantages': advantages.tolist(), 'returns': returns.tolist()},
                'scope': 'Provisional terminal accounting and synthetic-value interface check only; no neural learning'}
            (attempt / (case['case'] + '_CLOSURE_CHECK.json')).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
            cases.append({'case': case['case'], 'completed': ledger.completed, 'dropped': ledger.dropped,
                'unfinished': ledger.unfinished, 'tail_completed': sum(x.kind == 'completed' for x in ledger.tail_outcomes),
                'tail_dropped': sum(x.kind == 'dropped' for x in ledger.tail_outcomes),
                'all_generated_tasks_credited_once': True, 'raw_metrics_and_input_trace_unchanged': True,
                'all_completed_latency_sum_seconds': ledger.all_completed_latency_sum_seconds,
                'training_surrogate_tail_penalty_mean_seconds': ledger.tail_training_penalty_mean_seconds,
                'candidate_closed_last_reward': closed[-1]['reward']['team_reward'],
                'GAE_shape': list(advantages.shape), 'values_are_explicit_synthetic_zeros': True})
        assert all(sha(Path(path)) == expected for path, expected in inputs.items())
        assert 'torch' not in sys.modules
        sources = [Path(__file__), BASE / 'README_VISUAL_TERMINAL_GAE_中文.md',
            BASE / 'scripts/visual_temporal_gae.py', BASE / 'scripts/visual_terminal_accounting.py']
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'synthetic_GAE_fixtures': synthetic, 'cases': cases, 'read_only_inputs_sha256': inputs,
            'implementation_sha256': {str(path): sha(path) for path in sources},
            'candidate_settings': {'gamma_per_second': .99, 'lambda_per_second': .95,
                'terminal_weight': 1., 'unfinished_penalty_seconds': 1., 'drop_penalty_seconds': 1.,
                'latency_scale_seconds': 1., 'v': .5, 'migration_weight': .05},
            'candidate_closure_prepared': True, 'formal_terminal_protocol_frozen': False,
            'actual_neural_critic_executed': False, 'model_factory_executed': False,
            'independent_complete_entry_review': False, 'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Read-only reuse of five engineering traces; candidate terminal ledger and explicit synthetic-value GAE only, not model training, fresh service measurement, formal policy comparison or independent audit'}
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_TERMINAL_GAE_ENGINEERING_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'at': datetime.now().astimezone().isoformat(),
            'error': repr(error), 'policy_training_cells': 0, 'formal_policy_test_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
