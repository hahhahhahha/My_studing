"""Historical reward arithmetic/real past-receipt engineering, not neural work."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

from visual_learned_bindings import source_bindings
from visual_learning_buffer import VisualEpisodeBuffer, map_historical_action_output, HistoricalMAPPOBatchFacade
from visual_mean_variance_reward import MeanVarianceRecurrence, finalize_mean_variance_rewards
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger, finalize_training_rewards

BASE = Path(__file__).resolve().parents[1]
PRIOR = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_074310_240972_terminal_gae'
PRIOR_REPORT_SHA = 'f6b3dd387609fa8252d18a9ca9b589da2c2a3ddd9ca8502b550d58867012efe5'
TRACE_DIR = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_071415_395503_controller_loop'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError('MV receipt/range/method boundary was not enforced')


def source_formula_check():
    binding = source_bindings()
    path = next(Path(path) for path in binding if Path(path).name == 'paper_policy_baselines.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MVReward')
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = {'np': np, 'EPS': 1e-8, '__name__': 'isolated_historical_reward_arithmetic'}
    exec(compile(ast.fix_missing_locations(isolated), str(path), 'exec'), namespace)
    original = namespace['MVReward'](1.5)
    adapted = MeanVarianceRecurrence(1.5)
    # Explicit arithmetic fixtures; not evaluation ground truth or model data.
    inputs = [0., .2, .4, 1.7, 0., 2., .001, 0., 0., 1.5, .9] * 5
    rewards = []
    for latency in inputs:
        a, b = original(latency), adapted(latency)
        assert a == b
        assert original.count == adapted.count
        for name in ('mean', 'm2', 'prev_sr'):
            assert math.isclose(float(getattr(original, name)), float(getattr(adapted, name)), abs_tol=1e-12, rel_tol=0.)
        rewards.append(b)
    original.reset()
    adapted.reset()
    assert original(.3) == adapted(.3) == 0.
    refused(lambda: adapted(float('nan')))
    refused(lambda: adapted(-.1))
    assert 'torch' not in sys.modules
    return {'source_sha256': binding, 'isolated_original_reward_class_executed': True,
            'original_module_or_model_imported': False, 'synthetic_scalar_calls': len(inputs),
            'per_call_reward_and_state_match': True, 'source_class_only_not_performance_GT': True}


def prepare_synthetic_behavior_buffer(transitions, semantics):
    n = len(transitions[0]['portfolios'])
    packed = SimpleNamespace(actor=np.zeros((n, 19), dtype=np.float32), critic=np.zeros((n, 705), dtype=np.float32))
    critic_sha = hashlib.sha256(packed.critic.tobytes()).hexdigest()
    buffer = VisualEpisodeBuffer(n, 4, 19, 705, len(transitions), reward_semantics=semantics)
    adjusted = deepcopy(transitions)
    for row in adjusted:
        portfolios = np.asarray(row['portfolios'])
        # Deliberate artificial latent/logp/value fixtures ONLY. They are NOT
        # outputs of PPO, QLA or a learned actor; no model calls are made.
        latent = np.log(portfolios)
        zeros = np.zeros(n)
        sample = map_historical_action_output('PPO-MV-A', (latent, zeros, zeros, portfolios), n, 4)
        buffer.add_action(row['index'], row['time_seconds'], row['end_seconds'], packed, sample)
        row['pre_critic_sha256'] = critic_sha
    return buffer, adjusted


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_mv_reward')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert sha(PRIOR / 'RESULT.json') == PRIOR_REPORT_SHA
        previous = json.loads((PRIOR / 'RESULT.json').read_text(encoding='utf-8'))
        inputs = dict(previous['read_only_inputs_sha256'])
        inputs[str(PRIOR / 'RESULT.json')] = PRIOR_REPORT_SHA
        assert all(sha(Path(path)) == expected for path, expected in inputs.items())
        source_check = source_formula_check()
        schedule = json.loads((TRACE_DIR / 'SHARED_ENGINEERING_SCHEDULE.json').read_text(encoding='utf-8'))
        identities = tuple(IssuedIdentity(row['task_id'], row['user'], row['arrival_seconds']) for row in schedule['requests'])
        source_report = json.loads((TRACE_DIR / 'RESULT.json').read_text(encoding='utf-8'))
        cases = []
        for case in source_report['cases']:
            episode = json.loads((TRACE_DIR / (case['case'] + '_ENGINEERING_TRACE.json')).read_text(encoding='utf-8'))
            before = json.dumps(episode, sort_keys=True)
            end = schedule['decisions'] * schedule['decision_seconds'] + case['fixed_drain_seconds']
            ledger = build_closed_ledger(episode['final'], identities, end, schedule['users'])
            mv = finalize_mean_variance_rewards(episode['transitions'], ledger, identities, schedule['users'])
            counts = [sum(row['reward']['per_user_outcome_counts']) for row in mv]
            assert sum(counts) == ledger.generated == 102
            assert mv[-1]['reward']['shared_MV_count_after'] == 80 * 30
            binary = np.asarray([row['reward']['per_user_rewards'] for row in mv])
            assert binary.shape == (80, 30) and np.all((binary == 0.) | (binary == 1.))
            assert all(row['reward']['team_reward'] is None for row in mv)
            assert all(row['pre_critic_sha256'] == original['pre_critic_sha256']
                       for row, original in zip(mv, episode['transitions']))
            # Actual receipt-derived cost matrix independently replayed through
            # the isolated exact historical scalar recurrence.
            original_path = next(Path(path) for path in source_check['source_sha256'] if Path(path).name == 'paper_policy_baselines.py')
            tree = ast.parse(original_path.read_text(encoding='utf-8'))
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MVReward')
            namespace = {'np': np, 'EPS': 1e-8, '__name__': 'isolated_receipt_reward_reference'}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(original_path), 'exec'), namespace)
            reference = namespace['MVReward'](1.5)
            for row in mv:
                expected = [reference(value) for value in row['reward']['per_user_observed_or_proxy_cost_mean_seconds']]
                assert expected == row['reward']['per_user_rewards']
            refused(lambda: finalize_mean_variance_rewards(mv, ledger, identities, schedule['users']))
            duplicate = deepcopy(episode['transitions'])
            completed_step = next(row for row in duplicate if row['outcome_batch']['completed'])
            completed_step['outcome_batch']['completed'].append(completed_step['outcome_batch']['completed'][0])
            refused(lambda: finalize_mean_variance_rewards(duplicate, ledger, identities, schedule['users']))
            future = deepcopy(episode['transitions'])
            step = next(row for row in future if row['outcome_batch']['completed'])
            step['outcome_batch']['completed'][0]['completed_seconds'] = end + 1.
            refused(lambda: finalize_mean_variance_rewards(future, ledger, identities, schedule['users']))
            ppo_buffer, synthetic_mv = prepare_synthetic_behavior_buffer(mv, 'PPO-MV-A')
            prepared = ppo_buffer.prepare_closed_episode(synthetic_mv)
            assert prepared.reward_semantics == 'PPO-MV-A' and prepared.advantages.shape == (80, 30)
            refused(lambda: HistoricalMAPPOBatchFacade(prepared))
            qla_buffer, wrong = prepare_synthetic_behavior_buffer(mv, 'QLA-shared-receipt')
            refused(lambda: qla_buffer.prepare_closed_episode(wrong))
            qla_closed = finalize_training_rewards(episode['transitions'], ledger)
            ppo_buffer, wrong = prepare_synthetic_behavior_buffer(qla_closed, 'PPO-MV-A')
            refused(lambda: ppo_buffer.prepare_closed_episode(wrong))
            assert json.dumps(episode, sort_keys=True) == before
            output = {'closed_ledger': asdict(ledger), 'MV_adapted_transitions': mv,
                'behavior_buffer_fixture_logp_value_features_are_artificial': True,
                'source': 'Existing validation-derived engineering receipts, not PPO actions or training',
                'raw_metrics_unchanged': True}
            (attempt / (case['case'] + '_MV_ENGINEERING_CHECK.json')).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
            zero_inputs = sum(value == 0. for row in mv for value in row['reward']['per_user_observed_or_proxy_cost_mean_seconds'])
            cases.append({'case': case['case'], 'task_outcomes_accounted_once': sum(counts),
                'completed': ledger.completed, 'dropped': ledger.dropped, 'unfinished': ledger.unfinished,
                'mean_variance_scalar_updates': mv[-1]['reward']['shared_MV_count_after'],
                'empty_or_zero_input_user_calls': zero_inputs,
                'historical_formula_reward_match': True, 'method_range_guards_passed': True,
                'synthetic_behavior_buffer_GAE_shape': list(prepared.advantages.shape),
                'actual_neural_behavior_or_update': False})
        assert all(sha(Path(path)) == expected for path, expected in inputs.items())
        assert 'torch' not in sys.modules
        sources = [Path(__file__), BASE / 'README_VISUAL_MV_REWARD_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_mean_variance_reward.py', 'visual_learning_buffer.py', 'visual_neural_controller.py',
            'visual_terminal_accounting.py', 'visual_temporal_gae.py')]
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'historical_recurrence_check': source_check, 'cases': cases, 'read_only_inputs_sha256': inputs,
            'implementation_sha256': {str(path): sha(path) for path in sources},
            'candidate_mu_seconds': 1.5, 'user_order': 'ascending user ID, one shared historical statistic',
            'past_receipt_input_and_terminal_merge_is_disclosed_visual_adaptation': True,
            'PPO_MV_reward_buffer_adapter_connected_in_engineering': True,
            'actual_model_controller_constructed': False, 'actual_actor_or_critic_forward': False,
            'actual_optimizer_update': False, 'full_neural_collector_connected': False,
            'formal_protocol_frozen': False, 'independent_complete_entry_review': False,
            'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Isolated historical reward arithmetic and existing engineering-receipt adaptation plus explicit synthetic behavior-buffer checks only; no neural policy, performance, formal experiment or independent audit'}
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_MV_REWARD_ENGINEERING_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'at': datetime.now().astimezone().isoformat(),
            'error': repr(error), 'policy_training_cells': 0, 'formal_policy_test_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
