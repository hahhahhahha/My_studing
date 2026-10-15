"""Explicit artificial action/value assembly checks, not neural model smoke."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from fifo_execution import Request
from physical_mobility import PhysicalMobility, TrainingWindow
from replay_measured_fifo import independent_oracle
from visual_episode_collector import (EpisodeAssembly, EpisodeClosureCandidate,
    RealLearningEpisodeCollector, require_actual_cuda_controller, jsonable, write_evidence)
from visual_learning_buffer import map_historical_action_output, PreparedVisualEpisode
from visual_method_features import calibration_scales
from visual_neural_controller import NeuralVisualController
from visual_receipt_environment import VisualReceiptEnvironment
from visual_request_schedule import ScheduledRequest, VisualRequestSchedule
from visual_upload_link import UploadLinkConfig

BASE = Path(__file__).resolve().parents[1]
TRACE_DIR = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_071415_395503_controller_loop'
MV_REPORT = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_084523_355854_mv_reward/RESULT.json'
MV_REPORT_SHA = 'a00b114f8282283aab5e9d89e8fb1870c2740d5f1893eb26ade95db99811ce7c'
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError('Real-controller/assembly/phase guard was not enforced')


def artificial_sample(method, users, index, concentrated=False):
    # Literal interface fixtures ONLY: no distribution, model or learned data.
    latent = np.tile(np.array([20., -20., -20., -20.] if concentrated else [.3, -.4, .1, 0.]), (users, 1))
    ports = np.exp(latent - latent.max(axis=1, keepdims=True))
    ports /= ports.sum(axis=1, keepdims=True)
    logp = np.full(users, -2.)  # NOT an actual probability density/log likelihood
    values = np.arange(users) * .01 + index * .001  # NOT a critic forward output
    if method == 'QLA-MAPPO':
        return map_historical_action_output(method, (ports, latent, logp), users, 4, values)
    return map_historical_action_output(method, (latent, logp, values, ports), users, 4)


def make_environment(schedule, catalog, mobility, edges, capacity=32):
    return VisualReceiptEnvironment(schedule, catalog, mobility, edges, PROFILES, 0., .1, .1,
                                    capacity, UploadLinkConfig())


def static_real_entry_check(scales):
    refused(lambda: require_actual_cuda_controller(object()))
    uninitialized = object.__new__(NeuralVisualController)
    refused(lambda: require_actual_cuda_controller(uninitialized))
    refused(lambda: RealLearningEpisodeCollector(object(), scales))
    tree = ast.parse((BASE / 'scripts/visual_episode_collector.py').read_text(encoding='utf-8'))
    node = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == 'RealLearningEpisodeCollector')
    collect = next(row for row in node.body if isinstance(row, ast.FunctionDef) and row.name == 'collect_training_episode')
    calls = [row for row in ast.walk(collect) if isinstance(row, ast.Call)]
    actual_act = [row for row in calls if isinstance(row.func, ast.Attribute) and row.func.attr == 'act']
    assert len(actual_act) == 1 and len(actual_act[0].args) == 1
    assert isinstance(actual_act[0].args[0], ast.Name) and actual_act[0].args[0].id == 'packed'
    assert not any(isinstance(row.func, ast.Attribute) and row.func.attr in
                   ('update', 'update_closed_episode', '_update_ppo') for row in calls)
    assert not any(isinstance(row.func, ast.Name) and row.func.id == 'build_cuda_model' for row in calls)
    assert 'torch' not in sys.modules
    return {'rejected_fixture_or_uninitialized_controller': True,
            'static_actual_act_argument_is_only_pre_action_packed_features': True,
            'collector_does_not_call_update_or_model_factory': True,
            'actual_real_collector_instance_or_collect_executed': False}


def assembly_guard_check(schedule, catalog, mobility, edges, scales):
    env = make_environment(schedule, catalog, mobility, edges)
    assembly = EpisodeAssembly('QLA-MAPPO', schedule.users, 4, schedule.decisions, scales, EpisodeClosureCandidate())
    refused(lambda: assembly.close_after_fixed_drain(env))
    pre = env.observe()
    packed = assembly.features_for(pre)
    altered = packed.critic.copy()
    altered[0, 0] += .01
    refused(lambda: assembly.execute_validated_sample(env, pre, replace(packed, critic=altered),
                                                       artificial_sample('QLA-MAPPO', schedule.users, pre.index)))
    assert assembly.failed and not assembly.transitions and not assembly.buffer._rows
    refused(lambda: assembly.execute_validated_sample(env, pre, packed,
                                                       artificial_sample('QLA-MAPPO', schedule.users, pre.index)))
    refused(lambda: EpisodeAssembly('PPO-MV-A', schedule.users, 4, schedule.decisions,
                                    scales, EpisodeClosureCandidate(fixed_drain_seconds=.01)))
    return {'premature_closure_rejected': True, 'different_pre_action_critic_rejected': True,
            'failed_assembly_no_append_retry': True, 'too_short_fixed_drain_rejected': True}


def oracle_check(result, schedule, catalog, mobility, edges, capacity, drain):
    requests, services, available = [], [], [0.] * schedule.users
    by_index = {index: [] for index in range(1, schedule.decisions + 1)}
    for row in schedule.requests:
        by_index[schedule.decision_index(row)].append(row)
    link = UploadLinkConfig()
    for transition in result['ordinary_transitions']:
        now, portfolios = transition['time_seconds'], transition['portfolios']
        positions = mobility.positions(now)
        for row in by_index[transition['index']]:
            cumulative, chosen = 0., 3
            for server, probability in enumerate(portfolios[row.user]):
                cumulative += probability
                if row.assignment_uniform < cumulative:
                    chosen = server
                    break
            metadata = catalog.observable_metadata(row.image_id)
            upload = link.transmission_seconds(metadata.payload_bytes, positions[row.user], edges[chosen])
            ready = max(now, available[row.user]) + upload
            available[row.user] = ready
            measured = catalog.measured_pair_seconds(row.image_id, row.measured_session)
            services.append((row.task_id, row.image_id, row.measured_session, measured))
            requests.append(Request(row.task_id, now, ready-now, chosen, measured[PROFILES[chosen]],
                                    metadata.payload_bytes, row.image_id))
    oracle = independent_oracle(requests, 4, capacity)
    end = schedule.horizon_seconds + drain
    expected = {task: row['completed'] for task, row in oracle.items()
                if row['completed'] is not None and row['completed'] <= end}
    expected_drop = {task for task, row in oracle.items() if row['dropped'] and row['ready'] <= end}
    final = result['final']
    actual = {row['task_id']: row['physical_seconds'] for row in final['controller_events_reporting_only']
              if row['kind'] == 'completed'}
    actual_drop = {row['task_id'] for row in final['controller_events_reporting_only'] if row['kind'] == 'dropped'}
    assert set(actual) == set(expected) and actual_drop == expected_drop
    error = max((abs(actual[task]-expected[task]) for task in actual), default=0.)
    assert error < 1e-10
    lookup = {row.task_id: row for row in schedule.requests}
    wanted = sorted(time-lookup[task].arrival_seconds for task, time in expected.items())
    assert np.allclose(sorted(final['uncapped_latency_seconds']), wanted, rtol=0., atol=1e-10)
    assert final['tasks']['unfinished'] == len(schedule.requests)-len(expected)-len(expected_drop)
    mass_error = max(max(abs(row['offered_error']), abs(row['admitted_error'])) for row in final['work'])
    assert mass_error < 1e-10
    service_sha = hashlib.sha256(json.dumps(services, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'max_independent_completion_error_seconds': error, 'max_service_conservation_error_seconds': mass_error,
            'paired_service_witness_sha256': service_sha, 'true_arrival_uncapped_latency_matches_oracle': True}


def fixture_case(name, method, schedule, catalog, mobility, edges, scales, attempt, capacity=32, drain=30., concentrated=False):
    env = make_environment(schedule, catalog, mobility, edges, capacity)
    assembly = EpisodeAssembly(method, schedule.users, 4, schedule.decisions, scales,
                               EpisodeClosureCandidate(fixed_drain_seconds=drain))
    pre_actors, pre_critics, fixture_samples = [], [], []
    for _ in range(schedule.decisions):
        pre = env.observe()
        packed = assembly.features_for(pre)
        sample = artificial_sample(method, schedule.users, pre.index, concentrated)
        pre_actors.append(packed.actor.copy())
        pre_critics.append(packed.critic.copy())
        fixture_samples.append(sample)
        assembly.execute_validated_sample(env, pre, packed, sample)
    result = assembly.close_after_fixed_drain(env)
    prepared = result['prepared_episode']
    assert isinstance(prepared, PreparedVisualEpisode)
    assert prepared.advantages.shape == (80, 30) and prepared.flat()['actor_obs'].shape == (2400, 19)
    assert np.array_equal(prepared.time_user_fields['actor_obs'], np.stack(pre_actors))
    assert np.array_equal(prepared.time_user_fields['critic_obs'], np.stack(pre_critics))
    assert np.array_equal(prepared.time_user_fields['value'], np.stack([sample.values for sample in fixture_samples]))
    assert np.array_equal(prepared.time_user_fields['latent_action'], np.stack([sample.latent_actions for sample in fixture_samples]))
    assert np.all(prepared.time_user_fields['done'][-1]) and not np.any(prepared.time_user_fields['done'][:-1])
    expected_semantics = 'QLA-shared-receipt' if method == 'QLA-MAPPO' else 'PPO-MV-A'
    assert prepared.reward_semantics == expected_semantics and np.isfinite(prepared.advantages).all()
    assert result['ledger'].generated == len(schedule.requests) == 102
    refused(lambda: assembly.close_after_fixed_drain(env))
    refused(env.observe)
    refused(lambda: env.finish(drain+1.))
    oracle = oracle_check(result, schedule, catalog, mobility, edges, capacity, drain)
    path = attempt / (name + '_ARTIFICIAL_ACTION_ENGINEERING_TRACE.json')
    write_evidence(path, {key: value for key, value in result.items() if key != 'prepared_episode'} |
        {'behavior_kind': 'Explicit artificial latent/logp/value; NOT neural outputs or training'})
    np.savez_compressed(attempt / (name + '_ARTIFICIAL_BEHAVIOR_AND_GAE.npz'), **prepared.time_user_fields,
                        advantages=prepared.advantages, returns=prepared.returns)
    return {'case': name, 'reward_adapter_under_test': method, 'artificial_behavior_only': True,
        'trace_file_bytes_sha256': sha(path), 'generated': 102, 'completed': result['ledger'].completed,
        'dropped': result['ledger'].dropped, 'unfinished': result['ledger'].unfinished,
        'complete_behavior_samples': 2400, 'time_user_GAE_shape': [80, 30],
        'own_pre_action_feature_and_behavior_copies_match': True, 'fixed_drain_seconds': drain,
        'all_outcomes_credited_once_before_GAE': True, 'actual_model_calls': 0, **oracle}, result


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_episode_collector')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert sha(MV_REPORT) == MV_REPORT_SHA
        prior = json.loads(MV_REPORT.read_text(encoding='utf-8'))
        protected = dict(prior['read_only_inputs_sha256']) | prior['implementation_sha256'] | {str(MV_REPORT): MV_REPORT_SHA}
        assert all(sha(Path(path)) == expected for path, expected in protected.items())
        raw_schedule = json.loads((TRACE_DIR / 'SHARED_ENGINEERING_SCHEDULE.json').read_text(encoding='utf-8'))
        raw_schedule['requests'] = tuple(ScheduledRequest(**row) for row in raw_schedule['requests'])
        schedule = VisualRequestSchedule(**raw_schedule)
        calibration, validation = ExecutionServiceCatalog('calibration'), ExecutionServiceCatalog('validation')
        mobility = PhysicalMobility(users=30)
        window = TrainingWindow(0., 500.)
        edges = mobility.server_positions(window, 123, 4)
        scales = calibration_scales(calibration, mobility, window, edges, PROFILES)
        entry = static_real_entry_check(scales)
        guards = assembly_guard_check(schedule, validation, mobility, edges, scales)
        qla, qla_result = fixture_case('explicit_balanced_QL_reward', 'QLA-MAPPO', schedule, validation, mobility, edges, scales, attempt)
        ppo, ppo_result = fixture_case('explicit_balanced_MV_reward', 'PPO-MV-A', schedule, validation, mobility, edges, scales, attempt)
        assert qla_result['final'] == ppo_result['final']
        assert all(a['portfolios'] == b['portfolios'] and a['outcome_batch'] == b['outcome_batch']
                   for a, b in zip(qla_result['ordinary_transitions'], ppo_result['ordinary_transitions']))
        assert qla['paired_service_witness_sha256'] == ppo['paired_service_witness_sha256']
        drop, _ = fixture_case('explicit_concentrated_drop_QL_reward', 'QLA-MAPPO', schedule, validation,
            mobility, edges, scales, attempt, capacity=0, concentrated=True)
        unfinished, _ = fixture_case('explicit_concentrated_unfinished_MV_reward', 'PPO-MV-A', schedule,
            validation, mobility, edges, scales, attempt, drain=.05, concentrated=True)
        assert drop['dropped'] > 0 and unfinished['unfinished'] > 0
        assert 'torch' not in sys.modules and all(sha(Path(path)) == expected for path, expected in protected.items())
        sources = [Path(__file__), BASE / 'README_VISUAL_EPISODE_COLLECTOR_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_episode_collector.py', 'visual_neural_controller.py', 'visual_learning_buffer.py',
            'visual_mean_variance_reward.py', 'visual_terminal_accounting.py', 'visual_method_features.py',
            'visual_receipt_environment.py', 'visual_shared_reward.py', 'visual_temporal_gae.py')]
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'static_real_entry_and_rejection_checks': entry, 'assembly_phase_information_guards': guards,
            'cases': [qla, ppo, drop, unfinished], 'schedule_sha256': schedule.fingerprint(),
            'same_artificial_actions_and_exogenous_truth_give_same_raw_results_across_reward_adapters': True,
            'read_only_prior_evidence_sha256': protected,
            'calibration_sources_sha256': calibration.source_sha256, 'validation_sources_sha256': validation.source_sha256,
            'implementation_sha256': {str(path): sha(path) for path in sources},
            'real_learning_collector_source_prepared': True, 'real_learning_collector_constructed_or_executed': False,
            'actual_model_or_critic_forward': False, 'actual_optimizer_update': False,
            'artificial_logp_values_not_probabilities_or_neural_outputs': True,
            'MV_statistic_cross_episode_reset_rule_frozen': False, 'episode_budget_split_frozen': False,
            'test_measurements_accessed': False, 'formal_protocol_frozen': False,
            'independent_complete_entry_review': False, 'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Real validation-measurement FIFO episode assembly with explicitly artificial actions/logp/values only. Not neural rollout, model smoke, training, formal policy comparison, independent certification or 120 cells'}
        write_evidence(attempt / 'RESULT.json', report)
        print('VISUAL_EPISODE_ASSEMBLY_ENGINEERING_OK', attempt)
    except Exception as error:
        write_evidence(attempt / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(), 'error': repr(error),
            'policy_training_cells': 0, 'formal_policy_test_cells': 0})
        raise


if __name__ == '__main__':
    main()
