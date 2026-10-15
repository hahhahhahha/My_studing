"""Read-only real receipt arithmetic and explicit artificial assembly checks."""
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
import sys
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility, TrainingWindow
from visual_episode_collector import EpisodeClosureCandidate, write_evidence
from visual_learned_bindings import source_bindings
from visual_learning_buffer import map_historical_action_output
from visual_method_features import calibration_scales
from visual_mv_stream import PersistentMVStream
from visual_receipt_environment import VisualReceiptEnvironment
from visual_request_schedule import generate_schedule
from visual_streaming_episode import StreamingEpisodeAssembly, RealStreamingEpisodeCollector
from visual_terminal_accounting import IssuedIdentity, build_closed_ledger

BASE = Path(__file__).resolve().parents[1]
PRIOR = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_094233_888737_feasibility/RESULT.json'
PRIOR_SHA = '14fa17059cfc73a6250dfabca3e982293555d649ea21aa5dd2062b6b1065d4e5'
TRACES = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_071415_395503_controller_loop'
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError('Persistent statistic/order/role/failure guard was not enforced')


def historical_reward_class():
    binding = source_bindings()
    path = next(Path(path) for path in binding if Path(path).name == 'paper_policy_baselines.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == 'MVReward')
    namespace = {'np': np, 'EPS': 1e-8, '__name__': 'isolated_historical_MV_arithmetic'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['MVReward'], binding


def verify_against_original(closed, original):
    for row in closed:
        reward = row['reward']
        expected = [original(value) for value in reward['per_user_observed_or_proxy_cost_mean_seconds']]
        assert expected == reward['per_user_rewards']
        assert original.count == reward['shared_MV_count_after']
        for attr, field in (('mean', 'shared_MV_mean_after'), ('m2', 'shared_MV_m2_after'), ('prev_sr', 'shared_MV_ratio_after')):
            assert math.isclose(float(getattr(original, attr)), reward[field], abs_tol=1e-12, rel_tol=0.)


def read_only_receipt_stream(attempt):
    schedule = json.loads((TRACES / 'SHARED_ENGINEERING_SCHEDULE.json').read_text(encoding='utf-8'))
    identities = tuple(IssuedIdentity(row['task_id'], row['user'], row['arrival_seconds']) for row in schedule['requests'])
    cases = ('score_softmax', 'jseq_adapted', 'zero_waiting_buffer_drop_fixture', 'short_fixed_drain_unfinished_fixture')
    original_class, binding = historical_reward_class()
    original = original_class(1.5)
    stream = PersistentMVStream(30, purpose='engineering_validation_receipts')
    rows = []
    for index, case in enumerate(cases, 1):
        raw = json.loads((TRACES / (case+'_ENGINEERING_TRACE.json')).read_text(encoding='utf-8'))
        unchanged = json.dumps(raw, sort_keys=True)
        end = 4.05 if case.startswith('short_fixed') else 34.
        ledger = build_closed_ledger(raw['final'], identities, end, 30)
        result = stream.close_episode(index, 'validation-diag-'+case, 'validation', raw['transitions'], ledger, identities)
        assert result['state_before'].count == (index-1)*2400 and result['state_after'].count == index*2400
        verify_against_original(result['closed_transitions'], original)
        assert json.dumps(raw, sort_keys=True) == unchanged
        before_rejection = stream.state()
        refused(lambda: stream.close_episode(index, 'validation-diag-'+case, 'validation', raw['transitions'], ledger, identities))
        refused(lambda: stream.close_episode(index+2, 'out-of-order', 'validation', raw['transitions'], ledger, identities))
        refused(lambda: stream.close_episode(index+1, 'wrong-role', 'calibration', raw['transitions'], ledger, identities))
        assert stream.state() == before_rejection and not stream.failed
        write_evidence(attempt/(case+'_READ_ONLY_PERSISTENT_MV_CHECK.json'), result)
        rows.append({'source_case': case, 'generated_outcomes': ledger.generated, 'completed': ledger.completed,
            'dropped': ledger.dropped, 'unfinished': ledger.unfinished,
            'state_count_before': result['state_before'].count, 'state_count_after': result['state_after'].count,
            'exact_historical_reward_and_per_step_state_match': True, 'raw_unchanged': True})
    # Explicit scalar zero-input convention: no neural signal or task outcome.
    zeros = original_class(1.5)
    assert zeros(0.) == 0. and zeros(0.) == 1. and zeros.count == 2
    failed = PersistentMVStream(30, purpose='engineering_validation_receipts')
    bad = deepcopy(raw['transitions'])
    receipt_step = next(row for row in bad if row['outcome_batch']['completed'])
    receipt_step['outcome_batch']['completed'][0]['completed_seconds'] = float('nan')
    refused(lambda: failed.close_episode(1, 'explicit-invalid-receipt-fixture', 'validation', bad, ledger, identities))
    assert failed.failed and failed.failure_evidence is not None
    refused(lambda: failed.close_episode(1, 'new-key-is-not-a-retry-license', 'validation', raw['transitions'], ledger, identities))
    write_evidence(attempt/'EXPLICIT_FAILURE_GUARD_FIXTURE.json', failed.failure_evidence)
    return {'cases': rows, 'total_real_receipt_arithmetic_calls': original.count,
            'state_count_sequence': [0, 2400, 4800, 7200, 9600],
            'original_source_sha256': binding, 'order_duplicate_role_and_failure_guards_passed': True}


def calibration_literal_assembly(attempt):
    catalog = ExecutionServiceCatalog('calibration')
    mobility = PhysicalMobility(users=30)
    window = TrainingWindow(0., 500.)
    edges = mobility.server_positions(window, 123, 4)
    scales = calibration_scales(catalog, mobility, window, edges, PROFILES)
    stream = PersistentMVStream(30, purpose='training')
    original_class, _ = historical_reward_class()
    original = original_class(1.5)
    summaries = []
    for episode in (1, 2):
        schedule = generate_schedule(catalog, 30, 16, .05, 120+episode, 30.)
        env = VisualReceiptEnvironment(schedule, catalog, mobility, edges, PROFILES, float(episode-1))
        assembly = StreamingEpisodeAssembly('PPO-MV-A', 30, 4, 16, scales, EpisodeClosureCandidate(),
                    stream, episode, 'literal-calibration-assembly-'+str(episode))
        for _ in range(16):
            pre = env.observe()
            packed = assembly.features_for(pre)
            latent = np.tile(np.asarray([.3, -.4, .1, 0.]), (30, 1))
            ports = np.exp(latent-latent.max(axis=1, keepdims=True))
            ports /= ports.sum(axis=1, keepdims=True)
            # Explicit literal interface fixtures; not an actor/distribution.
            logp = np.full(30, -2.)
            values = np.arange(30)*.01+pre.index*.001
            sample = map_historical_action_output('PPO-MV-A', (latent, logp, values, ports), 30, 4)
            assembly.execute_validated_sample(env, pre, packed, sample)
        result = assembly.close_after_fixed_drain(env)
        prepared = result['prepared_episode']
        verify_against_original(result['closed_transitions'], original)
        assert prepared.advantages.shape == (16, 30) and prepared.reward_semantics == 'PPO-MV-A'
        assert result['MV_stream']['state_before'].count == (episode-1)*480
        assert result['MV_stream']['state_after'].count == episode*480
        assert np.array_equal(prepared.time_user_fields['reward'],
                              np.asarray([row['reward']['per_user_rewards'] for row in result['closed_transitions']]))
        assert np.isfinite(prepared.advantages).all()
        write_evidence(attempt/f'calibration_literal_episode_{episode}_NOT_NEURAL_TRACE.json',
            {key: value for key, value in result.items() if key != 'prepared_episode'} |
            {'actions_logp_value_are_explicit_artificial_not_neural': True})
        summaries.append({'episode': episode, 'generated': result['ledger'].generated,
            'completed': result['ledger'].completed, 'dropped': result['ledger'].dropped,
            'unfinished': result['ledger'].unfinished, 'count_before': (episode-1)*480, 'count_after': episode*480,
            'per_user_GAE_shape': [16, 30], 'actual_model_or_optimizer_calls': 0})
    refusal_state = stream.state()
    refused(lambda: stream.close_episode(3, 'cannot-use-validation', 'validation', [], None, ()))
    assert stream.state() == refusal_state
    refused(lambda: RealStreamingEpisodeCollector(object(), scales))
    # Static source check only; this is not an actual collector construction.
    tree = ast.parse((BASE/'scripts/visual_streaming_episode.py').read_text(encoding='utf-8'))
    cls = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == 'RealStreamingEpisodeCollector')
    collect = next(row for row in cls.body if isinstance(row, ast.FunctionDef) and row.name == 'collect_training_episode')
    calls = [row for row in ast.walk(collect) if isinstance(row, ast.Call)]
    act = [row for row in calls if isinstance(row.func, ast.Attribute) and row.func.attr == 'act']
    assert len(act) == 1 and len(act[0].args) == 1 and isinstance(act[0].args[0], ast.Name) and act[0].args[0].id == 'packed'
    assert not any(isinstance(row.func, ast.Attribute) and row.func.attr in ('update', 'update_closed_episode') for row in calls)
    return {'literal_calibration_episodes': summaries, 'stream_purpose_training_but_no_learner_executed': True,
            'continuous_MV_reward_used_before_GAE_in_assembly': True,
            'real_collector_source_prepared_not_constructed_or_executed': True,
            'calibration_sources_sha256': catalog.source_sha256, 'mobility_sources_sha256': mobility.source_hashes}


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_mv_stream')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert sha(PRIOR) == PRIOR_SHA
        previous = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = dict(previous['protected_input_and_source_sha256']) | {str(PRIOR): PRIOR_SHA}
        assert all(sha(Path(path)) == value for path, value in protected.items())
        arithmetic = read_only_receipt_stream(attempt)
        integration = calibration_literal_assembly(attempt)
        evidence = protected | integration['calibration_sources_sha256'] | integration['mobility_sources_sha256']
        assert all(sha(Path(path)) == value for path, value in evidence.items()) and 'torch' not in sys.modules
        sources = [Path(__file__), BASE/'README_VISUAL_MV_STREAM_中文.md', BASE/'scripts/visual_mv_stream.py',
                   BASE/'scripts/visual_streaming_episode.py']
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'read_only_receipt_arithmetic': arithmetic, 'literal_calibration_assembly': integration,
            'protected_input_and_source_sha256': evidence, 'new_implementation_sha256': {str(path): sha(path) for path in sources},
            'persistent_MV_stream_prepared_and_self_checked': True, 'old_reset_candidate_sources_unchanged': True,
            'persistent_state_same_as_historical_formula_not_exact_old_slot_input': True,
            'real_learning_collector_constructed_or_executed': False, 'actual_optimizer_update': False,
            'network_parameters_changed': False, 'test_measurements_accessed': False,
            'policy_training_cells': 0, 'formal_policy_test_cells': 0, 'formal_protocol_frozen': False,
            'independent_complete_entry_review': False,
            'scope': 'Persistent historical MV arithmetic and explicit artificial calibration assembly only; no learned behavior, model smoke, training or independent full-entry certification'}
        write_evidence(attempt/'RESULT.json', report)
        print('PERSISTENT_MV_STREAM_ENGINEERING_OK', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', {'at': datetime.now().astimezone().isoformat(), 'error': repr(error),
            'policy_training_cells': 0, 'formal_policy_test_cells': 0})
        raise


if __name__ == '__main__':
    main()
