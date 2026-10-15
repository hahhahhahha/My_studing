"""Real validation-derived engineering checks; no learning/performance claim."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from dataclasses import asdict, fields, replace
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
from predicted_fifo_feedback import RetrievedFeedback
from replay_measured_fifo import independent_oracle
from visual_controller_receipts import AdmissionAcknowledgment, ControllerEvent, ControllerReceiptFeedback
from visual_method_features import calibration_scales, pack_features
from visual_receipt_environment import ControllerDecisionObservation, VisualReceiptEnvironment
from visual_request_schedule import generate_schedule
from visual_rule_policies import JSEQVisualPolicy, ScoreSoftmaxPolicy
from visual_transition_loop import rule_engineering_episode
from visual_upload_link import UploadLinkConfig

BASE = Path(__file__).resolve().parents[1]
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, RuntimeError, TypeError):
        return
    raise AssertionError('Controller information/state guard was not enforced')


def make_environment(schedule, catalog, mobility, edges, capacity=32):
    return VisualReceiptEnvironment(schedule, catalog, mobility, edges, PROFILES,
                                     0., .1, .1, capacity, UploadLinkConfig())


def synthetic_fixtures(pre, packed, environment):
    feedback = ControllerReceiptFeedback(('CPU-1T', 'CPU-4T'), 0, .1, .1)
    feedback.submit_execution(Request('a', 0., .01, 0, .03, 100), .02)
    feedback.submit_execution(Request('b', 0., .01, 0, 100., 100), .06)
    assert feedback.controller_events_since(0) == ((), 0)  # future ready/completion unobservable
    feedback.advance(.01)
    events, cursor = feedback.controller_events_since(0)
    assert {(row.task_id, row.kind) for row in events} == {('a', 'admitted'), ('b', 'dropped')}
    assert all(row.physical_seconds <= .01 for row in events)
    assert {field.name for field in fields(ControllerEvent)} == {
        'kind', 'task_id', 'server', 'physical_seconds', 'predicted_service_seconds'}
    feedback.advance(.04)
    completed, cursor = feedback.controller_events_since(cursor)
    assert len(completed) == 1 and completed[0].kind == 'completed'
    assert math.isclose(completed[0].physical_seconds, .04, abs_tol=1e-15)
    assert feedback.controller_events_since(cursor) == ((), cursor)
    refused(lambda: feedback.controller_events_since(-1))
    refused(lambda: feedback.controller_events_since(cursor + 1))
    # A measured duration affects events only when an observed status changes;
    # admission acknowledgment carries the prediction, not actual requirement.
    alternate = ControllerReceiptFeedback(('CPU-1T', 'CPU-4T'), 0, .1, .1)
    alternate.submit_execution(Request('a', 0., .01, 0, 10., 100), .02)
    alternate.submit_execution(Request('b', 0., .01, 0, 1., 100), .06)
    alternate.advance(.01)
    assert alternate.controller_events_since(0)[0] == events
    # Explicit synthetic timestamp boundary: snapshot already includes the
    # admission exactly at its source time. Count only acknowledgments >source.
    source = RetrievedFeedback(.3, .2, False, (.4, 0., 0., 0.), (0,) * 4, (True, False, False, False))
    at_source = AdmissionAcknowledgment('source', 0, 0, .3, 9.)
    after_source = AdmissionAcknowledgment('later', 0, 0, .4, .2)
    view = replace(pre, physical_seconds=.5, feedback=source,
                   admission_acknowledgments=(at_source, after_source))
    jseq = JSEQVisualPolicy()
    estimate = jseq.estimated_backlog(view)
    assert np.allclose(estimate, (.4, 0., 0., 0.), atol=1e-12, rtol=0.)
    future = replace(after_source, admitted_seconds=.6)
    refused(lambda: jseq.estimated_backlog(replace(view, admission_acknowledgments=(future,))))
    refused(lambda: jseq.estimated_backlog(replace(view, admission_acknowledgments=(after_source, after_source))))
    refused(lambda: jseq.portfolios(environment, packed))
    refused(lambda: ScoreSoftmaxPolicy().portfolios(environment, packed))
    # Newly available historical acknowledgments do not leak into the current
    # critic or change the matched learning inputs.
    current = replace(pre, admission_acknowledgments=(AdmissionAcknowledgment('past', 0, 0, 0., .02),))
    assert np.array_equal(pack_features(current, environment._engineering_scales).critic, packed.critic)
    return 8


def replay_case(name, policy, schedule, catalog, mobility, edges, scales, attempt, capacity=32, drain_seconds=30.):
    env = make_environment(schedule, catalog, mobility, edges, capacity)
    result = rule_engineering_episode(env, scales, policy, drain_seconds)
    refused(env.observe)
    refused(env.settle_action)
    refused(lambda: env.finish(31.))
    requests, truth_rows = [], []
    available = [0.] * schedule.users
    link = UploadLinkConfig()
    by_index = {index: [] for index in range(1, schedule.decisions + 1)}
    for row in schedule.requests:
        by_index[schedule.decision_index(row)].append(row)
    for transition in result['transitions']:
        now, portfolios = transition['time_seconds'], transition['portfolios']
        positions = mobility.positions(now)
        for row in by_index[transition['index']]:
            cumulative, server = 0., 3
            for j, probability in enumerate(portfolios[row.user]):
                cumulative += probability
                if row.assignment_uniform < cumulative:
                    server = j
                    break
            metadata = catalog.observable_metadata(row.image_id)
            upload = link.transmission_seconds(metadata.payload_bytes, positions[row.user], edges[server])
            ready = max(now, available[row.user]) + upload
            available[row.user] = ready
            measured = catalog.measured_pair_seconds(row.image_id, row.measured_session)
            truth_rows.append((row.task_id, row.image_id, row.measured_session, measured))
            requests.append(Request(row.task_id, now, ready - now, server,
                        measured[PROFILES[server]], metadata.payload_bytes, row.image_id))
    oracle = independent_oracle(requests, 4, capacity)
    final = result['final']
    end = schedule.horizon_seconds + drain_seconds
    issued = {row.task_id: row for row in schedule.requests}
    expected = {task: row['completed'] for task, row in oracle.items()
                if row['completed'] is not None and row['completed'] <= end}
    actual = {row.task_id: row.physical_seconds for row in final['controller_events_reporting_only']
              if row.kind == 'completed'}
    assert set(actual) == set(expected)
    completion_error = max((abs(actual[task] - expected[task]) for task in actual), default=0.)
    assert completion_error < 1e-10
    completed_receipts = [row for transition in result['transitions']
                         for row in transition['outcome_batch']['completed']]
    assert len({row['task_id'] for row in completed_receipts}) == len(completed_receipts)
    max_receipt_error = max((abs(row['completed_seconds'] - expected[row['task_id']])
                            for row in completed_receipts), default=0.)
    assert max_receipt_error < 1e-10
    expected_drops = {task for task, row in oracle.items() if row['dropped'] and row['ready'] <= end}
    actual_drops = {row.task_id for row in final['controller_events_reporting_only'] if row.kind == 'dropped'}
    assert actual_drops == expected_drops
    assert final['tasks']['unfinished'] == len(schedule.requests) - len(expected) - len(expected_drops)
    dropped_receipts = [row for transition in result['transitions'] for row in transition['outcome_batch']['dropped']]
    assert len({row[0] for row in dropped_receipts}) == len(dropped_receipts)
    assert all(task in expected_drops and math.isclose(time, oracle[task]['ready'], abs_tol=1e-10, rel_tol=0.)
               for task, time in dropped_receipts)
    wanted_latencies = sorted(time - issued[task].arrival_seconds for task, time in expected.items())
    assert np.allclose(sorted(final['uncapped_latency_seconds']), wanted_latencies, rtol=0., atol=1e-10)
    max_mass_error = max(max(abs(row['offered_error']), abs(row['admitted_error'])) for row in final['work'])
    assert max_mass_error < 1e-10
    assert any(view['admission_acknowledgments'] for view in result['observations'])
    for view in result['observations']:
        assert all(row['admitted_seconds'] <= view['physical_seconds'] for row in view['admission_acknowledgments'])
    serialized = json.dumps(result, ensure_ascii=False, default=lambda value: asdict(value), indent=2)
    (attempt / (name + '_ENGINEERING_TRACE.json')).write_text(serialized, encoding='utf-8')
    trace_sha = hashlib.sha256(serialized.encode()).hexdigest()
    return {'engineering_adapter': policy.name, 'case': name, 'schedule_sha256': schedule.fingerprint(),
            'paired_service_witness_sha256': digest(truth_rows), 'trace_sha256': trace_sha,
            'generated': final['tasks']['generated'], 'completed': final['tasks']['completed'],
            'dropped': final['tasks']['dropped'], 'unfinished': final['tasks']['unfinished'],
            'waiting_capacity': capacity, 'fixed_drain_seconds': drain_seconds,
            'real_completed_receipts_rewarded': len(completed_receipts),
            'real_drop_receipts_rewarded': len(dropped_receipts),
            'controller_accounting': final['controller_accounting'],
            'max_independent_completion_error_seconds': completion_error,
            'max_completed_receipt_error_seconds': max_receipt_error,
            'max_service_conservation_error_seconds': max_mass_error}, result


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_controller_loop')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        calibration, validation = ExecutionServiceCatalog('calibration'), ExecutionServiceCatalog('validation')
        mobility = PhysicalMobility(users=30)
        window = TrainingWindow(0., 500.)
        edges = mobility.server_positions(window, 123, 4)
        scales = calibration_scales(calibration, mobility, window, edges, PROFILES)
        schedule = generate_schedule(validation, 30, 80, .05, 123, 30.)
        (attempt / 'SHARED_ENGINEERING_SCHEDULE.json').write_text(json.dumps(schedule.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        guard_env = make_environment(schedule, validation, mobility, edges)
        guard_env._engineering_scales = scales  # fixture helper only, never exported
        pre = guard_env.observe()
        assert isinstance(pre, ControllerDecisionObservation) and not pre.admission_acknowledgments
        packed = pack_features(pre, scales)
        synthetic_groups = synthetic_fixtures(pre, packed, guard_env)
        uniform = ((.25,) * 4,) * 30
        refused(guard_env.settle_action)
        guard_env.apply(uniform)
        refused(guard_env.observe)
        refused(lambda: guard_env.apply(uniform))
        guard_env.settle_action()
        refused(guard_env.settle_action)
        assert guard_env.observe().index == 2
        first, first_trace = replay_case('score_softmax', ScoreSoftmaxPolicy(), schedule, validation, mobility, edges, scales, attempt)
        repeat, repeat_trace = replay_case('score_softmax_repeat', ScoreSoftmaxPolicy(), schedule, validation, mobility, edges, scales, attempt)
        jseq, _ = replay_case('jseq_adapted', JSEQVisualPolicy(), schedule, validation, mobility, edges, scales, attempt)
        class ConcentratedRoutingFixture:
            name = 'EXPLICIT concentrated routing fixture, not a baseline'
            def portfolios(self, observation, packed):
                return np.tile((1., 0., 0., 0.), (len(observation.users), 1))
        drops, _ = replay_case('zero_waiting_buffer_drop_fixture', ConcentratedRoutingFixture(),
                        schedule, validation, mobility, edges, scales, attempt, capacity=0)
        unfinished, _ = replay_case('short_fixed_drain_unfinished_fixture', ConcentratedRoutingFixture(),
                        schedule, validation, mobility, edges, scales, attempt, drain_seconds=.05)
        assert drops['dropped'] > 0 and drops['real_drop_receipts_rewarded'] > 0
        assert unfinished['unfinished'] > 0  # reported, never relabeled completed/drop or hidden
        assert first_trace == repeat_trace
        assert first['paired_service_witness_sha256'] == jseq['paired_service_witness_sha256']
        assert 'torch' not in sys.modules
        sources = [Path(__file__), BASE / 'README_VISUAL_CONTROLLER_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_controller_receipts.py', 'visual_receipt_environment.py', 'visual_rule_policies.py',
            'visual_transition_loop.py', 'visual_environment_adapter.py', 'visual_shared_reward.py',
            'visual_method_features.py', 'predicted_fifo_feedback.py', 'fifo_execution.py',
            'visual_upload_link.py', 'visual_request_schedule.py', 'replay_measured_fifo.py')]
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'synthetic_receipt_information_groups': synthetic_groups, 'strict_phase_guard_groups': 5,
            'cases': [first, repeat, jseq, drops, unfinished], 'common_task_rows': len(schedule.requests),
            'physical_request_horizon_seconds': 4., 'last_action_interval_end_seconds': 4.05,
            'fixed_drain_end_seconds': 34., 'data_role': 'validation',
            'raw_measurement_sources': validation.source_sha256, 'calibration_sources': calibration.source_sha256,
            'implementation_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'real_receipt_reward_loop_connected': True, 'rule_adapters_executed': ['Score-softmax', 'JSEQ-A'],
            'actual_learning_methods_connected': False, 'model_factory_executed': False,
            'terminal_unfinished_training_rule_frozen': False, 'formal_protocol_frozen': False,
            'independent_complete_entry_review': False, 'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Two rule adapters, deterministic repeat and two explicitly labeled drop/unfinished routing fixtures in a four-second engineering schedule, NOT formal policy comparison, 120 cells, QLA/PPO outcome or independent audit'}
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_CONTROLLER_ENGINEERING_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'at': datetime.now().astimezone().isoformat(),
            'error': repr(error), 'policy_training_cells': 0, 'formal_policy_test_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
