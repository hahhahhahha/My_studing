"""Engineering integration fixtures, not learned-policy evaluation."""
from __future__ import annotations

from windows_runtime import initialize
initialize()

from dataclasses import asdict, fields, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from execution_service_catalog import ExecutionServiceCatalog
from fifo_execution import Request
from physical_mobility import PhysicalMobility, TrainingWindow
from replay_measured_fifo import independent_oracle
from visual_environment_adapter import VisualEnvironmentAdapter, ObservableRequest
from visual_request_schedule import generate_schedule
from visual_upload_link import UploadLinkConfig, SerializedUserUploads

BASE = Path(__file__).resolve().parents[1]
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def refused(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('State boundary was not enforced')


def make_environment(schedule, catalog, mobility, edges):
    return VisualEnvironmentAdapter(schedule, catalog, mobility, edges, PROFILES,
                                    mobility.common_start_seconds, .1, .1, 32, UploadLinkConfig())


def fixtures(schedule, catalog, mobility, edges):
    link = UploadLinkConfig()
    time = link.transmission_seconds(1000, (0., 0.), (100., 0.))
    assert math.isclose(2 * time, link.transmission_seconds(2000, (0., 0.), (100., 0.)), abs_tol=1e-15)
    assert time < link.transmission_seconds(1000, (0., 0.), (200., 0.))
    gain = 4.11 * (3e8 / (4 * math.pi * 915e6 * 100)) ** 2
    analytic = 2e6 * math.log2(1 + .25 * gain / 1e-10)
    assert math.isclose(link.bits_per_second((0., 0.), (100., 0.)), analytic, rel_tol=1e-12)
    uploader = SerializedUserUploads(2)
    assert uploader.enqueue(0, 0., .02) == .02
    assert uploader.enqueue(0, 0., .03) == .05
    assert uploader.enqueue(1, 0., .03) == .03
    changed = replace(schedule, requests=tuple(replace(row,
            measured_session=row.measured_session % 3 + 1,
            assignment_uniform=(row.assignment_uniform + .5) % 1) for row in schedule.requests))
    first = make_environment(schedule, catalog, mobility, edges)
    alternate = make_environment(changed, catalog, mobility, edges)
    view = first.observe()
    assert first.observe() is view
    assert alternate.observe() == view  # execution jitter/session/uniform absent from current view
    assert {field.name for field in fields(ObservableRequest)} == {
            'task_id', 'arrival_seconds', 'metadata', 'predicted_service_seconds', 'transmission_seconds'}
    uniform = ((.25,) * 4,) * 30
    refused(lambda: first.apply(((0.,) * 4,) * 30))
    assert first.observe() is view  # invalid input cannot expose the next batch
    first.apply(uniform)
    refused(lambda: first.apply(uniform))
    # Injected engineering fault: preserve a failed instance; never advance or
    # reapply a possibly partly processed batch.
    failing = make_environment(schedule, catalog, mobility, edges)
    while True:
        candidate = failing.observe()
        if any(user.requests for user in candidate.users):
            break
        failing.apply(uniform)
    with patch.object(catalog, 'measured_pair_seconds', side_effect=RuntimeError('synthetic injected fault')):
        refused(lambda: failing.apply(uniform))
    refused(failing.observe)
    refused(lambda: failing.apply(uniform))
    refused(lambda: failing.finish(30.))
    observation_failure = make_environment(schedule, catalog, mobility, edges)
    while not observation_failure._batches[observation_failure._index + 1]:
        observation_failure.observe()
        observation_failure.apply(uniform)
    with patch.object(observation_failure._predictor, 'predict_seconds',
                      side_effect=RuntimeError('synthetic observation failure')):
        refused(observation_failure.observe)
    refused(observation_failure.observe)
    refused(lambda: observation_failure.apply(uniform))
    refused(lambda: observation_failure.finish(30.))
    return 9


def replay(name, schedule, catalog, mobility, edges, probabilities):
    env = make_environment(schedule, catalog, mobility, edges)
    schedule_before = schedule.fingerprint()
    request_witness, observation_rows = [], []
    independent_upload_available = [0.] * schedule.users
    link = UploadLinkConfig()
    truth_witness = []
    for index in range(1, schedule.decisions + 1):
        view = env.observe()
        now = schedule.decision_time(index)
        assert view.index == index and view.physical_seconds == now
        assert all(request.arrival_seconds <= now for user in view.users for request in user.requests)
        observation_rows.append(asdict(view))
        locations = mobility.positions(mobility.common_start_seconds + now)
        for row in (r for r in schedule.requests if schedule.decision_index(r) == index):
            cumulative, destination = 0., 3
            for server, value in enumerate(probabilities[row.user]):
                cumulative += value
                if row.assignment_uniform < cumulative:
                    destination = server
                    break
            metadata = catalog.observable_metadata(row.image_id)
            duration = link.transmission_seconds(metadata.payload_bytes, locations[row.user], edges[destination])
            ready = max(now, independent_upload_available[row.user]) + duration
            independent_upload_available[row.user] = ready
            pair = catalog.measured_pair_seconds(row.image_id, row.measured_session)
            truth_witness.append((row.task_id, row.image_id, row.measured_session, pair))
            request_witness.append(Request(row.task_id, now, ready - now, destination,
                      pair[PROFILES[destination]], metadata.payload_bytes, row.image_id))
        env.apply(probabilities)
    result = env.finish(30.)  # fixed engineering drain, not a final protocol choice
    refused(lambda: env.finish(31.))
    refused(env.observe)
    assert result['schedule_sha256'] == schedule_before == schedule.fingerprint()
    expected = independent_oracle(request_witness, 4, 32)
    completed = env._feedback.execution_completion_times()  # privileged test witness only
    end = schedule.horizon_seconds + 30.
    expected_finished = {task: row['completed'] for task, row in expected.items()
                         if row['completed'] is not None and row['completed'] <= end}
    actual_finished = {task: time for task, time in completed.items() if time is not None}
    assert set(actual_finished) == set(expected_finished)
    max_error = max((abs(actual_finished[task] - expected_finished[task])
                     for task in actual_finished), default=0.)
    assert max_error < 1e-10
    original = {row.task_id: row for row in schedule.requests}
    wanted_latency = sorted(time - original[task].arrival_seconds for task, time in expected_finished.items())
    got_latency = sorted(result['uncapped_latency_seconds'])
    assert len(got_latency) == len(wanted_latency)
    max_latency_error = max((abs(a - b) for a, b in zip(got_latency, wanted_latency)), default=0.)
    assert max_latency_error < 1e-10
    dropped = sum(row['dropped'] and row['ready'] <= end for row in expected.values())
    unfinished = len(schedule.requests) - len(expected_finished) - dropped
    assert result['tasks']['generated'] == len(schedule.requests)
    assert result['tasks']['dropped'] == dropped and result['tasks']['unfinished'] == unfinished
    assert all(0 <= wait <= schedule.decision_seconds + 1e-12 for wait in result['decision_wait_seconds'])
    mass_error = max(max(abs(row['offered_error']), abs(row['admitted_error'])) for row in result['work'])
    assert mass_error < 1e-10
    return {'fixture': name, 'schedule_sha256': schedule_before,
            'paired_service_witness_sha256': sha(truth_witness),
            'observable_trace_sha256': sha(observation_rows),
            'generated': result['tasks']['generated'], 'completed': len(expected_finished),
            'dropped': dropped, 'unfinished': unfinished,
            'max_completion_error_seconds': max_error, 'max_true_latency_error_seconds': max_latency_error,
            'max_service_conservation_error_seconds': mass_error}, observation_rows


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_schedule_link')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        catalog = ExecutionServiceCatalog('validation')
        mobility = PhysicalMobility(users=30)
        edges = mobility.server_positions(TrainingWindow(0., 500.), 123, 4)
        schedule = generate_schedule(catalog, 30, 80, .05, 123, 30.)
        regenerated = generate_schedule(catalog, 30, 80, .05, 123, 30.)
        assert regenerated.fingerprint() == schedule.fingerprint()
        (attempt / 'SHARED_ENGINEERING_SCHEDULE.json').write_text(
            json.dumps(schedule.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        synthetic_groups = fixtures(schedule, catalog, mobility, edges)
        uniform = ((.25,) * 4,) * 30
        concentrated = ((1., 0., 0., 0.),) * 30
        first, observations = replay('uniform-routing fixture', schedule, catalog, mobility, edges, uniform)
        repeated, repeated_observations = replay('same-input deterministic repeat', schedule, catalog, mobility, edges, uniform)
        changed, _ = replay('concentrated-routing fixture', schedule, catalog, mobility, edges, concentrated)
        assert observations == repeated_observations
        assert first['paired_service_witness_sha256'] == changed['paired_service_witness_sha256']
        assert first['schedule_sha256'] == changed['schedule_sha256']
        (attempt / 'OBSERVABLE_ENGINEERING_TRACE.json').write_text(
            json.dumps(observations, ensure_ascii=False), encoding='utf-8')
        implementation = [Path(__file__), BASE / 'README_VISUAL_ENVIRONMENT_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_environment_adapter.py', 'visual_request_schedule.py', 'visual_upload_link.py',
            'physical_mobility.py', 'predicted_fifo_feedback.py', 'fifo_execution.py',
            'execution_service_catalog.py', 'metadata_service_predictor.py', 'replay_measured_fifo.py')]
        report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
            synthetic_integration_groups=synthetic_groups, cases=[first, repeated, changed],
            link_config=asdict(UploadLinkConfig()), physical_horizon_seconds=4., engineering_drain_seconds=30.,
            metadata_role='validation', model_fitting=False, test_measurements_accessed=False,
            measurement_sources=catalog.source_sha256, mobility_sources=mobility.source_hashes,
            implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation},
            formal_protocol_frozen=False, actual_learning_methods_connected=False,
            policy_training_cells=0, formal_policy_test_cells=0,
            scope='Engineering routing fixtures only; not QLA/PPO/JSEQ comparison, independent final audit or real wireless/deployment measurement')
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_ENVIRONMENT_ENGINEERING_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'error': repr(error), 'at': datetime.now().astimezone().isoformat(),
            'formal_policy_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
