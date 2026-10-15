"""Synthetic causality fixtures plus measured validation replay; no policy."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from execution_service_catalog import ExecutionServiceCatalog
from fifo_execution import Request
from metadata_service_predictor import MetadataServicePredictor
from predicted_fifo_feedback import PredictedFIFOFeedback, RetrievedFeedback

BASE = Path(__file__).resolve().parents[1]


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-10, rel_tol=0.), (a, b)


def fixtures():
    profiles = ('CPU-1T',)
    cold = PredictedFIFOFeedback(profiles, 32, .1, .1)
    cold.advance(.05)
    assert cold.retrieve().missing and cold.retrieve().source_seconds is None
    warm = PredictedFIFOFeedback(profiles, 32, .1, 0.)
    warm.advance(.05)
    assert not warm.retrieve().missing
    close(warm.retrieve().age_seconds, .05)  # zero nominal delay != zero age
    cold.advance(.3)
    close(cold.retrieve().source_seconds, .2)  # decimal boundary .3-.1

    # Different future hardware jitter, same alive statuses: same predictions.
    views = []
    for actual in (.4, .8):
        env = PredictedFIFOFeedback(profiles, 32, .1, 0.)
        env.submit_execution(Request('running', 0., 0., 0, actual, 100), .05)
        env.submit_execution(Request('waiting', 0., 0., 0, .02, 100), .02)
        env.advance(.1)
        view = env.retrieve()
        close(view.predicted_backlog_seconds[0], .02)
        assert view.busy == (True,) and view.waiting_counts == (1,)
        views.append(view)
    assert views[0] == views[1]  # no oracle remaining-time feature

    upload = PredictedFIFOFeedback(profiles, 32, .1, .1)
    upload.submit_execution(Request('long-upload', 0., .15, 0, .5, 100), .1)
    upload.advance(.1)
    old = upload.retrieve()
    assert old.busy == (False,) and old.predicted_backlog_seconds == (0.,)
    upload.advance(.25)
    assert upload.retrieve().source_seconds == .1
    assert upload.retrieve().busy == (False,)  # actual queue newer than received summary
    upload.advance(.3)
    new = upload.retrieve()
    assert new.source_seconds == .2 and new.busy == (True,)
    close(new.predicted_backlog_seconds[0], .05)
    assert old.busy == (False,)  # later evolution cannot mutate old observation
    try:
        new.missing = True
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('Feedback must be immutable')

    tie = PredictedFIFOFeedback(profiles, 32, .1, 0.)
    tie.submit_execution(Request('finishes-on-broadcast', 0., 0., 0, .1, 100), .1)
    tie.advance(.1)
    assert tie.retrieve().busy == (False,)  # completion before snapshot
    before = tie.execution_diagnostics()['tasks']['generated']
    try:
        tie.submit_execution(Request('bad-prediction', .1, 0., 0, .1, 100), float('nan'))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid prediction accepted')
    assert tie.execution_diagnostics()['tasks']['generated'] == before
    names = {field.name for field in fields(RetrievedFeedback)}
    assert names == {'source_seconds', 'age_seconds', 'missing', 'predicted_backlog_seconds',
                     'waiting_counts', 'busy'}
    return 7


def measured_replay():
    catalog = ExecutionServiceCatalog('validation')
    predictor = MetadataServicePredictor()
    profiles = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')
    images = catalog.image_ids()[:64]
    # A fixed engineering request list, not a formal load/horizon/schedule.
    requests = []
    predicted = []
    for index, image in enumerate(images):
        metadata = catalog.observable_metadata(image)
        estimate = predictor.predict_seconds(metadata)
        truth = catalog.measured_pair_seconds(image, 1)
        server = index % 4
        arrival = round(index * .005, 6)
        upload = metadata.payload_bytes * 8. / 20_000_000.
        requests.append(Request(f'request-{index:03d}', arrival, upload, server,
                                truth[profiles[server]], metadata.payload_bytes, image))
        predicted.append(estimate[profiles[server]])
    # Separate serial oracle by upload-ready/task-ID; <=16 requests/server
    # means waiting capacity32 cannot bind in this specific fixture.
    expected = {}
    for server in range(4):
        clock = 0.
        for request in sorted((r for r in requests if r.server == server),
                              key=lambda r: (r.arrival + r.upload, r.task_id)):
            clock = max(clock, request.arrival + request.upload) + request.service
            expected[request.task_id] = clock
    drain_end = max(expected.values()) + 1.  # oracle fixture only, not a final drain rule
    cases = []
    for delay in (0., .1, .4):
        env = PredictedFIFOFeedback(profiles, 32, .1, delay)
        observations = []
        for request, estimate in zip(requests, predicted):
            env.advance(request.arrival)
            view = env.retrieve()  # pre-action boundary
            if not view.missing:
                assert view.source_seconds <= request.arrival - delay + 1e-12
                close(view.age_seconds, request.arrival - view.source_seconds)
            observations.append(view)
            env.submit_execution(request, estimate)
        # Continue 50ms engineering decisions after the last arrival: the .4s
        # case must also exercise receipt of real past broadcasts, not only
        # startup-missing placeholders during the .315s arrival interval.
        post_views = []
        tick = math.floor(env.time / .05) + 1
        while round(tick * .05, 8) <= drain_end:
            now = round(tick * .05, 8)
            env.advance(now)
            view = env.retrieve()
            if not view.missing:
                assert view.source_seconds <= now - delay + 1e-12
                close(view.age_seconds, now - view.source_seconds)
            post_views.append(view)
            tick += 1
        assert any(not view.missing for view in post_views)
        env.advance(drain_end)
        diagnostic = env.execution_diagnostics()
        assert diagnostic['tasks']['completed'] == 64
        assert diagnostic['tasks']['dropped'] == diagnostic['tasks']['unfinished'] == 0
        actual = env.execution_completion_times()
        error = max(abs(actual[task] - expected[task]) for task in expected)
        assert error < 1e-10
        mass_error = max(max(abs(row['offered_error']), abs(row['admitted_error']))
                         for row in diagnostic['work'])
        assert mass_error < 1e-10
        cases.append({'nominal_delay_seconds': delay, 'observations': len(observations),
                      'startup_missing_observations': sum(view.missing for view in observations),
                      'post_request_decision_observations': len(post_views),
                      'post_request_available_feedback': sum(not view.missing for view in post_views),
                      'generated': 64, 'completed': 64, 'dropped': 0, 'unfinished': 0,
                      'max_serial_completion_error_seconds': error,
                      'max_service_conservation_error_seconds': mass_error})
    return cases, catalog.source_sha256, predictor.model_sha256


def main():
    count = fixtures()
    cases, sources, model_sha = measured_replay()
    implementation = [Path(__file__), BASE / 'scripts/predicted_fifo_feedback.py',
                      BASE / 'scripts/fifo_execution.py', BASE / 'scripts/metadata_service_predictor.py',
                      BASE / 'scripts/execution_service_catalog.py', BASE / 'README_VISUAL_FEEDBACK_中文.md']
    report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
                  synthetic_causality_groups=count, real_validation_images=64, cases=cases,
                  source_sha256=sources, model_sha256=model_sha,
                  implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in implementation},
                  policy_training_cells=0, formal_policy_test_cells=0, test_measurements_accessed=False,
                  scope='Execution/feedback engineering boundary and validation replay only; no integrated policy, independent final certification or QLA effect')
    path = BASE / 'protocol' / ('PREDICTED_FIFO_FEEDBACK_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PREDICTED_FIFO_FEEDBACK_FOUNDATION_OK', path)


if __name__ == '__main__':
    main()
