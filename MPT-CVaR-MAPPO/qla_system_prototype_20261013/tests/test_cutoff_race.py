"""Regression tests for the cutoff-window double-outcome race (review B1).

Written as ``unittest.TestCase`` so that BOTH the author unit runner
(``tools/verify_unit.py``, which uses ``unittest discover``) and pytest collect
and run them. A pytest-function-only style would be silently skipped by the
author unit runner.

These use a synthetic, service-free fixture: no real JPEG/ResNet/HTTP. They
assert the invariant that every offered task receives exactly one formal cutoff
outcome, even when the response thread is descheduled between recording its
response time and acquiring the outcome lock.
"""
import hashlib
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from systemlab.common import write_new  # noqa: E402
from systemlab.client import replay  # noqa: E402
from systemlab.analysis import analyze  # noqa: E402


class FixtureRouter:
    receipt = {'method': 'SYNTHETIC_ONLY'}

    def __init__(self, *args):
        pass

    def portfolio(self, *args):
        return [1.], dict(actor_observation=[], expected_costs_s=[0], feedback_age_s=[0])

    def verify_frozen(self):
        return {'actor_executed': False, 'synthetic_fixture': True}


class FixtureCache:
    def __init__(self, *args):
        pass

    def observation(self, *args):
        return {'fixture_edge': dict(source_epoch_ns=0, version=1)}


class FixtureSocket:
    def __init__(self, *args):
        pass

    def bind(self, *args):
        pass

    def settimeout(self, *args):
        pass

    def recvfrom(self, *args):
        time.sleep(.005)
        raise socket.timeout()

    def close(self):
        pass


def _setup(root, response_delay_s=0.0):
    asset = root / 'NOT_A_JPEG_fixture.bin'
    asset.write_bytes(b'SYNTHETIC_FIXTURE_ONLY')
    manifest = root / 'manifest.json'
    write_new(manifest, dict(schema='systemlab-images-v1', role='pilot', image_root=str(root),
        images=[dict(image_id=asset.name, image_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                     payload_bytes=asset.stat().st_size)]))
    schedule = root / 'schedule.json'
    write_new(schedule, dict(schema='systemlab-schedule-v1', role='pilot', users=1, seed=1, duration_s=.05,
        image_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        tasks=[dict(task_id='fixture_task', user_id=0, image_id=asset.name, arrival_s=.001, assignment_uniform=.5)]))
    cfg = dict(users=1, seed=1, edges=[dict(id='fixture_edge', url='fixture://no-network')],
        waiting_capacity=2, weight_sha256='fixture', warmup=0,
        feature_scales={}, score_temperature_s=.05, controller_url='fixture://no-controller',
        feedback=dict(mode='periodic', budget_bytes_per_s=1, burst_bytes=1, udp_port=0, injected_loss=0),
        feedback_start_timeout_s=1, max_inflight=1,
        fixed_drain_s=.05, decision_period_s=.01, client_timeout_s=1)

    def fake_request(url, **kwargs):
        if url.endswith('/health'):
            return 200, dict(ready=True, session='SYNTHETIC')
        if response_delay_s:
            time.sleep(response_delay_s)
        return 200, dict(status='completed', task_id='fixture_task', server_id='fixture_edge',
            queue_s=0., server_service_s=.001, preprocess_s=0., compute_s=.001)

    return cfg, schedule, manifest, fake_request


def _run(root, cfg, schedule, manifest, fake_request, pause_after_end=False):
    paused = [False]

    def trace(frame, event, arg):
        if (event == 'line' and frame.f_code.co_name == 'send_task' and
                'end' in frame.f_locals and pause_after_end and not paused[0]):
            paused[0] = True
            deadline = time.monotonic() + 2
            while not (root / 'client' / 'CUTOFF.json').exists() and time.monotonic() < deadline:
                time.sleep(.005)
        return trace

    if pause_after_end:
        threading.settrace(trace)
    try:
        with patch('systemlab.client.Router', FixtureRouter), \
             patch('systemlab.client.FeedbackCache', FixtureCache), \
             patch('systemlab.client.socket.socket', FixtureSocket), \
             patch('systemlab.client.request_json', fake_request):
            replay(cfg, schedule, manifest, 'score_softmax', root / 'client')
    finally:
        if pause_after_end:
            threading.settrace(None)


def _outcomes(root):
    return [json.loads(s) for s in (root / 'client' / 'tasks.jsonl').read_text().splitlines()]


class CutoffRaceTests(unittest.TestCase):
    def test_cutoff_race_single_outcome(self):
        """Response thread descheduled after 'end' but before the lock: still one outcome."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, schedule, manifest, fake_request = _setup(root)
            _run(root, cfg, schedule, manifest, fake_request, pause_after_end=True)
            outcomes = _outcomes(root)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(len({x['task_id'] for x in outcomes}), 1)
            self.assertEqual(outcomes[0]['status'], 'completed')
            # The previous double-write made analyze reject with "exactly one outcome".
            analyze(root / 'client', root / 'analysis.json')
            analysis = json.load(open(root / 'analysis.json'))
            self.assertEqual(analysis['completed'], 1)
            self.assertEqual(analysis['status_counts'], {'completed': 1})

    def test_normal_completion_single_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, schedule, manifest, fake_request = _setup(root)
            _run(root, cfg, schedule, manifest, fake_request)
            outcomes = _outcomes(root)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]['status'], 'completed')

    def test_late_response_recorded_as_unfinished(self):
        """Response genuinely after cutoff: one 'unfinished_at_cutoff' + late evidence."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, schedule, manifest, fake_request = _setup(root, response_delay_s=0.2)
            _run(root, cfg, schedule, manifest, fake_request)
            outcomes = _outcomes(root)
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]['status'], 'unfinished_at_cutoff')
            late = [json.loads(s) for s in (root / 'client' / 'late_responses.jsonl').read_text().splitlines()]
            self.assertEqual(len(late), 1)
            # analyze still resolves to exactly one offered outcome
            analyze(root / 'client', root / 'analysis.json')


if __name__ == '__main__':
    unittest.main()
