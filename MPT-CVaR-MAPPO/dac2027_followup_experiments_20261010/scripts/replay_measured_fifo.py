"""Measured validation-service FIFO replay against an independent serial oracle.

No actor/policy, fitting, test outcome selection, or physical network claim.
"""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from fifo_execution import FIFOExecution, Request

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
RAW = ROOT / 'dac_visual_calibration_20261010_v2/measurements/validation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_oracle(requests, servers, capacity):
    """Sort uploads independently; serialized accepted finish times define FIFO."""
    result = {}
    for server in range(servers):
        active_ends = []
        for request in sorted((r for r in requests if r.server == server), key=lambda r: (r.arrival + r.upload, r.task_id)):
            ready = request.arrival + request.upload
            active_ends = [end for end in active_ends if end > ready]
            if active_ends and len(active_ends) - 1 >= capacity:
                result[request.task_id] = dict(ready=ready, admitted=None, started=None, completed=None, dropped=True)
            else:
                start = max(ready, active_ends[-1]) if active_ends else ready
                end = start + request.service
                active_ends.append(end)
                result[request.task_id] = dict(ready=ready, admitted=ready, started=start, completed=end, dropped=False)
    return result


def check_case(requests, servers, capacity, name):
    expected = independent_oracle(requests, servers, capacity)
    execution = FIFOExecution(servers, capacity)
    for request in requests:
        execution.submit(request)
    end = max((r['completed'] or r['ready']) for r in expected.values()) + .1
    # Independent observed cuts include upload-in-flight and partial service.
    cuts = sorted(set([.01, .025] + [min(end, .05 * i) for i in range(1, math.ceil(end / .05) + 1)] + [end]))
    max_mass_error = 0.
    comparisons = 0
    for cut in cuts:
        execution.advance(cut)
        for row in execution.work_accounting():
            max_mass_error = max(max_mass_error, abs(row['offered_error']), abs(row['admitted_error']))
        for task_id, wanted in expected.items():
            got = execution.records[task_id]
            if cut < wanted['ready']:
                status = 'uploading'
            elif wanted['dropped']:
                status = 'dropped'
            elif cut < wanted['started']:
                status = 'waiting'
            elif cut < wanted['completed']:
                status = 'running'
            else:
                status = 'completed'
            if got['status'] != status:
                raise AssertionError((name, task_id, cut, got['status'], status))
            for field in ('admitted', 'started', 'completed'):
                actual = got[field]
                should = wanted[field]
                visible = should is not None and should <= cut
                if visible:
                    if actual is None or not math.isclose(actual, should, rel_tol=0., abs_tol=1e-11):
                        raise AssertionError((name, task_id, field, actual, should))
                elif actual is not None:
                    raise AssertionError((name, task_id, 'future event visible', field))
            comparisons += 1
    metrics = execution.results()
    if metrics['unfinished'] != 0 or metrics['generated'] != metrics['completed'] + metrics['dropped']:
        raise AssertionError('Final task conservation/drain failed')
    return dict(case=name, servers=servers, waiting_capacity=capacity, tasks=len(requests),
                completed=metrics['completed'], dropped=metrics['dropped'], observation_cuts=len(cuts),
                record_state_comparisons=comparisons, max_service_seconds_mass_error=max_mass_error)


def run():
    files = sorted(RAW.glob('validation_s1_*.csv'))
    measured = {}
    hashes = {}
    for path in files:
        hashes[str(path)] = sha(path)
        with path.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row['image_split'] != 'validation' or row['error_status'] != 'ok':
                    raise AssertionError('Wrong measured held-out role/error')
                key = (row['image_id'], row['profile_id'])
                if key in measured:
                    raise AssertionError('Duplicate image/profile')
                measured[key] = row
    ids = sorted({key[0] for key in measured})[:128]  # metadata order, not timing/outcomes
    if len(ids) != 128 or any((image, p) not in measured for image in ids for p in ('CPU-1T', 'CPU-4T')):
        raise AssertionError('Incomplete paired validation data')
    cases = []
    for servers, capacity in [(1, 0), (1, 1), (1, 32), (4, 0), (4, 1), (4, 32)]:
        requests = []
        for i, image in enumerate(ids):
            server = i % servers
            profile = 'CPU-1T' if server % 2 == 0 else 'CPU-4T'
            row = measured[(image, profile)]
            # Stress replay fixture only: physical seconds and 20Mbps upload.
            # These are not formal-policy workload or link settings.
            requests.append(Request(f'task_{i:04d}', .001 * i, int(row['payload_bytes']) / (20e6 / 8), server,
                                    float(row['service_ms']) / 1000., int(row['payload_bytes']), image))
        cases.append(check_case(requests, servers, capacity, f'servers{servers}_capacity{capacity}'))
    report = dict(passed=True, at=datetime.now().astimezone().isoformat(), raw_hashes=hashes, validation_images=128,
                  cases=cases, predictor_or_policy_fitted=False, retained_test_used=False,
                  evidence='Measured validation-service execution replay only; not integrated simulator, physical network or QLA gains',
                  execution_sha256=sha(BASE / 'scripts/fifo_execution.py'), oracle_sha256=sha(__file__))
    target = BASE / 'protocol' / ('MEASURED_FIFO_REPLAY_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with target.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('MEASURED_VALIDATION_FIFO_REPLAY_OK', target, flush=True)


if __name__ == '__main__':
    run()
