"""Engineering fixtures for physical mobility, not a policy evaluation."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile

from physical_mobility import PhysicalMobility, TrainingWindow

BASE = Path(__file__).resolve().parents[1]


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError):
        return True
    raise AssertionError('Required boundary was not enforced')


def main():
    # Artificial positions are labeled fixtures, never appended to real traces.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'fixture.txt'
        path.write_text('0 0 0\n10 100 20\n20 10000 20000\n', encoding='utf-8')
        fixture = PhysicalMobility(temporary, users=1)
        assert fixture.position(0, .05) == (.5, .1)
        assert fixture.position(0, 5.) == (50., 10.)
        assert fixture.position(0, 10.) == (100., 20.)
        bounds = fixture.training_bounds(TrainingWindow(0., 10.))
        assert bounds == ((0., 0.), (100., 20.))  # Later held-out extreme excluded.
        servers = fixture.server_positions(TrainingWindow(0., 10.), 123, 4)
        assert servers == fixture.server_positions(TrainingWindow(0., 10.), 123, 4)
        assert all(0 <= x <= 100 and 0 <= y <= 20 for x, y in servers)
        refused(lambda: fixture.position(0, 20.05))
        refused(lambda: fixture.position(0, -.05))
        refused(lambda: fixture.training_bounds((0., 10.)))
        path.write_text('0 0 0\n1 100 20\n', encoding='utf-8')
        refused(lambda: PhysicalMobility(temporary, users=1))

    mobility = PhysicalMobility(users=30)
    # Engineering check only, NOT an adopted final split or server seed.
    check_start = mobility.common_start_seconds
    check_end = check_start + 500.
    if check_end > mobility.common_end_seconds:
        raise RuntimeError('Insufficient physical-time support for engineering check')
    window = TrainingWindow(check_start, check_end)
    edges = mobility.server_positions(window, 123, 4)
    errors = []
    speed_errors = []
    for user in range(30):
        for offset in (0., 10., 100., 490.):
            time = check_start + offset
            start = mobility.position(user, time)
            end = mobility.position(user, time + 10.)
            middle = mobility.position(user, time + 5.)
            near = mobility.position(user, time + .05)
            expected_mid = tuple((a + b) * .5 for a, b in zip(start, end))
            expected_near = tuple(a + .005 * (b - a) for a, b in zip(start, end))
            errors.extend(abs(a - b) for a, b in zip(middle, expected_mid))
            errors.extend(abs(a - b) for a, b in zip(near, expected_near))
            # 50ms displacement must be 0.005 of a raw 10s displacement,
            # not one whole old trace step per new policy decision.
            speed_errors.append(abs(math.dist(start, near) - .005 * math.dist(start, end)))
    assert max(errors) < 1e-9 and max(speed_errors) < 1e-9
    refused(lambda: mobility.positions(mobility.common_end_seconds + .05))
    report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
                  real_users=30, raw_sample_period_seconds=10., policy_grid_fixture_seconds=.05,
                  common_start_seconds=mobility.common_start_seconds,
                  common_end_seconds=mobility.common_end_seconds,
                  engineering_window_seconds=[check_start, check_end],
                  engineering_only_server_seed=123, engineering_server_positions=edges,
                  max_interpolation_error_meters=max(errors),
                  max_displacement_scale_error_meters=max(speed_errors),
                  source_sha256=mobility.source_hashes,
                  implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (Path(__file__), BASE / 'scripts/physical_mobility.py',
                                                      BASE / 'README_VISUAL_MOBILITY_中文.md')},
                  final_trace_split_frozen=False, formal_policy_cells=0,
                  scope='Physical-time engineering adapter fixtures only; no policy, test selection or deployment result')
    output = BASE / 'protocol' / ('PHYSICAL_MOBILITY_FOUNDATION_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PHYSICAL_MOBILITY_FOUNDATION_OK', output)


if __name__ == '__main__':
    main()
