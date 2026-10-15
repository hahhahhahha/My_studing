"""Raw time/x/y mobility in physical seconds; no legacy index/time compression.

This is an engineering adapter, not a frozen visual-policy split protocol.
The caller must provide the eventual pre-registered train/validation windows.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
KAIST = ROOT / 'Traces_TimeXY_10sec_txt/KAIST'


@dataclass(frozen=True)
class TrainingWindow:
    start_seconds: float
    end_seconds: float

    def __post_init__(self):
        if (not all(math.isfinite(t) for t in (self.start_seconds, self.end_seconds))
                or self.start_seconds < 0 or self.end_seconds <= self.start_seconds):
            raise ValueError('Positive-duration physical training window required')


class PhysicalMobility:
    def __init__(self, folder=KAIST, users=30):
        if type(users) is not int or users < 1:
            raise ValueError('Positive integer user count required')
        files = sorted(Path(folder).glob('*.txt'))[:users]
        if len(files) != users:
            raise ValueError('Insufficient real traces; no synthetic user fallback')
        self.source_hashes = {}
        self._traces = []
        for path in files:
            raw = path.read_bytes()
            self.source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            rows = []
            for line in raw.decode('utf-8').splitlines():
                fields = line.split()
                if len(fields) != 3:
                    raise ValueError(f'Time/x/y fields required: {path}')
                row = tuple(map(float, fields))
                if not all(map(math.isfinite, row)):
                    raise ValueError(f'Nonfinite raw mobility: {path}')
                if rows and not math.isclose(row[0] - rows[-1][0], 10., abs_tol=1e-9, rel_tol=0.):
                    raise ValueError(f'Raw KAIST time grid is not exactly 10 seconds: {path}')
                rows.append(row)
            if len(rows) < 2:
                raise ValueError(f'Trace too short: {path}')
            self._traces.append(tuple(rows))
        self._times = [tuple(row[0] for row in rows) for rows in self._traces]
        self.users = users
        self.common_start_seconds = max(times[0] for times in self._times)
        self.common_end_seconds = min(times[-1] for times in self._times)
        if self.common_end_seconds <= self.common_start_seconds:
            raise ValueError('No shared physical-time support')

    def _require_time(self, seconds):
        if (not math.isfinite(seconds) or
                not self.common_start_seconds <= seconds <= self.common_end_seconds):
            raise ValueError('Physical time outside shared trace; no wrap, hold or extrapolation')

    def position(self, user, seconds):
        self._require_time(seconds)
        if type(user) is not int or not 0 <= user < self.users:
            raise ValueError('Invalid user index')
        times, rows = self._times[user], self._traces[user]
        left = bisect_right(times, seconds) - 1
        if times[left] == seconds:
            return rows[left][1:]
        right = left + 1
        fraction = (seconds - times[left]) / (times[right] - times[left])
        return tuple(rows[left][axis] + fraction * (rows[right][axis] - rows[left][axis])
                     for axis in (1, 2))

    def positions(self, seconds):
        return tuple(self.position(user, seconds) for user in range(self.users))

    def training_bounds(self, window):
        if not isinstance(window, TrainingWindow):
            raise TypeError('Explicit training window required; no whole-trace placement bounds')
        self._require_time(window.start_seconds)
        self._require_time(window.end_seconds)
        points = list(self.positions(window.start_seconds)) + list(self.positions(window.end_seconds))
        for rows in self._traces:
            points.extend(row[1:] for row in rows
                          if window.start_seconds < row[0] < window.end_seconds)
        # Piecewise-linear extrema occur at the window endpoints or raw vertices.
        return tuple(min(point[axis] for point in points) for axis in (0, 1)), \
               tuple(max(point[axis] for point in points) for axis in (0, 1))

    def server_positions(self, window, seed, servers=4):
        if type(seed) is not int or type(servers) is not int or servers < 1:
            raise ValueError('Integer seed and positive server count required')
        low, high = self.training_bounds(window)
        rng = random.Random(seed)
        return tuple(tuple(rng.uniform(low[axis], low[axis] + max(high[axis] - low[axis], 1.))
                           for axis in (0, 1)) for _ in range(servers))
