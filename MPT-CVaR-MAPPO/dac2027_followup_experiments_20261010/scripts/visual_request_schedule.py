"""Method-independent engineering request schedule; retained test disabled."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from execution_service_catalog import SESSIONS


@dataclass(frozen=True)
class ScheduledRequest:
    task_id: str
    user: int
    arrival_seconds: float
    image_id: str
    measured_session: int
    assignment_uniform: float


@dataclass(frozen=True)
class VisualRequestSchedule:
    role: str
    users: int
    decision_seconds: float
    decisions: int
    seed: int
    requests: tuple[ScheduledRequest, ...]
    arrival_rule: str

    def __post_init__(self):
        if self.role not in SESSIONS:
            raise ValueError('Engineering schedule cannot enable retained test')
        if type(self.users) is not int or self.users < 1 or type(self.decisions) is not int or self.decisions < 1:
            raise ValueError('Positive user/decision counts required')
        if not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0:
            raise ValueError('Physical decision duration required')
        tasks = set()
        for row in self.requests:
            if (row.task_id in tasks or not 0 <= row.user < self.users
                    or not math.isfinite(row.arrival_seconds)
                    or not 0 <= row.arrival_seconds < self.horizon_seconds
                    or row.measured_session not in SESSIONS[self.role]
                    or not math.isfinite(row.assignment_uniform) or not 0 <= row.assignment_uniform < 1):
                raise ValueError('Invalid shared request row')
            tasks.add(row.task_id)

    @property
    def horizon_seconds(self):
        return float(self.decisions * Fraction(str(self.decision_seconds)))

    def decision_time(self, index):
        if type(index) is not int or not 1 <= index <= self.decisions:
            raise ValueError('Decision index out of schedule')
        return float(index * Fraction(str(self.decision_seconds)))

    def decision_index(self, request):
        # Arrivals in the preceding interval are observed at the next decision;
        # never reveal not-yet-arrived images during an earlier portfolio choice.
        return max(1, math.ceil(Fraction(str(request.arrival_seconds)) /
                               Fraction(str(self.decision_seconds))))

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(',', ':')).encode('utf-8')).hexdigest()


def generate_schedule(catalog, users, decisions, decision_seconds, seed,
                      total_requests_per_second, burst_probability=.1, burst_multiplier=2.):
    if (not math.isfinite(total_requests_per_second) or total_requests_per_second <= 0
            or not 0 <= burst_probability <= 1 or not math.isfinite(burst_multiplier)
            or burst_multiplier < 1):
        raise ValueError('Valid unconditional rate and burst parameters required')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    # Streams are isolated from policy actions, hardware truth and each other.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    counts_rng, offsets_rng, images_rng, sessions_rng, uniform_rng = streams
    ids = catalog.image_ids()
    sessions = sorted(SESSIONS[catalog.role])
    base = total_requests_per_second / users / (1 + burst_probability * (burst_multiplier - 1))
    rows = []
    period = Fraction(str(decision_seconds))
    for step in range(decisions):
        for user in range(users):
            burst = counts_rng.random() < burst_probability
            count = int(counts_rng.poisson(base * decision_seconds * (burst_multiplier if burst else 1)))
            for _ in range(count):
                arrival = float(step * period) + float(offsets_rng.random()) * decision_seconds
                # Uniform is exclusive of1; protect an extremely rare rounded
                # boundary so the row remains within its generation interval.
                arrival = min(arrival, math.nextafter(float((step + 1) * period), -math.inf))
                rows.append(ScheduledRequest(f'task-{len(rows):08d}', user, arrival,
                            ids[int(images_rng.integers(len(ids)))],
                            sessions[int(sessions_rng.integers(len(sessions)))], float(uniform_rng.random())))
    rows.sort(key=lambda row: (row.arrival_seconds, row.user, row.task_id))
    return VisualRequestSchedule(catalog.role, users, decision_seconds, decisions, seed, tuple(rows),
            f'Per-user interval Poisson-bursty; unconditional total rate={total_requests_per_second!r}/s; '
            f'p={burst_probability!r}, multiplier={burst_multiplier!r}; no request truncation')
