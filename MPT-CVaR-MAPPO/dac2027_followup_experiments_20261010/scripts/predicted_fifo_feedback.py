"""Route-B execution/feedback boundary, not a full policy environment.

Actual measured service belongs to FIFO execution. Broadcasts contain only
metadata-predicted outstanding service and observed queue counts/busy flags.
No actor, critic, QL rule or learned method receives this execution object.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math

from fifo_execution import FIFOExecution, Request
from metadata_service_predictor import PROFILES


@dataclass(frozen=True)
class QueueBroadcast:
    source_seconds: float
    predicted_backlog_seconds: tuple[float, ...]
    waiting_counts: tuple[int, ...]
    busy: tuple[bool, ...]


@dataclass(frozen=True)
class RetrievedFeedback:
    source_seconds: float | None
    age_seconds: float | None
    missing: bool
    predicted_backlog_seconds: tuple[float, ...]
    waiting_counts: tuple[int, ...]
    busy: tuple[bool, ...]


class PredictedFIFOFeedback:
    def __init__(self, profiles, waiting_capacity, broadcast_seconds, nominal_delay_seconds):
        profiles = tuple(profiles)
        if not profiles or any(profile not in PROFILES for profile in profiles):
            raise ValueError('Registered CPU profiles required')
        if (not math.isfinite(broadcast_seconds) or broadcast_seconds <= 0
                or not math.isfinite(nominal_delay_seconds) or nominal_delay_seconds < 0):
            raise ValueError('Positive physical broadcast period and nonnegative delay required')
        self.profiles = profiles
        self._period = Fraction(str(broadcast_seconds))
        self._delay = Fraction(str(nominal_delay_seconds))
        self._execution = FIFOExecution(len(profiles), waiting_capacity)
        self._predicted = {}
        self._history = [self._snapshot()]
        self._source_times = [0.]
        self._next_broadcast_index = 1

    @property
    def time(self):
        return self._execution.time

    def submit_execution(self, request, predicted_service_seconds):
        """Environment-only submission; actor inputs are constructed separately."""
        if not isinstance(request, Request):
            raise TypeError('Execution request required')
        if not math.isfinite(predicted_service_seconds) or predicted_service_seconds <= 0:
            raise ValueError('Positive metadata prediction required before submitting truth')
        self._execution.submit(request)
        self._predicted[request.task_id] = float(predicted_service_seconds)

    def _snapshot(self):
        backlog, waiting_counts, busy = [], [], []
        for server in range(len(self.profiles)):
            waiting = self._execution.waiting[server]
            running = self._execution.running[server]
            predicted = math.fsum(self._predicted[task] for task in waiting)
            if running is not None:
                elapsed = self.time - self._execution.records[running]['started']
                predicted += max(self._predicted[running] - elapsed, 0.)
            # Status and start time are observable at the server. No actual
            # service or actual remaining duration is read for this summary.
            backlog.append(predicted)
            waiting_counts.append(len(waiting))
            busy.append(running is not None)
        return QueueBroadcast(self.time, tuple(backlog), tuple(waiting_counts), tuple(busy))

    def advance(self, until):
        if not math.isfinite(until) or until < self.time:
            raise ValueError('Physical execution time cannot run backwards')
        # Integer broadcast indices times an exact decimal-period fraction
        # avoid .1+.1+.1 timestamp drift at the .3 decision boundary.
        while float(self._next_broadcast_index * self._period) <= until:
            source = float(self._next_broadcast_index * self._period)
            self._execution.advance(source)
            self._history.append(self._snapshot())
            self._source_times.append(source)
            self._next_broadcast_index += 1
        self._execution.advance(until)

    def retrieve(self):
        """The same immutable timestamped feedback surface is for all methods.

        A snapshot is usable only if source <= current time - nominal delay.
        Missing startup feedback is explicitly flagged, never backdated.
        """
        cutoff = float(Fraction(str(self.time)) - self._delay)
        index = bisect_right(self._source_times, cutoff) - 1
        if index < 0:
            zeros = (0.,) * len(self.profiles)
            return RetrievedFeedback(None, None, True, zeros,
                                     (0,) * len(self.profiles), (False,) * len(self.profiles))
        snapshot = self._history[index]
        return RetrievedFeedback(snapshot.source_seconds, self.time - snapshot.source_seconds,
                                 False, snapshot.predicted_backlog_seconds,
                                 snapshot.waiting_counts, snapshot.busy)

    def execution_diagnostics(self):
        """Privileged engineering/reporting output, NOT actor/QL/critic features."""
        return {'tasks': self._execution.results(),
                'work': self._execution.work_accounting()}

    def execution_completion_times(self):
        """Privileged replay check; never passed to a policy."""
        return {task: record['completed'] for task, record in self._execution.records.items()}
