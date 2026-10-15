"""Two-phase observable/execution adapter; learners/reward not yet integrated.

Policies receive DecisionObservation only, never this environment, schedule,
measurement catalog, assignment uniforms, future positions or actual durations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from fifo_execution import Request
from metadata_service_predictor import MetadataServicePredictor, ObservableImage
from predicted_fifo_feedback import PredictedFIFOFeedback, RetrievedFeedback
from visual_upload_link import UploadLinkConfig, SerializedUserUploads


@dataclass(frozen=True)
class ObservableRequest:
    task_id: str
    arrival_seconds: float
    metadata: ObservableImage
    predicted_service_seconds: tuple[float, ...]
    transmission_seconds: tuple[float, ...]


@dataclass(frozen=True)
class ObservableUser:
    user: int
    location_meters: tuple[float, float]
    previous_portfolio: tuple[float, ...]
    existing_upload_wait_seconds: float
    requests: tuple[ObservableRequest, ...]


@dataclass(frozen=True)
class DecisionObservation:
    index: int
    physical_seconds: float
    feedback: RetrievedFeedback
    users: tuple[ObservableUser, ...]
    joint_predicted_reference_demand_seconds: float


class VisualEnvironmentAdapter:
    def __init__(self, schedule, catalog, mobility, edge_positions, profiles,
                 mobility_start_seconds, broadcast_seconds=.1, nominal_delay_seconds=.1,
                 waiting_capacity=32, link=None):
        if schedule.role != catalog.role or mobility.users != schedule.users:
            raise ValueError('Shared measurement role and user geometry mismatch')
        self._edges = tuple(tuple(point) for point in edge_positions)
        self._profiles = tuple(profiles)
        if len(self._edges) != len(self._profiles) or not self._edges:
            raise ValueError('One declared measured profile per virtual server required')
        mobility.positions(mobility_start_seconds)
        mobility.positions(mobility_start_seconds + schedule.horizon_seconds)
        self._schedule, self._catalog, self._mobility = schedule, catalog, mobility
        self._mobility_start = mobility_start_seconds
        self._predictor = MetadataServicePredictor()
        self._link = link or UploadLinkConfig()
        self._feedback = PredictedFIFOFeedback(self._profiles, waiting_capacity,
                                               broadcast_seconds, nominal_delay_seconds)
        self._uploads = SerializedUserUploads(schedule.users)
        self._batches = {index: [] for index in range(1, schedule.decisions + 1)}
        for row in schedule.requests:
            catalog.observable_metadata(row.image_id)  # invalid IDs fail, never silently omitted
            self._batches[schedule.decision_index(row)].append(row)
        self._index = 0
        self._pending = None
        self._failed = False
        self._finished = False
        self._last = tuple((1. / len(self._edges),) * len(self._edges) for _ in range(schedule.users))
        self._issued = {}

    def observe(self):
        if self._failed or self._finished:
            raise RuntimeError('Failed/finished engineering execution cannot continue or restart')
        if self._pending is not None:
            return self._pending  # repeat reads cannot move to a future request batch
        if self._index >= self._schedule.decisions:
            raise RuntimeError('All decision intervals consumed; no implicit schedule wrap')
        try:
            return self._build_observation()
        except Exception:
            self._failed = True
            raise

    def _build_observation(self):
        self._index += 1
        now = self._schedule.decision_time(self._index)
        self._feedback.advance(now)
        locations = self._mobility.positions(self._mobility_start + now)
        requests = [[] for _ in range(self._schedule.users)]
        demand = []
        for row in self._batches[self._index]:
            if row.arrival_seconds > now:
                raise RuntimeError('Future image exposed at current decision')
            metadata = self._catalog.observable_metadata(row.image_id)
            estimate = self._predictor.predict_seconds(metadata)
            service = tuple(estimate[profile] for profile in self._profiles)
            transmission = tuple(self._link.transmission_seconds(metadata.payload_bytes,
                                 locations[row.user], edge) for edge in self._edges)
            requests[row.user].append(ObservableRequest(row.task_id, row.arrival_seconds,
                                                       metadata, service, transmission))
            demand.append(estimate['CPU-1T'])  # hardware-independent reference seconds, not JPEG bytes
        users = tuple(ObservableUser(user, locations[user], self._last[user],
                      self._uploads.existing_wait(user, now), tuple(requests[user]))
                      for user in range(self._schedule.users))
        self._pending = DecisionObservation(self._index, now, self._feedback.retrieve(),
                                             users, math.fsum(demand))
        return self._pending

    def apply(self, portfolios):
        if self._failed or self._finished or self._pending is None:
            raise RuntimeError('Must observe current decision before applying; no double apply')
        if len(portfolios) != self._schedule.users:
            raise ValueError('A portfolio for every user is required')
        probabilities = []
        for portfolio in portfolios:
            values = tuple(float(value) for value in portfolio)
            total = math.fsum(values)
            if (len(values) != len(self._edges) or not all(math.isfinite(value) and value >= 0 for value in values)
                    or not math.isclose(total, 1., abs_tol=1e-7, rel_tol=0.)):
                raise ValueError('Finite nonnegative normalized portfolios required')
            probabilities.append(tuple(value / total for value in values))
        pending, rows = self._pending, self._batches[self._index]
        views = {request.task_id: request for user in pending.users for request in user.requests}
        # Mark consumed before execution. A partial runtime fault is not an
        # invitation to append/auto-retry the same action batch.
        self._pending = None
        try:
            for row in rows:
                probability = probabilities[row.user]
                cumulative, chosen = 0., len(probability) - 1
                for server, value in enumerate(probability):
                    cumulative += value
                    if row.assignment_uniform < cumulative:
                        chosen = server
                        break
                view = views[row.task_id]
                upload = self._uploads.enqueue(row.user, pending.physical_seconds,
                                               view.transmission_seconds[chosen])
                pair = self._catalog.measured_pair_seconds(row.image_id, row.measured_session)
                self._feedback.submit_execution(Request(row.task_id, pending.physical_seconds, upload,
                              chosen, pair[self._profiles[chosen]], view.metadata.payload_bytes, row.image_id),
                              view.predicted_service_seconds[chosen])
                self._issued[row.task_id] = row
        except Exception:
            self._failed = True
            raise
        self._last = tuple(probabilities)

    def finish(self, drain_seconds):
        if self._failed or self._finished or self._pending is not None or self._index != self._schedule.decisions:
            raise RuntimeError('Every observed decision needs an action before drain')
        if not math.isfinite(drain_seconds) or drain_seconds < 0:
            raise ValueError('Common explicit drain duration required')
        self._finished = True  # no outcome-driven extension of a completed drain
        self._feedback.advance(self._schedule.horizon_seconds + drain_seconds)
        diagnostics = self._feedback.execution_diagnostics()
        completed = self._feedback.execution_completion_times()
        # FIFO arrival is issuance at the decision (must not submit in its past).
        # True latency includes the wait from exogenous arrival to that decision.
        true_latencies = tuple(time - self._issued[task].arrival_seconds
                               for task, time in completed.items() if time is not None)
        issuance_latencies = tuple(diagnostics['tasks']['latency_seconds'])
        diagnostics['tasks']['latency_seconds'] = list(true_latencies)
        return {'schedule_sha256': self._schedule.fingerprint(), 'tasks': diagnostics['tasks'],
                'work': diagnostics['work'], 'uncapped_latency_seconds': true_latencies,
                'issuance_latency_seconds_diagnostic_only': issuance_latencies,
                'decision_wait_seconds': tuple(self._schedule.decision_time(self._schedule.decision_index(row))
                                              - row.arrival_seconds for row in self._issued.values())}
