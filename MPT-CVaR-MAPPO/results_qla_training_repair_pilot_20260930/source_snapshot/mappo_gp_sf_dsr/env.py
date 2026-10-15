from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from repro_risk_averse_edge.data import TraceSet

from .config import MAPPOSystemConfig, ServerScoreConfig
from .canonical_protocol import CanonicalSchedule
from .feedback_buffer import FeedbackBuffer


@dataclass
class TaskRecord:
    user: int
    task: int
    edge: int
    upload_s: float
    package_mb: float
    compute_mb: float
    tran_s: float = 0.0
    queue_s: float = 0.0
    comp_s: float = 0.0
    dropped: bool = False
    drop_penalty_s: float = 10.0

    @property
    def latency_s(self) -> float:
        return self.tran_s + self.queue_s + self.comp_s

    @property
    def training_latency_s(self) -> float:
        return float(self.drop_penalty_s if self.dropped else self.latency_s)


@dataclass
class ServerFeedback:
    score: np.ndarray
    queue_task_count_norm: np.ndarray
    available_resource_norm: np.ndarray
    capacity_norm: np.ndarray
    wait_norm: np.ndarray
    available_resource_mb: np.ndarray
    wait_s: np.ndarray
    # Slot at which this summary was generated.  The timestamp travels with
    # the feedback object through FeedbackBuffer; a global "last broadcast"
    # timestamp is not sufficient when a delayed buffer entry is selected.
    source_slot: int = 0


@dataclass
class QueueWorkItem:
    """One FCFS item stored in a server queue.

    Task workload is atomic at admission.  Background workload is divisible
    and is tracked separately so it is never counted as an unfinished task.
    """

    remaining_mb: float
    is_task: bool


@dataclass
class AdmissionResult:
    queue_before_mb: np.ndarray
    served_mb: np.ndarray
    task_offered_mb: np.ndarray
    task_accepted_mb: np.ndarray
    task_dropped_mb: np.ndarray
    background_offered_mb: np.ndarray
    background_accepted_mb: np.ndarray
    background_dropped_mb: np.ndarray
    queue_after_mb: np.ndarray
    n_dropped_tasks: int
    mass_ok: bool
    mass_error: float

    @property
    def offered_mb(self) -> np.ndarray:
        return self.task_offered_mb + self.background_offered_mb

    @property
    def accepted_mb(self) -> np.ndarray:
        return self.task_accepted_mb + self.background_accepted_mb

    @property
    def dropped_mb(self) -> np.ndarray:
        return self.task_dropped_mb + self.background_dropped_mb


@dataclass
class PortfolioStats:
    entropy: np.ndarray
    concentration: np.ndarray
    mean_entropy: float
    mean_concentration: float
    demand: np.ndarray
    demand_share: np.ndarray
    global_demand_concentration: float
    max_server_expected_demand_share: float


@dataclass
class SlotResult:
    records: list[TaskRecord]
    per_user_mean_latency: np.ndarray
    per_user_total_latency: np.ndarray
    per_user_tasks: np.ndarray
    system_average_latency: float
    portfolios: np.ndarray
    portfolio_stats: PortfolioStats
    server_feedback: ServerFeedback
    server_assignment_counts: np.ndarray
    server_arrival_load_mb: np.ndarray
    queue_load_before_mb: np.ndarray
    queue_load_after_mb: np.ndarray
    queue_task_count_after: np.ndarray
    overflow_count: int
    decision_latency_s: float
    served_load_mb: np.ndarray | None = None
    accepted_load_mb: np.ndarray | None = None
    dropped_load_mb: np.ndarray | None = None
    background_load_mb: np.ndarray | None = None
    n_dropped_tasks: int = 0
    mass_conservation_ok: bool = True
    mass_conservation_error: float = 0.0
    task_offered_load_mb: np.ndarray | None = None
    task_accepted_load_mb: np.ndarray | None = None
    task_dropped_load_mb: np.ndarray | None = None
    background_accepted_load_mb: np.ndarray | None = None
    background_dropped_load_mb: np.ndarray | None = None
    slot_index: int = 0

    @property
    def latencies(self) -> list[float]:
        return [record.latency_s for record in self.records if not record.dropped]

    @property
    def accepted_records(self) -> list[TaskRecord]:
        return [record for record in self.records if not record.dropped]

    @property
    def dropped_records(self) -> list[TaskRecord]:
        return [record for record in self.records if record.dropped]


@dataclass
class DrainResult:
    """Outcome of the drain-only episode tail (Phase 0.2).

    The drain phase continues stepping with `generating=False` (no new arrivals)
    until every server queue is empty or `max_horizon` is reached. Unfinished
    workload is reported so CVaR can be reported as uncapped latency plus an
    explicit drop probability instead of being silently biased low.
    """

    drain_horizon: int
    finished: bool
    n_unfinished_tasks: int
    workload_unfinished_mb_per_server: np.ndarray
    workload_served_mb_total: float
    workload_served_mb_during_drain: float = 0.0


class JointMAPPOEdgeEnv:
    """Joint multi-user extension of the paper MEC environment.

    All users emit portfolios for the same slot first. The environment then
    samples task-level server assignments and computes FCFS delays jointly.
    """

    def __init__(
        self,
        config: MAPPOSystemConfig,
        traces: TraceSet,
        request_distribution: str = "uniform",
        seed: int = 42,
        queue_constraint: str = "resample",
        score_config: ServerScoreConfig | None = None,
        feedback_delay: int = 0,
        broadcast_period: int = 1,
        poisson_lambda: float | None = None,
        burst_prob: float = 0.0,
        burst_multiplier: float = 2.0,
        edge_capacities_mb_s: np.ndarray | None = None,
        background_load: float = 0.0,
    ) -> None:
        self.cfg = config
        self.traces = traces
        self.request_distribution = request_distribution
        self.queue_constraint = queue_constraint
        self.score_config = score_config or ServerScoreConfig()
        self.feedback_delay = max(0, int(feedback_delay))
        self.broadcast_period = max(1, int(broadcast_period))
        self.feedback_buffer: FeedbackBuffer[ServerFeedback] = FeedbackBuffer(self.feedback_delay)
        self.poisson_lambda = poisson_lambda
        self.burst_prob = max(0.0, float(burst_prob))
        self.burst_multiplier = max(1.0, float(burst_multiplier))
        self.edge_capacities_mb_s = (
            np.asarray(edge_capacities_mb_s, dtype=np.float32)
            if edge_capacities_mb_s is not None
            else np.full(self.cfg.n_edges, self.cfg.edge_compute_mb_s, dtype=np.float32)
        )
        if self.edge_capacities_mb_s.shape != (self.cfg.n_edges,):
            raise ValueError(f"edge_capacities_mb_s must have shape {(self.cfg.n_edges,)}")
        self.background_load = max(0.0, float(background_load))
        self.last_broadcast_feedback: ServerFeedback | None = None
        self.last_broadcast_slot_index: int = -1
        self.rng = np.random.default_rng(seed)
        self.slot_index = 0
        self.queues_mb = np.zeros(self.cfg.n_edges, dtype=np.float32)
        self.queue_task_counts = np.zeros(self.cfg.n_edges, dtype=np.int64)
        self._queue_items: list[deque[QueueWorkItem]] = [deque() for _ in range(self.cfg.n_edges)]
        self.requests = np.zeros(self.cfg.n_users, dtype=np.int64)
        self.completed_latencies: list[float] = []
        self.last_result: SlotResult | None = None
        self.generating: bool = True
        self.workload_generated_mb_total = np.zeros(self.cfg.n_edges, dtype=np.float64)
        self.workload_accepted_mb_total = np.zeros(self.cfg.n_edges, dtype=np.float64)
        self.workload_served_mb_total = np.zeros(self.cfg.n_edges, dtype=np.float64)
        self.workload_dropped_mb_total = np.zeros(self.cfg.n_edges, dtype=np.float64)
        self.n_generated_tasks_total: int = 0
        self.n_accepted_tasks_total: int = 0
        self.n_dropped_tasks_total: int = 0
        self.split_start_slot: int | None = None
        self.split_n_slots: int | None = None
        self.canonical_schedule: CanonicalSchedule | None = None
        self.schedule_cursor: int = 0

        coords = np.concatenate(
            [self.traces.trajectories.reshape(-1, 2), self.traces.edge_positions.reshape(-1, 2)],
            axis=0,
        )
        self.coord_min = coords.min(axis=0).astype(np.float32)
        self.coord_span = np.maximum(coords.max(axis=0) - self.coord_min, 1.0).astype(np.float32)
        self.max_distance = float(np.linalg.norm(self.coord_span))
        self.reset()

    def reset(self) -> np.ndarray:
        self.slot_index = 0
        self.schedule_cursor = 0
        self.queues_mb.fill(0.0)
        self.queue_task_counts.fill(0)
        self._queue_items = [deque() for _ in range(self.cfg.n_edges)]
        self.completed_latencies.clear()
        self.last_result = None
        self.generating = True
        self.workload_generated_mb_total.fill(0.0)
        self.workload_accepted_mb_total.fill(0.0)
        self.workload_served_mb_total.fill(0.0)
        self.workload_dropped_mb_total.fill(0.0)
        self.n_generated_tasks_total = 0
        self.n_accepted_tasks_total = 0
        self.n_dropped_tasks_total = 0
        self.requests = self._sample_requests()
        self.feedback_buffer.clear()
        self.last_broadcast_feedback = self.server_feedback()
        self.last_broadcast_slot_index = 0
        self.feedback_buffer.push(self.last_broadcast_feedback)
        return self.actor_observations()

    def install_canonical_schedule(self, schedule: CanonicalSchedule) -> None:
        if schedule.requests.shape[1] != self.cfg.n_users:
            raise ValueError("Schedule user count does not match environment")
        if schedule.assignment_uniforms.shape[1:] != (
            self.cfg.n_users,
            self.cfg.max_requests_per_user,
        ):
            raise ValueError("Schedule assignment-uniform shape does not match environment")
        if schedule.trace_indices.size and np.max(schedule.trace_indices) >= self.traces.trajectories.shape[1]:
            raise ValueError("Schedule trace index exceeds available trajectory")
        self.canonical_schedule = schedule
        self.reset()

    def _sample_requests(self) -> np.ndarray:
        if self.canonical_schedule is not None:
            if self.schedule_cursor >= self.canonical_schedule.n_steps:
                return np.zeros(self.cfg.n_users, dtype=np.int64)
            return self.canonical_schedule.requests[self.schedule_cursor].astype(np.int64).copy()
        high = self.cfg.max_requests_per_user
        if self.request_distribution == "uniform":
            return self.rng.integers(0, high + 1, size=self.cfg.n_users, dtype=np.int64)
        if self.request_distribution == "gaussian":
            values = np.rint(self.rng.normal(loc=high / 2.0, scale=high / 6.0, size=self.cfg.n_users))
            return np.clip(values, 0, high).astype(np.int64)
        if self.request_distribution in {"poisson", "poisson_bursty", "bursty"}:
            lam = float(self.poisson_lambda if self.poisson_lambda is not None else high / 2.0)
            if self.request_distribution in {"poisson_bursty", "bursty"} and self.rng.random() < self.burst_prob:
                lam *= self.burst_multiplier
            values = self.rng.poisson(lam=lam, size=self.cfg.n_users)
            return np.clip(values, 0, high).astype(np.int64)
        raise ValueError(f"Unknown request distribution: {self.request_distribution}")

    def _locations(self) -> np.ndarray:
        if self.canonical_schedule is not None and self.canonical_schedule.n_steps > 0:
            cursor = min(self.schedule_cursor, self.canonical_schedule.n_steps - 1)
            trace_index = int(self.canonical_schedule.trace_indices[cursor])
            return self.traces.trajectories[:, trace_index]
        if self.split_start_slot is None or self.split_n_slots is None:
            return self.traces.trajectories[:, self.slot_index % self.traces.trajectories.shape[1]]
        # Split-aware: no wrap-around. slot_index is already an absolute offset
        # into [split_start_slot, split_start_slot + split_n_slots).
        return self.traces.trajectories[:, self.split_start_slot + self.slot_index]

    def _advance_slot(self) -> None:
        """Move slot_index forward by one, honoring split bounds when set.

        With split_start_slot/split_n_slots set (Phase 1 schedule mode), the
        index stops at split_n_slots and the env switches to ``generating=False``
        so the caller can run drain(). Without a split, legacy wrap-around
        behavior is preserved.
        """
        if self.canonical_schedule is not None:
            self.schedule_cursor += 1
            self.slot_index += 1
            if self.schedule_cursor >= self.canonical_schedule.n_steps:
                self.generating = False
            return
        if self.split_start_slot is None or self.split_n_slots is None:
            self.slot_index = (self.slot_index + 1) % self.traces.trajectories.shape[1]
            return
        if self.slot_index + 1 >= self.split_n_slots:
            self.generating = False
            self.slot_index = self.split_n_slots - 1
        else:
            self.slot_index += 1

    def _server_feedback_from_state(
        self,
        queues_mb: np.ndarray,
        queue_task_counts: np.ndarray | None = None,
        source_slot: int | None = None,
    ) -> ServerFeedback:
        eps = self.score_config.eps
        remain = np.asarray(queues_mb, dtype=np.float64)
        capacity = self.edge_capacities_mb_s.astype(np.float64)
        available = np.maximum(0.0, capacity * self.cfg.slot_s - remain)
        wait = remain / np.maximum(capacity, eps)

        if queue_task_counts is None:
            task_count = np.ceil(remain / max(self.cfg.task_compute_mb, eps))
        else:
            task_count = np.asarray(queue_task_counts, dtype=np.float64)
        task_norm = self._norm_by_max(task_count, eps)
        avail_norm = self._norm_by_max(available, eps)
        cap_norm = self._norm_by_max(capacity, eps)
        wait_norm = self._norm_by_max(wait, eps)

        score = (
            self.score_config.resource_weight * avail_norm
            + self.score_config.capacity_weight * cap_norm
            - self.score_config.task_count_weight * task_norm
            - self.score_config.wait_weight * wait_norm
        )
        return ServerFeedback(
            score=score.astype(np.float32),
            queue_task_count_norm=task_norm.astype(np.float32),
            available_resource_norm=avail_norm.astype(np.float32),
            capacity_norm=cap_norm.astype(np.float32),
            wait_norm=wait_norm.astype(np.float32),
            available_resource_mb=available.astype(np.float32),
            wait_s=wait.astype(np.float32),
            source_slot=int(self.slot_index if source_slot is None else source_slot),
        )

    def server_feedback(self) -> ServerFeedback:
        return self._server_feedback_from_state(
            self.queues_mb,
            self.queue_task_counts,
            source_slot=self.slot_index,
        )

    def feedback_source_age(self, feedback: ServerFeedback | None = None) -> int:
        """Return the actual source age of the selected broadcast summary."""
        selected = feedback if feedback is not None else self.feedback_buffer.get(self.feedback_delay)
        return max(0, int(self.slot_index) - int(selected.source_slot))

    @staticmethod
    def _norm_by_max(values: np.ndarray, eps: float) -> np.ndarray:
        max_value = float(np.max(values)) if values.size else 0.0
        if max_value <= eps:
            return np.zeros_like(values, dtype=np.float64)
        return values / (max_value + eps)

    def actor_observation_for_user(self, user: int, feedback: ServerFeedback | None = None) -> np.ndarray:
        feedback = feedback if feedback is not None else self.feedback_buffer.get(self.feedback_delay)
        locs = self._locations()
        loc = locs[user]
        distances = np.linalg.norm(self.traces.edge_positions - loc[None, :], axis=1)
        req = self._request_summary(user)
        loc_norm = (loc.astype(np.float32) - self.coord_min) / self.coord_span
        distances_norm = distances.astype(np.float32) / max(self.max_distance, 1.0)
        parts = [loc_norm, distances_norm, req, feedback.score]
        if self.cfg.include_feedback_age:
            age = float(self.feedback_source_age(feedback)) / max(float(self.cfg.feedback_age_norm), 1e-8)
            parts.append(np.asarray([age], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def actor_observations(self) -> np.ndarray:
        feedback = self.feedback_buffer.get(self.feedback_delay)
        obs = [self.actor_observation_for_user(user, feedback=feedback) for user in range(self.cfg.n_users)]
        return np.asarray(obs, dtype=np.float32)

    def _request_summary(self, user: int) -> np.ndarray:
        k = float(self.requests[user])
        max_k = float(max(1, self.cfg.max_requests_per_user))
        return np.asarray(
            [
                k / max_k,
                (k * self.cfg.task_package_mb) / (max_k * self.cfg.task_package_mb),
                (k * self.cfg.task_compute_mb) / (max_k * self.cfg.task_compute_mb),
            ],
            dtype=np.float32,
        )

    def portfolio_stats(self, portfolios: np.ndarray) -> PortfolioStats:
        w = self._normalize_portfolios(portfolios)
        eps = 1e-8
        entropy = -np.sum(w * np.log(w + eps), axis=1) / np.log(self.cfg.n_edges)
        raw_conc = np.sum(w * w, axis=1)
        concentration = (raw_conc - 1.0 / self.cfg.n_edges) / (1.0 - 1.0 / self.cfg.n_edges)
        concentration = np.clip(concentration, 0.0, 1.0)

        user_loads = self.requests.astype(np.float64) * self.cfg.task_compute_mb
        demand = np.sum(user_loads[:, None] * w, axis=0)
        demand_total = float(np.sum(demand))
        demand_share = demand / (demand_total + eps) if demand_total > eps else np.zeros_like(demand)
        global_raw = float(np.sum(demand_share * demand_share))
        global_conc = (global_raw - 1.0 / self.cfg.n_edges) / (1.0 - 1.0 / self.cfg.n_edges)
        global_conc = float(np.clip(global_conc, 0.0, 1.0))
        return PortfolioStats(
            entropy=entropy.astype(np.float32),
            concentration=concentration.astype(np.float32),
            mean_entropy=float(np.mean(entropy)) if entropy.size else 0.0,
            mean_concentration=float(np.mean(concentration)) if concentration.size else 0.0,
            demand=demand.astype(np.float32),
            demand_share=demand_share.astype(np.float32),
            global_demand_concentration=global_conc,
            max_server_expected_demand_share=float(np.max(demand_share)) if demand_share.size else 0.0,
        )

    def critic_observations(
        self,
        portfolios: np.ndarray,
        stats: PortfolioStats | None = None,
        feedback: ServerFeedback | None = None,
    ) -> np.ndarray:
        stats = stats or self.portfolio_stats(portfolios)
        feedback = feedback or self.server_feedback()
        server_part = np.concatenate(
            [
                feedback.score,
                feedback.queue_task_count_norm,
                feedback.available_resource_norm,
                feedback.capacity_norm,
                feedback.wait_norm,
            ]
        ).astype(np.float32)
        system = self._system_summary()
        locs = self._locations()
        rows = []
        for user in range(self.cfg.n_users):
            loc_norm = (locs[user].astype(np.float32) - self.coord_min) / self.coord_span
            req = self._request_summary(user)
            user_feature = np.concatenate([loc_norm, req]).astype(np.float32)
            portfolio_part = np.asarray(
                [
                    stats.mean_entropy,
                    stats.mean_concentration,
                    stats.global_demand_concentration,
                    stats.max_server_expected_demand_share,
                    float(stats.entropy[user]),
                    float(stats.concentration[user]),
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate([server_part, system, portfolio_part, user_feature]))
        return np.asarray(rows, dtype=np.float32)

    def _system_summary(self) -> np.ndarray:
        max_tasks = max(1, self.cfg.n_users * self.cfg.max_requests_per_user)
        total_tasks = float(np.sum(self.requests))
        mean_tasks = float(np.mean(self.requests)) / max(1, self.cfg.max_requests_per_user)
        max_user_tasks = float(np.max(self.requests)) / max(1, self.cfg.max_requests_per_user)
        total_data = total_tasks * self.cfg.task_package_mb
        total_compute = total_tasks * self.cfg.task_compute_mb
        max_total_data = max_tasks * self.cfg.task_package_mb
        max_total_compute = max_tasks * self.cfg.task_compute_mb
        return np.asarray(
            [
                mean_tasks,
                max_user_tasks,
                total_tasks / max_tasks,
                total_data / max(max_total_data, 1e-8),
                total_compute / max(max_total_compute, 1e-8),
                float(np.max(self.queues_mb / self.cfg.queue_upper_mb)),
            ],
            dtype=np.float32,
        )

    def joint_step(
        self,
        portfolios: np.ndarray,
        stats: PortfolioStats | None = None,
        decision_latency_s: float | None = None,
    ) -> SlotResult:
        start = perf_counter()
        weights = self._normalize_portfolios(portfolios)
        feedback = self.server_feedback()
        stats = stats or self.portfolio_stats(weights)
        locs = self._locations()
        records: list[TaskRecord] = []
        slot_load = np.zeros(self.cfg.n_edges, dtype=np.float32)
        assignment_counts = np.zeros(self.cfg.n_edges, dtype=np.int64)
        overflow_count = 0

        for user in range(self.cfg.n_users):
            for task_idx in range(int(self.requests[user])):
                edge, overflowed = self._sample_edge(weights[user], slot_load, user=user, task_idx=task_idx)
                overflow_count += int(overflowed)
                tran_s = self._transmission_delay_s(locs[user], edge)
                records.append(
                    TaskRecord(
                        user=user,
                        task=task_idx,
                        edge=edge,
                        upload_s=tran_s,
                        package_mb=self.cfg.task_package_mb,
                        compute_mb=self.cfg.task_compute_mb,
                        tran_s=tran_s,
                    comp_s=self.cfg.task_compute_mb / max(float(self.edge_capacities_mb_s[edge]), 1e-8),
                    )
                )
                slot_load[edge] += self.cfg.task_compute_mb
                assignment_counts[edge] += 1

        admission = self._serve_and_admit(records, slot_load, decision_latency_s)

        per_user_total = np.zeros(self.cfg.n_users, dtype=np.float32)
        per_user_tasks = np.zeros(self.cfg.n_users, dtype=np.int64)
        for record in records:
            per_user_total[record.user] += record.training_latency_s
            per_user_tasks[record.user] += 1
        per_user_mean = np.divide(
            per_user_total,
            np.maximum(per_user_tasks, 1),
            out=np.zeros_like(per_user_total),
            where=per_user_tasks > 0,
        )
        denom = int(np.sum(per_user_tasks))
        system_average = float(np.sum(per_user_total) / max(1, denom))

        result = SlotResult(
            records=records,
            per_user_mean_latency=per_user_mean,
            per_user_total_latency=per_user_total,
            per_user_tasks=per_user_tasks,
            system_average_latency=system_average,
            portfolios=weights.astype(np.float32),
            portfolio_stats=stats,
            server_feedback=feedback,
            server_assignment_counts=assignment_counts,
            server_arrival_load_mb=admission.offered_mb.astype(np.float32),
            queue_load_before_mb=admission.queue_before_mb.astype(np.float32),
            queue_load_after_mb=self.queues_mb.copy(),
            queue_task_count_after=self.queue_task_counts.copy(),
            overflow_count=overflow_count,
            decision_latency_s=float(decision_latency_s if decision_latency_s is not None else perf_counter() - start),
            served_load_mb=admission.served_mb.astype(np.float32),
            accepted_load_mb=admission.accepted_mb.astype(np.float32),
            dropped_load_mb=admission.dropped_mb.astype(np.float32),
            background_load_mb=admission.background_offered_mb.astype(np.float32),
            n_dropped_tasks=admission.n_dropped_tasks,
            mass_conservation_ok=admission.mass_ok,
            mass_conservation_error=admission.mass_error,
            task_offered_load_mb=admission.task_offered_mb.astype(np.float32),
            task_accepted_load_mb=admission.task_accepted_mb.astype(np.float32),
            task_dropped_load_mb=admission.task_dropped_mb.astype(np.float32),
            background_accepted_load_mb=admission.background_accepted_mb.astype(np.float32),
            background_dropped_load_mb=admission.background_dropped_mb.astype(np.float32),
            slot_index=int(self.slot_index),
        )
        self.last_result = result
        self.completed_latencies.extend(result.latencies)
        self._advance_slot()
        self.requests = self._sample_requests()
        true_feedback = self.server_feedback()
        if self.last_broadcast_feedback is None or self.slot_index % self.broadcast_period == 0:
            self.last_broadcast_feedback = true_feedback
            self.last_broadcast_slot_index = int(self.slot_index)
        self.feedback_buffer.push(self.last_broadcast_feedback)
        return result

    def sequential_step(
        self,
        policy_fn,
        decision_latency_s: float | None = None,
    ) -> SlotResult:
        """Execute one slot with sequential user decisions.

        Earlier users' sampled incoming load is exposed to later users when
        feedback_delay is zero. The physical queue update still happens once at
        the end of the slot, so this isolates the information/execution-order
        effect from extra service-time artifacts.
        """

        start = perf_counter()
        feedback = self.server_feedback()
        locs = self._locations()
        records: list[TaskRecord] = []
        slot_load = np.zeros(self.cfg.n_edges, dtype=np.float32)
        portfolios = np.zeros((self.cfg.n_users, self.cfg.n_edges), dtype=np.float32)
        assignment_counts = np.zeros(self.cfg.n_edges, dtype=np.int64)
        overflow_count = 0

        for user in range(self.cfg.n_users):
            if self.feedback_delay == 0:
                effective_queue = self.queues_mb + slot_load
                user_feedback = self._server_feedback_from_state(effective_queue)
            else:
                user_feedback = self.feedback_buffer.get(self.feedback_delay)
            obs = self.actor_observation_for_user(user, feedback=user_feedback)
            user_weights = np.asarray(policy_fn(user, obs), dtype=np.float32)
            if user_weights.shape != (self.cfg.n_edges,):
                raise ValueError(f"Sequential policy must return shape {(self.cfg.n_edges,)}, got {user_weights.shape}")
            user_weights = np.where(np.isfinite(user_weights), np.maximum(user_weights, 0.0), 0.0)
            if float(np.sum(user_weights)) <= 0.0:
                user_weights = np.ones(self.cfg.n_edges, dtype=np.float32)
            user_weights = (user_weights / np.sum(user_weights)).astype(np.float32)
            portfolios[user] = user_weights

            for task_idx in range(int(self.requests[user])):
                edge, overflowed = self._sample_edge(user_weights, slot_load, user=user, task_idx=task_idx)
                overflow_count += int(overflowed)
                tran_s = self._transmission_delay_s(locs[user], edge)
                records.append(
                    TaskRecord(
                        user=user,
                        task=task_idx,
                        edge=edge,
                        upload_s=tran_s,
                        package_mb=self.cfg.task_package_mb,
                        compute_mb=self.cfg.task_compute_mb,
                        tran_s=tran_s,
                        comp_s=self.cfg.task_compute_mb / max(float(self.edge_capacities_mb_s[edge]), 1e-8),
                    )
                )
                slot_load[edge] += self.cfg.task_compute_mb
                assignment_counts[edge] += 1

        stats = self.portfolio_stats(portfolios)
        admission = self._serve_and_admit(records, slot_load, decision_latency_s)

        per_user_total = np.zeros(self.cfg.n_users, dtype=np.float32)
        per_user_tasks = np.zeros(self.cfg.n_users, dtype=np.int64)
        for record in records:
            per_user_total[record.user] += record.training_latency_s
            per_user_tasks[record.user] += 1
        per_user_mean = np.divide(
            per_user_total,
            np.maximum(per_user_tasks, 1),
            out=np.zeros_like(per_user_total),
            where=per_user_tasks > 0,
        )
        denom = int(np.sum(per_user_tasks))
        system_average = float(np.sum(per_user_total) / max(1, denom))

        result = SlotResult(
            records=records,
            per_user_mean_latency=per_user_mean,
            per_user_total_latency=per_user_total,
            per_user_tasks=per_user_tasks,
            system_average_latency=system_average,
            portfolios=portfolios,
            portfolio_stats=stats,
            server_feedback=feedback,
            server_assignment_counts=assignment_counts,
            server_arrival_load_mb=admission.offered_mb.astype(np.float32),
            queue_load_before_mb=admission.queue_before_mb.astype(np.float32),
            queue_load_after_mb=self.queues_mb.copy(),
            queue_task_count_after=self.queue_task_counts.copy(),
            overflow_count=overflow_count,
            decision_latency_s=float(decision_latency_s if decision_latency_s is not None else perf_counter() - start),
            served_load_mb=admission.served_mb.astype(np.float32),
            accepted_load_mb=admission.accepted_mb.astype(np.float32),
            dropped_load_mb=admission.dropped_mb.astype(np.float32),
            background_load_mb=admission.background_offered_mb.astype(np.float32),
            n_dropped_tasks=admission.n_dropped_tasks,
            mass_conservation_ok=admission.mass_ok,
            mass_conservation_error=admission.mass_error,
            task_offered_load_mb=admission.task_offered_mb.astype(np.float32),
            task_accepted_load_mb=admission.task_accepted_mb.astype(np.float32),
            task_dropped_load_mb=admission.task_dropped_mb.astype(np.float32),
            background_accepted_load_mb=admission.background_accepted_mb.astype(np.float32),
            background_dropped_load_mb=admission.background_dropped_mb.astype(np.float32),
            slot_index=int(self.slot_index),
        )
        self.last_result = result
        self.completed_latencies.extend(result.latencies)
        self._advance_slot()
        self.requests = self._sample_requests()
        true_feedback = self.server_feedback()
        if self.last_broadcast_feedback is None or self.slot_index % self.broadcast_period == 0:
            self.last_broadcast_feedback = true_feedback
            self.last_broadcast_slot_index = int(self.slot_index)
        self.feedback_buffer.push(self.last_broadcast_feedback)
        return result

    def _serve_and_admit(
        self,
        records: list[TaskRecord],
        slot_load: np.ndarray,
        decision_latency_s: float | None = None,
    ) -> AdmissionResult:
        """Serve old backlog, then atomically admit tasks and background load.

        Canonical ordering is fixed as follows: old backlog is served first;
        newly uploaded tasks are considered FCFS by ``upload_s`` and are
        accepted only as whole tasks; divisible background workload uses the
        remaining buffer afterwards.  This ordering makes task counts,
        accepted workload, dropped workload, and the queue transition agree.
        ``slot_load`` is retained in the signature for compatibility and is
        validated against the task records in finite-buffer mode.
        """
        cfg = self.cfg
        eps = 1e-12
        self._ensure_queue_ledger()
        queue_before = self.queues_mb.astype(np.float64).copy()
        capacity = self.edge_capacities_mb_s.astype(np.float64)
        slot_s = float(cfg.slot_s)
        queue_max = float(cfg.queue_upper_mb)

        # 1. Service from existing FCFS backlog.
        served = self._serve_queue_items(capacity * slot_s)
        residual = self._queue_workload_from_ledger()

        # 2. Offered task and background arrivals.
        task_offered = np.zeros(self.cfg.n_edges, dtype=np.float64)
        for record in records:
            record.dropped = False
            task_offered[record.edge] += float(record.compute_mb)
        if cfg.finite_buffer_protocol:
            np.testing.assert_allclose(
                np.asarray(slot_load, dtype=np.float64),
                task_offered,
                rtol=1e-6,
                atol=max(float(cfg.finite_buffer_tolerance), 1e-7),
                err_msg="slot_load must equal workload represented by task records",
            )
        background_offered = (
            (self.background_load * capacity * slot_s)
            if self.background_load > 0.0
            else np.zeros(self.cfg.n_edges, dtype=np.float64)
        )
        free = np.maximum(0.0, queue_max - residual)

        task_accepted = np.zeros(self.cfg.n_edges, dtype=np.float64)
        task_dropped = np.zeros(self.cfg.n_edges, dtype=np.float64)
        background_accepted = np.zeros(self.cfg.n_edges, dtype=np.float64)
        background_dropped = np.zeros(self.cfg.n_edges, dtype=np.float64)
        n_dropped_tasks = 0

        # 3. Atomic task admission, followed by divisible background load.
        for edge in range(self.cfg.n_edges):
            edge_records = [r for r in records if r.edge == edge]
            edge_records.sort(key=lambda r: r.upload_s)
            waiting_load = float(residual[edge])
            cap_edge = float(capacity[edge])
            for record in edge_records:
                work = float(record.compute_mb)
                if cfg.finite_buffer_protocol and work > float(free[edge]) + eps:
                    record.dropped = True
                    record.drop_penalty_s = float(cfg.drop_penalty_s)
                    task_dropped[edge] += work
                    n_dropped_tasks += 1
                    continue
                free[edge] -= work
                task_accepted[edge] += work
                record.queue_s = waiting_load / max(cap_edge, eps)
                waiting_load += work
                self._queue_items[edge].append(QueueWorkItem(remaining_mb=work, is_task=True))

            bg = float(background_offered[edge])
            if cfg.finite_buffer_protocol:
                bg_accept = min(bg, max(0.0, float(free[edge])))
            else:
                bg_accept = bg
            background_accepted[edge] = bg_accept
            background_dropped[edge] = max(0.0, bg - bg_accept)
            if bg_accept > eps:
                self._queue_items[edge].append(QueueWorkItem(remaining_mb=bg_accept, is_task=False))

        # Legacy mode preserves its historical silent clipping only when
        # explicitly requested.  Canonical runs must set finite_buffer_protocol.
        if not cfg.finite_buffer_protocol:
            for edge in range(self.cfg.n_edges):
                overflow = max(0.0, self._queue_workload_for_edge(edge) - queue_max)
                if overflow > eps:
                    self._discard_queue_tail(edge, overflow)

        self._sync_queue_arrays_from_ledger()
        queue_after = self.queues_mb.astype(np.float64).copy()

        # 4. Workload conservation.  In canonical mode every dropped task is
        # represented by its full compute demand; no partial task is admitted.
        offered = task_offered + background_offered
        dropped = task_dropped + background_dropped
        lhs = queue_before + offered
        rhs = served + dropped + queue_after
        total_workload = float(np.maximum(lhs, rhs).max())
        mass_err = float(np.max(np.abs(lhs - rhs)))
        tol = max(cfg.finite_buffer_tolerance * max(1.0, total_workload), 1e-9)
        mass_ok = bool(mass_err <= tol)

        # 5. Episode accounting.
        accepted = task_accepted + background_accepted
        self.workload_generated_mb_total += offered
        self.workload_accepted_mb_total += accepted
        self.workload_served_mb_total += served
        self.workload_dropped_mb_total += dropped
        self.n_generated_tasks_total += len(records)
        self.n_accepted_tasks_total += sum(1 for r in records if not r.dropped)
        self.n_dropped_tasks_total += n_dropped_tasks

        return AdmissionResult(
            queue_before_mb=queue_before,
            served_mb=served,
            task_offered_mb=task_offered,
            task_accepted_mb=task_accepted,
            task_dropped_mb=task_dropped,
            background_offered_mb=background_offered,
            background_accepted_mb=background_accepted,
            background_dropped_mb=background_dropped,
            queue_after_mb=queue_after,
            n_dropped_tasks=n_dropped_tasks,
            mass_ok=mass_ok,
            mass_error=mass_err,
        )

    def _ensure_queue_ledger(self) -> None:
        """Synchronize manually injected queue arrays into the FCFS ledger.

        Production code mutates queues only through the ledger.  This fallback
        keeps diagnostics that seed an initial backlog well-defined without
        pretending that anonymous injected workload is a real task.
        """
        ledger = self._queue_workload_from_ledger()
        target = self.queues_mb.astype(np.float64)
        if np.allclose(ledger, target, rtol=0.0, atol=1e-7):
            return
        self._queue_items = [deque() for _ in range(self.cfg.n_edges)]
        for edge, amount in enumerate(target):
            if float(amount) > 0.0:
                self._queue_items[edge].append(QueueWorkItem(float(amount), is_task=False))
        self._sync_queue_arrays_from_ledger()

    def _queue_workload_for_edge(self, edge: int) -> float:
        return float(sum(max(0.0, item.remaining_mb) for item in self._queue_items[edge]))

    def _queue_workload_from_ledger(self) -> np.ndarray:
        return np.asarray(
            [self._queue_workload_for_edge(edge) for edge in range(self.cfg.n_edges)],
            dtype=np.float64,
        )

    def _sync_queue_arrays_from_ledger(self) -> None:
        self.queues_mb = self._queue_workload_from_ledger().astype(np.float32)
        self.queue_task_counts = np.asarray(
            [sum(1 for item in items if item.is_task and item.remaining_mb > 1e-12) for items in self._queue_items],
            dtype=np.int64,
        )

    def _serve_queue_items(self, service_budget_mb: np.ndarray) -> np.ndarray:
        served = np.zeros(self.cfg.n_edges, dtype=np.float64)
        for edge in range(self.cfg.n_edges):
            budget = max(0.0, float(service_budget_mb[edge]))
            items = self._queue_items[edge]
            while budget > 1e-12 and items:
                item = items[0]
                take = min(budget, max(0.0, float(item.remaining_mb)))
                item.remaining_mb -= take
                budget -= take
                served[edge] += take
                if item.remaining_mb <= 1e-12:
                    items.popleft()
        self._sync_queue_arrays_from_ledger()
        return served

    def _discard_queue_tail(self, edge: int, amount_mb: float) -> None:
        """Legacy-only silent truncation, isolated from canonical execution."""
        remaining = max(0.0, float(amount_mb))
        items = self._queue_items[edge]
        while remaining > 1e-12 and items:
            item = items[-1]
            take = min(remaining, max(0.0, float(item.remaining_mb)))
            item.remaining_mb -= take
            remaining -= take
            if item.remaining_mb <= 1e-12:
                items.pop()

    def drain(self, max_horizon: int | None = None) -> DrainResult:
        """Drain-only episode tail (Phase 0.2).

        Continues stepping with `generating=False` (no new arrivals) until
        either all queues hit zero or `max_horizon` is reached.
        """
        if max_horizon is None:
            max_horizon = int(self.cfg.drain_max_horizon)
        self.generating = False
        eps = 1e-12
        self._ensure_queue_ledger()
        capacity = self.edge_capacities_mb_s.astype(np.float64)
        slot_s = float(self.cfg.slot_s)
        horizon = 0
        served_before = float(np.sum(self.workload_served_mb_total))
        for _ in range(max_horizon):
            if float(np.max(self.queues_mb)) <= eps:
                break
            served = self._serve_queue_items(capacity * slot_s)
            self.workload_served_mb_total += served
            horizon += 1
        finished = bool(float(np.max(self.queues_mb)) <= eps)
        unfinished_mb = self.queues_mb.astype(np.float64).copy()
        n_unfinished = int(
            sum(
                1
                for items in self._queue_items
                for item in items
                if item.is_task and item.remaining_mb > eps
            )
        )
        served_total = float(np.sum(self.workload_served_mb_total))
        return DrainResult(
            drain_horizon=horizon,
            finished=finished,
            n_unfinished_tasks=n_unfinished,
            workload_unfinished_mb_per_server=unfinished_mb,
            workload_served_mb_total=served_total,
            workload_served_mb_during_drain=served_total - served_before,
        )

    def _normalize_portfolios(self, portfolios: np.ndarray) -> np.ndarray:
        w = np.asarray(portfolios, dtype=np.float64)
        if w.shape != (self.cfg.n_users, self.cfg.n_edges):
            raise ValueError(f"Expected portfolios shape {(self.cfg.n_users, self.cfg.n_edges)}, got {w.shape}")
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
        row_sums = w.sum(axis=1, keepdims=True)
        bad = row_sums.squeeze(-1) <= 0.0
        if np.any(bad):
            w[bad] = 1.0
            row_sums = w.sum(axis=1, keepdims=True)
        return w / row_sums

    def _sample_edge(
        self,
        weights: np.ndarray,
        slot_load: np.ndarray,
        *,
        user: int | None = None,
        task_idx: int | None = None,
    ) -> tuple[int, bool]:
        if self.cfg.finite_buffer_protocol:
            service = self.edge_capacities_mb_s * self.cfg.slot_s
            residual = np.maximum(0.0, self.queues_mb - service)
            feasible = residual + slot_load + self.cfg.task_compute_mb <= self.cfg.queue_upper_mb + 1e-8
        else:
            feasible = self.queues_mb + slot_load + self.cfg.task_compute_mb <= self.cfg.queue_upper_mb
        probs = np.where(feasible, weights, 0.0)
        overflowed = False
        if probs.sum() <= 0:
            overflowed = True
            if self.queue_constraint == "penalty":
                probs = weights
            else:
                probs = feasible.astype(np.float64)
        if probs.sum() <= 0:
            probs = np.ones(self.cfg.n_edges, dtype=np.float64)
        probs = probs / probs.sum()
        if (
            self.canonical_schedule is not None
            and user is not None
            and task_idx is not None
            and self.schedule_cursor < self.canonical_schedule.n_steps
        ):
            u = float(self.canonical_schedule.assignment_uniforms[self.schedule_cursor, user, task_idx])
            return int(np.searchsorted(np.cumsum(probs), min(u, np.nextafter(1.0, 0.0)), side="right")), overflowed
        return int(self.rng.choice(self.cfg.n_edges, p=probs)), overflowed

    def _throughput_mb_s(self, loc: np.ndarray, edge: int) -> float:
        distance = max(float(np.linalg.norm(self.traces.edge_positions[edge] - loc)), self.cfg.min_distance_m)
        h = self.cfg.antenna_gain * (self.cfg.light_speed_m_s / (4.0 * np.pi * self.cfg.carrier_hz * distance)) ** 2
        snr = self.cfg.tx_power_w * h / self.cfg.noise_w**2
        bits_s = self.cfg.bandwidth_hz * np.log2(1.0 + snr)
        return float(bits_s / 8.0 / 1024.0 / 1024.0)

    def _transmission_delay_s(self, loc: np.ndarray, edge: int) -> float:
        return self.cfg.task_package_mb / max(self._throughput_mb_s(loc, edge), 1e-12)
