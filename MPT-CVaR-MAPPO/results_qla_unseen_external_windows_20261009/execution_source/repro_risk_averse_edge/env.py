from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SystemConfig
from .data import TraceSet


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

    @property
    def latency_s(self) -> float:
        return self.tran_s + self.queue_s + self.comp_s


class RiskAverseEdgeEnv:
    """Paper-faithful MEC offloading environment.

    This is a single-agent/shared-policy environment. Each `act_for_current_user`
    call consumes one user decision in a random per-slot order. The policy emits
    a portfolio over edge servers; the environment samples one server per task.
    """

    def __init__(
        self,
        config: SystemConfig,
        traces: TraceSet,
        request_distribution: str = "uniform",
        seed: int = 42,
        queue_constraint: str = "resample",
    ) -> None:
        self.cfg = config
        self.traces = traces
        self.request_distribution = request_distribution
        self.queue_constraint = queue_constraint
        self.rng = np.random.default_rng(seed)
        self.slot_index = 0
        self.user_order: list[int] = []
        self.order_pos = 0
        self.queues_mb = np.zeros(self.cfg.n_edges, dtype=np.float32)
        self.requests = np.zeros(self.cfg.n_users, dtype=np.int64)
        self.current_tasks: list[TaskRecord] = []
        self.completed_latencies: list[float] = []
        self.last_slot_records: list[TaskRecord] = []
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
        self.queues_mb.fill(0.0)
        self.completed_latencies.clear()
        self.last_slot_records.clear()
        self._start_slot()
        return self.observe()

    def _sample_requests(self) -> np.ndarray:
        high = self.cfg.max_requests_per_user
        if self.request_distribution == "uniform":
            return self.rng.integers(0, high + 1, size=self.cfg.n_users)
        if self.request_distribution == "gaussian":
            values = np.rint(self.rng.normal(loc=high / 2.0, scale=high / 6.0, size=self.cfg.n_users))
            return np.clip(values, 0, high).astype(np.int64)
        raise ValueError(f"Unknown request distribution: {self.request_distribution}")

    def _start_slot(self) -> None:
        self.requests = self._sample_requests()
        self.user_order = self.rng.permutation(self.cfg.n_users).tolist()
        self.order_pos = 0
        self.current_tasks = []

    @property
    def current_user(self) -> int:
        return self.user_order[self.order_pos]

    def observe(self) -> np.ndarray:
        user = self.current_user
        loc = self.traces.trajectories[user, self.slot_index % self.traces.trajectories.shape[1]]
        distances = np.linalg.norm(self.traces.edge_positions - loc[None, :], axis=1)
        k = int(self.requests[user])
        req = np.zeros(1 + 2 * self.cfg.max_requests_per_user, dtype=np.float32)
        req[0] = k / max(1, self.cfg.max_requests_per_user)
        for idx in range(k):
            base = 1 + 2 * idx
            req[base] = 1.0
            req[base + 1] = 1.0
        loc_norm = (loc.astype(np.float32) - self.coord_min) / self.coord_span
        distances_norm = distances.astype(np.float32) / max(self.max_distance, 1.0)
        queues_norm = self.queues_mb.astype(np.float32) / self.cfg.queue_upper_mb
        obs = np.concatenate(
            [
                loc_norm,
                distances_norm,
                req,
                queues_norm,
            ]
        )
        return obs.astype(np.float32)

    def action_mask(self) -> np.ndarray:
        # Edge server admissibility under q_upper for at least one new task.
        return (self.queues_mb + self.cfg.task_compute_mb <= self.cfg.queue_upper_mb).astype(np.float32)

    def act_for_current_user(self, weights: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        user = self.current_user
        weights = self._normalize_weights(weights)
        k = int(self.requests[user])
        records: list[TaskRecord] = []
        for task_idx in range(k):
            edge = self._sample_edge(weights)
            loc = self.traces.trajectories[user, self.slot_index % self.traces.trajectories.shape[1]]
            tran_s = self._transmission_delay_s(loc, edge)
            record = TaskRecord(
                user=user,
                task=task_idx,
                edge=edge,
                upload_s=tran_s,
                package_mb=self.cfg.task_package_mb,
                compute_mb=self.cfg.task_compute_mb,
                tran_s=tran_s,
                comp_s=self.cfg.task_compute_mb / self.cfg.edge_compute_mb_s,
            )
            records.append(record)
            self.current_tasks.append(record)

        self.order_pos += 1
        slot_done = self.order_pos >= self.cfg.n_users
        info: dict = {
            "user": user,
            "n_tasks": k,
            "sampled_edges": [r.edge for r in records],
            "portfolio": weights.copy(),
        }
        user_latency_sum = 0.0
        if slot_done:
            finished_user_order = list(self.user_order)
            self._finish_slot()
            user_latency_sum = sum(r.latency_s for r in self.last_slot_records if r.user == user)
            per_user_latency = {}
            per_user_mean_latency = {}
            per_user_tasks = {}
            for uid in range(self.cfg.n_users):
                user_records = [r for r in self.last_slot_records if r.user == uid]
                per_user_tasks[uid] = len(user_records)
                total_latency = float(sum(r.latency_s for r in user_records))
                per_user_latency[uid] = total_latency
                per_user_mean_latency[uid] = total_latency / max(1, len(user_records))
            info["slot_latencies"] = [r.latency_s for r in self.last_slot_records]
            info["per_user_latency"] = per_user_latency
            info["per_user_mean_latency"] = per_user_mean_latency
            info["per_user_tasks"] = per_user_tasks
            info["slot_tasks"] = len(self.last_slot_records)
            info["slot_user_order"] = finished_user_order
            self.slot_index = (self.slot_index + 1) % self.traces.trajectories.shape[1]
            self._start_slot()
        else:
            user_latency_sum = 0.0

        next_obs = self.observe()
        mean_user_latency = user_latency_sum / max(1, k)
        return next_obs, mean_user_latency, False, info

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != self.cfg.n_edges:
            raise ValueError(f"Expected {self.cfg.n_edges} portfolio weights, got {w.size}")
        mask = self.action_mask().astype(bool)
        w = np.where(mask, np.maximum(w, 0.0), 0.0)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = mask.astype(np.float64)
        if w.sum() <= 0:
            w = np.ones(self.cfg.n_edges, dtype=np.float64)
        return (w / w.sum()).astype(np.float32)

    def _sample_edge(self, weights: np.ndarray) -> int:
        # Queue upper bound is enforced against initial queue + already sampled
        # tasks in this slot, measured in computing load MB.
        slot_load = np.zeros(self.cfg.n_edges, dtype=np.float32)
        for record in self.current_tasks:
            slot_load[record.edge] += record.compute_mb
        feasible = self.queues_mb + slot_load + self.cfg.task_compute_mb <= self.cfg.queue_upper_mb
        probs = np.where(feasible, weights, 0.0)
        if probs.sum() <= 0:
            if self.queue_constraint == "penalty":
                probs = weights
            else:
                probs = np.ones(self.cfg.n_edges, dtype=np.float64) / self.cfg.n_edges
        probs = probs / probs.sum()
        return int(self.rng.choice(self.cfg.n_edges, p=probs))

    def _throughput_mb_s(self, loc: np.ndarray, edge: int) -> float:
        distance = max(float(np.linalg.norm(self.traces.edge_positions[edge] - loc)), self.cfg.min_distance_m)
        h = self.cfg.antenna_gain * (self.cfg.light_speed_m_s / (4.0 * np.pi * self.cfg.carrier_hz * distance)) ** 2
        snr = self.cfg.tx_power_w * h / self.cfg.noise_w ** 2
        bits_s = self.cfg.bandwidth_hz * np.log2(1.0 + snr)
        return float(bits_s / 8.0 / 1024.0 / 1024.0)

    def _transmission_delay_s(self, loc: np.ndarray, edge: int) -> float:
        return self.cfg.task_package_mb / max(self._throughput_mb_s(loc, edge), 1e-12)

    def _finish_slot(self) -> None:
        records = self.current_tasks
        for edge in range(self.cfg.n_edges):
            edge_records = [r for r in records if r.edge == edge]
            edge_records.sort(key=lambda r: r.upload_s)
            waiting_load = float(self.queues_mb[edge])
            for record in edge_records:
                record.queue_s = waiting_load / self.cfg.edge_compute_mb_s
                waiting_load += record.compute_mb

        arrivals = np.zeros(self.cfg.n_edges, dtype=np.float32)
        for record in records:
            arrivals[record.edge] += record.compute_mb
        self.queues_mb = np.maximum(
            0.0,
            self.queues_mb + arrivals - self.cfg.edge_compute_mb_s * self.cfg.slot_s,
        ).astype(np.float32)
        self.queues_mb = np.minimum(self.queues_mb, self.cfg.queue_upper_mb)
        self.last_slot_records = list(records)
        self.completed_latencies.extend([r.latency_s for r in records])
