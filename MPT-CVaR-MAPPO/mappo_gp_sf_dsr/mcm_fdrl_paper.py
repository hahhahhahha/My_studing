"""Paper-structure MCM-FDRL adapted to the repository's MEC environment.

The model and runtime flow follow Chen et al., "Multi-Agent Collaboration for
Vehicular Task Offloading Using Federated Deep Reinforcement Learning":

* local execution plus three nearest edge candidates;
* state ``<compute, load, transmission, density, current plan>``;
* six fixed add/decrease actions over the three adjacent edges;
* counterfactual reward ``latency_before - latency_after``;
* invalid-action masking in DQN targets and runtime decisions;
* two 128-unit ReLU hidden layers, Huber loss, replay and FedAvg; and
* runtime stopping when all valid Q-values are below ``tau``.

The physical queue and mobility interface is the current KAIST-driven MEC
environment.  Local execution is added only inside this isolated adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, ROOT, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv, SlotResult, TaskRecord
from .metrics import phase_diagnostics, slot_metrics, summarize_results
from .plots import make_training_plots
from .train import edge_capacities


METHOD = "MCM-FDRL-paper"


class PaperDQN(nn.Module):
    """Paper DQN: input, two 128-unit ReLU layers, output."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PaperReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_valid = np.zeros((capacity, action_dim), dtype=np.bool_)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_valid: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.next_valid[self.position] = next_valid
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "next_valid": self.next_valid[indices],
            "dones": self.dones[indices],
        }


@dataclass
class PaperDQNAgent:
    q: PaperDQN
    target: PaperDQN
    optimizer: torch.optim.Optimizer
    replay: PaperReplayBuffer


class LocalAdjacentMECAdapter:
    """Map the paper's local-plus-three-edge plan into the current simulator."""

    candidate_edges = 3
    plan_dim = 4

    def __init__(
        self,
        environment: JointMAPPOEdgeEnv,
        local_compute_ratio: float,
        density_radius: float,
    ) -> None:
        self.env = environment
        self.local_capacities = (
            np.full(environment.cfg.n_users, environment.cfg.edge_compute_mb_s * local_compute_ratio)
        ).astype(np.float32)
        self.local_queues = np.zeros(environment.cfg.n_users, dtype=np.float32)
        self.density_radius = float(density_radius)

    def reset(self) -> None:
        self.env.reset()
        self.local_queues.fill(0.0)

    def initial_plans(self) -> np.ndarray:
        plans = np.zeros((self.env.cfg.n_users, self.plan_dim), dtype=np.float32)
        plans[:, 0] = 1.0
        return plans

    def adjacent_edges(self) -> np.ndarray:
        locations = self.env._locations()
        distances = np.linalg.norm(
            locations[:, None, :] - self.env.traces.edge_positions[None, :, :],
            axis=-1,
        )
        return np.argsort(distances, axis=1)[:, : self.candidate_edges].astype(np.int64)

    def edge_densities(self, adjacent: np.ndarray) -> np.ndarray:
        locations = self.env._locations()
        edges = self.env.traces.edge_positions
        densities = np.zeros((self.env.cfg.n_users, self.candidate_edges), dtype=np.float32)
        for user in range(self.env.cfg.n_users):
            for index, edge in enumerate(adjacent[user]):
                center = edges[edge]
                vehicles = int(np.sum(np.linalg.norm(locations - center[None, :], axis=1) <= self.density_radius))
                nearby_edges = int(np.sum(np.linalg.norm(edges - center[None, :], axis=1) <= self.density_radius))
                densities[user, index] = vehicles / max(1, nearby_edges)
        return densities

    def states(self, plans: np.ndarray, adjacent: np.ndarray | None = None) -> np.ndarray:
        adjacent = self.adjacent_edges() if adjacent is None else adjacent
        densities = self.edge_densities(adjacent)
        locations = self.env._locations()
        rows: list[np.ndarray] = []
        for user in range(self.env.cfg.n_users):
            edge_ids = adjacent[user]
            edge_caps = self.env.edge_capacities_mb_s[edge_ids]
            compute = np.concatenate([[self.local_capacities[user]], edge_caps])
            compute = compute / max(float(np.max(self.env.edge_capacities_mb_s)), 1e-8)
            local_load = self.local_queues[user] / max(
                float(self.local_capacities[user] * self.env.cfg.slot_s), 1e-8
            )
            edge_load = self.env.queues_mb[edge_ids] / np.maximum(
                edge_caps * self.env.cfg.slot_s, 1e-8
            )
            load = np.concatenate([[local_load], edge_load])
            rates = np.asarray(
                [self.env._throughput_mb_s(locations[user], int(edge)) for edge in edge_ids],
                dtype=np.float32,
            )
            rates = rates / max(float(np.max(rates)), 1e-8)
            density = np.clip(densities[user] / 5.0, 0.0, 1.0)
            rows.append(
                np.concatenate([compute, load, rates, density, plans[user]]).astype(np.float32)
            )
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def valid_actions(plan: np.ndarray, delta: float) -> np.ndarray:
        valid = np.zeros(6, dtype=np.bool_)
        for edge_index in range(3):
            valid[2 * edge_index] = plan[0] + 1e-8 >= delta and plan[edge_index + 1] + delta <= 1.0 + 1e-8
            valid[2 * edge_index + 1] = (
                plan[edge_index + 1] + 1e-8 >= delta and plan[0] + delta <= 1.0 + 1e-8
            )
        return valid

    @classmethod
    def apply_action(cls, plan: np.ndarray, action: int, delta: float) -> np.ndarray:
        updated = plan.astype(np.float64).copy()
        if not cls.valid_actions(updated, delta)[action]:
            return plan.copy()
        edge_index = action // 2 + 1
        if action % 2 == 0:
            updated[0] -= delta
            updated[edge_index] += delta
        else:
            updated[edge_index] -= delta
            updated[0] += delta
        updated[np.abs(updated) < 1e-8] = 0.0
        updated = np.clip(updated, 0.0, 1.0)
        return (updated / max(float(updated.sum()), 1e-8)).astype(np.float32)

    def estimated_response_times(self, plans: np.ndarray, adjacent: np.ndarray) -> np.ndarray:
        locations = self.env._locations()
        estimates = np.zeros(self.env.cfg.n_users, dtype=np.float32)
        task_compute = self.env.cfg.task_compute_mb
        for user in range(self.env.cfg.n_users):
            local_latency = (
                self.local_queues[user] / max(float(self.local_capacities[user]), 1e-8)
                + task_compute / max(float(self.local_capacities[user]), 1e-8)
            )
            candidates = [local_latency]
            for edge in adjacent[user]:
                capacity = float(self.env.edge_capacities_mb_s[edge])
                candidates.append(
                    self.env._transmission_delay_s(locations[user], int(edge))
                    + float(self.env.queues_mb[edge]) / max(capacity, 1e-8)
                    + task_compute / max(capacity, 1e-8)
                )
            estimates[user] = float(np.dot(plans[user], np.asarray(candidates, dtype=np.float32)))
        return estimates

    def _edge_portfolios(self, plans: np.ndarray, adjacent: np.ndarray) -> np.ndarray:
        portfolios = np.zeros((self.env.cfg.n_users, self.env.cfg.n_edges), dtype=np.float32)
        for user in range(self.env.cfg.n_users):
            for index, edge in enumerate(adjacent[user]):
                portfolios[user, edge] += plans[user, index + 1]
            if portfolios[user].sum() <= 1e-8:
                portfolios[user, adjacent[user]] = 1.0 / self.candidate_edges
            else:
                portfolios[user] /= portfolios[user].sum()
        return portfolios

    def joint_step(
        self,
        plans: np.ndarray,
        adjacent: np.ndarray,
        decision_latency_s: float,
    ) -> SlotResult:
        start = perf_counter()
        edge_portfolios = self._edge_portfolios(plans, adjacent)
        feedback = self.env.server_feedback()
        stats = self.env.portfolio_stats(edge_portfolios)
        locations = self.env._locations()
        records: list[TaskRecord] = []
        edge_slot_load = np.zeros(self.env.cfg.n_edges, dtype=np.float32)
        local_slot_load = np.zeros(self.env.cfg.n_users, dtype=np.float32)
        assignment_counts = np.zeros(self.env.cfg.n_edges, dtype=np.int64)
        overflow_count = 0

        for user in range(self.env.cfg.n_users):
            for task in range(int(self.env.requests[user])):
                feasible = np.ones(self.plan_dim, dtype=np.bool_)
                feasible[0] = (
                    self.local_queues[user] + local_slot_load[user] + self.env.cfg.task_compute_mb
                    <= self.env.cfg.queue_upper_mb
                )
                for index, edge in enumerate(adjacent[user]):
                    feasible[index + 1] = (
                        self.env.queues_mb[edge] + edge_slot_load[edge] + self.env.cfg.task_compute_mb
                        <= self.env.cfg.queue_upper_mb
                    )
                probabilities = np.maximum(
                    np.where(feasible, plans[user], 0.0).astype(np.float64),
                    0.0,
                )
                if probabilities.sum() <= 0.0:
                    overflow_count += 1
                    probabilities = feasible.astype(np.float64)
                if probabilities.sum() <= 0.0:
                    probabilities = np.ones(self.plan_dim, dtype=np.float64)
                choice = int(self.env.rng.choice(self.plan_dim, p=probabilities / probabilities.sum()))
                if choice == 0:
                    capacity = float(self.local_capacities[user])
                    queue_s = float(self.local_queues[user] + local_slot_load[user]) / max(capacity, 1e-8)
                    records.append(
                        TaskRecord(
                            user=user,
                            task=task,
                            edge=-1,
                            upload_s=0.0,
                            package_mb=self.env.cfg.task_package_mb,
                            compute_mb=self.env.cfg.task_compute_mb,
                            tran_s=0.0,
                            queue_s=queue_s,
                            comp_s=self.env.cfg.task_compute_mb / max(capacity, 1e-8),
                        )
                    )
                    local_slot_load[user] += self.env.cfg.task_compute_mb
                else:
                    edge = int(adjacent[user, choice - 1])
                    transmission = self.env._transmission_delay_s(locations[user], edge)
                    records.append(
                        TaskRecord(
                            user=user,
                            task=task,
                            edge=edge,
                            upload_s=transmission,
                            package_mb=self.env.cfg.task_package_mb,
                            compute_mb=self.env.cfg.task_compute_mb,
                            tran_s=transmission,
                            comp_s=self.env.cfg.task_compute_mb
                            / max(float(self.env.edge_capacities_mb_s[edge]), 1e-8),
                        )
                    )
                    edge_slot_load[edge] += self.env.cfg.task_compute_mb
                    assignment_counts[edge] += 1

        edge_queue_before = self.env.queues_mb.copy()
        for edge in range(self.env.cfg.n_edges):
            edge_records = [record for record in records if record.edge == edge]
            edge_records.sort(key=lambda record: record.upload_s)
            waiting = float(self.env.queues_mb[edge])
            for record in edge_records:
                record.queue_s = waiting / max(float(self.env.edge_capacities_mb_s[edge]), 1e-8)
                waiting += record.compute_mb

        arrivals = edge_slot_load.copy()
        if self.env.background_load > 0.0:
            arrivals += (
                self.env.background_load * self.env.edge_capacities_mb_s * self.env.cfg.slot_s
            ).astype(np.float32)
        self.env.queues_mb = np.maximum(
            0.0,
            self.env.queues_mb + arrivals - self.env.edge_capacities_mb_s * self.env.cfg.slot_s,
        ).astype(np.float32)
        self.env.queues_mb = np.minimum(self.env.queues_mb, self.env.cfg.queue_upper_mb)
        self.env.queue_task_counts = np.ceil(
            self.env.queues_mb / max(self.env.cfg.task_compute_mb, 1e-8)
        ).astype(np.int64)
        self.local_queues = np.maximum(
            0.0,
            self.local_queues + local_slot_load - self.local_capacities * self.env.cfg.slot_s,
        ).astype(np.float32)
        self.local_queues = np.minimum(self.local_queues, self.env.cfg.queue_upper_mb)

        per_user_total = np.zeros(self.env.cfg.n_users, dtype=np.float32)
        per_user_tasks = np.zeros(self.env.cfg.n_users, dtype=np.int64)
        for record in records:
            per_user_total[record.user] += record.latency_s
            per_user_tasks[record.user] += 1
        per_user_mean = np.divide(
            per_user_total,
            np.maximum(per_user_tasks, 1),
            out=np.zeros_like(per_user_total),
            where=per_user_tasks > 0,
        )
        result = SlotResult(
            records=records,
            per_user_mean_latency=per_user_mean,
            per_user_total_latency=per_user_total,
            per_user_tasks=per_user_tasks,
            system_average_latency=float(np.sum(per_user_total) / max(1, np.sum(per_user_tasks))),
            portfolios=edge_portfolios,
            portfolio_stats=stats,
            server_feedback=feedback,
            server_assignment_counts=assignment_counts,
            server_arrival_load_mb=arrivals,
            queue_load_before_mb=edge_queue_before,
            queue_load_after_mb=self.env.queues_mb.copy(),
            queue_task_count_after=self.env.queue_task_counts.copy(),
            overflow_count=overflow_count,
            decision_latency_s=float(decision_latency_s or perf_counter() - start),
        )
        self.env.last_result = result
        self.env.completed_latencies.extend(result.latencies)
        self.env.slot_index = (self.env.slot_index + 1) % self.env.traces.trajectories.shape[1]
        self.env.requests = self.env._sample_requests()
        true_feedback = self.env.server_feedback()
        if self.env.last_broadcast_feedback is None or self.env.slot_index % self.env.broadcast_period == 0:
            self.env.last_broadcast_feedback = true_feedback
        self.env.feedback_buffer.push(self.env.last_broadcast_feedback)
        return result


def _select_training_action(
    agent: PaperDQNAgent,
    state: np.ndarray,
    valid: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise RuntimeError("At least one adjustment must be valid")
    if rng.random() < epsilon:
        return int(rng.choice(valid_indices))
    with torch.no_grad():
        values = agent.q(torch.as_tensor(state[None, :], dtype=torch.float32, device=device)).squeeze(0)
        mask = torch.as_tensor(valid, dtype=torch.bool, device=device)
        values = values.masked_fill(~mask, -torch.inf)
        return int(torch.argmax(values).item())


def _runtime_action(
    agent: PaperDQNAgent,
    state: np.ndarray,
    valid: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int | None:
    with torch.no_grad():
        values = agent.q(torch.as_tensor(state[None, :], dtype=torch.float32, device=device)).squeeze(0)
    q = values.cpu().numpy()
    q[~valid] = -np.inf
    valid_q = q[np.isfinite(q)]
    if valid_q.size == 0 or np.all(valid_q < tau):
        return None
    maximum = float(np.max(valid_q))
    ties = np.flatnonzero(np.isfinite(q) & np.isclose(q, maximum, rtol=1e-6, atol=1e-8))
    return int(rng.choice(ties))


def _update_agent(
    agent: PaperDQNAgent,
    batch_size: int,
    gamma: float,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if agent.replay.size < batch_size:
        return 0.0
    batch = agent.replay.sample(batch_size, rng)
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=device)
    next_valid = torch.as_tensor(batch["next_valid"], dtype=torch.bool, device=device)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
    predicted = agent.q(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = agent.target(next_states).masked_fill(~next_valid, -torch.inf)
        has_valid = next_valid.any(dim=1)
        next_max = torch.where(has_valid, next_q.max(dim=1).values, torch.zeros_like(rewards))
        target = rewards + gamma * (1.0 - dones) * next_max
    loss = nn.functional.smooth_l1_loss(predicted, target, beta=1.0)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)
    agent.optimizer.step()
    return float(loss.detach().item())


def _fedavg(agents: list[PaperDQNAgent]) -> None:
    with torch.no_grad():
        states = [agent.q.state_dict() for agent in agents]
        averaged = {
            key: torch.stack([state[key].detach().float() for state in states]).mean(dim=0)
            for key in states[0]
        }
        for agent in agents:
            agent.q.load_state_dict(averaged)
            agent.target.load_state_dict(averaged)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_adapter(args: argparse.Namespace, seed: int) -> tuple[LocalAdjacentMECAdapter, MAPPOSystemConfig]:
    system = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        task_package_mb=args.task_package_mb,
        max_supported_users=max(args.users, args.max_supported_users),
    )
    capacities = edge_capacities(system.n_edges, system.edge_compute_mb_s, args.capacity_mode, args.seed)
    traces = load_trace_set(
        args.dataset,
        args.users,
        system.n_edges,
        system.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    environment = JointMAPPOEdgeEnv(
        system,
        traces,
        request_distribution=args.distribution,
        seed=seed,
        queue_constraint=args.queue_constraint,
        score_config=ServerScoreConfig(),
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    return LocalAdjacentMECAdapter(environment, args.local_compute_ratio, args.density_radius), system


def _agents(args: argparse.Namespace, state_dim: int, device: torch.device) -> list[PaperDQNAgent]:
    agents: list[PaperDQNAgent] = []
    for user in range(args.users):
        torch.manual_seed(args.seed + 101 * user)
        q = PaperDQN(state_dim, 6).to(device)
        target = PaperDQN(state_dim, 6).to(device)
        target.load_state_dict(q.state_dict())
        agents.append(
            PaperDQNAgent(
                q=q,
                target=target,
                optimizer=torch.optim.Adam(q.parameters(), lr=args.learning_rate),
                replay=PaperReplayBuffer(args.replay_size, state_dim, 6),
            )
        )
    return agents


def _record_latencies(rows: list[dict[str, object]], result: SlotResult, phase: str, slot: int) -> None:
    for record in result.records:
        rows.append(
            {
                "phase": phase,
                "slot": slot,
                "step": slot * result.portfolios.shape[0],
                "user": record.user,
                "task": record.task,
                "edge": record.edge,
                "is_local": int(record.edge < 0),
                "latency": record.latency_s,
                "tran_s": record.tran_s,
                "queue_s": record.queue_s,
                "comp_s": record.comp_s,
            }
        )


def _evaluate(
    adapter: LocalAdjacentMECAdapter,
    agents: list[PaperDQNAgent],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[SlotResult], float, float]:
    adapter.reset()
    plans = adapter.initial_plans()
    results: list[SlotResult] = []
    stopped = 0
    decisions = 0
    local_fractions: list[float] = []
    for _ in range(args.test_steps):
        adjacent = adapter.adjacent_edges()
        states = adapter.states(plans, adjacent)
        started = perf_counter()
        for user, agent in enumerate(agents):
            valid = adapter.valid_actions(plans[user], args.adjust_delta)
            action = _runtime_action(agent, states[user], valid, args.tau, adapter.env.rng, device)
            decisions += 1
            if action is None:
                stopped += 1
            else:
                plans[user] = adapter.apply_action(plans[user], action, args.adjust_delta)
        decision_latency = perf_counter() - started
        results.append(adapter.joint_step(plans.copy(), adjacent, decision_latency))
        local_fractions.append(float(np.mean(plans[:, 0])))
    return results, stopped / max(1, decisions), float(np.mean(local_fractions))


def train_once(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter, system = _build_adapter(args, args.seed)
    plans = adapter.initial_plans()
    adjacent = adapter.adjacent_edges()
    state_dim = int(adapter.states(plans, adjacent).shape[1])
    if state_dim != 18:
        raise RuntimeError(f"Paper state mapping must have dimension 18, got {state_dim}")
    agents = _agents(args, state_dim, device)
    output = Path(args.output_dir) / args.dataset / f"{args.users}u" / args.distribution
    output.mkdir(parents=True, exist_ok=True)
    learning_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    epsilon = args.epsilon_start
    total_steps = args.episodes * args.steps_per_episode
    global_step = 0
    started_all = perf_counter()

    for episode in range(1, args.episodes + 1):
        adapter.reset()
        plans = adapter.initial_plans()
        for episode_step in range(1, args.steps_per_episode + 1):
            global_step += 1
            adjacent = adapter.adjacent_edges()
            states = adapter.states(plans, adjacent)
            actions = np.zeros(system.n_users, dtype=np.int64)
            before = adapter.estimated_response_times(plans, adjacent)
            action_started = perf_counter()
            for user, agent in enumerate(agents):
                valid = adapter.valid_actions(plans[user], args.adjust_delta)
                actions[user] = _select_training_action(
                    agent,
                    states[user],
                    valid,
                    epsilon,
                    rng,
                    device,
                )
                plans[user] = adapter.apply_action(plans[user], int(actions[user]), args.adjust_delta)
            after = adapter.estimated_response_times(plans, adjacent)
            rewards = before - after
            result = adapter.joint_step(
                plans.copy(),
                adjacent,
                perf_counter() - action_started,
            )
            next_adjacent = adapter.adjacent_edges()
            next_states = adapter.states(plans, next_adjacent)
            done = episode_step == args.steps_per_episode
            losses: list[float] = []
            for user, agent in enumerate(agents):
                next_valid = adapter.valid_actions(plans[user], args.adjust_delta)
                agent.replay.add(
                    states[user],
                    int(actions[user]),
                    float(rewards[user]),
                    next_states[user],
                    next_valid,
                    done,
                )
                losses.append(_update_agent(agent, args.batch_size, args.gamma, rng, device))
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
            _record_latencies(latency_rows, result, "train", global_step)

            if global_step == 1 or global_step % args.log_interval == 0 or global_step == total_steps:
                positive_losses = [loss for loss in losses if loss > 0.0]
                row: dict[str, object] = {
                    "episode": episode,
                    "episode_step": episode_step,
                    "slot": global_step,
                    "step": global_step * system.n_users,
                    "mean_reward": float(np.mean(rewards)),
                    "epsilon": epsilon,
                    "loss": float(np.mean(positive_losses)) if positive_losses else 0.0,
                    "valid_action_fraction": float(
                        np.mean([adapter.valid_actions(plan, args.adjust_delta).mean() for plan in plans])
                    ),
                    "mean_local_fraction": float(np.mean(plans[:, 0])),
                    "elapsed_s": perf_counter() - started_all,
                }
                row.update(slot_metrics(result))
                learning_rows.append(row)

        if episode % args.fed_every_episodes == 0:
            _fedavg(agents)

    _fedavg(agents)
    evaluation_adapter, _ = _build_adapter(args, args.seed + 10_000)
    evaluation_results, stop_rate, mean_local_fraction = _evaluate(evaluation_adapter, agents, args, device)
    for slot, result in enumerate(evaluation_results, start=1):
        _record_latencies(latency_rows, result, "test", slot)
    summary: dict[str, object] = summarize_results(evaluation_results)
    summary.update(phase_diagnostics(evaluation_results, system))
    summary.update(
        {
            "method": METHOD,
            "source_paper_doi": "10.1109/TMC.2025.3557898",
            "architecture": "per-user DQN 18-128-128-6 with FedAvg",
            "state_mapping": "compute4+load4+transmission3+density3+plan4",
            "action_mapping": "add/decrease 4% normalized task fraction on three nearest edges",
            "source_adjust_volume": 0.028,
            "reward": "counterfactual latency_before-latency_after",
            "invalid_action_masking": True,
            "runtime_q_stopping_threshold": args.tau,
            "runtime_stop_rate": stop_rate,
            "mean_local_fraction": mean_local_fraction,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "train_steps": total_steps,
            "test_steps": args.test_steps,
            "fed_every_episodes": args.fed_every_episodes,
            "adjust_delta": args.adjust_delta,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
        }
    )
    _write_csv(output / f"learning_{METHOD}.csv", learning_rows)
    _write_csv(output / f"latency_samples_{METHOD}.csv", latency_rows)
    _write_csv(output / f"summary_{METHOD}.csv", [summary])
    with (output / f"config_{METHOD}.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    torch.save(
        {
            "q_models": [agent.q.state_dict() for agent in agents],
            "state_dim": state_dim,
            "action_dim": 6,
            "config": vars(args),
        },
        output / f"checkpoint_{METHOD}.pt",
    )
    try:
        make_training_plots(output, METHOD)
    except Exception:
        pass
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-structure MCM-FDRL on the current MEC environment.")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--episodes", type=int, default=7_200)
    parser.add_argument("--steps-per-episode", type=int, default=25)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_mcm_fdrl_paper"))
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--feedback-delay", type=int, default=0)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=80)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous")
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--local-compute-ratio", type=float, default=0.25)
    parser.add_argument("--density-radius", type=float, default=500.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fed-every-episodes", type=int, default=10)
    parser.add_argument(
        "--adjust-delta",
        type=float,
        default=0.04,
        help="Normalized plan step. Paper 0.028 task volume / lambda 0.7 = 0.04.",
    )
    parser.add_argument("--tau", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train_once(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
