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
from repro_risk_averse_edge.config import ROOT

from .config import MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results
from .plots import make_training_plots
from .train import edge_capacities


class DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool = False) -> None:
        self.states[self.pos] = state.astype(np.float32)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = next_state.astype(np.float32)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


@dataclass
class DQNAgent:
    q: DQN
    target: DQN
    optimizer: torch.optim.Optimizer
    replay: ReplayBuffer


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, row: dict) -> None:
    _write_csv(path, [row])


def _state(actor_obs: np.ndarray, portfolios: np.ndarray) -> np.ndarray:
    return np.concatenate([actor_obs.astype(np.float32), portfolios.astype(np.float32)], axis=1)


def _apply_adjustment(portfolios: np.ndarray, user: int, action: int, delta: float) -> None:
    n_edges = portfolios.shape[1]
    edge = int(action // 2) % n_edges
    add = action % 2 == 0
    if add:
        portfolios[user, edge] += delta
    else:
        portfolios[user, edge] = max(0.0, portfolios[user, edge] - delta)
    total = float(np.sum(portfolios[user]))
    if total <= 1e-8:
        portfolios[user] = 1.0 / n_edges
    else:
        portfolios[user] /= total


def _fedavg(agents: list[DQNAgent]) -> None:
    with torch.no_grad():
        states = [agent.q.state_dict() for agent in agents]
        avg = {}
        for key in states[0]:
            avg[key] = torch.stack([state[key].detach().float() for state in states], dim=0).mean(dim=0)
        for agent in agents:
            agent.q.load_state_dict(avg)
            agent.target.load_state_dict(avg)


def _sync_targets(agents: list[DQNAgent]) -> None:
    for agent in agents:
        agent.target.load_state_dict(agent.q.state_dict())


def _select_actions(
    agents: list[DQNAgent],
    states: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    actions = np.zeros(len(agents), dtype=np.int64)
    for user, agent in enumerate(agents):
        if rng.random() < epsilon:
            actions[user] = int(rng.integers(0, agent.q.net[-1].out_features))
            continue
        with torch.no_grad():
            s = torch.as_tensor(states[user : user + 1], dtype=torch.float32, device=device)
            actions[user] = int(torch.argmax(agent.q(s), dim=-1).item())
    return actions


def _update_agent(
    agent: DQNAgent,
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
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
    q_values = agent.q(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        targets = rewards + gamma * (1.0 - dones) * agent.target(next_states).max(dim=1).values
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)
    agent.optimizer.step()
    return float(loss.detach().cpu().item())


def _run_policy(
    env: JointMAPPOEdgeEnv,
    agents: list[DQNAgent],
    steps: int,
    adjust_delta: float,
    device: torch.device,
) -> list[SlotResult]:
    results: list[SlotResult] = []
    env.reset()
    portfolios = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
    for _ in range(steps):
        states = _state(env.actor_observations(), portfolios)
        start = perf_counter()
        actions = _select_actions(agents, states, epsilon=0.0, rng=env.rng, device=device)
        for user, action in enumerate(actions):
            _apply_adjustment(portfolios, user, int(action), adjust_delta)
        decision_latency = perf_counter() - start
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios.copy(), stats=stats, decision_latency_s=decision_latency))
    return results


def train_once(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    sys_cfg = MAPPOSystemConfig(
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
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, args.seed)
    traces = load_trace_set(
        args.dataset,
        args.users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=args.seed,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    state_dim = sys_cfg.actor_obs_dim + sys_cfg.n_edges
    action_dim = 2 * sys_cfg.n_edges
    agents: list[DQNAgent] = []
    for _ in range(sys_cfg.n_users):
        q = DQN(state_dim, action_dim, args.hidden).to(device)
        target = DQN(state_dim, action_dim, args.hidden).to(device)
        target.load_state_dict(q.state_dict())
        agents.append(
            DQNAgent(
                q=q,
                target=target,
                optimizer=torch.optim.Adam(q.parameters(), lr=args.learning_rate),
                replay=ReplayBuffer(args.replay_size, state_dim),
            )
        )

    out_dir = Path(args.output_dir) / args.dataset / f"{args.users}u" / args.distribution
    out_dir.mkdir(parents=True, exist_ok=True)
    learning_rows: list[dict] = []
    latency_rows: list[dict] = []
    env.reset()
    portfolios = np.full((sys_cfg.n_users, sys_cfg.n_edges), 1.0 / sys_cfg.n_edges, dtype=np.float32)
    prev_latency = np.zeros(sys_cfg.n_users, dtype=np.float32)
    epsilon = args.epsilon_start
    start_all = perf_counter()
    for slot in range(1, args.train_steps + 1):
        states = _state(env.actor_observations(), portfolios)
        t0 = perf_counter()
        actions = _select_actions(agents, states, epsilon=epsilon, rng=rng, device=device)
        for user, action in enumerate(actions):
            _apply_adjustment(portfolios, user, int(action), args.adjust_delta)
        decision_latency = perf_counter() - t0
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios.copy(), stats=stats, decision_latency_s=decision_latency)
        next_states = _state(env.actor_observations(), portfolios)

        current_latency = result.per_user_mean_latency.astype(np.float32)
        active = result.per_user_tasks > 0
        if args.reward_mode == "improvement":
            rewards = (prev_latency - current_latency) / max(args.latency_norm, 1e-8)
            rewards[~active] = 0.0
        else:
            rewards = -current_latency / max(args.latency_norm, 1e-8)
            rewards[~active] = 0.0
        prev_latency = np.where(active, current_latency, prev_latency)
        losses = []
        for user, agent in enumerate(agents):
            agent.replay.add(states[user], int(actions[user]), float(rewards[user]), next_states[user])
            losses.append(_update_agent(agent, args.batch_size, args.gamma, rng, device))
        if slot % args.target_update_interval == 0:
            _sync_targets(agents)
        if slot % args.fed_interval == 0:
            _fedavg(agents)
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        for record in result.records:
            latency_rows.append(
                {
                    "phase": "train",
                    "slot": slot,
                    "step": slot * sys_cfg.n_users,
                    "user": record.user,
                    "task": record.task,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                }
            )
        if slot == 1 or slot % args.log_interval == 0 or slot == args.train_steps:
            row = {
                "slot": slot,
                "step": slot * sys_cfg.n_users,
                "average_latency": result.system_average_latency,
                "mean_reward": float(np.mean(rewards)),
                "epsilon": float(epsilon),
                "loss": float(np.mean([loss for loss in losses if loss > 0.0])) if any(loss > 0.0 for loss in losses) else 0.0,
                "elapsed_s": perf_counter() - start_all,
            }
            row.update(
                {
                    "mean_portfolio_entropy": result.portfolio_stats.mean_entropy,
                    "global_demand_concentration": result.portfolio_stats.global_demand_concentration,
                    "server_selection_gini": float(np.mean(result.server_assignment_counts > 0)),
                    "slot_tasks": int(np.sum(result.per_user_tasks)),
                }
            )
            learning_rows.append(row)

    _fedavg(agents)
    eval_env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=args.seed + 10_000,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    eval_results = _run_policy(eval_env, agents, args.test_steps, args.adjust_delta, device)
    for slot, result in enumerate(eval_results, start=1):
        for record in result.records:
            latency_rows.append(
                {
                    "phase": "test",
                    "slot": slot,
                    "step": slot * sys_cfg.n_users,
                    "user": record.user,
                    "task": record.task,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                }
            )
    summary = summarize_results(eval_results)
    summary.update(phase_diagnostics(eval_results, sys_cfg))
    summary.update(
        {
            "method": "MCM-FDRL",
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
            "slot_s": args.slot_s,
            "task_compute_mb": args.task_compute_mb,
            "background_load": args.background_load,
            "capacity_mode": args.capacity_mode,
            "fed_interval": args.fed_interval,
            "adjust_delta": args.adjust_delta,
            "reward_mode": args.reward_mode,
        }
    )
    _write_csv(out_dir / "learning_MCM-FDRL.csv", learning_rows)
    _write_csv(out_dir / "latency_samples_MCM-FDRL.csv", latency_rows)
    _write_summary(out_dir / "summary_MCM-FDRL.csv", summary)
    torch.save(
        {
            "q": agents[0].q.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "config": vars(args),
        },
        out_dir / "checkpoint_MCM-FDRL.pt",
    )
    with (out_dir / "config_MCM-FDRL.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    try:
        make_training_plots(out_dir, "MCM-FDRL")
    except Exception:
        pass
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MCM-FDRL-style federated DQN portfolio adjuster.")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty", choices=["uniform", "gaussian", "poisson", "poisson_bursty", "bursty"])
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_mcm_fdrl"))
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
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fed-interval", type=int, default=250)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--adjust-delta", type=float, default=0.10)
    parser.add_argument("--reward-mode", default="improvement", choices=["improvement", "latency"])
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    summary = train_once(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
