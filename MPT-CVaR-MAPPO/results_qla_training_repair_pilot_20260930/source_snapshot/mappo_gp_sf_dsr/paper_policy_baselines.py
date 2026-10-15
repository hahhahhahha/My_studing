"""Isolated paper-inspired policy baselines for the V4-AgeGate scenario.

The file intentionally avoids changing the V4-AgeGate implementation. It
contains:

* Towards-PPO-DSR and Towards-PPO-MV: shared Gaussian latent + softmax PPO
  adapted from "Towards Risk-Averse Edge Computing With DRL".
* Fed-MADDPG: a compact federated MADDPG adaptation inspired by
  "Federated MADDPG-Based Collaborative Scheduling Strategy in VEC".

These are algorithm-core reproductions in the current RSU-MEC environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, ROOT, TRACE_ROOT, ServerScoreConfig
from .canonical_protocol import install_schedule_for_env, write_run_manifest
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import finalize_evaluation, slot_metrics
from .train import build_output_dir, edge_capacities, eval_diagnostics, write_csv, write_summary


EPS = 1e-8


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class DSRReward:
    def __init__(self, mu: float = 1.5, eta: float = 0.01, temperature: float = 5.0) -> None:
        self.mu = float(mu)
        self.eta = float(eta)
        self.temperature = float(temperature)
        self.a = 0.0
        self.b = 0.0

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0

    def __call__(self, latency: float) -> float:
        ret = self.mu - float(latency)
        da = ret - self.a
        db = ret * ret - self.b
        prev_a = self.a
        prev_b = self.b
        denom = max((prev_b - prev_a * prev_a), EPS) ** 1.5
        reward = (prev_b * da - 0.5 * prev_a * db) / denom
        self.a = prev_a + self.eta * da
        self.b = prev_b + self.eta * db
        return float(np.tanh(reward / max(self.temperature, EPS)))


class MVReward:
    def __init__(self, mu: float = 1.5) -> None:
        self.mu = float(mu)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.prev_sr: float | None = None

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.prev_sr = None

    def __call__(self, latency: float) -> float:
        ret = self.mu - float(latency)
        self.count += 1
        delta = ret - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (ret - self.mean)
        var = self.m2 / self.count
        sr = self.mean / max(np.sqrt(var), EPS)
        reward = 0.0 if self.prev_sr is None else (1.0 if sr >= self.prev_sr else 0.0)
        self.prev_sr = sr
        return float(reward)


class SharedGaussianPPO(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, log_std_min: float = -1.5, log_std_max: float = 1.0) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = mlp(obs_dim, 1)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mu)
        return torch.distributions.Normal(mu, log_std.exp())

    def act(self, obs_np: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = self.distribution(obs)
            z = dist.mean if deterministic else dist.rsample()
            logp = dist.log_prob(z).sum(dim=-1)
            values = self.critic(obs).squeeze(-1)
            portfolios = torch.softmax(z, dim=-1)
        return (
            z.cpu().numpy().astype(np.float32),
            logp.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
            portfolios.cpu().numpy().astype(np.float32),
        )


@dataclass
class PPOItem:
    obs: np.ndarray
    z: np.ndarray
    logp: float
    value: float
    reward: float


class PPOBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.data: list[PPOItem] = []

    def add_batch(self, obs: np.ndarray, z: np.ndarray, logp: np.ndarray, values: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(obs.shape[0]):
            self.data.append(PPOItem(obs[i].copy(), z[i].copy(), float(logp[i]), float(values[i]), float(rewards[i])))
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size :]

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def tensors(self, gamma: float = 0.99, lam: float = 0.95, device: str | None = None) -> dict[str, torch.Tensor]:
        rewards = np.asarray([x.reward for x in self.data], dtype=np.float32)
        values = np.asarray([x.value for x in self.data], dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": torch.as_tensor(np.stack([x.obs for x in self.data]), dtype=torch.float32, device=device),
            "z": torch.as_tensor(np.stack([x.z for x in self.data]), dtype=torch.float32, device=device),
            "logp": torch.as_tensor([x.logp for x in self.data], dtype=torch.float32, device=device),
            "adv": torch.as_tensor(adv, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        }


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.data: deque = deque(maxlen=max_size)

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray) -> None:
        self.data.append((obs.astype(np.float32), actions.astype(np.float32), rewards.astype(np.float32), next_obs.astype(np.float32)))

    def sample(self, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.data), size=batch, replace=False)
        obs, act, rew, nxt = zip(*(self.data[i] for i in idx))
        return np.stack(obs), np.stack(act), np.stack(rew), np.stack(nxt)

    def __len__(self) -> int:
        return len(self.data)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = mlp(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(obs), dim=-1)


class CentralCritic(nn.Module):
    def __init__(self, n_users: int, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = mlp(n_users * (obs_dim + action_dim), 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=-1)
        return self.net(x).squeeze(-1)


def _build_env(args: argparse.Namespace, seed_offset: int = 0) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig, np.ndarray]:
    sys_cfg = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        task_package_mb=args.task_package_mb,
        max_requests_per_user=args.max_requests_per_user,
        max_supported_users=max(args.max_supported_users, args.users),
    )
    traces = load_trace_set(
        args.dataset,
        args.users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, args.seed)
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=args.seed + seed_offset,
        queue_constraint=args.queue_constraint,
        score_config=ServerScoreConfig(
            resource_weight=args.score_resource_weight,
            capacity_weight=args.score_capacity_weight,
            task_count_weight=args.score_task_count_weight,
            wait_weight=args.score_wait_weight,
        ),
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    is_training = int(seed_offset) == 0
    install_schedule_for_env(
        env,
        root=Path(
            getattr(
                args,
                "canonical_schedule_root",
                ROOT / "results_qla_canonical_v1" / "shared_schedules",
            )
        ),
        distribution=args.distribution,
        role="train" if is_training else "test",
        seed=int(args.seed + seed_offset),
        n_steps=int(args.train_steps if is_training else args.test_steps),
        max_feedback_delay=int(getattr(args, "canonical_max_feedback_delay", 8)),
    )
    return env, sys_cfg, capacities


def _write_outputs(out_dir: Path, method: str, learning_rows: list[dict], latency_rows: list[dict], summary: dict, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"learning_{method}.csv", learning_rows)
    write_csv(out_dir / f"latency_samples_{method}.csv", latency_rows)
    write_summary(out_dir / f"summary_{method}.csv", summary)
    with (out_dir / f"config_{method}.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


def _evaluate_portfolio_policy(env: JointMAPPOEdgeEnv, policy_fn, test_steps: int, sys_cfg: MAPPOSystemConfig, capacities: np.ndarray) -> tuple[list[SlotResult], dict, list[dict]]:
    obs = env.reset()
    results: list[SlotResult] = []
    latency_rows: list[dict] = []
    for step in range(test_steps):
        start = perf_counter()
        portfolios = policy_fn(obs)
        result = env.joint_step(portfolios, decision_latency_s=perf_counter() - start)
        results.append(result)
        for record in result.records:
            latency_rows.append(
                {
                    "step": step,
                    "user": record.user,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "dropped": int(record.dropped),
                }
            )
        obs = env.actor_observations()
    summary, _ = finalize_evaluation(env, results, sys_cfg)
    summary.update(eval_diagnostics(results, capacities, sys_cfg.slot_s))
    return results, summary, latency_rows


def train_towards_ppo(args: argparse.Namespace) -> dict:
    method = args.method
    out_dir = build_output_dir(Path(args.output_dir) / f"{method}_delay_{args.feedback_delay}", args.dataset, args.users, args.distribution)
    env, sys_cfg, capacities = _build_env(args)
    obs = env.reset()
    device = args.device
    policy = SharedGaussianPPO(obs.shape[1], sys_cfg.n_edges, log_std_min=args.log_std_min).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    buffer = PPOBuffer(args.buffer_size)
    rewarder = DSRReward(args.dsr_mu, args.dsr_eta, args.dsr_temperature) if method == "Towards-PPO-DSR" else MVReward(args.dsr_mu)
    learning_rows: list[dict] = []

    for step in range(1, args.train_steps + 1):
        z, logp, values, portfolios = policy.act(obs, deterministic=False)
        result = env.joint_step(portfolios)
        rewards = np.asarray([rewarder(float(x)) for x in result.per_user_mean_latency], dtype=np.float32)
        buffer.add_batch(obs, z, logp, values, rewards)
        if len(buffer) >= args.rollout_size:
            data = buffer.tensors(args.gamma, args.gae_lambda, device)
            for _ in range(args.update_epochs):
                idx = torch.randperm(data["obs"].shape[0], device=device)
                for start in range(0, idx.numel(), args.batch_size):
                    batch = idx[start : start + args.batch_size]
                    dist = policy.distribution(data["obs"][batch])
                    new_logp = dist.log_prob(data["z"][batch]).sum(dim=-1)
                    ratio = torch.exp(new_logp - data["logp"][batch])
                    surr1 = ratio * data["adv"][batch]
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * data["adv"][batch]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    values_t = policy.critic(data["obs"][batch]).squeeze(-1)
                    critic_loss = F.mse_loss(values_t, data["returns"][batch])
                    entropy_bonus = dist.entropy().sum(dim=-1).mean()
                    loss = actor_loss + 0.5 * critic_loss - args.entropy_coef * entropy_bonus
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optim.step()
            buffer.clear()
        if step % args.log_interval == 0:
            row = {"slot": step, "step": step * args.users, "mean_reward": float(np.mean(rewards)), **slot_metrics(result)}
            learning_rows.append(row)
        obs = env.actor_observations()

    eval_env, _, eval_cap = _build_env(args, seed_offset=10_000)

    def policy_fn(eval_obs: np.ndarray) -> np.ndarray:
        return policy.act(eval_obs, deterministic=True)[3]

    _, summary, latency_rows = _evaluate_portfolio_policy(eval_env, policy_fn, args.test_steps, sys_cfg, eval_cap)
    summary.update(
        {
            "method": method,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
            "slot_s": args.slot_s,
            "task_compute_mb": sys_cfg.task_compute_mb,
            "background_load": args.background_load,
            "capacity_mode": args.capacity_mode,
            "paper_source": "Towards Risk-Averse Edge Computing With DRL",
        }
    )
    if env.canonical_schedule is not None:
        summary["train_split_id"] = env.canonical_schedule.schedule_id
        summary["train_schedule_sha256"] = env.canonical_schedule.content_sha256
    _write_outputs(out_dir, method, learning_rows, latency_rows, summary, args)
    checkpoint_path = out_dir / f"checkpoint_{method}.pt"
    config_path = out_dir / f"config_{method}.json"
    torch.save(policy.state_dict(), checkpoint_path)
    write_run_manifest(
        output_dir=out_dir,
        method=method,
        args=args,
        summary=summary,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    return summary


def _fedavg_actors(actors: list[DeterministicActor]) -> None:
    with torch.no_grad():
        keys = actors[0].state_dict().keys()
        avg = {}
        for key in keys:
            avg[key] = torch.stack([a.state_dict()[key].detach().cpu() for a in actors], dim=0).mean(dim=0)
        for actor in actors:
            actor.load_state_dict(avg)


def train_fed_maddpg(args: argparse.Namespace) -> dict:
    method = "Fed-MADDPG"
    out_dir = build_output_dir(Path(args.output_dir) / f"{method}_delay_{args.feedback_delay}", args.dataset, args.users, args.distribution)
    env, sys_cfg, capacities = _build_env(args)
    obs = env.reset()
    obs_dim = obs.shape[1]
    n_users = sys_cfg.n_users
    n_edges = sys_cfg.n_edges
    device = args.device
    actors = [DeterministicActor(obs_dim, n_edges).to(device) for _ in range(n_users)]
    target_actors = [DeterministicActor(obs_dim, n_edges).to(device) for _ in range(n_users)]
    critics = [CentralCritic(n_users, obs_dim, n_edges).to(device) for _ in range(n_users)]
    target_critics = [CentralCritic(n_users, obs_dim, n_edges).to(device) for _ in range(n_users)]
    for a, ta in zip(actors, target_actors):
        ta.load_state_dict(a.state_dict())
    for c, tc in zip(critics, target_critics):
        tc.load_state_dict(c.state_dict())
    actor_opts = [torch.optim.Adam(a.parameters(), lr=args.learning_rate) for a in actors]
    critic_opts = [torch.optim.Adam(c.parameters(), lr=args.learning_rate) for c in critics]
    replay = ReplayBuffer(args.buffer_size)
    learning_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)

    def act(obs_np: np.ndarray, noise: float) -> np.ndarray:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        outs = []
        with torch.no_grad():
            for i, actor in enumerate(actors):
                w = actor(obs_t[i : i + 1]).squeeze(0).cpu().numpy()
                if noise > 0:
                    w = np.maximum(w + rng.normal(0.0, noise, size=w.shape), 1e-6)
                    w = w / np.sum(w)
                outs.append(w.astype(np.float32))
        return np.stack(outs)

    for step in range(1, args.train_steps + 1):
        noise = max(args.noise_final, args.noise_start * (1.0 - step / max(1, args.train_steps)))
        actions = act(obs, noise)
        result = env.joint_step(actions)
        # Potential-function adaptation: higher reward when global latency is lower,
        # with local shaping to retain each agent's latency signal.
        global_reward = -float(result.system_average_latency) / max(args.latency_norm, EPS)
        local = -result.per_user_mean_latency.astype(np.float32) / max(args.latency_norm, EPS)
        rewards = (0.7 * global_reward + 0.3 * local).astype(np.float32)
        next_obs = env.actor_observations()
        replay.add(obs, actions, rewards, next_obs)
        if len(replay) >= args.batch_size and step % args.update_every == 0:
            b_obs, b_act, b_rew, b_next = replay.sample(args.batch_size)
            obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
            act_t = torch.as_tensor(b_act, dtype=torch.float32, device=device)
            rew_t = torch.as_tensor(b_rew, dtype=torch.float32, device=device)
            next_t = torch.as_tensor(b_next, dtype=torch.float32, device=device)
            with torch.no_grad():
                next_actions = torch.stack([target_actors[i](next_t[:, i, :]) for i in range(n_users)], dim=1)
            for i in range(n_users):
                with torch.no_grad():
                    target_q = rew_t[:, i] + args.gamma * target_critics[i](next_t, next_actions)
                q = critics[i](obs_t, act_t)
                critic_loss = F.mse_loss(q, target_q)
                critic_opts[i].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critics[i].parameters(), 1.0)
                critic_opts[i].step()

                current_actions = torch.stack([actors[j](obs_t[:, j, :]) for j in range(n_users)], dim=1)
                actor_loss = -critics[i](obs_t, current_actions).mean()
                actor_opts[i].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actors[i].parameters(), 1.0)
                actor_opts[i].step()
                for p, tp in zip(actors[i].parameters(), target_actors[i].parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
                for p, tp in zip(critics[i].parameters(), target_critics[i].parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
        if step % args.fed_interval == 0:
            _fedavg_actors(actors)
            for a, ta in zip(actors, target_actors):
                ta.load_state_dict(a.state_dict())
        if step % args.log_interval == 0:
            learning_rows.append({"slot": step, "step": step * n_users, "mean_reward": float(np.mean(rewards)), **slot_metrics(result)})
        obs = next_obs

    eval_env, _, eval_cap = _build_env(args, seed_offset=10_000)

    def policy_fn(eval_obs: np.ndarray) -> np.ndarray:
        return act(eval_obs, noise=0.0)

    _, summary, latency_rows = _evaluate_portfolio_policy(eval_env, policy_fn, args.test_steps, sys_cfg, eval_cap)
    summary.update(
        {
            "method": method,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
            "slot_s": args.slot_s,
            "task_compute_mb": sys_cfg.task_compute_mb,
            "background_load": args.background_load,
            "capacity_mode": args.capacity_mode,
            "fed_interval": args.fed_interval,
            "paper_source": "Federated MADDPG-Based Collaborative Scheduling Strategy in VEC",
        }
    )
    _write_outputs(out_dir, method, learning_rows, latency_rows, summary, args)
    checkpoint_path = out_dir / f"checkpoint_{method}.pt"
    config_path = out_dir / f"config_{method}.json"
    torch.save({"actors": [a.state_dict() for a in actors], "critics": [c.state_dict() for c in critics]}, checkpoint_path)
    write_run_manifest(
        output_dir=out_dir,
        method=method,
        args=args,
        summary=summary,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["Towards-PPO-DSR", "Towards-PPO-MV", "Fed-MADDPG"])
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty", choices=["uniform", "gaussian", "poisson", "poisson_bursty", "bursty"])
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_paper_algorithms_v4scenario"))
    parser.add_argument(
        "--canonical-schedule-root",
        default=str(ROOT / "results_qla_canonical_v1" / "shared_schedules"),
    )
    parser.add_argument("--canonical-max-feedback-delay", type=int, default=8)
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
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--buffer-size", type=int, default=20_000)
    parser.add_argument("--rollout-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--dsr-mu", type=float, default=1.5)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument("--dsr-temperature", type=float, default=5.0)
    parser.add_argument("--log-std-min", type=float, default=-1.5)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--fed-interval", type=int, default=250)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--noise-start", type=float, default=0.08)
    parser.add_argument("--noise-final", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method in {"Towards-PPO-DSR", "Towards-PPO-MV"}:
        summary = train_towards_ppo(args)
    else:
        summary = train_fed_maddpg(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
