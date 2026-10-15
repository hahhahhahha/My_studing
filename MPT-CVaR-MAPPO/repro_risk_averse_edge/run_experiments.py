from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .agents import DQNPolicy, DiscreteSACPolicy, PGPolicy, PPOPolicy, Transition
from .config import ExperimentConfig, PPOConfig, SystemConfig
from .data import load_trace_set
from .env import RiskAverseEdgeEnv
from .metrics import latency_summary
from .rewards import DSRReward, MVReward, latency_reward


def make_reward(method: str, sys_cfg: SystemConfig):
    if method == "PPO-DSR":
        return DSRReward(mu=sys_cfg.mu_return, eta=sys_cfg.dsr_eta)
    if method == "PPO-MV":
        return MVReward(mu=sys_cfg.mu_return)
    return None


def reward_for(method: str, latency: float, reward_state, sys_cfg: SystemConfig) -> float:
    if method == "PPO-DSR":
        return reward_state(latency)
    if method == "PPO-MV":
        return reward_state(latency)
    return latency_reward(latency)


def make_agent(method: str, sys_cfg: SystemConfig, ppo_cfg: PPOConfig, seed: int):
    if method in {"PPO-DSR", "PPO-MV", "PPO-L"}:
        return PPOPolicy(sys_cfg, ppo_cfg, seed=seed)
    if method == "PG":
        return PGPolicy(sys_cfg, ppo_cfg, seed=seed)
    if method == "DQN":
        return DQNPolicy(sys_cfg, ppo_cfg, seed=seed, recurrent=False)
    if method == "DRQN":
        return DQNPolicy(sys_cfg, ppo_cfg, seed=seed, recurrent=True)
    if method == "SAC":
        return DiscreteSACPolicy(sys_cfg, ppo_cfg, seed=seed)
    raise ValueError(method)


def train_method(
    method: str,
    dataset: str,
    n_users: int,
    distribution: str,
    train_steps: int,
    seed: int,
    output_dir: Path,
    ppo_rollout_size: int | None = None,
    learning_rate: float | None = None,
    entropy_coef: float | None = None,
    dsr_eta: float | None = None,
    mu_return: float | None = None,
) -> tuple[object, list[dict]]:
    sys_cfg = SystemConfig(n_users=n_users)
    if dsr_eta is not None or mu_return is not None:
        sys_cfg = replace(
            sys_cfg,
            dsr_eta=sys_cfg.dsr_eta if dsr_eta is None else dsr_eta,
            mu_return=sys_cfg.mu_return if mu_return is None else mu_return,
        )
    base_ppo_cfg = PPOConfig()
    ppo_cfg = replace(
        base_ppo_cfg,
        rollout_size=base_ppo_cfg.rollout_size if ppo_rollout_size is None else ppo_rollout_size,
        learning_rate=base_ppo_cfg.learning_rate if learning_rate is None else learning_rate,
        entropy_coef=base_ppo_cfg.entropy_coef if entropy_coef is None else entropy_coef,
    )
    traces = load_trace_set(dataset, n_users, sys_cfg.n_edges, sys_cfg.slot_s, seed)
    env = RiskAverseEdgeEnv(sys_cfg, traces, distribution, seed=seed)
    agent = make_agent(method, sys_cfg, ppo_cfg, seed)
    reward_state = make_reward(method, sys_cfg)
    if reward_state is not None:
        reward_state.reset()

    curves = []
    pending = []
    obs = env.observe()
    start = time.perf_counter()
    for step in range(train_steps):
        weights, log_prob, value = agent.act(obs)
        next_obs, _, _, info = env.act_for_current_user(weights)
        actions = info["sampled_edges"]
        if isinstance(agent, (PPOPolicy, PGPolicy)):
            latent_action = agent.last_latent_action.copy()
            pending.append((obs, latent_action, actions, log_prob, value, next_obs))
        else:
            pending.append((obs, None, actions, 0.0, 0.0, next_obs))

        if "per_user_mean_latency" in info:
            user_latency = info["per_user_mean_latency"]
            finished_order = info["slot_user_order"]
            slot_rewards: list[float] = []
            for idx, (p_obs, p_latent, p_actions, p_logp, p_value, p_next_obs) in enumerate(pending):
                uid = finished_order[idx] if idx < len(finished_order) else idx
                latency = user_latency.get(uid, 0.0)
                reward = reward_for(method, latency, reward_state, sys_cfg)
                slot_rewards.append(reward)
                if isinstance(agent, (PPOPolicy, PGPolicy)):
                    assert p_latent is not None
                    agent.add(Transition(p_obs, p_latent, p_actions, p_logp, p_value, reward, False, p_next_obs))
                else:
                    agent.add(p_obs, p_actions, reward, p_next_obs, False)
            pending.clear()
            avg_slot_latency = float(np.mean(info["slot_latencies"])) if info["slot_latencies"] else 0.0
            curves.append(
                {
                    "step": step + 1,
                    "average_latency": avg_slot_latency,
                    "slot_tasks": info["slot_tasks"],
                    "elapsed_s": time.perf_counter() - start,
                    "episode_reward": float(np.sum(slot_rewards)) if slot_rewards else 0.0,
                    "mean_reward": float(np.mean(slot_rewards)) if slot_rewards else 0.0,
                    "min_reward": float(np.min(slot_rewards)) if slot_rewards else 0.0,
                    "max_reward": float(np.max(slot_rewards)) if slot_rewards else 0.0,
                }
            )
        if isinstance(agent, (PPOPolicy, PGPolicy)):
            if len(agent.buffer) >= ppo_cfg.rollout_size:
                agent.update()
        elif (step + 1) % max(ppo_cfg.batch_size, n_users) == 0:
            agent.update()
        obs = next_obs

    if isinstance(agent, (PPOPolicy, PGPolicy)) and len(agent.buffer) >= ppo_cfg.batch_size:
        agent.update()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"learning_{method}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "average_latency",
                "episode_reward",
                "mean_reward",
                "min_reward",
                "max_reward",
                "slot_tasks",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        writer.writerows(curves)
    return agent, curves


def evaluate_method(
    agent,
    method: str,
    dataset: str,
    n_users: int,
    distribution: str,
    test_steps: int,
    seed: int,
) -> dict:
    sys_cfg = SystemConfig(n_users=n_users)
    traces = load_trace_set(dataset, n_users, sys_cfg.n_edges, sys_cfg.slot_s, seed + 10_000)
    env = RiskAverseEdgeEnv(sys_cfg, traces, distribution, seed=seed + 10_000)
    obs = env.observe()
    inference_times = []
    for _ in range(test_steps):
        t0 = time.perf_counter()
        weights, _, _ = agent.act(obs, deterministic=True)
        inference_times.append(time.perf_counter() - t0)
        obs, _, _, _ = env.act_for_current_user(weights)
    summary = latency_summary(env.completed_latencies)
    summary.update(
        {
            "method": method,
            "dataset": dataset,
            "users": n_users,
            "distribution": distribution,
            "average_inference_ms": float(np.mean(inference_times) * 1000.0),
        }
    )
    return summary | {"latencies": env.completed_latencies}


def plot_learning(curves_by_method: dict[str, list[dict]], output: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method, rows in curves_by_method.items():
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ys = [r["average_latency"] for r in rows]
        plt.plot(xs, ys, label=method, linewidth=1.5)
    plt.xlabel("training user-decisions")
    plt.ylabel("average latency per task (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_pdf(latencies_by_method: dict[str, list[float]], output: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method, values in latencies_by_method.items():
        if len(values) < 2:
            continue
        plt.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, label=method)
    plt.xlabel("latency per task (s)")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def run_suite(args: argparse.Namespace) -> None:
    exp = ExperimentConfig(
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    methods = tuple(args.methods.split(",")) if args.methods else exp.methods
    datasets = tuple(args.datasets.split(",")) if args.datasets else exp.datasets
    users = tuple(int(x) for x in args.users.split(",")) if args.users else exp.users
    distributions = tuple(args.distributions.split(",")) if args.distributions else exp.distributions

    all_rows = []
    for dataset in datasets:
        for n_users in users:
            for dist in distributions:
                combo_dir = exp.output_dir / dataset / f"{n_users}u" / dist
                curves_by_method = {}
                latencies_by_method = {}
                for method in methods:
                    print(f"[{dataset} {n_users}u {dist}] training {method}")
                    agent, curves = train_method(
                        method,
                        dataset,
                        n_users,
                        dist,
                        exp.train_steps,
                        exp.seed,
                        combo_dir,
                        ppo_rollout_size=args.ppo_rollout_size,
                        learning_rate=args.learning_rate,
                        entropy_coef=args.entropy_coef,
                        dsr_eta=args.dsr_eta,
                        mu_return=args.mu_return,
                    )
                    curves_by_method[method] = curves
                    result = evaluate_method(agent, method, dataset, n_users, dist, exp.test_steps, exp.seed)
                    latencies_by_method[method] = result.pop("latencies")
                    all_rows.append(result)
                plot_learning(curves_by_method, combo_dir / "learning_curves.png")
                plot_pdf(latencies_by_method, combo_dir / "latency_pdf.png")

    exp.output_dir.mkdir(parents=True, exist_ok=True)
    with (exp.output_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "users",
            "distribution",
            "method",
            "average_latency",
            "cvar95",
            "cvar99",
            "n_tasks",
            "average_inference_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (exp.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Towards Risk-Averse Edge Computing with DRL")
    parser.add_argument("--train-steps", type=int, default=80_000)
    parser.add_argument("--test-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ExperimentConfig().output_dir))
    parser.add_argument("--methods", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--users", default="")
    parser.add_argument("--distributions", default="")
    parser.add_argument(
        "--ppo-rollout-size",
        type=int,
        default=None,
        help="PPO update interval. Defaults to PPOConfig.rollout_size=20000; try 2000-5000 for tuning.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--dsr-eta", type=float, default=None)
    parser.add_argument("--mu-return", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_suite(parse_args())
