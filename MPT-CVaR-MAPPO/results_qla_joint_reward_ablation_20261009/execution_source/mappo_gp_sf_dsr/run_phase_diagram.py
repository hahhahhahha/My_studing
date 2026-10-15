from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv
from .evaluate import evaluate_policy
from .metrics import phase_diagnostics, summarize_results
from .models import MAPPOPolicy


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["users"]), int(row["feedback_delay"])), []).append(row)

    metrics = [
        "average_latency",
        "cvar95",
        "cvar99",
        "queue_oscillation",
        "portfolio_overlap",
        "portfolio_migration_rate",
        "mean_portfolio_entropy",
        "mean_portfolio_concentration",
        "global_demand_concentration",
        "server_selection_gini",
    ]
    out = []
    for (users, delay), group in sorted(grouped.items()):
        item: dict[str, float | int] = {"users": users, "feedback_delay": delay, "runs": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values))
        out.append(item)
    return out


def _plot_heatmaps(agg_rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    users = sorted({int(row["users"]) for row in agg_rows})
    delays = sorted({int(row["feedback_delay"]) for row in agg_rows})
    metrics = [
        ("cvar99_mean", "CVaR99"),
        ("queue_oscillation_mean", "Queue oscillation"),
        ("portfolio_overlap_mean", "Portfolio overlap"),
        ("portfolio_migration_rate_mean", "Portfolio migration rate"),
        ("global_demand_concentration_mean", "Global demand concentration"),
        ("server_selection_gini_mean", "Server selection Gini"),
    ]
    for key, title in metrics:
        matrix = np.full((len(users), len(delays)), np.nan, dtype=np.float64)
        for row in agg_rows:
            i = users.index(int(row["users"]))
            j = delays.index(int(row["feedback_delay"]))
            matrix[i, j] = float(row[key])

        fig, ax = plt.subplots(figsize=(7, 4.8))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(delays)), labels=delays)
        ax.set_yticks(np.arange(len(users)), labels=users)
        ax.set_xlabel("feedback delay δ (slots)")
        ax.set_ylabel("users N")
        ax.set_title(title)
        for i in range(len(users)):
            for j in range(len(delays)):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        filename = key.replace("_mean", "")
        fig.savefig(out_dir / f"heatmap_{filename}.png", dpi=180)
        plt.close(fig)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _evaluate_score_policy(env: JointMAPPOEdgeEnv, steps: int, temperature: float) -> list:
    results = []
    env.reset()
    for _ in range(steps):
        obs = env.actor_observations()
        scores = obs[:, -env.cfg.n_edges :]
        start = perf_counter()
        portfolios = _softmax(scores * temperature).astype(np.float32)
        decision_latency = perf_counter() - start
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency))
    return results


def _apply_score_ablation(obs: np.ndarray, n_edges: int, mode: str) -> np.ndarray:
    if mode == "none":
        return obs
    out = obs.copy()
    if mode == "zero":
        out[:, -n_edges:] = 0.0
    elif mode == "mean":
        out[:, -n_edges:] = np.mean(out[:, -n_edges:], axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown score ablation: {mode}")
    return out


def _evaluate_checkpoint_policy(
    policy: MAPPOPolicy,
    env: JointMAPPOEdgeEnv,
    steps: int,
    deterministic: bool,
    score_ablation: str,
) -> list:
    results = []
    env.reset()
    for _ in range(steps):
        obs = env.actor_observations()
        obs = _apply_score_ablation(obs, env.cfg.n_edges, score_ablation)
        start = perf_counter()
        portfolios, _, _ = policy.act(obs, deterministic=deterministic)
        decision_latency = perf_counter() - start
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency))
    return results


def run_one(args: argparse.Namespace, users: int, delay: int, seed: int) -> dict:
    sys_cfg = MAPPOSystemConfig(
        n_users=users,
        max_supported_users=max(users, 50),
        edge_compute_mb_s=63.0 * args.edge_compute_scale,
        task_compute_mb=10.0 * args.task_compute_scale,
    )
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    alg_cfg = MAPPOConfig(
        twin_critic=True,
        auto_portfolio_entropy=True,
        log_std_min=args.log_std_min,
        score=score_cfg,
    )
    traces = load_trace_set(
        args.dataset,
        users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        seed,
        trace_root=Path(args.trace_root),
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=seed,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=delay,
    )
    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --policy checkpoint")
        policy = MAPPOPolicy(sys_cfg, alg_cfg, seed=seed, device=args.device)
        policy.load_checkpoint(args.checkpoint, actor_only=True)
        if args.score_ablation == "none":
            results = evaluate_policy(policy, env, args.test_steps, deterministic=not args.stochastic_policy)
        else:
            results = _evaluate_checkpoint_policy(
                policy,
                env,
                args.test_steps,
                deterministic=not args.stochastic_policy,
                score_ablation=args.score_ablation,
            )
    elif args.policy == "score-softmax":
        results = _evaluate_score_policy(env, args.test_steps, args.score_policy_temperature)
    elif args.policy == "uniform":
        from .evaluate import random_policy_results

        results = random_policy_results(env, args.test_steps)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    summary = summarize_results(results)
    summary.update(phase_diagnostics(results, sys_cfg))
    summary.update(
        {
            "dataset": args.dataset,
            "distribution": args.distribution,
            "users": users,
            "feedback_delay": delay,
            "seed": seed,
            "test_steps": args.test_steps,
            "checkpoint": str(args.checkpoint),
            "policy": args.policy,
            "edge_compute_scale": args.edge_compute_scale,
            "task_compute_scale": args.task_compute_scale,
            "score_policy_temperature": args.score_policy_temperature,
            "score_ablation": args.score_ablation,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation-only N x delay phase diagram pilot.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--policy", default="checkpoint", choices=["checkpoint", "score-softmax", "uniform"])
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--distribution", default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--users", default="5,10,20,30,50")
    parser.add_argument("--delays", default="0,1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--test-steps", type=int, default=2000)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default="results_phase_diagram")
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--stochastic-policy", action="store_true")
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--score-policy-temperature", type=float, default=5.0)
    parser.add_argument("--score-ablation", default="none", choices=["none", "zero", "mean"])
    parser.add_argument("--log-std-min", type=float, default=-1.5)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.dataset / args.distribution
    out_dir.mkdir(parents=True, exist_ok=True)
    users_list = _parse_int_list(args.users)
    delays = _parse_int_list(args.delays)
    seeds = _parse_int_list(args.seeds)

    rows = []
    for users in users_list:
        for delay in delays:
            for seed in seeds:
                print({"users": users, "feedback_delay": delay, "seed": seed})
                try:
                    rows.append(run_one(args, users, delay, seed))
                    _write_csv(out_dir / "phase_runs.csv", rows)
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "distribution": args.distribution,
                            "users": users,
                            "feedback_delay": delay,
                            "seed": seed,
                            "error": str(exc),
                        }
                    )
                    _write_csv(out_dir / "phase_runs.csv", rows)

    valid_rows = [row for row in rows if "error" not in row]
    agg_rows = _aggregate(valid_rows)
    _write_csv(out_dir / "phase_summary.csv", agg_rows)
    _plot_heatmaps(agg_rows, out_dir)
    with (out_dir / "phase_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
