from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv
from .evaluate import random_policy_results
from .metrics import summarize_results
from .train import edge_capacities, eval_diagnostics, write_summary


ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "average_latency",
        "cvar99",
        "queue_oscillation",
        "portfolio_migration_rate",
        "global_demand_concentration",
        "mean_portfolio_entropy",
        "inference_time_ms",
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out: list[dict[str, object]] = []
    for variant, group in sorted(grouped.items()):
        item: dict[str, object] = {"variant": variant, "runs": len(group)}
        for metric in metrics:
            vals = []
            for row in group:
                try:
                    vals.append(float(row[metric]))
                except Exception:
                    pass
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
        out.append(item)
    return out


def build_env(args: argparse.Namespace, seed: int, delay: int) -> tuple[JointMAPPOEdgeEnv, np.ndarray]:
    sys_cfg = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_package_mb=args.task_package_mb,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        max_supported_users=max(args.max_supported_users, args.users),
        include_feedback_age=True,
        feedback_age_norm=args.feedback_age_norm,
    )
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, seed)
    traces = load_trace_set(
        args.dataset,
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        seed=seed,
        trace_root=TRACE_ROOT,
    )
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=seed + 10_000,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    return env, capacities.astype(np.float64)


def run_one(args: argparse.Namespace, seed: int, delay: int) -> dict[str, object]:
    env, capacities = build_env(args, seed, delay)
    results = random_policy_results(env, args.test_steps)
    summary: dict[str, object] = summarize_results(results)
    summary.update(eval_diagnostics(results, capacities, args.slot_s))
    summary.update(
        {
            "variant": "Uniform-Portfolio",
            "seed": seed,
            "delay": delay,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "test_steps": args.test_steps,
            "inference_time_ms": 0.0,
        }
    )
    out_dir = args.output_dir / "uniform" / "Uniform-Portfolio" / f"seed_{seed}" / f"delay_{delay}" / args.dataset / f"{args.users}u" / args.distribution
    write_summary(out_dir / "summary_uniform.csv", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic uniform portfolio baseline.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_uniform_portfolio_eval")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous")
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--queue-constraint", default=None)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for seed in parse_int_list(args.seeds):
        for delay in parse_int_list(args.delays):
            print(f"[run] Uniform-Portfolio seed={seed} delay={delay}", flush=True)
            rows.append(run_one(args, seed, delay))
    write_rows(args.output_dir / "uniform_raw.csv", rows)
    write_rows(args.output_dir / "uniform_summary.csv", aggregate_rows(rows))
    print(f"Uniform portfolio evaluation finished: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
