from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, ROOT, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, slot_metrics, summarize_results
from .train import apply_lyapunov_prior, edge_capacities, eval_diagnostics


DEFAULT_OUTPUT = ROOT / "results_facr_ablation"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def build_env(args: argparse.Namespace, seed: int, delay: int) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig, np.ndarray]:
    sys_cfg = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_package_mb=args.task_package_mb,
        task_compute_mb=args.task_compute_mb,
        max_supported_users=max(args.users, args.max_supported_users),
        include_feedback_age=True,
        feedback_age_norm=args.feedback_age_norm,
    )
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    capacities = edge_capacities(args.n_edges, args.edge_compute_mb_s, args.capacity_mode, seed)
    traces = load_trace_set(
        args.dataset,
        args.users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        seed,
        trace_root=Path(args.trace_root),
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
    return env, sys_cfg, capacities


def run_prior_only(env: JointMAPPOEdgeEnv, args: argparse.Namespace) -> tuple[list[SlotResult], dict[str, list[float]]]:
    lyap_args = argparse.Namespace(
        lyapunov_prior_weight=1.0,
        lyapunov_prior_v=args.lyapunov_prior_v,
        lyapunov_prior_temperature=args.lyapunov_prior_temperature,
        lyapunov_prior_smoothing=args.lyapunov_prior_smoothing,
        lyapunov_prior_max_step=args.lyapunov_prior_max_step,
        lyapunov_prior_fresh_state=False,
    )
    results: list[SlotResult] = []
    diagnostics: dict[str, list[float]] = {}
    previous: np.ndarray | None = None
    env.reset()
    dummy_actor = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
    for _ in range(args.test_steps):
        started = perf_counter()
        portfolios = apply_lyapunov_prior(dummy_actor, env, lyap_args, previous, diagnostics=diagnostics)
        decision_latency_s = perf_counter() - started
        previous = portfolios.copy()
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s))
    return results, diagnostics


def summarize_run(
    results: list[SlotResult],
    sys_cfg: MAPPOSystemConfig,
    capacities: np.ndarray,
    diagnostics: dict[str, list[float]] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = summarize_results(results)
    summary.update(phase_diagnostics(results, sys_cfg))
    summary.update(eval_diagnostics(results, capacities, sys_cfg.slot_s))
    if diagnostics:
        for key, values in diagnostics.items():
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            summary[key] = float(np.mean(finite)) if finite.size else float("nan")
    if results:
        last = slot_metrics(results[-1])
        for key in [
            "mean_portfolio_entropy",
            "mean_portfolio_concentration",
            "global_demand_concentration",
            "max_server_expected_demand_share",
            "server_selection_gini",
        ]:
            summary.setdefault(key, last.get(key))
    return summary


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior-only FACR ablation controller.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
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
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--queue-constraint", type=float, default=10.0)
    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--lyapunov-prior-v", type=float, default=1.0)
    parser.add_argument("--lyapunov-prior-temperature", type=float, default=2.5)
    parser.add_argument("--lyapunov-prior-smoothing", type=float, default=0.80)
    parser.add_argument("--lyapunov-prior-max-step", type=float, default=0.03)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve() / "Prior-only"
    seeds = parse_csv_ints(args.seeds)
    delays = parse_csv_ints(args.delays)
    total = len(seeds) * len(delays)
    done = 0
    print(f"Running {total} Prior-only FACR ablation evaluations under {output_root}", flush=True)
    for seed in seeds:
        for delay in delays:
            out_dir = output_root / f"seed_{seed}" / f"delay_{delay}" / args.dataset / f"{args.users}u" / args.distribution
            summary_path = out_dir / "summary_Prior-only.csv"
            log_dir = output_root / f"seed_{seed}" / f"delay_{delay}" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "purpose": "FACR-MAPPO core ablation: prior-only controller",
                "variant": "Prior-only",
                "seed": seed,
                "feedback_delay": delay,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "note": "No actor training; stale Lyapunov prior with smoothing/max-step projection only.",
            }
            with (log_dir / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            if summary_path.exists() and not args.force:
                print(f"[skip] Prior-only seed={seed} delay={delay}", flush=True)
                done += 1
                continue
            print(f"[run] Prior-only seed={seed} delay={delay}", flush=True)
            env, sys_cfg, capacities = build_env(args, seed, delay)
            results, diagnostics = run_prior_only(env, args)
            summary = summarize_run(results, sys_cfg, capacities, diagnostics)
            summary.update(
                {
                    "method": "Prior-only",
                    "dataset": args.dataset,
                    "users": args.users,
                    "distribution": args.distribution,
                    "seed": seed,
                    "feedback_delay": delay,
                    "lyapunov_prior_weight": 1.0,
                    "lyapunov_prior_smoothing": args.lyapunov_prior_smoothing,
                    "lyapunov_prior_max_step": args.lyapunov_prior_max_step,
                }
            )
            write_summary(summary_path, summary)
            learning_rows = []
            for slot, result in enumerate(results, start=1):
                if slot == 1 or slot % 100 == 0 or slot == len(results):
                    row = {"slot": slot, "step": slot * sys_cfg.n_users}
                    row.update(slot_metrics(result))
                    learning_rows.append(row)
            write_csv(out_dir / "learning_Prior-only.csv", learning_rows)
            done += 1
            print(f"[done] {done}/{total} Prior-only seed={seed} delay={delay}", flush=True)
    print(f"Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
