from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results
from .models import MAPPOPolicy
from .rewards import MAPPORewarder
from .train import (
    augment_lcc_price_actor_observations,
    edge_capacities,
    eval_diagnostics,
    normalize_portfolios,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = ROOT / "results_priceedge_monoqt_formal_5x5" / "formal" / "PriceEdgeBias-QT-P0p75-Final"


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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


def checkpoint_path(root: Path, seed: int, delay: int) -> Path:
    return (
        root
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "checkpoint_LCC-MAPPO.pt"
    )


def build_system(users: int, args: argparse.Namespace) -> tuple[MAPPOSystemConfig, ServerScoreConfig, np.ndarray]:
    extra_actor_obs_dim = args.n_edges + 1
    sys_cfg = MAPPOSystemConfig(
        n_users=users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_package_mb=args.task_package_mb,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        max_supported_users=max(users, args.max_supported_users),
        include_feedback_age=True,
        feedback_age_norm=args.feedback_age_norm,
        extra_actor_obs_dim=extra_actor_obs_dim,
    )
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, args.seed)
    return sys_cfg, score_cfg, capacities


def build_env(dataset: str, distribution: str, users: int, delay: int, args: argparse.Namespace) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig, np.ndarray]:
    sys_cfg, score_cfg, capacities = build_system(users, args)
    traces = load_trace_set(
        dataset,
        users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=distribution,
        seed=args.seed,
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


def make_policy(sys_cfg: MAPPOSystemConfig, args: argparse.Namespace) -> MAPPOPolicy:
    cfg = MAPPOConfig(
        auto_portfolio_entropy=True,
        target_portfolio_entropy=args.target_portfolio_entropy,
        edgewise_price_actor=True,
        lcc_enabled=True,
        lcc_price_observation=True,
        lcc_price_scale_observation=True,
        lcc_edgewise_price_actor=True,
        lcc_price_components=args.lcc_price_components,
        lcc_price_logit_bias=args.lcc_price_logit_bias,
        lcc_v=args.lcc_v,
        lcc_migration_weight=args.lcc_migration_weight,
        lcc_cvar_alpha=args.lcc_cvar_alpha,
        lcc_cvar_budget=args.lcc_cvar_budget,
        lcc_var_lr=args.lcc_var_lr,
        lcc_concentration_factor=args.lcc_concentration_factor,
        lcc_stale_margin=args.lcc_stale_margin,
    )
    return MAPPOPolicy(sys_cfg, cfg, seed=args.checkpoint_seed, device=args.device)


def latency_rows(results: list[SlotResult], meta: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot, result in enumerate(results):
        for record in result.records:
            rows.append(
                {
                    **meta,
                    "slot": slot,
                    "user": record.user,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                }
            )
    return rows


def evaluate_priceedge(env: JointMAPPOEdgeEnv, sys_cfg: MAPPOSystemConfig, args: argparse.Namespace) -> list[SlotResult]:
    policy = make_policy(sys_cfg, args)
    ckpt = checkpoint_path(Path(args.checkpoint_root), args.checkpoint_seed, env.feedback_delay)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    policy.load_checkpoint(ckpt, actor_only=True)
    reward_cfg = MAPPOConfig(
        lcc_enabled=True,
        lcc_v=args.lcc_v,
        lcc_migration_weight=args.lcc_migration_weight,
        lcc_cvar_alpha=args.lcc_cvar_alpha,
        lcc_cvar_budget=args.lcc_cvar_budget,
        lcc_var_lr=args.lcc_var_lr,
        lcc_concentration_factor=args.lcc_concentration_factor,
        lcc_stale_margin=args.lcc_stale_margin,
        lcc_price_components=args.lcc_price_components,
        lcc_price_logit_bias=args.lcc_price_logit_bias,
    )
    rewarder = MAPPORewarder(reward_cfg, method="LCC-MAPPO")
    state: dict[str, np.ndarray] = {}
    previous: np.ndarray | None = None
    results: list[SlotResult] = []
    env.reset()
    for _ in range(args.test_steps):
        base_obs = env.actor_observations()
        actor_obs = augment_lcc_price_actor_observations(
            base_obs,
            env,
            args,
            rewarder,
            previous_portfolios=previous,
            state=state,
        )
        start = perf_counter()
        portfolios, _, _ = policy.act(actor_obs, deterministic=True)
        portfolios = normalize_portfolios(portfolios)
        decision_latency_s = perf_counter() - start
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s)
        prev_for_reward = None if previous is None else previous.copy()
        previous = np.asarray(result.portfolios, dtype=np.float32)
        rewarder.compute(
            result.per_user_mean_latency,
            result.per_user_tasks,
            result.system_average_latency,
            result.portfolio_stats,
            portfolios=result.portfolios,
            queues_before_mb=result.queue_load_before_mb,
            sys_cfg=sys_cfg,
            previous_portfolios=prev_for_reward,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )
        results.append(result)
    return results


def run_one(method: str, dataset: str, distribution: str, users: int, delay: int, args: argparse.Namespace) -> tuple[dict[str, object], list[SlotResult]]:
    if method != "PriceEdge-QT-Bias":
        raise ValueError(f"Unsupported method: {method}")
    env, sys_cfg, capacities = build_env(dataset, distribution, users, delay, args)
    results = evaluate_priceedge(env, sys_cfg, args)
    row: dict[str, object] = summarize_results(results)
    row.update(phase_diagnostics(results, sys_cfg))
    row.update(eval_diagnostics(results, capacities.astype(np.float64), sys_cfg.slot_s))
    row.update(
        {
            "method": method,
            "dataset": dataset,
            "distribution": distribution,
            "users": users,
            "feedback_delay": delay,
            "seed": args.seed,
            "checkpoint_seed": args.checkpoint_seed,
            "status": "ok",
            "checkpoint_protocol": "fixed KAIST/N=30/poisson-bursty checkpoint selected by feedback delay; no retraining in target environment",
            "lcc_price_components": args.lcc_price_components,
            "lcc_price_logit_bias": args.lcc_price_logit_bias,
        }
    )
    return row, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-checkpoint robustness for PriceEdge-QT-Bias MAPPO.")
    parser.add_argument("--output-dir", default="results_priceedge_qtbias_fixed_checkpoint_robustness")
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--datasets", default="KAIST,NCSU,NewYork")
    parser.add_argument("--distributions", default="uniform,gaussian")
    parser.add_argument("--users", default="15,20,25,30")
    parser.add_argument("--delays", default="0,8")
    parser.add_argument("--methods", default="PriceEdge-QT-Bias")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=80)
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)

    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--target-portfolio-entropy", type=float, default=0.99)
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--lcc-v", type=float, default=0.5)
    parser.add_argument("--lcc-migration-weight", type=float, default=0.05)
    parser.add_argument("--lcc-cvar-alpha", type=float, default=0.99)
    parser.add_argument("--lcc-cvar-budget", type=float, default=1.30)
    parser.add_argument("--lcc-var-lr", type=float, default=0.05)
    parser.add_argument("--lcc-concentration-factor", type=float, default=2.0)
    parser.add_argument("--lcc-stale-margin", type=float, default=1.0)
    parser.add_argument("--lcc-price-components", default="QT", choices=["QT", "QTZ", "QTH", "Full", "FullSafe"])
    parser.add_argument("--lcc-price-logit-bias", type=float, default=0.75)
    parser.add_argument("--lcc-price-intervention", default="aligned", choices=["aligned", "shuffle", "zero", "constant"])
    parser.add_argument("--lcc-price-shuffle", action="store_true")
    parser.add_argument("--lcc-no-stale-margin", action="store_true")
    parser.add_argument("--lcc-no-z-queue", action="store_true")
    parser.add_argument("--lcc-no-h-queue", action="store_true")

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # The trained PriceEdge-QT-Bias checkpoints use
    # [base stale observation, normalized QT price vector, log price scale].
    # The augmentation helper reads this flag from argparse.Namespace.
    args.lcc_price_scale_observation = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "priceedge_qtbias_fixed_checkpoint_raw.csv"
    latency_path = out_dir / "priceedge_qtbias_fixed_checkpoint_latency_samples.csv"
    rows: list[dict[str, object]] = []
    lat_rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, int, int, str]] = set()
    if summary_path.exists() and not args.force:
        import pandas as pd

        old = pd.read_csv(summary_path)
        rows = old.to_dict("records")
        for row in rows:
            if str(row.get("status", "ok")) == "ok":
                completed.add((str(row["dataset"]), str(row["distribution"]), int(row["users"]), int(row["feedback_delay"]), str(row["method"])))
        print(f"[resume] {len(completed)} completed rows", flush=True)

    datasets = parse_csv_list(args.datasets, str)
    distributions = parse_csv_list(args.distributions, str)
    users_list = parse_csv_list(args.users, int)
    delays = parse_csv_list(args.delays, int)
    methods = parse_csv_list(args.methods, str)
    total = len(datasets) * len(distributions) * len(users_list) * len(delays) * len(methods)
    done = 0
    for dataset in datasets:
        for distribution in distributions:
            for users in users_list:
                for delay in delays:
                    for method in methods:
                        done += 1
                        key = (dataset, distribution, users, delay, method)
                        print(f"[run] {done}/{total} {method} {dataset} {distribution} users={users} delay={delay}", flush=True)
                        if key in completed:
                            print("[skip] completed", flush=True)
                            continue
                        meta = {"method": method, "dataset": dataset, "distribution": distribution, "users": users, "feedback_delay": delay, "seed": args.seed}
                        try:
                            row, results = run_one(method, dataset, distribution, users, delay, args)
                            if delay == 8 and users in {30, 40}:
                                lat_rows.extend(latency_rows(results, meta))
                        except Exception as exc:
                            row = {**meta, "status": "error", "error": str(exc)}
                            print(f"[error] {row}", flush=True)
                        rows.append(row)
                        write_rows(summary_path, rows)
    write_rows(summary_path, rows)
    if lat_rows:
        write_rows(latency_path, lat_rows)
    print(f"Results: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
