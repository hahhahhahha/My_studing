"""Validation-grid tuning for the Lyapunov-Q-local-stale-tuned baseline.

The tuned variant shares the exact algorithmic core of the paper-facing
Lyapunov-Q-local-stale baseline (stale local-candidate drift-plus-penalty
controller, no tail memory, no fresh state, no AgeGate, no CVaR reward, no
crowding penalty). Only (candidate_edges, V, temperature) are tuned, and the
tuning is performed on a separate validation grid (seed 0, delays {0, 4}).
The selected configuration is then re-evaluated on the test grid
(seeds 1-5, delays {0, 1, 2, 4, 8}) for the appendix table.

This script reports the full validation grid (all configs, not just the
winner) so reviewers can verify no cherry-picking.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .recent_mec_baselines import LYAPUNOV_LOCAL_STALE_TUNED
from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
METHOD = LYAPUNOV_LOCAL_STALE_TUNED
METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "portfolio_migration_rate",
    "mean_portfolio_entropy",
    "inference_time_ms",
]
DEFAULT_VALIDATION_SEEDS = [0]
DEFAULT_VALIDATION_DELAYS = [0, 4]
DEFAULT_CONFIRMATION_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_CONFIRMATION_DELAYS = [0, 1, 2, 4, 8]
DEFAULT_QUEUE_LIMIT = 0.52
DEFAULT_MIGRATION_LIMIT = 0.035


@dataclass(frozen=True)
class GridPoint:
    candidate_edges: int
    tradeoff_v: float
    temperature: float

    @property
    def name(self) -> str:
        return (
            f"edges{self.candidate_edges}"
            f"_v{self.tradeoff_v:.1f}"
            f"_t{self.temperature:.1f}"
        )


def grid() -> list[GridPoint]:
    points: list[GridPoint] = []
    for edges, v, temp in product((3, 5), (1.0, 2.0, 5.0), (1.0, 2.5)):
        points.append(GridPoint(edges, v, temp))
    return points


def out_base(args: argparse.Namespace, point: GridPoint, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / point.name / f"seed_{seed}" / f"delay_{delay}"


def summary_path(args: argparse.Namespace, point: GridPoint, seed: int, delay: int) -> Path:
    return (
        out_base(args, point, seed, delay)
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{METHOD}.csv"
    )


def run_one(
    args: argparse.Namespace,
    point: GridPoint,
    seed: int,
    delay: int,
) -> None:
    summary = summary_path(args, point, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] {point.name} seed={seed} delay={delay}", flush=True)
        return
    base = out_base(args, point, seed, delay)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    values = {
        **COMMON_ARGS,
        "method": METHOD,
        "seed": seed,
        "feedback-delay": delay,
        "train-steps": args.train_steps,
        "test-steps": args.test_steps,
        "output-dir": str(base),
        "log-interval": args.log_interval,
        "lyapunov-local-candidates": point.candidate_edges,
        "lyapunov-local-v": point.tradeoff_v,
        "lyapunov-local-temperature": point.temperature,
    }
    if args.device:
        values["device"] = args.device
    cmd = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.recent_mec_baselines"]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    print(f"[run] {point.name} seed={seed} delay={delay}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_summary(
    args: argparse.Namespace, point: GridPoint, seed: int, delay: int
) -> dict[str, object]:
    path = summary_path(args, point, seed, delay)
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    output: dict[str, object] = {
        "candidate": point.name,
        "candidate_edges": point.candidate_edges,
        "tradeoff_v": point.tradeoff_v,
        "temperature": point.temperature,
        "seed": seed,
        "feedback_delay": delay,
    }
    for metric in METRICS:
        try:
            output[metric] = float(row.get(metric, "nan"))
        except ValueError:
            output[metric] = float("nan")
    return output


def collect(
    args: argparse.Namespace,
    points: list[GridPoint],
    seeds: list[int],
    delays: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for point in points:
        for seed in seeds:
            for delay in delays:
                path = summary_path(args, point, seed, delay)
                if path.exists():
                    rows.append(read_summary(args, point, seed, delay))
                else:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing[:50]))
    return pd.DataFrame(rows)


def rank(
    raw: pd.DataFrame,
    queue_limit: float,
    migration_limit: float,
) -> pd.DataFrame:
    grouped = raw.groupby(
        ["candidate", "candidate_edges", "tradeoff_v", "temperature"], as_index=False
    )[["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate", "global_demand_concentration"]].mean()
    grouped["queue_ok"] = grouped["queue_oscillation"] <= queue_limit
    grouped["migration_ok"] = grouped["portfolio_migration_rate"] <= migration_limit
    grouped["feasible"] = grouped["queue_ok"] & grouped["migration_ok"]
    grouped = grouped.sort_values(
        ["feasible", "average_latency", "cvar99"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-grid tuning for Lyapunov-Q-local-stale-tuned."
    )
    parser.add_argument("--output-dir", default="results_lyapunov_local_stale_tuning")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--queue-limit", type=float, default=DEFAULT_QUEUE_LIMIT)
    parser.add_argument("--migration-limit", type=float, default=DEFAULT_MIGRATION_LIMIT)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--validation-seeds",
        default=",".join(str(s) for s in DEFAULT_VALIDATION_SEEDS),
        help="Comma-separated seeds used ONLY for tuning selection (must be disjoint from confirmation seeds).",
    )
    parser.add_argument(
        "--validation-delays",
        default=",".join(str(d) for d in DEFAULT_VALIDATION_DELAYS),
        help="Comma-separated delays used ONLY for tuning selection.",
    )
    parser.add_argument(
        "--confirmation-seeds",
        default=",".join(str(s) for s in DEFAULT_CONFIRMATION_SEEDS),
        help="Comma-separated seeds for the post-selection test grid.",
    )
    parser.add_argument(
        "--confirmation-delays",
        default=",".join(str(d) for d in DEFAULT_CONFIRMATION_DELAYS),
        help="Comma-separated delays for the post-selection test grid.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Stop after validation ranking; do not run the confirmation grid.",
    )
    args = parser.parse_args()

    val_seeds = [int(s) for s in args.validation_seeds.split(",") if s.strip()]
    val_delays = [int(d) for d in args.validation_delays.split(",") if d.strip()]
    conf_seeds = [int(s) for s in args.confirmation_seeds.split(",") if s.strip()]
    conf_delays = [int(d) for d in args.confirmation_delays.split(",") if d.strip()]

    overlap = set(val_seeds) & set(conf_seeds)
    if overlap:
        raise ValueError(
            f"Validation seeds must be disjoint from confirmation seeds; overlap={sorted(overlap)}"
        )

    points = grid()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for point in points:
            for seed in val_seeds:
                for delay in val_delays:
                    run_one(args, point, seed, delay)

    val_raw = collect(args, points, val_seeds, val_delays)
    val_raw.to_csv(Path(args.output_dir) / "validation_grid_raw.csv", index=False)
    val_ranking = rank(val_raw, args.queue_limit, args.migration_limit)
    val_ranking.to_csv(Path(args.output_dir) / "validation_grid_ranking.csv", index=False)
    print("[validation ranking]")
    print(val_ranking.to_string(index=False))

    best_row = val_ranking.iloc[0]
    best_point = GridPoint(
        candidate_edges=int(best_row["candidate_edges"]),
        tradeoff_v=float(best_row["tradeoff_v"]),
        temperature=float(best_row["temperature"]),
    )
    best_config = {
        "candidate": best_point.name,
        "candidate_edges": best_point.candidate_edges,
        "tradeoff_v": best_point.tradeoff_v,
        "temperature": best_point.temperature,
        "validation_average_latency": float(best_row["average_latency"]),
        "validation_cvar99": float(best_row["cvar99"]),
        "validation_queue_oscillation": float(best_row["queue_oscillation"]),
        "validation_portfolio_migration_rate": float(best_row["portfolio_migration_rate"]),
        "validation_global_demand_concentration": float(best_row["global_demand_concentration"]),
        "validation_feasible": bool(best_row["feasible"]),
        "validation_seeds": val_seeds,
        "validation_delays": val_delays,
        "confirmation_seeds": conf_seeds,
        "confirmation_delays": conf_delays,
        "queue_limit": args.queue_limit,
        "migration_limit": args.migration_limit,
    }
    with (Path(args.output_dir) / "best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)
    print(f"[best] {best_point.name}")

    if args.skip_confirmation:
        return

    if not args.collect_only:
        for seed in conf_seeds:
            for delay in conf_delays:
                run_one(args, best_point, seed, delay)

    conf_raw = collect(args, [best_point], conf_seeds, conf_delays)
    conf_raw.to_csv(Path(args.output_dir) / "confirmation_raw.csv", index=False)
    conf_summary = (
        conf_raw[["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate", "global_demand_concentration"]]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"index": "statistic"})
    )
    conf_summary.to_csv(Path(args.output_dir) / "confirmation_summary.csv", index=False)
    print("[confirmation summary]")
    print(conf_summary.to_string(index=False))


if __name__ == "__main__":
    main()
