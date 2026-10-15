"""Constrained V4-AgeGate tuning against strong Lyapunov baselines.

Objective:
    Improve average latency and CVaR99 while keeping V4's queue oscillation and
    portfolio migration close to the current paper setting.

Default constraints:
    queue_oscillation <= 0.52
    portfolio_migration_rate <= 0.035

This script is intentionally small-candidate / high-budget.  The earlier 5k
factorial screen proved too noisy for final selection, so these candidates use
the same 10k training budget as the paper comparison.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "DSR-MTG-Qpen-v4-agegate"
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


@dataclass(frozen=True)
class Candidate:
    name: str
    method: str = SUMMARY_NAME
    args: dict[str, object] = field(default_factory=dict)
    note: str = ""


def candidates() -> list[Candidate]:
    base = {
        "alpha": 0.30,
        "q-penalty-weight": 0.08,
        "kappa-i": 0.02,
        "kappa-g": 0.02,
        "target-portfolio-entropy": 0.99,
        "feedback-gate-eta": 0.10,
        "max-supported-users": 50,
    }
    tail = {"beta-mean": 0.35, "beta-tail": 0.50, "beta-gap": 0.15}
    return [
        Candidate("v4_current", args=base, note="Current coordinate-selected V4 paper setting."),
        Candidate(
            "tail_push_smooth",
            args={
                **base,
                **tail,
                "alpha": 0.35,
                "q-penalty-weight": 0.10,
                "target-portfolio-entropy": 0.99,
            },
            note="More tail-sensitive, but keeps the high entropy target to protect migration.",
        ),
        Candidate(
            "tail_push_eta005",
            args={
                **base,
                **tail,
                "alpha": 0.35,
                "q-penalty-weight": 0.10,
                "feedback-gate-eta": 0.05,
                "target-portfolio-entropy": 0.99,
            },
            note="More willing to use stale score information; tests latency gain without lowering entropy.",
        ),
        Candidate(
            "entropy_985_tail",
            args={
                **base,
                **tail,
                "alpha": 0.35,
                "q-penalty-weight": 0.10,
                "target-portfolio-entropy": 0.985,
            },
            note="Slightly relaxes the nearly-uniform portfolio while keeping concentration penalty unchanged.",
        ),
        Candidate(
            "kappa_015_entropy_985",
            args={
                **base,
                **tail,
                "alpha": 0.35,
                "q-penalty-weight": 0.10,
                "kappa-i": 0.015,
                "kappa-g": 0.015,
                "target-portfolio-entropy": 0.985,
            },
            note="Relax both concentration and entropy slightly; highest latency upside, higher risk.",
        ),
        Candidate(
            "high_q_tail_smooth",
            args={
                **base,
                **tail,
                "alpha": 0.35,
                "q-penalty-weight": 0.12,
                "target-portfolio-entropy": 0.99,
            },
            note="Stronger stale-queue penalty while preserving smooth portfolios.",
        ),
        Candidate(
            "lyapprior_w10_smooth80",
            method="DSR-MTG-Qpen-v4-agegate-lyapprior",
            args={
                **base,
                "lyapunov-prior-weight": 0.10,
                "lyapunov-prior-v": 1.0,
                "lyapunov-prior-temperature": 2.5,
                "lyapunov-prior-smoothing": 0.80,
                "lyapunov-prior-max-step": 0.03,
            },
            note="Weak stale Lyapunov action prior with strong portfolio smoothing.",
        ),
        Candidate(
            "lyapprior_w20_smooth80",
            method="DSR-MTG-Qpen-v4-agegate-lyapprior",
            args={
                **base,
                "lyapunov-prior-weight": 0.20,
                "lyapunov-prior-v": 1.0,
                "lyapunov-prior-temperature": 2.5,
                "lyapunov-prior-smoothing": 0.80,
                "lyapunov-prior-max-step": 0.03,
            },
            note="Moderate stale Lyapunov prior, same migration trust region.",
        ),
        Candidate(
            "lyapprior_w20_smooth70",
            method="DSR-MTG-Qpen-v4-agegate-lyapprior",
            args={
                **base,
                "lyapunov-prior-weight": 0.20,
                "lyapunov-prior-v": 1.0,
                "lyapunov-prior-temperature": 2.5,
                "lyapunov-prior-smoothing": 0.70,
                "lyapunov-prior-max-step": 0.035,
            },
            note="Moderate prior with slightly faster adaptation; highest latency upside among safe candidates.",
        ),
        Candidate(
            "lyapprior_w30_smooth85",
            method="DSR-MTG-Qpen-v4-agegate-lyapprior",
            args={
                **base,
                "lyapunov-prior-weight": 0.30,
                "lyapunov-prior-v": 1.0,
                "lyapunov-prior-temperature": 2.5,
                "lyapunov-prior-smoothing": 0.85,
                "lyapunov-prior-max-step": 0.03,
            },
            note="Stronger prior but heavily smoothed to protect migration.",
        ),
    ]


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def out_base(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / candidate.name / f"seed_{seed}" / f"delay_{delay}"


def summary_path(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return (
        out_base(args, candidate, seed, delay)
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{candidate.method}.csv"
    )


def run_one(
    args: argparse.Namespace,
    candidate: Candidate,
    seed: int,
    delay: int,
) -> None:
    summary = summary_path(args, candidate, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] {candidate.name} seed={seed} delay={delay}", flush=True)
        return
    base = out_base(args, candidate, seed, delay)
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    values = {
        **COMMON_ARGS,
        **candidate.args,
        "method": candidate.method,
        "seed": seed,
        "feedback-delay": delay,
        "train-steps": args.train_steps,
        "test-steps": args.test_steps,
        "output-dir": str(base),
        "log-interval": args.log_interval,
    }
    if args.device:
        values["device"] = args.device
    cmd = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train"]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    print(f"[run] {candidate.name} seed={seed} delay={delay}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_summary(
    args: argparse.Namespace, candidate: Candidate, seed: int, delay: int
) -> dict[str, object]:
    path = summary_path(args, candidate, seed, delay)
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    output: dict[str, object] = {
        "candidate": candidate.name,
        "method": candidate.method,
        "seed": seed,
        "feedback_delay": delay,
        "note": candidate.note,
    }
    output.update(candidate.args)
    for metric in METRICS:
        try:
            output[metric] = float(row.get(metric, "nan"))
        except ValueError:
            output[metric] = float("nan")
    return output


def collect(args: argparse.Namespace, selected: list[Candidate], seeds: list[int], delays: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for candidate in selected:
        for seed in seeds:
            for delay in delays:
                path = summary_path(args, candidate, seed, delay)
                if path.exists():
                    rows.append(read_summary(args, candidate, seed, delay))
                else:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing[:50]))
    return pd.DataFrame(rows)


def rank(raw: pd.DataFrame, queue_limit: float, migration_limit: float) -> pd.DataFrame:
    grouped = raw.groupby("candidate", as_index=False)[
        ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate", "global_demand_concentration"]
    ].mean()
    grouped["queue_ok"] = grouped["queue_oscillation"] <= queue_limit
    grouped["migration_ok"] = grouped["portfolio_migration_rate"] <= migration_limit
    grouped["feasible"] = grouped["queue_ok"] & grouped["migration_ok"]
    grouped["latency_delta_vs_current_pct"] = np.nan
    grouped["cvar99_delta_vs_current_pct"] = np.nan
    current = grouped.loc[grouped["candidate"] == "v4_current"]
    if not current.empty:
        c = current.iloc[0]
        grouped["latency_delta_vs_current_pct"] = (grouped["average_latency"] / c["average_latency"] - 1.0) * 100.0
        grouped["cvar99_delta_vs_current_pct"] = (grouped["cvar99"] / c["cvar99"] - 1.0) * 100.0
    grouped = grouped.sort_values(
        ["feasible", "average_latency", "cvar99"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_v4_agegate_constrained_latency_tuning")
    parser.add_argument("--candidate", action="append", default=None, help="Run only named candidate(s).")
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--delays", default="0,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--queue-limit", type=float, default=0.52)
    parser.add_argument("--migration-limit", type=float, default=0.035)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_candidates = candidates()
    if args.candidate:
        wanted = set(args.candidate)
        selected = [candidate for candidate in all_candidates if candidate.name in wanted]
        missing = wanted - {candidate.name for candidate in selected}
        if missing:
            raise ValueError(f"Unknown candidates: {sorted(missing)}")
    else:
        selected = all_candidates
    seeds = _parse_ints(args.seeds)
    delays = _parse_ints(args.delays)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        for candidate in selected:
            for seed in seeds:
                for delay in delays:
                    run_one(args, candidate, seed, delay)
    raw = collect(args, selected, seeds, delays)
    raw.to_csv(Path(args.output_dir) / "constrained_tuning_raw.csv", index=False)
    ranking = rank(raw, args.queue_limit, args.migration_limit)
    ranking.to_csv(Path(args.output_dir) / "constrained_tuning_ranking.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
