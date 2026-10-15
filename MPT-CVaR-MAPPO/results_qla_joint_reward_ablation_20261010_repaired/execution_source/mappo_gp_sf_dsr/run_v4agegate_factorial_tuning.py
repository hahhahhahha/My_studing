"""Formal 2^(5-1) fractional-factorial parameter study for V4+LyapPrior.

This script is intentionally confirmatory rather than exploratory.  It reruns
the original five-factor fractional-factorial design using the accepted
V4+LyapPrior model, the paper training budget, and blocked main effects with
metric-scale standardization.

Default design:
    16 parameter combinations x 3 seeds x 3 feedback delays = 144 runs.

Factors:
    A alpha:                    0.20 / 0.30
    B q_penalty_weight:         0.05 / 0.08
    C kappa_i = kappa_g:        0.01 / 0.02
    D target_portfolio_entropy: 0.95 / 0.99
    E feedback_gate_eta:        0.10 / 0.25

Lyapunov-prior parameters are fixed to the accepted formal setting:
    prior weight = 0.20, smoothing = 0.80, temperature = 2.5, max step = 0.03.

Generator:
    E = -A*B*C*D

AgeGate implementation note:
    env.py appends delta / feedback_age_norm to the actor observation, but
    models.py multiplies it back by feedback_age_norm before applying the gate.
    Therefore the implemented gate is g(delta)=exp(-eta*delta), not
    exp(-eta*delta/8).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "DSR-MTG-Qpen-v4-agegate-lyapprior"
FIXED_LYAP_ARGS: dict[str, object] = {
    "lyapunov-prior-weight": 0.20,
    "lyapunov-prior-v": 1.0,
    "lyapunov-prior-temperature": 2.5,
    "lyapunov-prior-smoothing": 0.80,
    "lyapunov-prior-max-step": 0.03,
}
PRIMARY_METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "portfolio_migration_rate",
]
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
FACTORS = {
    "alpha": "a",
    "q_penalty_weight": "b",
    "kappa": "c",
    "target_portfolio_entropy": "d",
    "feedback_gate_eta": "e",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    alpha: float
    q_penalty_weight: float
    kappa: float
    target_portfolio_entropy: float
    feedback_gate_eta: float
    a: int
    b: int
    c: int
    d: int
    e: int


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def factorial_candidates() -> list[Candidate]:
    levels = {
        "alpha": {-1: 0.20, 1: 0.30},
        "q_penalty_weight": {-1: 0.05, 1: 0.08},
        "kappa": {-1: 0.01, 1: 0.02},
        "target_portfolio_entropy": {-1: 0.95, 1: 0.99},
        "feedback_gate_eta": {-1: 0.10, 1: 0.25},
    }
    candidates: list[Candidate] = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                for d in [-1, 1]:
                    e = -a * b * c * d
                    candidates.append(
                        Candidate(
                            name=(
                                f"A{'H' if a > 0 else 'L'}"
                                f"_B{'H' if b > 0 else 'L'}"
                                f"_C{'H' if c > 0 else 'L'}"
                                f"_D{'H' if d > 0 else 'L'}"
                                f"_E{'H' if e > 0 else 'L'}"
                            ),
                            alpha=levels["alpha"][a],
                            q_penalty_weight=levels["q_penalty_weight"][b],
                            kappa=levels["kappa"][c],
                            target_portfolio_entropy=levels["target_portfolio_entropy"][d],
                            feedback_gate_eta=levels["feedback_gate_eta"][e],
                            a=a,
                            b=b,
                            c=c,
                            d=d,
                            e=e,
                        )
                    )
    return sorted(candidates, key=lambda candidate: candidate.name)


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def method_file(method: str) -> str:
    return method.replace("/", "_")


def out_base(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / candidate.name / f"seed_{seed}" / f"delay_{delay}"


def summary_path(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return (
        out_base(args, candidate, seed, delay)
        / args.dataset
        / f"{args.users}u"
        / args.distribution
        / f"summary_{method_file(SUMMARY_NAME)}.csv"
    )


def build_command(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> list[str]:
    values: dict[str, object] = {
        **COMMON_ARGS,
        **FIXED_LYAP_ARGS,
        "method": SUMMARY_NAME,
        "dataset": args.dataset,
        "users": args.users,
        "distribution": args.distribution,
        "train-steps": args.train_steps,
        "test-steps": args.test_steps,
        "seed": seed,
        "feedback-delay": delay,
        "alpha": candidate.alpha,
        "q-penalty-weight": candidate.q_penalty_weight,
        "kappa-i": candidate.kappa,
        "kappa-g": candidate.kappa,
        "target-portfolio-entropy": candidate.target_portfolio_entropy,
        "feedback-gate-eta": candidate.feedback_gate_eta,
        "max-supported-users": args.max_supported_users,
        "output-dir": str(out_base(args, candidate, seed, delay)),
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
    return cmd


def run_one(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> None:
    summary = summary_path(args, candidate, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] {candidate.name} seed={seed} delay={delay}", flush=True)
        return
    log_dir = out_base(args, candidate, seed, delay) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {candidate.name} seed={seed} delay={delay}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(build_command(args, candidate, seed, delay), cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_summary(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> dict[str, object]:
    path = summary_path(args, candidate, seed, delay)
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    output: dict[str, object] = {
        "candidate": candidate.name,
        "seed": seed,
        "feedback_delay": delay,
        **asdict(candidate),
    }
    for metric in METRICS:
        output[metric] = safe_float(row.get(metric))
    return output


def collect(args: argparse.Namespace, candidates: list[Candidate], seeds: list[int], delays: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for candidate in candidates:
        for seed in seeds:
            for delay in delays:
                path = summary_path(args, candidate, seed, delay)
                if path.exists():
                    rows.append(read_summary(args, candidate, seed, delay))
                else:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing[:80]))
    return pd.DataFrame(rows)


def ci95(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    mean = float(np.mean(vals))
    half = 1.96 * float(np.std(vals, ddof=1)) / math.sqrt(vals.size)
    return mean - half, mean + half


def candidate_ranking(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby("candidate", as_index=False)[PRIMARY_METRICS].mean()
    metric_std = raw[PRIMARY_METRICS].std(ddof=1).replace(0.0, np.nan)
    standardized = grouped.copy()
    standardized_components: list[np.ndarray] = []
    for metric in PRIMARY_METRICS:
        best = float(grouped[metric].min())
        component = (grouped[metric].to_numpy(dtype=np.float64) - best) / float(metric_std[metric])
        standardized[f"{metric}_std_from_best"] = component
        standardized_components.append(component)
    mat = np.vstack(standardized_components).T
    standardized["standardized_rms_score"] = np.sqrt(np.nanmean(mat * mat, axis=1))
    standardized["standardized_max_abs_score"] = np.nanmax(np.abs(mat), axis=1)
    standardized = standardized.sort_values(["standardized_rms_score", "cvar99", "average_latency"]).reset_index(drop=True)
    standardized["rank"] = np.arange(1, len(standardized) + 1)
    return standardized


def estimate_factor_effects(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_std = raw[PRIMARY_METRICS].std(ddof=1).replace(0.0, np.nan)
    effect_rows: list[dict[str, object]] = []
    block_cols = ["seed", "feedback_delay"]
    for metric in PRIMARY_METRICS:
        for factor, code in FACTORS.items():
            block_effects: list[float] = []
            for _, block in raw.groupby(block_cols):
                high = block.loc[block[code] > 0, metric].mean()
                low = block.loc[block[code] < 0, metric].mean()
                block_effects.append(float(high - low))
            effects = np.asarray(block_effects, dtype=np.float64)
            lo, hi = ci95(effects)
            mean = float(np.mean(effects))
            std = float(metric_std[metric])
            effect_rows.append(
                {
                    "metric": metric,
                    "factor": factor,
                    "effect_high_minus_low": mean,
                    "effect_ci95_low": lo,
                    "effect_ci95_high": hi,
                    "metric_std": std,
                    "standardized_effect": mean / std,
                    "standardized_ci95_low": lo / std,
                    "standardized_ci95_high": hi / std,
                    "abs_standardized_effect": abs(mean / std),
                    "n_blocks": int(len(effects)),
                }
            )
    effects = pd.DataFrame(effect_rows)
    summary = (
        effects.groupby("factor", as_index=False)
        .agg(
            max_abs_standardized_effect=("abs_standardized_effect", "max"),
            rms_standardized_effect=("standardized_effect", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .sort_values(["max_abs_standardized_effect", "rms_standardized_effect"], ascending=False)
        .reset_index(drop=True)
    )
    summary["rank_by_max_abs"] = np.arange(1, len(summary) + 1)
    summary = summary.sort_values(["rms_standardized_effect", "max_abs_standardized_effect"], ascending=False).reset_index(drop=True)
    summary["rank_by_rms"] = np.arange(1, len(summary) + 1)
    return effects, summary


def write_schedule(args: argparse.Namespace, candidates: list[Candidate], seeds: list[int], delays: list[int]) -> list[tuple[Candidate, int, int]]:
    schedule = [(candidate, seed, delay) for candidate in candidates for seed in seeds for delay in delays]
    if args.randomize_order:
        rng = np.random.default_rng(args.schedule_seed)
        rng.shuffle(schedule)
    rows = [
        {
            "order": idx + 1,
            "candidate": candidate.name,
            "seed": seed,
            "feedback_delay": delay,
            **asdict(candidate),
        }
        for idx, (candidate, seed, delay) in enumerate(schedule)
    ]
    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "run_schedule.csv", index=False)
    return schedule


def write_report(args: argparse.Namespace, raw: pd.DataFrame, ranking: pd.DataFrame, effects: pd.DataFrame, factor_summary: pd.DataFrame) -> None:
    output = Path(args.output_dir)
    lines = [
        "# Formal V4+LyapPrior Fractional-Factorial Parameter Study",
        "",
        "## Design",
        "",
        "- Design: 2^(5-1) fractional factorial with generator `E = -A*B*C*D`.",
        f"- Runs: `{len(factorial_candidates())}` candidates x `{args.seeds}` seeds x `{args.delays}` feedback delays.",
        f"- Training budget: `{args.train_steps}` slots; test budget: `{args.test_steps}` slots.",
        "- Model: accepted `DSR-MTG-Qpen-v4-agegate-lyapprior` variant.",
        "- Fixed Lyapunov-prior parameters: `weight=0.20`, `smoothing=0.80`, `temperature=2.5`, `max_step=0.03`.",
        "- Blocking unit for main-effect confidence intervals: one complete seed-delay block.",
        "- AgeGate implementation: `g(delta)=exp(-eta*delta)`, because the normalized actor input `delta/8` is multiplied back by `feedback_age_norm` inside the actor.",
        "- Composite ranking uses standardized metrics; no raw multi-metric L2 norm is used.",
        "",
        "## Candidate ranking",
        "",
        ranking[
            [
                "rank",
                "candidate",
                "standardized_rms_score",
                "standardized_max_abs_score",
                *PRIMARY_METRICS,
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Factor sensitivity summary",
        "",
        factor_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Main effects with 95% CI",
        "",
        effects.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output / "V4_AGEGATE_FACTORIAL_TUNING.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal V4+LyapPrior fractional-factorial parameter study.")
    parser.add_argument("--output-dir", default="results_v4_lyapprior_factorial_formal")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--randomize-order", action="store_true", help="Randomize run order and save the schedule.")
    parser.add_argument("--schedule-seed", type=int, default=20260712)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = factorial_candidates()
    seeds = parse_ints(args.seeds)
    delays = parse_ints(args.delays)
    (output / "candidate_design.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], indent=2),
        encoding="utf-8",
    )
    (output / "fixed_lyapprior_args.json").write_text(
        json.dumps(FIXED_LYAP_ARGS, indent=2),
        encoding="utf-8",
    )
    schedule = write_schedule(args, candidates, seeds, delays)

    if not args.collect_only:
        for candidate, seed, delay in schedule:
            run_one(args, candidate, seed, delay)

    raw = collect(args, candidates, seeds, delays)
    ranking = candidate_ranking(raw)
    effects, factor_summary = estimate_factor_effects(raw)
    raw.to_csv(output / "factorial_raw.csv", index=False)
    ranking.to_csv(output / "factorial_candidate_ranking.csv", index=False)
    effects.to_csv(output / "factorial_main_effects_standardized.csv", index=False)
    factor_summary.to_csv(output / "factorial_factor_sensitivity_summary.csv", index=False)
    write_report(args, raw, ranking, effects, factor_summary)
    print(ranking[["rank", "candidate", "standardized_rms_score", "average_latency", "cvar99"]].to_string(index=False))
    print("\nFactor sensitivity:")
    print(factor_summary.to_string(index=False))
    print(f"\nWrote {output / 'V4_AGEGATE_FACTORIAL_TUNING.md'}")


if __name__ == "__main__":
    main()
