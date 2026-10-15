"""Staged parameter study for the improved V4 + stale Lyapunov-prior model.

The study separates reward-risk parameters from prior-stability parameters so
the final paper can argue that only a small number of meaningful knobs are
tuned jointly.

Round 1:
    Fix H_target=0.99, eta=0.25, kappa_i=kappa_g=0.02 and tune
    alpha x q_penalty_weight over a 3x3 grid.

Round 2:
    Fix the selected Round-1 alpha/q setting and tune prior_weight x smoothing
    over a 3x3 grid.

Round 3:
    Fix the selected Round-1/2 setting and run one-at-a-time local validation
    for eta, kappa, and target entropy.  This is a trend check, not a full
    5-seed x 5-delay confirmatory run.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
METHOD = "DSR-MTG-Qpen-v4-agegate-lyapprior"
PRIMARY_METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "portfolio_migration_rate",
]
METRICS = [
    *PRIMARY_METRICS,
    "cvar95",
    "mean_portfolio_entropy",
    "inference_time_ms",
]


BASE_PARAMS: dict[str, object] = {
    "alpha": 0.20,
    "q-penalty-weight": 0.08,
    "kappa-i": 0.02,
    "kappa-g": 0.02,
    "target-portfolio-entropy": 0.99,
    "feedback-gate-eta": 0.25,
    "max-supported-users": 50,
    "lyapunov-prior-weight": 0.20,
    "lyapunov-prior-v": 1.0,
    "lyapunov-prior-temperature": 2.5,
    "lyapunov-prior-smoothing": 0.80,
    "lyapunov-prior-max-step": 0.03,
}


@dataclass(frozen=True)
class Candidate:
    phase: str
    name: str
    args: dict[str, object] = field(default_factory=dict)
    note: str = ""


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def round1_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for alpha in [0.20, 0.30, 0.40]:
        for q_weight in [0.04, 0.08, 0.12]:
            candidates.append(
                Candidate(
                    phase="round1",
                    name=f"alpha{alpha:.2f}_q{q_weight:.2f}",
                    args={
                        "alpha": alpha,
                        "q-penalty-weight": q_weight,
                        "kappa-i": 0.02,
                        "kappa-g": 0.02,
                        "target-portfolio-entropy": 0.99,
                        "feedback-gate-eta": 0.25,
                    },
                    note="Reward-risk joint tuning with fixed entropy, age gate, and concentration penalty.",
                )
            )
    return candidates


def round2_candidates(alpha: float, q_weight: float) -> list[Candidate]:
    candidates: list[Candidate] = []
    for prior_weight in [0.10, 0.20, 0.30]:
        for smoothing in [0.60, 0.80, 0.90]:
            candidates.append(
                Candidate(
                    phase="round2",
                    name=f"w{prior_weight:.2f}_rho{smoothing:.2f}",
                    args={
                        "alpha": alpha,
                        "q-penalty-weight": q_weight,
                        "lyapunov-prior-weight": prior_weight,
                        "lyapunov-prior-smoothing": smoothing,
                    },
                    note="Lyapunov-prior strength and temporal smoothing tuning.",
                )
            )
    return candidates


def round3_candidates(alpha: float, q_weight: float, prior_weight: float, smoothing: float) -> list[Candidate]:
    fixed = {
        "alpha": alpha,
        "q-penalty-weight": q_weight,
        "lyapunov-prior-weight": prior_weight,
        "lyapunov-prior-smoothing": smoothing,
    }
    candidates: list[Candidate] = []
    for eta in [0.05, 0.10, 0.25, 0.40]:
        candidates.append(
            Candidate(
                phase="round3",
                name=f"eta{eta:.2f}",
                args={**fixed, "feedback-gate-eta": eta},
                note="One-at-a-time feedback-age gate validation.",
            )
        )
    for kappa in [0.01, 0.02, 0.03]:
        candidates.append(
            Candidate(
                phase="round3",
                name=f"kappa{kappa:.2f}",
                args={**fixed, "kappa-i": kappa, "kappa-g": kappa},
                note="One-at-a-time concentration penalty validation.",
            )
        )
    for entropy in [0.95, 0.99]:
        candidates.append(
            Candidate(
                phase="round3",
                name=f"entropy{entropy:.2f}",
                args={**fixed, "target-portfolio-entropy": entropy},
                note="One-at-a-time target entropy validation.",
            )
        )
    # Keep names unique if the fixed/default point appears in multiple local checks.
    deduped: dict[str, Candidate] = {}
    for candidate in candidates:
        suffix = 1
        name = candidate.name
        while name in deduped:
            suffix += 1
            name = f"{candidate.name}_{suffix}"
        deduped[name] = Candidate(candidate.phase, name, candidate.args, candidate.note)
    return list(deduped.values())


def out_base(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / candidate.phase / candidate.name / f"seed_{seed}" / f"delay_{delay}"


def summary_path(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> Path:
    return (
        out_base(args, candidate, seed, delay)
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{METHOD}.csv"
    )


def build_command(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int, train_steps: int, test_steps: int) -> list[str]:
    values: dict[str, object] = {
        **COMMON_ARGS,
        **BASE_PARAMS,
        **candidate.args,
        "method": METHOD,
        "seed": seed,
        "feedback-delay": delay,
        "train-steps": train_steps,
        "test-steps": test_steps,
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


def run_one(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int, train_steps: int, test_steps: int) -> None:
    summary = summary_path(args, candidate, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] {candidate.phase} {candidate.name} seed={seed} delay={delay}", flush=True)
        return
    log_dir = out_base(args, candidate, seed, delay) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {candidate.phase} {candidate.name} seed={seed} delay={delay}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(build_command(args, candidate, seed, delay, train_steps, test_steps), cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_summary(args: argparse.Namespace, candidate: Candidate, seed: int, delay: int) -> dict[str, object]:
    with summary_path(args, candidate, seed, delay).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    output: dict[str, object] = {
        "phase": candidate.phase,
        "candidate": candidate.name,
        "seed": seed,
        "feedback_delay": delay,
        "note": candidate.note,
        **BASE_PARAMS,
        **candidate.args,
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


def rank(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["phase", "candidate"], as_index=False)[PRIMARY_METRICS].mean()
    metric_std = raw[PRIMARY_METRICS].std(ddof=1).replace(0.0, np.nan)
    components: list[np.ndarray] = []
    ranked = grouped.copy()
    for metric in PRIMARY_METRICS:
        best = float(grouped[metric].min())
        comp = (grouped[metric].to_numpy(dtype=np.float64) - best) / float(metric_std[metric])
        ranked[f"{metric}_std_from_best"] = comp
        components.append(comp)
    mat = np.vstack(components).T
    ranked["standardized_rms_score"] = np.sqrt(np.nanmean(mat * mat, axis=1))
    ranked["standardized_max_abs_score"] = np.nanmax(np.abs(mat), axis=1)
    ranked = ranked.sort_values(["phase", "standardized_rms_score", "cvar99", "average_latency"]).reset_index(drop=True)
    ranked["rank_within_phase"] = ranked.groupby("phase").cumcount() + 1
    return ranked


def _best_from_ranking(path: Path, phase: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if not path.exists():
        return fallback
    df = pd.read_csv(path)
    sub = df[df["phase"] == phase].sort_values("rank_within_phase")
    if sub.empty:
        return fallback
    candidate = str(sub.iloc[0]["candidate"])
    if phase == "round1":
        # candidate format: alpha0.30_q0.08
        alpha = float(candidate.split("_")[0].removeprefix("alpha"))
        q_weight = float(candidate.split("_")[1].removeprefix("q"))
        return alpha, q_weight
    if phase == "round2":
        prior_weight = float(candidate.split("_")[0].removeprefix("w"))
        smoothing = float(candidate.split("_")[1].removeprefix("rho"))
        return prior_weight, smoothing
    return fallback


def run_phase(
    args: argparse.Namespace,
    candidates: list[Candidate],
    seeds: list[int],
    delays: list[int],
    train_steps: int,
    test_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not args.collect_only:
        for candidate in candidates:
            for seed in seeds:
                for delay in delays:
                    run_one(args, candidate, seed, delay, train_steps, test_steps)
    raw = collect(args, candidates, seeds, delays)
    ranked = rank(raw)
    return raw, ranked


def write_outputs(args: argparse.Namespace, phase: str, raw: pd.DataFrame, ranked: pd.DataFrame) -> None:
    output = Path(args.output_dir)
    raw.to_csv(output / f"{phase}_raw.csv", index=False)
    ranked.to_csv(output / f"{phase}_ranking.csv", index=False)
    lines = [
        f"# {phase} V4 + LyapPrior Parameter Study",
        "",
        "All multi-metric rankings use metric-standardized scores rather than raw L2 norms.",
        "",
        ranked[
            [
                "rank_within_phase",
                "candidate",
                "standardized_rms_score",
                "standardized_max_abs_score",
                *PRIMARY_METRICS,
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output / f"{phase}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged parameter study for V4 + Lyapunov prior.")
    parser.add_argument("--output-dir", default="results_v4_lyapprior_staged_parameter_study")
    parser.add_argument("--phase", choices=["round1", "round2", "round3", "all"], default="round1")
    parser.add_argument("--round1-seeds", default="1,2,3")
    parser.add_argument("--round2-seeds", default="1,2,3")
    parser.add_argument("--round3-seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--round1-best-alpha", type=float, default=None)
    parser.add_argument("--round1-best-q", type=float, default=None)
    parser.add_argument("--round2-best-prior-weight", type=float, default=None)
    parser.add_argument("--round2-best-smoothing", type=float, default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    delays = parse_ints(args.delays)
    (output / "base_params.json").write_text(json.dumps(BASE_PARAMS, indent=2), encoding="utf-8")

    if args.phase in {"round1", "all"}:
        raw, ranked = run_phase(args, round1_candidates(), parse_ints(args.round1_seeds), delays, args.train_steps, args.test_steps)
        write_outputs(args, "round1", raw, ranked)
        print("Round 1 ranking:")
        print(ranked[["rank_within_phase", "candidate", "standardized_rms_score", "average_latency", "cvar99"]].to_string(index=False))

    alpha = args.round1_best_alpha
    q_weight = args.round1_best_q
    if alpha is None or q_weight is None:
        alpha, q_weight = _best_from_ranking(output / "round1_ranking.csv", "round1", (0.30, 0.08))

    if args.phase in {"round2", "all"}:
        raw, ranked = run_phase(args, round2_candidates(alpha, q_weight), parse_ints(args.round2_seeds), delays, args.train_steps, args.test_steps)
        write_outputs(args, "round2", raw, ranked)
        print("Round 2 ranking:")
        print(ranked[["rank_within_phase", "candidate", "standardized_rms_score", "average_latency", "cvar99"]].to_string(index=False))

    prior_weight = args.round2_best_prior_weight
    smoothing = args.round2_best_smoothing
    if prior_weight is None or smoothing is None:
        prior_weight, smoothing = _best_from_ranking(output / "round2_ranking.csv", "round2", (0.20, 0.80))

    if args.phase in {"round3", "all"}:
        raw, ranked = run_phase(
            args,
            round3_candidates(alpha, q_weight, prior_weight, smoothing),
            parse_ints(args.round3_seeds),
            delays,
            args.train_steps,
            args.test_steps,
        )
        write_outputs(args, "round3", raw, ranked)
        print("Round 3 ranking:")
        print(ranked[["rank_within_phase", "candidate", "standardized_rms_score", "average_latency", "cvar99"]].to_string(index=False))


if __name__ == "__main__":
    main()
