from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    alpha: float = 0.3
    q_penalty_weight: float = 0.05
    kappa_i: float = 0.01
    kappa_g: float = 0.01
    target_portfolio_entropy: float = 0.95
    feedback_gate_eta: float = 0.25


CANDIDATES = [
    Candidate("baseline"),
    Candidate("alpha020", alpha=0.20),
    Candidate("qpen003", q_penalty_weight=0.03),
    Candidate("qpen010", q_penalty_weight=0.10),
    Candidate("conc005", kappa_i=0.005, kappa_g=0.005),
    Candidate("entropy090", target_portfolio_entropy=0.90),
    Candidate("gateeta040", feedback_gate_eta=0.40),
]


SUMMARY_METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "p90_queue_lifetime",
    "global_demand_concentration",
    "server_selection_gini",
    "portfolio_migration_rate",
]


def summary_path(root: Path, cand: Candidate, delay: int, users: int, distribution: str) -> Path:
    return (
        root
        / cand.name
        / f"delay_{delay}"
        / "KAIST"
        / f"{users}u"
        / distribution
        / "summary_DSR-MTG-Qpen-v4-agegate.csv"
    )


def run_candidate(args: argparse.Namespace, cand: Candidate, delay: int) -> None:
    out_base = Path(args.output_dir) / cand.name / f"delay_{delay}"
    target = summary_path(Path(args.output_dir), cand, delay, args.users, args.distribution)
    if target.exists() and not args.force:
        print(f"[skip] {cand.name} delay={delay}: {target}")
        return

    cmd = [
        sys.executable,
        "-B",
        "-m",
        "mappo_gp_sf_dsr.train",
        "--method",
        "DSR-MTG-Qpen-v4-agegate",
        "--dataset",
        "KAIST",
        "--users",
        str(args.users),
        "--distribution",
        args.distribution,
        "--train-steps",
        str(args.train_steps),
        "--test-steps",
        str(args.test_steps),
        "--feedback-delay",
        str(delay),
        "--broadcast-period",
        str(args.broadcast_period),
        "--slot-s",
        str(args.slot_s),
        "--task-compute-mb",
        str(args.task_compute_mb),
        "--poisson-lambda",
        str(args.poisson_lambda),
        "--burst-prob",
        str(args.burst_prob),
        "--output-dir",
        str(out_base),
        "--log-interval",
        str(args.log_interval),
        "--alpha",
        str(cand.alpha),
        "--q-penalty-weight",
        str(cand.q_penalty_weight),
        "--kappa-i",
        str(cand.kappa_i),
        "--kappa-g",
        str(cand.kappa_g),
        "--target-portfolio-entropy",
        str(cand.target_portfolio_entropy),
        "--feedback-gate-eta",
        str(cand.feedback_gate_eta),
        "--seed",
        str(args.seed),
    ]
    print(f"[run] {cand.name} delay={delay}")
    log_dir = out_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "train_stdout.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_one(path: Path, cand: Candidate, delay: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    out: dict[str, object] = {
        "candidate": cand.name,
        "feedback_delay": delay,
        "alpha": cand.alpha,
        "q_penalty_weight": cand.q_penalty_weight,
        "kappa_i": cand.kappa_i,
        "kappa_g": cand.kappa_g,
        "target_portfolio_entropy": cand.target_portfolio_entropy,
        "feedback_gate_eta": cand.feedback_gate_eta,
    }
    for key in SUMMARY_METRICS:
        out[key] = float(row.get(key, "nan"))
    return out


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    score_metrics = [
        ("average_latency", 0.25),
        ("cvar99", 0.35),
        ("queue_oscillation", 0.20),
        ("p90_queue_lifetime", 0.05),
        ("global_demand_concentration", 0.15),
    ]
    scored = df.copy()
    scored["score"] = 0.0
    for metric, weight in score_metrics:
        values = scored[metric].astype(float)
        lo = values.min()
        hi = values.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            norm = np.zeros(len(scored))
        else:
            norm = (values - lo) / (hi - lo)
        scored[f"{metric}_norm"] = norm
        scored["score"] += weight * norm
    return scored


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored.groupby("candidate", as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_worst=("score", "max"),
            average_latency=("average_latency", "mean"),
            cvar99=("cvar99", "mean"),
            queue_oscillation=("queue_oscillation", "mean"),
            p90_queue_lifetime=("p90_queue_lifetime", "mean"),
            global_demand_concentration=("global_demand_concentration", "mean"),
            server_selection_gini=("server_selection_gini", "mean"),
            portfolio_migration_rate=("portfolio_migration_rate", "mean"),
            alpha=("alpha", "first"),
            q_penalty_weight=("q_penalty_weight", "first"),
            kappa_i=("kappa_i", "first"),
            kappa_g=("kappa_g", "first"),
            target_portfolio_entropy=("target_portfolio_entropy", "first"),
            feedback_gate_eta=("feedback_gate_eta", "first"),
        )
        .sort_values(["score_mean", "score_worst"], ascending=True)
    )
    return grouped


def write_report(out_dir: Path, agg: pd.DataFrame, scored: pd.DataFrame) -> None:
    best = agg.iloc[0]
    baseline = agg[agg["candidate"] == "baseline"].iloc[0]
    lines = [
        "# V4-AgeGate Stage-1 Tuning",
        "",
        "Screening setting: KAIST, N=30, poisson_bursty, task=15MB, K=2, delay={0,8}.",
        "Lower composite score is better. Score weights: CVaR99 0.35, average latency 0.25, queue oscillation 0.20, global demand concentration 0.15, P90 queue lifetime 0.05.",
        "",
        "## Best Candidate",
        "",
        f"- Best: `{best['candidate']}`",
        f"- Mean score: {best['score_mean']:.4f} vs baseline {baseline['score_mean']:.4f}",
        f"- Mean latency: {best['average_latency']:.4f}s vs baseline {baseline['average_latency']:.4f}s",
        f"- Mean CVaR99: {best['cvar99']:.4f}s vs baseline {baseline['cvar99']:.4f}s",
        f"- Mean queue oscillation: {best['queue_oscillation']:.4f} vs baseline {baseline['queue_oscillation']:.4f}",
        f"- Mean global demand concentration: {best['global_demand_concentration']:.6f} vs baseline {baseline['global_demand_concentration']:.6f}",
        "",
        "## Candidate Ranking",
        "",
        agg.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-Delay Scores",
        "",
        scored[
            [
                "candidate",
                "feedback_delay",
                "score",
                "average_latency",
                "cvar99",
                "queue_oscillation",
                "p90_queue_lifetime",
                "global_demand_concentration",
            ]
        ]
        .sort_values(["feedback_delay", "score"])
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Tuning Rule",
        "",
        "Use the best Stage-1 candidate only as a direction signal. Promote candidates whose mean score improves over baseline and whose worst-delay score does not degrade. Then re-run the best 2 candidates with full delay={0,1,2,4,8}, longer train steps, and multiple seeds.",
    ]
    (out_dir / "V4_AGEGATE_TUNING_STAGE1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_v4_agegate_tuning_stage1")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--test-steps", type=int, default=300)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for cand in CANDIDATES:
            for delay in args.delays:
                run_candidate(args, cand, delay)

    rows = []
    missing = []
    for cand in CANDIDATES:
        for delay in args.delays:
            path = summary_path(out_dir, cand, delay, args.users, args.distribution)
            if path.exists():
                rows.append(read_one(path, cand, delay))
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing))

    raw = pd.DataFrame(rows)
    scored = score_rows(raw)
    agg = aggregate(scored)
    raw.to_csv(out_dir / "stage1_raw_metrics.csv", index=False)
    scored.to_csv(out_dir / "stage1_scored_by_delay.csv", index=False)
    agg.to_csv(out_dir / "stage1_candidate_ranking.csv", index=False)
    write_report(out_dir, agg, scored)
    print(agg[["candidate", "score_mean", "score_worst", "average_latency", "cvar99", "queue_oscillation"]].to_string(index=False))
    print(f"Wrote {out_dir / 'V4_AGEGATE_TUNING_STAGE1.md'}")


if __name__ == "__main__":
    main()
