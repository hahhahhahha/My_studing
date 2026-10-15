from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Params:
    alpha: float = 0.3
    q_penalty_weight: float = 0.05
    kappa_i: float = 0.01
    kappa_g: float = 0.01
    target_portfolio_entropy: float = 0.95
    feedback_gate_eta: float = 0.25


@dataclass(frozen=True)
class Sweep:
    name: str
    values: list[float]
    field: str
    paired_field: str | None = None


SWEEPS = [
    Sweep("q_penalty_weight", [round(x, 2) for x in np.arange(0.01, 0.101, 0.01)], "q_penalty_weight"),
    Sweep("alpha", [round(x, 1) for x in np.arange(0.1, 0.61, 0.1)], "alpha"),
    Sweep("kappa_i_g", [0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020], "kappa_i", "kappa_g"),
    Sweep("target_portfolio_entropy", [round(x, 2) for x in np.arange(0.85, 1.001, 0.02)], "target_portfolio_entropy"),
    Sweep("feedback_gate_eta", [round(x, 1) for x in np.arange(0.1, 0.51, 0.1)], "feedback_gate_eta"),
]


SCORE_WEIGHTS = {
    "cvar99": 0.35,
    "average_latency": 0.25,
    "queue_oscillation": 0.20,
    "global_demand_concentration": 0.15,
    "p90_queue_lifetime": 0.05,
}


READ_METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "p90_queue_lifetime",
    "global_demand_concentration",
    "server_selection_gini",
    "portfolio_migration_rate",
    "mean_portfolio_entropy",
]


def set_sweep_value(params: Params, sweep: Sweep, value: float) -> Params:
    kwargs = {sweep.field: value}
    if sweep.paired_field is not None:
        kwargs[sweep.paired_field] = value
    return replace(params, **kwargs)


def run_one(args: argparse.Namespace, sweep: Sweep, value: float, params: Params, delay: int) -> Path:
    value_tag = str(value).replace(".", "p")
    out_base = Path(args.output_dir) / sweep.name / f"value_{value_tag}" / f"delay_{delay}"
    summary = out_base / "KAIST" / f"{args.users}u" / args.distribution / "summary_DSR-MTG-Qpen-v4-agegate.csv"
    if summary.exists() and not args.force:
        print(f"[skip] {sweep.name}={value} delay={delay}", flush=True)
        return summary

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
        "--alpha",
        str(params.alpha),
        "--q-penalty-weight",
        str(params.q_penalty_weight),
        "--kappa-i",
        str(params.kappa_i),
        "--kappa-g",
        str(params.kappa_g),
        "--target-portfolio-entropy",
        str(params.target_portfolio_entropy),
        "--feedback-gate-eta",
        str(params.feedback_gate_eta),
        "--output-dir",
        str(out_base),
        "--log-interval",
        str(args.log_interval),
        "--seed",
        str(args.seed),
    ]
    print(f"[run] {sweep.name}={value} delay={delay}", flush=True)
    log_dir = out_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "train_stdout.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return summary


def read_summary(path: Path, sweep: Sweep, value: float, params: Params, delay: int, stage: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    out: dict[str, object] = {
        "stage": stage,
        "sweep": sweep.name,
        "value": value,
        "feedback_delay": delay,
        "alpha": params.alpha,
        "q_penalty_weight": params.q_penalty_weight,
        "kappa_i": params.kappa_i,
        "kappa_g": params.kappa_g,
        "target_portfolio_entropy": params.target_portfolio_entropy,
        "feedback_gate_eta": params.feedback_gate_eta,
    }
    for metric in READ_METRICS:
        out[metric] = float(row.get(metric, "nan"))
    return out


def score_delay_rows(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score"] = 0.0
    for metric, weight in SCORE_WEIGHTS.items():
        vals = scored[metric].astype(float)
        lo, hi = vals.min(), vals.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            norm = np.zeros(len(vals))
        else:
            norm = (vals - lo) / (hi - lo)
        scored[f"{metric}_norm"] = norm
        scored["score"] += weight * norm
    return scored


def summarize_sweep(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["stage", "sweep", "value"], as_index=False)
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
            mean_portfolio_entropy=("mean_portfolio_entropy", "mean"),
            alpha=("alpha", "first"),
            q_penalty_weight=("q_penalty_weight", "first"),
            kappa_i=("kappa_i", "first"),
            kappa_g=("kappa_g", "first"),
            target_portfolio_entropy=("target_portfolio_entropy", "first"),
            feedback_gate_eta=("feedback_gate_eta", "first"),
        )
        .sort_values(["stage", "score_mean", "score_worst"])
    )


def plot_coordinate(summary: pd.DataFrame, out_path: Path) -> None:
    sweeps = summary["sweep"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(15, 3.2 * len(sweeps)))
    if len(sweeps) == 1:
        axes = np.asarray([axes])
    for row_idx, sweep in enumerate(sweeps):
        sub = summary[summary["sweep"] == sweep].sort_values("value")
        best = sub.loc[sub["score_mean"].idxmin()]
        ax = axes[row_idx, 0]
        ax.plot(sub["value"], sub["score_mean"], marker="o", color="#2166ac", linewidth=2)
        ax.scatter([best["value"]], [best["score_mean"]], s=90, color="#b2182b", zorder=4, label="best")
        ax.set_title(f"{sweep}: composite score", fontsize=13)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Score, lower is better")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)

        ax2 = axes[row_idx, 1]
        ax2.plot(sub["value"], sub["cvar99"], marker="o", label="CVaR99", color="#b2182b", linewidth=2)
        ax2b = ax2.twinx()
        ax2b.plot(sub["value"], sub["average_latency"], marker="s", label="Avg latency", color="#2166ac", linewidth=2)
        ax2.set_title(f"{sweep}: key metrics", fontsize=13)
        ax2.set_xlabel("Parameter value")
        ax2.set_ylabel("CVaR99 (s)", color="#b2182b")
        ax2b.set_ylabel("Avg latency (s)", color="#2166ac")
        ax2.grid(alpha=0.25)
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, frameon=True, loc="best")
    fig.suptitle("V4-AgeGate Coordinate Hyperparameter Search", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_v4_agegate_coordinate_tuning")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--test-steps", type=int, default=200)
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
    current = Params()
    all_rows: list[dict] = []
    chosen_rows: list[dict] = []

    for stage, sweep in enumerate(SWEEPS, start=1):
        stage_rows = []
        print(f"\n=== Stage {stage}: sweep {sweep.name}, current={current} ===", flush=True)
        for value in sweep.values:
            params = set_sweep_value(current, sweep, value)
            for delay in args.delays:
                if not args.collect_only:
                    path = run_one(args, sweep, value, params, delay)
                else:
                    value_tag = str(value).replace(".", "p")
                    path = (
                        out_dir
                        / sweep.name
                        / f"value_{value_tag}"
                        / f"delay_{delay}"
                        / "KAIST"
                        / f"{args.users}u"
                        / args.distribution
                        / "summary_DSR-MTG-Qpen-v4-agegate.csv"
                    )
                row = read_summary(path, sweep, value, params, delay, stage)
                stage_rows.append(row)
                all_rows.append(row)
        stage_df = pd.DataFrame(stage_rows)
        stage_scored = score_delay_rows(stage_df)
        stage_summary = summarize_sweep(stage_scored)
        best = stage_summary.iloc[0]
        current = Params(
            alpha=float(best["alpha"]),
            q_penalty_weight=float(best["q_penalty_weight"]),
            kappa_i=float(best["kappa_i"]),
            kappa_g=float(best["kappa_g"]),
            target_portfolio_entropy=float(best["target_portfolio_entropy"]),
            feedback_gate_eta=float(best["feedback_gate_eta"]),
        )
        chosen_rows.append(
            {
                "stage": stage,
                "sweep": sweep.name,
                "chosen_value": float(best["value"]),
                "score_mean": float(best["score_mean"]),
                "score_worst": float(best["score_worst"]),
                **current.__dict__,
            }
        )
        print(
            f"[choose] {sweep.name}={best['value']} score={best['score_mean']:.4f} current={current}",
            flush=True,
        )

    raw = pd.DataFrame(all_rows)
    scored = raw.groupby("stage", group_keys=False).apply(score_delay_rows)
    summary = summarize_sweep(scored)
    chosen = pd.DataFrame(chosen_rows)
    raw.to_csv(out_dir / "coordinate_raw_metrics.csv", index=False)
    scored.to_csv(out_dir / "coordinate_scored_by_delay.csv", index=False)
    summary.to_csv(out_dir / "coordinate_sweep_summary.csv", index=False)
    chosen.to_csv(out_dir / "coordinate_chosen_path.csv", index=False)
    plot_coordinate(summary, out_dir / "v4_agegate_coordinate_tuning.png")
    report = [
        "# V4-AgeGate Coordinate Hyperparameter Tuning",
        "",
        "This run uses coordinate search: each stage fixes all other parameters, scans one parameter over a reasonable range, selects the best value by weighted composite score, then carries that value into the next stage.",
        "",
        "Score weights: CVaR99 0.35, average latency 0.25, queue oscillation 0.20, global demand concentration 0.15, P90 queue lifetime 0.05.",
        "",
        "## Chosen Path",
        "",
        chosen.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Final Low-Cost Candidate",
        "",
        f"`{current}`",
        "",
        "This is still a low-cost screening result. It must be confirmed with full delay={0,1,2,4,8}, train_steps=10000, test_steps=1000 before replacing the default V4-AgeGate setting.",
    ]
    (out_dir / "V4_AGEGATE_COORDINATE_TUNING.md").write_text("\n".join(report), encoding="utf-8")
    print("\nChosen path:")
    print(chosen.to_string(index=False))
    print(f"Wrote {out_dir / 'v4_agegate_coordinate_tuning.png'}")
    print(f"Wrote {out_dir / 'V4_AGEGATE_COORDINATE_TUNING.md'}")


if __name__ == "__main__":
    main()
