from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Policy:
    name: str
    alpha: float
    q_penalty_weight: float
    kappa_i: float
    kappa_g: float
    target_portfolio_entropy: float
    feedback_gate_eta: float


POLICIES = [
    Policy(
        name="default_v4_agegate",
        alpha=0.30,
        q_penalty_weight=0.05,
        kappa_i=0.01,
        kappa_g=0.01,
        target_portfolio_entropy=0.95,
        feedback_gate_eta=0.25,
    ),
    Policy(
        name="coordinate_tuned",
        alpha=0.30,
        q_penalty_weight=0.08,
        kappa_i=0.02,
        kappa_g=0.02,
        target_portfolio_entropy=0.99,
        feedback_gate_eta=0.10,
    ),
]


METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "p90_queue_lifetime",
    "global_demand_concentration",
    "server_selection_gini",
    "portfolio_migration_rate",
    "mean_portfolio_entropy",
    "inference_time_ms",
]


PLOT_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("global_demand_concentration", "Global demand concentration"),
]


def out_base(args: argparse.Namespace, policy: Policy, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / policy.name / f"seed_{seed}" / f"delay_{delay}"


def summary_path(args: argparse.Namespace, policy: Policy, seed: int, delay: int) -> Path:
    return (
        out_base(args, policy, seed, delay)
        / "KAIST"
        / f"{args.users}u"
        / args.distribution
        / "summary_DSR-MTG-Qpen-v4-agegate.csv"
    )


def run_one(args: argparse.Namespace, policy: Policy, seed: int, delay: int) -> None:
    summary = summary_path(args, policy, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] {policy.name} seed={seed} delay={delay}", flush=True)
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
        "--alpha",
        str(policy.alpha),
        "--q-penalty-weight",
        str(policy.q_penalty_weight),
        "--kappa-i",
        str(policy.kappa_i),
        "--kappa-g",
        str(policy.kappa_g),
        "--target-portfolio-entropy",
        str(policy.target_portfolio_entropy),
        "--feedback-gate-eta",
        str(policy.feedback_gate_eta),
        "--seed",
        str(seed),
        "--output-dir",
        str(out_base(args, policy, seed, delay)),
        "--log-interval",
        str(args.log_interval),
    ]
    print(f"[run] {policy.name} seed={seed} delay={delay}", flush=True)
    log_dir = out_base(args, policy, seed, delay) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "train_stdout.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def read_summary(path: Path, policy: Policy, seed: int, delay: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    out: dict[str, object] = {
        "policy": policy.name,
        "seed": seed,
        "feedback_delay": delay,
        "alpha": policy.alpha,
        "q_penalty_weight": policy.q_penalty_weight,
        "kappa_i": policy.kappa_i,
        "kappa_g": policy.kappa_g,
        "target_portfolio_entropy": policy.target_portfolio_entropy,
        "feedback_gate_eta": policy.feedback_gate_eta,
    }
    for metric in METRICS:
        out[metric] = float(row.get(metric, "nan"))
    return out


def collect(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    missing = []
    for policy in POLICIES:
        for seed in args.seeds:
            for delay in args.delays:
                path = summary_path(args, policy, seed, delay)
                if path.exists():
                    rows.append(read_summary(path, policy, seed, delay))
                else:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing summaries:\n" + "\n".join(missing))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_delay = (
        df.groupby(["policy", "feedback_delay"], as_index=False)
        .agg({metric: ["mean", "std"] for metric in METRICS})
    )
    by_delay.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in by_delay.columns
    ]
    mean = (
        df.groupby("policy", as_index=False)
        .agg({metric: ["mean", "std"] for metric in METRICS})
    )
    mean.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in mean.columns
    ]
    return by_delay, mean


def plot(by_delay: pd.DataFrame, out_dir: Path) -> None:
    colors = {"default_v4_agegate": "#2166ac", "coordinate_tuned": "#b2182b"}
    labels = {"default_v4_agegate": "Default V4-AgeGate", "coordinate_tuned": "Coordinate tuned"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.ravel()
    for ax, (metric, ylabel) in zip(axes, PLOT_METRICS):
        for policy in ["default_v4_agegate", "coordinate_tuned"]:
            sub = by_delay[by_delay["policy"] == policy].sort_values("feedback_delay")
            x = sub["feedback_delay"].to_numpy()
            y = sub[f"{metric}_mean"].to_numpy()
            e = sub[f"{metric}_std"].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2.2, markersize=6.5, color=colors[policy], label=labels[policy])
            ax.fill_between(x, y - e, y + e, color=colors[policy], alpha=0.16)
        ax.set_xlabel("Feedback delay", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=14)
        ax.grid(alpha=0.28)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Multi-Seed Confirmation: Tuned V4-AgeGate vs Default", fontsize=16, y=0.995)
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.08, hspace=0.36, wspace=0.25)
    fig.savefig(out_dir / "tuned_vs_default_multiseed.png", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, mean: pd.DataFrame) -> None:
    default = mean[mean["policy"] == "default_v4_agegate"].iloc[0]
    tuned = mean[mean["policy"] == "coordinate_tuned"].iloc[0]
    lines = [
        "# Multi-Seed Confirmation: Coordinate Tuned V4-AgeGate",
        "",
        "Setting: KAIST, N=30, poisson_bursty, task=15MB, K=2, delay={0,1,2,4,8}, seeds={0,1,2}, train_steps=10000, test_steps=1000.",
        "",
        "## Mean Across Seeds and Delays",
        "",
        mean.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Relative Change: Tuned vs Default",
        "",
        "| metric | change |",
        "|---|---:|",
    ]
    for metric in ["average_latency", "cvar99", "queue_oscillation", "global_demand_concentration", "portfolio_migration_rate"]:
        d = float(default[f"{metric}_mean"])
        t = float(tuned[f"{metric}_mean"])
        change = (t / d - 1.0) * 100.0 if abs(d) > 1e-12 else float("nan")
        lines.append(f"| {metric} | {change:.2f}% |")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Promote the tuned version only if it improves mean latency, CVaR99, and queue oscillation without increasing global demand concentration across the multi-seed average.",
        ]
    )
    if (
        tuned["average_latency_mean"] < default["average_latency_mean"]
        and tuned["cvar99_mean"] < default["cvar99_mean"]
        and tuned["queue_oscillation_mean"] < default["queue_oscillation_mean"]
        and tuned["global_demand_concentration_mean"] <= default["global_demand_concentration_mean"]
    ):
        lines.append("")
        lines.append("Decision: promote `coordinate_tuned` as the new tuned V4-AgeGate candidate.")
    else:
        lines.append("")
        lines.append("Decision: keep default V4-AgeGate as the main method; use tuned results as sensitivity evidence.")
    (out_dir / "MULTISEED_TUNED_VS_DEFAULT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_v4_agegate_tuned_multiseed_compare")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        for policy in POLICIES:
            for seed in args.seeds:
                for delay in args.delays:
                    run_one(args, policy, seed, delay)
    df = collect(args)
    by_delay, mean = aggregate(df)
    df.to_csv(out_dir / "tuned_vs_default_multiseed_raw.csv", index=False)
    by_delay.to_csv(out_dir / "tuned_vs_default_multiseed_by_delay.csv", index=False)
    mean.to_csv(out_dir / "tuned_vs_default_multiseed_mean.csv", index=False)
    plot(by_delay, out_dir)
    write_report(out_dir, mean)
    print(mean[["policy", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "global_demand_concentration_mean"]].to_string(index=False))
    print(f"Wrote {out_dir / 'MULTISEED_TUNED_VS_DEFAULT.md'}")
    print(f"Wrote {out_dir / 'tuned_vs_default_multiseed.png'}")


if __name__ == "__main__":
    main()
