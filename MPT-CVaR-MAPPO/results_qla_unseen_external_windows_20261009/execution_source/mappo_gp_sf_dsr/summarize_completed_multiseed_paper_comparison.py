from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "V4-AgeGate",
    "Towards-PPO-MV",
    "Towards-PPO-DSR",
    "M2PPO",
    "MCM-FDRL",
    "Fed-MADDPG",
]

SUMMARY_NAMES = {
    "V4-AgeGate": "DSR-MTG-Qpen-v4-agegate",
    "M2PPO": "M2PPO",
    "MCM-FDRL": "MCM-FDRL",
    "Towards-PPO-DSR": "Towards-PPO-DSR",
    "Towards-PPO-MV": "Towards-PPO-MV",
    "Fed-MADDPG": "Fed-MADDPG",
}

METRICS = [
    ("average_latency", "Average latency (s)", False),
    ("cvar99", "CVaR99 (s)", False),
    ("queue_oscillation", "Queue oscillation", False),
    ("global_demand_concentration", "Global demand concentration", False),
    ("portfolio_migration_rate", "Portfolio migration rate", False),
    ("p90_queue_lifetime", "P90 queue lifetime", False),
]

PLOT_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("global_demand_concentration", "Global demand concentration"),
    ("portfolio_migration_rate", "Portfolio migration rate"),
    ("p90_queue_lifetime", "P90 queue lifetime"),
]

COLORS = {
    "V4-AgeGate": "#1f77b4",
    "Towards-PPO-MV": "#8c564b",
    "Towards-PPO-DSR": "#9467bd",
    "M2PPO": "#2ca02c",
    "MCM-FDRL": "#d62728",
    "Fed-MADDPG": "#ff7f0e",
}

MARKERS = {
    "V4-AgeGate": "o",
    "Towards-PPO-MV": "P",
    "Towards-PPO-DSR": "D",
    "M2PPO": "s",
    "MCM-FDRL": "^",
    "Fed-MADDPG": "X",
}


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def parse_seed_delay(path: Path, root: Path) -> tuple[int | None, int | None]:
    seed = None
    delay = None
    for part in path.relative_to(root).parts:
        if part.startswith("seed_"):
            seed = int(part.split("_", 1)[1])
        elif part.startswith("delay_"):
            delay = int(part.split("_", 1)[1])
    return seed, delay


def to_float(x: object) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def collect_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary_*.csv")):
        rel = summary_path.relative_to(root)
        policy = rel.parts[0]
        seed, path_delay = parse_seed_delay(summary_path, root)
        raw = read_one_csv(summary_path)
        delay = int(to_float(raw.get("feedback_delay", path_delay)))
        row: dict[str, object] = {
            "policy": policy,
            "seed": seed,
            "feedback_delay": delay,
            "summary_path": str(summary_path),
        }
        for metric, _, _ in METRICS:
            row[metric] = to_float(raw.get(metric))
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary CSVs under {root}")
    return pd.DataFrame(rows)


def keep_complete(df: pd.DataFrame, seeds: list[int], delays: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_seeds = set(seeds)
    expected_delays = set(delays)
    complete = []
    status_rows = []
    for policy, sub in df.groupby("policy"):
        got_seeds = set(int(x) for x in sub["seed"].dropna().astype(int).unique())
        got_delays = set(int(x) for x in sub["feedback_delay"].dropna().astype(int).unique())
        grid = sub.groupby(["seed", "feedback_delay"]).size()
        done = int(len(grid))
        total = len(expected_seeds) * len(expected_delays)
        is_complete = got_seeds == expected_seeds and got_delays == expected_delays and done == total
        status_rows.append({
            "policy": policy,
            "completed_cells": done,
            "expected_cells": total,
            "seeds": ",".join(map(str, sorted(got_seeds))),
            "delays": ",".join(map(str, sorted(got_delays))),
            "included": is_complete,
        })
        if is_complete:
            complete.append(policy)
    return df[df["policy"].isin(complete)].copy(), pd.DataFrame(status_rows)


def order_df(df: pd.DataFrame) -> pd.DataFrame:
    order = {p: i for i, p in enumerate(METHOD_ORDER)}
    df = df.copy()
    df["_order"] = df["policy"].map(order).fillna(999)
    sort_cols = ["_order", "policy"]
    for col in ["feedback_delay", "seed"]:
        if col in df.columns:
            sort_cols.append(col)
    return df.sort_values(sort_cols).drop(columns=["_order"])


def aggregate_by_delay(df: pd.DataFrame) -> pd.DataFrame:
    keys = [m[0] for m in METRICS]
    out = df.groupby(["policy", "feedback_delay"], as_index=False)[keys].agg(["mean", "std"])
    out.columns = ["_".join(c).strip("_") for c in out.columns.to_flat_index()]
    return order_df(out)


def aggregate_overall(by_delay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in METHOD_ORDER:
        sub = by_delay[by_delay["policy"] == policy]
        if sub.empty:
            continue
        row = {"policy": policy}
        for metric, _, _ in METRICS:
            row[f"{metric}_mean_over_delay"] = float(sub[f"{metric}_mean"].mean())
            row[f"{metric}_worst_delay"] = int(sub.loc[sub[f"{metric}_mean"].idxmax(), "feedback_delay"])
            row[f"{metric}_worst_mean"] = float(sub[f"{metric}_mean"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "---"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} 卤 {std:.{digits}f}"


def make_delay_table(by_delay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in METHOD_ORDER:
        for delay in sorted(by_delay["feedback_delay"].unique()):
            sub = by_delay[(by_delay["policy"] == policy) & (by_delay["feedback_delay"] == delay)]
            if sub.empty:
                continue
            row = {"Method": policy, "Delay": int(delay)}
            for metric, label, _ in METRICS:
                row[label] = fmt(float(sub[f"{metric}_mean"].iloc[0]), float(sub[f"{metric}_std"].iloc[0]))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_by_delay(by_delay: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.8))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, PLOT_METRICS):
        for policy in METHOD_ORDER:
            sub = by_delay[by_delay["policy"] == policy].sort_values("feedback_delay")
            if sub.empty:
                continue
            x = sub["feedback_delay"].to_numpy()
            y = sub[f"{metric}_mean"].to_numpy()
            ax.plot(
                x, y,
                marker=MARKERS.get(policy, "o"),
                color=COLORS.get(policy),
                linewidth=2.3,
                markersize=6.5,
                label=policy,
            )
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Feedback delay", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Fair multi-seed comparison under stale feedback (mean, seeds 1-5)", fontsize=17, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=5, fontsize=12, frameon=True)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.87, hspace=0.38, wspace=0.28)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_overall(overall: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("average_latency_mean_over_delay", "Mean latency"),
        ("cvar99_mean_over_delay", "Mean CVaR99"),
        ("queue_oscillation_mean_over_delay", "Mean queue oscillation"),
        ("global_demand_concentration_mean_over_delay", "Mean demand concentration"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    policies = [p for p in METHOD_ORDER if p in set(overall["policy"])]
    for ax, (metric, label) in zip(axes, metrics):
        vals = [float(overall.loc[overall["policy"] == p, metric].iloc[0]) for p in policies]
        ax.bar(policies, vals, color=[COLORS.get(p) for p in policies], alpha=0.88)
        ax.set_title(label, fontsize=14)
        ax.tick_params(axis="x", labelrotation=25, labelsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Overall multi-seed comparison averaged over delays", fontsize=17, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.90, hspace=0.38, wspace=0.25)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(status: pd.DataFrame, by_delay: pd.DataFrame, overall: pd.DataFrame, out_path: Path) -> None:
    table = make_delay_table(by_delay)
    included = ", ".join([p for p in METHOD_ORDER if p in set(by_delay["policy"])])
    excluded = status[~status["included"]]
    lines = [
        "# Completed Multi-Seed Paper Baseline Comparison",
        "",
        "This report excludes incomplete methods and compares only methods with a full 5-seed x 5-delay grid.",
        "",
        "Setting: KAIST, N=30, poisson_bursty workload, task_compute=15MB, background_load=0, broadcast_period K=2, delays={0,1,2,4,8}, seeds={1,2,3,4,5}, train_steps=10000, test_steps=1000.",
        "",
        f"Included methods: {included}.",
        "",
        "## Completion Status",
        "",
        status.sort_values("policy").to_markdown(index=False),
        "",
    ]
    if not excluded.empty:
        lines.extend([
            "Incomplete methods are not used in the main fair comparison table/figures. In the current run, Fed-MADDPG is incomplete and should be reported separately after it finishes.",
            "",
        ])
    lines.extend([
        "## Overall Mean Across Delays",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-Delay Results",
        "",
        table.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Lower average latency means better mean task completion performance.",
        "- Lower CVaR99 means lower extreme tail latency risk.",
        "- Lower queue oscillation means more stable queue dynamics under stale feedback.",
        "- Lower global demand concentration means less systemic herding toward a small subset of servers.",
        "- Lower portfolio migration rate indicates fewer abrupt portfolio shifts; it is good only when latency and risk are also low.",
        "- Lower P90 queue lifetime means queues clear faster and congestion episodes are shorter.",
        "",
        "Across the completed 5-seed baselines, V4-AgeGate is the only method explicitly designed for stale server feedback: it combines age-aware score usage, queue-pressure penalty, and concentration control. The plot therefore should be read as a fair multi-seed same-environment comparison for the completed methods, not as a single-seed smoke test.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results_paper_algorithms_v4scenario_multiseed")
    p.add_argument("--out-dir", default="results_paper_algorithms_v4scenario_multiseed_completed")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_rows(root)
    complete, status = keep_complete(raw, args.seeds, args.delays)
    complete = order_df(complete)
    by_delay = aggregate_by_delay(complete)
    overall = aggregate_overall(by_delay)

    raw.to_csv(out_dir / "all_available_raw.csv", index=False)
    status.to_csv(out_dir / "method_completion_status.csv", index=False)
    complete.to_csv(out_dir / "completed_methods_raw.csv", index=False)
    by_delay.to_csv(out_dir / "completed_methods_by_delay_mean_std.csv", index=False)
    overall.to_csv(out_dir / "completed_methods_overall_mean.csv", index=False)
    plot_by_delay(by_delay, out_dir / "completed_multiseed_key_metrics_by_delay.png")
    plot_overall(overall, out_dir / "completed_multiseed_overall_bars.png")
    write_report(status, by_delay, overall, out_dir / "COMPLETED_MULTI_SEED_BASELINE_COMPARISON.md")

    print(f"Wrote {out_dir / 'completed_methods_by_delay_mean_std.csv'}")
    print(f"Wrote {out_dir / 'completed_methods_overall_mean.csv'}")
    print(f"Wrote {out_dir / 'completed_multiseed_key_metrics_by_delay.png'}")
    print(f"Wrote {out_dir / 'completed_multiseed_overall_bars.png'}")
    print(f"Wrote {out_dir / 'COMPLETED_MULTI_SEED_BASELINE_COMPARISON.md'}")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()

