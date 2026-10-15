"""Create compact V4-AgeGate ablation tables and figures from existing results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "score-softmax": "score-softmax",
    "DSR-MTG-Qpen-v4": "V4",
    "DSR-MTG-Qpen-v4-agegate": "V4-AgeGate",
    "DSR-MTG-Qpen-v4-agegate-noConc": "AgeGate noConc",
    "DSR-MTG-Qpen-v4-agegate-noQpen": "AgeGate noQpen",
}

METRICS = [
    ("average_latency_mean_over_delays", "Avg latency"),
    ("cvar99_mean_over_delays", "CVaR99"),
    ("queue_oscillation_mean_over_delays", "Queue oscillation"),
    ("global_demand_concentration_mean_over_delays", "Demand concentration"),
    ("server_selection_gini_mean_over_delays", "Selection Gini"),
    ("portfolio_migration_rate_mean_over_delays", "Migration rate"),
]


def _friendly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    order = list(METHOD_LABELS.values())
    df["_order"] = df["label"].apply(lambda x: order.index(x) if x in order else 99)
    return df.sort_values("_order").drop(columns=["_order"])


def _plot(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[df["label"] != "score-softmax"].copy()
    colors = ["#4c78a8", "#1f77b4", "#f58518", "#e45756"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=False)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, METRICS):
        ax.bar(plot_df["label"], plot_df[col], color=colors[: len(plot_df)])
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="x", rotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("V4-AgeGate compact ablation summary", fontsize=20, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.9, hspace=0.35, wspace=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_md(df: pd.DataFrame, pct: pd.DataFrame, out_path: Path, plot_name: str) -> None:
    compact_cols = ["label"] + [col for col, _ in METRICS]
    pct = pct.copy()
    pct["label"] = pct["method"].map(METHOD_LABELS).fillna(pct["method"])
    lines = [
        "# V4-AgeGate Compact Ablation Summary",
        "",
        f"Plot: `{plot_name}`",
        "",
        "## Mean Over Delay Grid",
        "",
        df[compact_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Relative Change vs V4-AgeGate (%)",
        "",
        pct.drop(columns=["method"], errors="ignore").to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Reading",
        "",
        "- V4-AgeGate has the lowest mean latency and CVaR99 among the compact MAPPO variants.",
        "- Removing Q pressure hurts more than removing concentration: noQpen increases CVaR99 by about 10.86%, while noConc increases it by about 5.14%.",
        "- V4 without AgeGate is stable but has higher CVaR99, so AgeGate mainly improves tail-risk behavior while preserving queue stability.",
        "- score-softmax is kept as a stress reference, not a learned MAPPO ablation; it is much worse on latency, queue oscillation, demand concentration, and migration.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results_phase3_compact_ablation/phase3_compact_ablation_summary.csv")
    parser.add_argument("--pct", default="results_phase3_compact_ablation/phase3_compact_ablation_vs_agegate_pct.csv")
    parser.add_argument("--out-dir", default="results_phase3_compact_ablation/clean_summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _friendly(pd.read_csv(args.summary))
    pct = pd.read_csv(args.pct)
    df.to_csv(out_dir / "v4_agegate_ablation_clean_summary.csv", index=False)
    plot_path = out_dir / "v4_agegate_ablation_clean_summary.png"
    _plot(df, plot_path)
    _write_md(df, pct, out_dir / "V4_AGEGATE_ABLATION_CLEAN_SUMMARY.md", plot_path.name)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
