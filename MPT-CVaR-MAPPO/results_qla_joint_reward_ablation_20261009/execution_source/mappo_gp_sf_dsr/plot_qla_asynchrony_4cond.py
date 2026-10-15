"""Bar chart of QLA-MAPPO 4-condition asynchrony experiment (TODO-EXP-9 extended).

Reads ``qla_shared_vs_heterogeneous_age_summary.csv`` produced by
``run_qla_heterogeneous_age_evidence`` and writes a 1×3 panel of avg_latency /
cvar99 / queue_oscillation with condition on x-axis and 5-seed SD bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results_qla_simplified_rerun" / "heterogeneous_age_5seeds" / "qla_shared_vs_heterogeneous_age_summary.csv"
OUT_DIR = ROOT / "results_qla_simplified_rerun" / "paper_figures"

CONDITIONS = ["sequential_fresh", "sync_fresh", "heterogeneous_age", "shared_stale"]
LABELS = {
    "sequential_fresh": "Sequential\nfresh δ=0",
    "sync_fresh": "Sync\nfresh δ=0",
    "heterogeneous_age": "Heterogeneous\nage",
    "shared_stale": "Sync shared\nstale δ=8",
}
COLORS = {
    "sequential_fresh": "#1b9e77",
    "sync_fresh": "#4daf4a",
    "heterogeneous_age": "#ff7f00",
    "shared_stale": "#e41a1c",
}
METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
]


def main() -> None:
    df = pd.read_csv(SRC)
    df["order"] = df["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    df = df.sort_values("order").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for ax, (metric, label) in zip(axes, METRICS):
        x = list(range(len(CONDITIONS)))
        means = [df.loc[df["condition"].eq(c), f"{metric}_mean"].iloc[0] for c in CONDITIONS]
        stds = [df.loc[df["condition"].eq(c), f"{metric}_std"].iloc[0] for c in CONDITIONS]
        colors = [COLORS[c] for c in CONDITIONS]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("QLA-MAPPO 4-condition asynchrony (KAIST, N=30, 5 eval seeds; same δ=8 checkpoint)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_asynchrony_4cond.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 12] -> fig_asynchrony_4cond.{{png,pdf,svg}}", flush=True)


if __name__ == "__main__":
    main()
