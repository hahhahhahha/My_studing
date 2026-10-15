"""4×4 panel figure for §6.7 cross-env breakdown.

Rows: 4 metrics (avg_latency, cvar99, queue_oscillation, portfolio_migration_rate)
Cols: 4 cuts — by dataset (3-level), by distribution (2-level), by N (4-level), by δ (2-level)

Each panel bars the group mean ± SD across the 48-cell fixed-checkpoint grid, so
readers see which axis drives the largest swings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results_qla_simplified_rerun" / "paper_figures" / "crossenv_breakdown.csv"
OUT_DIR = ROOT / "results_qla_simplified_rerun" / "paper_figures"

METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("portfolio_migration_rate", "Portfolio migration rate"),
]
GROUPS = [
    ("dataset", ["KAIST", "NCSU", "NewYork"], "Dataset"),
    ("distribution", ["uniform", "gaussian"], "Distribution"),
    ("users", [15, 20, 25, 30], "User count N"),
    ("feedback_delay", [0, 8], "Feedback delay δ"),
]


def main() -> None:
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(4, 4, figsize=(13.5, 11.0))
    color = "#0072B2"

    for r, (metric, mlabel) in enumerate(METRICS):
        for c, (col, levels, xlabel) in enumerate(GROUPS):
            ax = axes[r, c]
            means = [df.loc[df[col].eq(lv), metric].mean() for lv in levels]
            stds = [df.loc[df[col].eq(lv), metric].std(ddof=1) for lv in levels]
            x = np.arange(len(levels))
            ax.bar(x, means, yerr=stds, capsize=4, color=color, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([str(lv) for lv in levels], fontsize=8)
            if r == 0:
                ax.set_title(xlabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    fig.suptitle(
        "QLA-MAPPO fixed-checkpoint cross-env breakdown (48 cells; rows=metrics, cols=cuts)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"fig_crossenv_4x4_panel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 6b] -> fig_crossenv_4x4_panel.{{png,pdf,svg}}", flush=True)


if __name__ == "__main__":
    main()
