"""Create the citation-audited related-work positioning matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .config import ROOT


def main() -> None:
    output = ROOT / "results_qla_simplified_rerun" / "paper_figures"
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        "Risk-aware PPO [18]",
        "Lyapunov MEC [12,22]",
        "AoI-aware MEC [14–17,29]",
        "Delayed-info MFC/JSEQ [34,35]",
        "QLA-MAPPO (ours)",
    ]
    columns = ["Tail risk", "Feedback age", "Simultaneous users", "Portfolio action", "No fresh queue"]
    # 0=no, 1=partial/different object, 2=yes
    data = np.asarray(
        [
            [2, 0, 1, 2, 2],
            [1, 0, 2, 0, 0],
            [1, 2, 2, 1, 1],
            [0, 2, 2, 0, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=int,
    )
    cmap = ListedColormap(["#E9ECEF", "#F5C86A", "#2A9D8F"])
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), labels=columns, fontsize=10)
    ax.set_yticks(np.arange(len(rows)), labels=rows, fontsize=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    symbols = {0: "—", 1: "Partial", 2: "✓"}
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(col, row, symbols[int(data[row, col])], ha="center", va="center", fontsize=11,
                    color="white" if data[row, col] == 2 else "#343A40", fontweight="bold")
    for edge in np.arange(-0.5, data.shape[0], 1):
        ax.axhline(edge, color="white", linewidth=1.5)
    for edge in np.arange(-0.5, data.shape[1], 1):
        ax.axvline(edge, color="white", linewidth=1.5)
    ax.set_title("Positioning against citation-audited method families", fontsize=13, pad=18, fontweight="bold")
    ax.text(0.0, -0.16, "Partial includes a different freshness object or only a subset of the setting.",
            transform=ax.transAxes, fontsize=9, color="#555555")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"fig_literature_positioning.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
