"""Generate split introduction figures for the FACR-MAPPO manuscript.

Fig. 1: conceptual mechanism of stale feedback induced crowding.
Fig. 2: empirical score-softmax degradation under N=30 as feedback age grows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA = ROOT / "results_learned_policy_scale_n20_30_40" / "learned_policy_scale_by_delay.csv"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)


def _rounded_box(ax, xy, wh, color, label, sublabel="", lw=1.5, fc=None):
    x, y = xy
    w, h = wh
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        linewidth=lw,
        edgecolor=color,
        facecolor=fc if fc else (*color[:3], 0.08),
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.76, label, ha="center", va="center", weight="bold", color=color, fontsize=10.2)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.45, sublabel, ha="center", va="center", color="#333333", fontsize=8.5)
    return box


def make_mechanism_figure() -> None:
    _style()
    blue = (0.05, 0.28, 0.62)
    orange = (0.86, 0.36, 0.03)
    red = (0.78, 0.12, 0.19)
    green = (0.0, 0.45, 0.22)

    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _rounded_box(
        ax,
        (0.035, 0.21),
        (0.245, 0.58),
        blue,
        "Stale edge summary",
        r"$B(t-\delta)$: score / wait / queue" + "\n" + "broadcast once per period",
    )
    _rounded_box(
        ax,
        (0.385, 0.21),
        (0.245, 0.58),
        orange,
        "Synchronized response",
        "many users chase the same\napparently low-load edge",
    )
    _rounded_box(
        ax,
        (0.735, 0.21),
        (0.225, 0.58),
        red,
        "Queue spike",
        "transient overload\nand tail latency",
    )

    # User icons.
    user_xs = np.linspace(0.39, 0.61, 5)
    for k, x in enumerate(user_xs):
        ax.add_patch(patches.Circle((x, 0.78), 0.018, color=green, alpha=0.9))
        ax.plot([x, x], [0.70, 0.755], color=green, lw=1.2)
        ax.plot([x - 0.018, x + 0.018], [0.73, 0.73], color=green, lw=1.0)
        ax.plot([x, x - 0.015], [0.70, 0.66], color=green, lw=1.0)
        ax.plot([x, x + 0.015], [0.70, 0.66], color=green, lw=1.0)

    # Edge/server glyphs and queue bars.
    for idx, x in enumerate([0.085, 0.155, 0.225]):
        ax.add_patch(patches.Rectangle((x, 0.245), 0.038, 0.13, ec=blue, fc="white", lw=1.2))
        ax.add_patch(patches.Rectangle((x + 0.007, 0.345), 0.024, 0.010, color=blue, alpha=0.55))
        ax.add_patch(patches.Rectangle((x + 0.007, 0.322), 0.024, 0.010, color=blue, alpha=0.35))

    ax.annotate("", xy=(0.365, 0.50), xytext=(0.29, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color=blue))
    ax.annotate("", xy=(0.715, 0.50), xytext=(0.64, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color=orange))

    # Queue spike bars.
    bx = np.array([0.79, 0.825, 0.86, 0.895])
    heights = [0.12, 0.20, 0.39, 0.50]
    for x, h in zip(bx, heights):
        ax.add_patch(patches.Rectangle((x, 0.34), 0.02, h, color=red, alpha=0.78))
    ax.plot([0.78, 0.925], [0.34, 0.34], color="#333333", lw=0.9)
    ax.text(0.852, 0.83, "tail events", ha="center", va="center", color=red, fontsize=8.8)

    # Feedback-age arrow annotation.
    ax.annotate(
        "larger feedback age weakens\nstate-current correlation",
        xy=(0.49, 0.18),
        xytext=(0.49, 0.05),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#777777"),
    )
    _save(fig, "fig_intro_stale_feedback_mechanism")


def make_score_softmax_evidence() -> None:
    _style()
    df = pd.read_csv(DATA)
    sub = df[(df["users"] == 30) & (df["method"] == "score-softmax")].sort_values("feedback_delay")
    if sub.empty:
        raise RuntimeError("No score-softmax N=30 rows found in learned_policy_scale_by_delay.csv")

    x = sub["feedback_delay"].to_numpy()
    cvar = sub["cvar99_mean"].to_numpy()
    cvar_std = sub["cvar99_std"].to_numpy()
    qosc = sub["queue_oscillation_mean"].to_numpy()
    qosc_std = sub["queue_oscillation_std"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    colors = {"cvar": "#D95F02", "qosc": "#1B9E77"}

    axes[0].errorbar(x, cvar, yerr=cvar_std, marker="o", lw=2.0, capsize=3, color=colors["cvar"])
    axes[0].set_xlabel(r"Feedback age $\delta$ (slots)")
    axes[0].set_ylabel("CVaR99 latency (s)")
    axes[0].set_xticks(x)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].errorbar(x, qosc, yerr=qosc_std, marker="s", lw=2.0, capsize=3, color=colors["qosc"])
    axes[1].set_xlabel(r"Feedback age $\delta$ (slots)")
    axes[1].set_ylabel("Queue oscillation")
    axes[1].set_xticks(x)
    axes[1].grid(axis="y", alpha=0.25)

    for ax, vals in zip(axes, [cvar, qosc]):
        ax.annotate(
            f"+{(vals[-1] / vals[0] - 1) * 100:.0f}%",
            xy=(x[-1], vals[-1]),
            xytext=(-26, 12),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"),
        )

    fig.text(0.02, 0.965, "(a)", weight="bold", fontsize=10)
    fig.text(0.515, 0.965, "(b)", weight="bold", fontsize=10)
    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig_intro_scoresoftmax_delay_evidence")


def main() -> None:
    make_mechanism_figure()
    make_score_softmax_evidence()


if __name__ == "__main__":
    main()
