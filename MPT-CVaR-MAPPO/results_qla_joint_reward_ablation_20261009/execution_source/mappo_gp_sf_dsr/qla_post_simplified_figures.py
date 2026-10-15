"""Regenerate Fig 4 (ablation), Fig 6 (cross-env), Fig 7 (N=40) from simplified rerun.

These figures replace the v4 paper-ready figures referenced in §6.4/§6.7/§6.8
so that the figures match the freshly rewritten Tables 5/6/7/8.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_qla_simplified_rerun" / "paper_figures"
SEEDS = [1, 2, 3, 4, 5]
DELAYS = [0, 1, 2, 4, 8]
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
METRIC_LABELS = {
    "average_latency": "Average latency (s)",
    "cvar99": "CVaR99 (s)",
    "queue_oscillation": "Queue oscillation",
    "portfolio_migration_rate": "Portfolio migration rate",
}
ABLATION_VARIANTS = {
    "QLA-MAPPO":      ("main",     "QLA-MAPPO-NoMargin"),
    "QL-NoBias":      ("ablation", "QL-NoBias-NoMargin"),
    "QL-Bias-Shuffle":("ablation", "QL-Bias-Shuffle-NoMargin"),
    "Flat-Bias-QL":   ("ablation", "Flat-Bias-QL-NoMargin"),
}
COLORS = {
    "QLA-MAPPO":        "#0072B2",
    "QL-NoBias":        "#D55E00",
    "QL-Bias-Shuffle":  "#009E73",
    "Flat-Bias-QL":     "#CC79A7",
}


def save_publication_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_variant(phase: str, dirname: str) -> pd.DataFrame:
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / phase / dirname
    for seed in SEEDS:
        for delay in DELAYS:
            csv = base / f"seed_{seed}" / f"delay_{delay}" / "KAIST" / "30u" / "poisson_bursty" / "summary_LCC-MAPPO.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if len(df) == 0:
                continue
            rec = df.iloc[0]
            rows.append({
                "method": dirname,
                "seed": seed,
                "feedback_delay": delay,
                "average_latency": float(rec["average_latency"]),
                "cvar99": float(rec["cvar99"]),
                "queue_oscillation": float(rec["queue_oscillation"]),
                "portfolio_migration_rate": float(rec["portfolio_migration_rate"]),
            })
    return pd.DataFrame(rows)


def fig4_ablation() -> None:
    """4-panel bar chart: variants × {avg_latency, cvar99, queue_osc, migration}.

    Error bars are 25-cell sample SD (not 95% CI). The caption states this
    explicitly. For 95% CI see Table 6 (paired) or core_ablation_seed_block_summary.csv.
    """
    data = {label: load_variant(phase, dirname) for label, (phase, dirname) in ABLATION_VARIANTS.items()}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, metric in zip(axes, METRICS):
        labels = list(data.keys())
        means = [data[l][metric].mean() for l in labels]
        stds = [data[l][metric].std(ddof=1) for l in labels]
        colors = [COLORS[l] for l in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Value labels above each error bar.
        y_max = max(m + s for m, s in zip(means, stds))
        offset = 0.04 * y_max
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + offset,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)
        # Headroom so labels are not clipped.
        ax.set_ylim(top=y_max + 2 * offset)
    fig.suptitle("QLA-MAPPO ablation (simplified rerun; 5 seeds × 5 delays = 25 cells per variant; "
                 "error bars = 25-cell sample SD)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_publication_figure(fig, "fig_ablation_simplified")
    print(f"[Fig 4] -> fig_ablation_simplified.{{png,pdf,svg}}", flush=True)


def load_crossenv() -> pd.DataFrame:
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / "crossenv" / "QLA-MAPPO-NoMargin"
    for ds in ["KAIST", "NCSU", "NewYork"]:
        for dist in ["uniform", "gaussian"]:
            for n in [15, 20, 25, 30]:
                for delay in [0, 8]:
                    csv = base / ds / dist / f"n{n}" / f"delay_{delay}" / ds / f"{n}u" / dist / "summary_LCC-MAPPO.csv"
                    if not csv.exists():
                        continue
                    df = pd.read_csv(csv)
                    if len(df) == 0:
                        continue
                    rec = df.iloc[0]
                    rows.append({
                        "dataset": ds,
                        "distribution": dist,
                        "users": n,
                        "feedback_delay": delay,
                        "average_latency": float(rec["average_latency"]),
                        "cvar99": float(rec["cvar99"]),
                        "queue_oscillation": float(rec["queue_oscillation"]),
                        "portfolio_migration_rate": float(rec["portfolio_migration_rate"]),
                    })
    return pd.DataFrame(rows)


def fig6_crossenv() -> None:
    """2x2 panel: by dataset, by distribution, by N, by δ."""
    df = load_crossenv()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    metrics_plot = [("average_latency", "Average latency (s)"), ("cvar99", "CVaR99 (s)")]
    datasets = ["KAIST", "NCSU", "NewYork"]
    user_counts = [15, 20, 25, 30]

    # Row 1: by dataset (left), by distribution (right)
    for ax, metric, label in [(axes[0, 0], "average_latency", "Average latency (s)"),
                              (axes[0, 1], "cvar99", "CVaR99 (s)")]:
        # By dataset
        means = [df.loc[df["dataset"].eq(d), metric].mean() for d in datasets]
        stds = [df.loc[df["dataset"].eq(d), metric].std(ddof=1) for d in datasets]
        x = np.arange(len(datasets))
        ax.bar(x, means, yerr=stds, capsize=4, color="#0072B2", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by dataset (48 cells total)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Row 2: by N (left), by δ (right)
    for ax, metric, label in [(axes[1, 0], "average_latency", "Average latency (s)"),
                              (axes[1, 1], "cvar99", "CVaR99 (s)")]:
        means = [df.loc[df["users"].eq(n), metric].mean() for n in user_counts]
        stds = [df.loc[df["users"].eq(n), metric].std(ddof=1) for n in user_counts]
        x = np.arange(len(user_counts))
        ax.bar(x, means, yerr=stds, capsize=4, color="#009E73", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in user_counts])
        ax.set_ylabel(label)
        ax.set_title(f"{label} by user count N (fixed KAIST checkpoint)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Cross-environment fixed-checkpoint test (QLA-MAPPO-NoMargin; 48 cells)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_publication_figure(fig, "fig_crossenv_simplified")
    print(f"[Fig 6] -> fig_crossenv_simplified.{{png,pdf,svg}}", flush=True)


def load_n40() -> pd.DataFrame:
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / "n40" / "QLA-MAPPO-NoMargin"
    for dist in ["uniform", "gaussian"]:
        for seed in SEEDS:
            csv = base / dist / f"seed_{seed}" / "KAIST" / "40u" / dist / "summary_LCC-MAPPO.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if len(df) == 0:
                continue
            rec = df.iloc[0]
            rows.append({
                "distribution": dist,
                "seed": seed,
                "average_latency": float(rec["average_latency"]),
                "cvar99": float(rec["cvar99"]),
                "queue_oscillation": float(rec["queue_oscillation"]),
                "portfolio_migration_rate": float(rec["portfolio_migration_rate"]),
            })
    return pd.DataFrame(rows)


def fig7_n40() -> None:
    df = load_n40()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, metric in zip(axes, METRICS):
        labels = sorted(df["distribution"].unique())
        means = [df.loc[df["distribution"].eq(d), metric].mean() for d in labels]
        stds = [df.loc[df["distribution"].eq(d), metric].std(ddof=1) for d in labels]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=["#D55E00", "#0072B2"], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n(5 seeds)" for d in labels], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("KAIST N=40 stress test (QLA-MAPPO-NoMargin; 5 seeds × 2 distributions = 10 cells)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_publication_figure(fig, "fig_n40_simplified")
    print(f"[Fig 7] -> fig_n40_simplified.{{png,pdf,svg}}", flush=True)


def fig3_main_by_delay() -> None:
    """QLA-MAPPO avg_latency and cvar99 by feedback delay, with SD bars."""
    df = load_variant("main", "QLA-MAPPO-NoMargin")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, metric, label in [(axes[0], "average_latency", "Average latency (s)"),
                              (axes[1], "cvar99", "CVaR99 (s)")]:
        means = [df.loc[df["feedback_delay"].eq(d), metric].mean() for d in DELAYS]
        stds = [df.loc[df["feedback_delay"].eq(d), metric].std(ddof=1) for d in DELAYS]
        x = np.arange(len(DELAYS))
        ax.errorbar(x, means, yerr=stds, fmt="o-", color="#0072B2", capsize=4, markersize=6, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"δ={d}" for d in DELAYS])
        ax.set_xlabel("Feedback delay (slots)")
        ax.set_ylabel(label)
        ax.grid(linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("QLA-MAPPO-NoMargin performance by feedback delay (KAIST, N=30, 5 seeds per delay)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_publication_figure(fig, "fig_main_by_delay_simplified")
    print(f"[Fig 3] -> fig_main_by_delay_simplified.{{png,pdf,svg}}", flush=True)


def main() -> None:
    fig3_main_by_delay()
    fig4_ablation()
    fig6_crossenv()
    fig7_n40()


if __name__ == "__main__":
    main()
