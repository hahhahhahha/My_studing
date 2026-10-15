from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results_priceedge_v4_missing_evidence"
MAIN_RAW = ROOT / "results_priceedge_qtbias_paper_ready" / "main_baseline_raw_no_facr.csv"
OUT = EVIDENCE / "figures"
MECHANISM = ROOT / "results_priceedge_v4_problem_mechanism"
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_forest() -> None:
    df = pd.read_csv(EVIDENCE / "main_baseline_paired_stratified.csv")
    order = ["PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for panel, (ax, metric, title) in enumerate(
        zip(axes, ["average_latency", "cvar99"], ["Average latency improvement (s)", "CVaR99 improvement (s)"])
    ):
        sub = df[df["metric"].eq(metric)].set_index("baseline").reindex(order).reset_index()
        y = np.arange(len(sub))
        mean = sub["mean_improvement_baseline_minus_main"].to_numpy(float)
        low = sub["stratified_seed_bootstrap95_low"].to_numpy(float)
        high = sub["stratified_seed_bootstrap95_high"].to_numpy(float)
        ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o", capsize=3, color=OKABE_ITO[panel], lw=1.3)
        ax.axvline(0, color="0.35", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["baseline"])
        ax.invert_yaxis()
        ax.set_xlabel("Baseline − PriceEdge-QT-Bias")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.text(0.01, -0.22, "95% CI: seed-stratified paired bootstrap; n=5 seeds × 5 delays", transform=ax.transAxes, fontsize=6.5)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_main_baseline_paired_forest")


def feedback_age_figure() -> None:
    raw = pd.read_csv(MAIN_RAW)
    main = raw[raw["method"].eq("PriceEdge-QT-Bias")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, title in zip(axes, ["average_latency", "cvar99"], ["Average latency (s)", "CVaR99 (s)"]):
        for seed, group in main.groupby("seed"):
            group = group.sort_values("feedback_delay")
            ax.plot(group["feedback_delay"], group[metric], color="#999999", alpha=0.42, lw=0.8)
        grouped = main.groupby("feedback_delay")[metric]
        x = np.asarray(sorted(grouped.groups), dtype=float)
        mean = np.asarray([grouped.get_group(v).mean() for v in x], dtype=float)
        sd = np.asarray([grouped.get_group(v).std(ddof=1) for v in x], dtype=float)
        ax.errorbar(x, mean, yerr=sd, marker="o", color=OKABE_ITO[0], lw=1.8, capsize=3, label="Mean ± SD")
        ax.set_xticks(x)
        ax.set_xlabel(r"Feedback delay $\delta$")
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    diag = pd.read_csv(EVIDENCE / "feedback_age_diagnostics_summary.csv")
    avg = diag[(diag["method"].eq("PriceEdge-QT-Bias")) & (diag["metric"].eq("average_latency"))].iloc[0]
    cvar = diag[(diag["method"].eq("PriceEdge-QT-Bias")) & (diag["metric"].eq("cvar99"))].iloc[0]
    fig.text(
        0.5,
        -0.02,
        f"Relative δ=0→8 degradation: latency {100*avg['relative_stale_degradation_mean']:.1f}%; "
        f"CVaR99 {100*cvar['relative_stale_degradation_mean']:.1f}% (5 training seeds)",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_feedback_age_diagnostics")


def robustness_decomposition() -> None:
    specs = [
        ("fixed_checkpoint_robustness_by_dataset.csv", "dataset", ["KAIST", "NCSU", "NewYork"]),
        ("fixed_checkpoint_robustness_by_distribution.csv", "distribution", ["uniform", "gaussian"]),
        ("fixed_checkpoint_robustness_by_users.csv", "users", [15, 20, 25, 30]),
        ("fixed_checkpoint_robustness_by_delay.csv", "feedback_delay", [0, 8]),
    ]
    methods = ["PriceEdge-QT-Bias", "PPO-MV-A", "PPO-DSR-A", "QECO-A"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (filename, dim, order) in zip(axes.flat, specs):
        df = pd.read_csv(EVIDENCE / filename)
        for index, method in enumerate(methods):
            sub = df[df["method"].eq(method)].copy()
            sub["_order"] = sub[dim].apply(lambda value: order.index(int(value) if dim in {"users", "feedback_delay"} else value))
            sub = sub.sort_values("_order")
            ax.plot(np.arange(len(sub)), sub["cvar99_mean"], marker=["o", "s", "^", "D"][index], lw=1.5, color=OKABE_ITO[index], label=method)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel(dim.replace("_", " "))
        ax.set_ylabel("CVaR99 (s)")
        ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_fixed_checkpoint_robustness_decomposed")


def asynchrony_diagnostic() -> None:
    raw = pd.read_csv(MECHANISM / "shared_stale_asynchrony_raw.csv")
    order = ["synchronous_fresh", "synchronous_shared_stale", "heterogeneous_user_age", "sequential_fresh_updates"]
    labels = ["Sync fresh", "Sync shared stale", "Heterogeneous age", "Sequential fresh"]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, metric, title in zip(
        axes,
        ["average_latency", "cvar99", "queue_oscillation"],
        ["Avg latency (s)", "CVaR99 (s)", "Queue oscillation"],
    ):
        grouped = raw.groupby("condition")[metric]
        mean = np.asarray([grouped.get_group(key).mean() for key in order], dtype=float)
        sd = np.asarray([grouped.get_group(key).std(ddof=1) for key in order], dtype=float)
        bars = ax.bar(np.arange(4), mean, yerr=sd, capsize=2.5, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, mean):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Shared-stale synchronization mechanism (mean ± SD, 5 seeds)", y=1.04)
    fig.tight_layout()
    save(fig, "fig_shared_stale_asynchrony_colorblind")


def main() -> None:
    style()
    paired_forest()
    feedback_age_figure()
    robustness_decomposition()
    asynchrony_diagnostic()
    print(f"Wrote publication figures to {OUT}")


if __name__ == "__main__":
    main()
