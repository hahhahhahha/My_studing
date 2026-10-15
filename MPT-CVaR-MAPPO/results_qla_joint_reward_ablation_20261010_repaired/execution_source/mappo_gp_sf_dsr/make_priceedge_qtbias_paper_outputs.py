from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_priceedge_qtbias_paper_ready"
FIG = OUT / "figures"

METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
METRIC_LABELS = {
    "average_latency": "Avg latency (s)",
    "cvar99": "CVaR99 (s)",
    "queue_oscillation": "Queue osc.",
    "portfolio_migration_rate": "Migration",
    "global_demand_concentration": "Global conc.",
    "mean_portfolio_entropy": "Entropy",
    "lcc_price_action_spearman": "Spearman(-price,w)",
}


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil((1.0 - alpha) * arr.size)))
    return float(np.mean(np.sort(arr)[-k:]))


def mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["runs"] = len(group)
        for metric in metrics:
            if metric not in group.columns:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def add_bar_labels(ax, bars, fmt="{:.3f}") -> None:
    ylim = ax.get_ylim()
    offset = 0.015 * (ylim[1] - ylim[0])
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, fmt.format(h), ha="center", va="bottom", fontsize=7.5, rotation=0)


def load_main_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    pe = pd.read_csv(ROOT / "results_priceedge_monoqt_formal_5x5" / "formal_raw.csv").copy()
    pe["method"] = "PriceEdge-QT-Bias"
    pe["feedback_delay"] = pe["delay"].astype(int)
    base = pd.read_csv(ROOT / "results_selected_baseline_comparison" / "selected_baseline_common_grid_raw.csv").copy()
    base = base[~base["method"].eq("FACR-MAPPO")].copy()
    common_cols = sorted(set(pe.columns).intersection(base.columns))
    df = pd.concat([pe[common_cols], base[common_cols]], ignore_index=True, sort=False)
    for metric in METRICS + ["global_demand_concentration", "mean_portfolio_entropy", "inference_time_ms"]:
        if metric in df:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    summary = mean_std(
        df,
        ["method"],
        METRICS + ["global_demand_concentration", "mean_portfolio_entropy", "inference_time_ms"],
    )
    order = ["PriceEdge-QT-Bias", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    summary["order"] = summary["method"].apply(lambda x: order.index(x) if x in order else 99)
    summary = summary.sort_values("order").drop(columns="order")
    by_delay = mean_std(df, ["method", "feedback_delay"], METRICS)
    by_delay["order"] = by_delay["method"].apply(lambda x: order.index(x) if x in order else 99)
    by_delay = by_delay.sort_values(["order", "feedback_delay"]).drop(columns="order")
    df.to_csv(OUT / "main_baseline_raw_no_facr.csv", index=False)
    summary.to_csv(OUT / "main_baseline_summary_no_facr.csv", index=False)
    by_delay.to_csv(OUT / "main_baseline_by_delay_no_facr.csv", index=False)
    return summary, by_delay


def plot_main_baseline(summary: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.6))
    colors = ["#2C7FB8", "#7FCDBB", "#C7E9B4", "#FDD49E", "#F4A582", "#B2ABD2"]
    for ax, metric in zip(axes, METRICS):
        vals = summary[f"{metric}_mean"].to_numpy(float)
        bars = ax.bar(np.arange(len(summary)), vals, color=colors[: len(summary)], edgecolor="white", linewidth=0.6)
        add_bar_labels(ax, bars)
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary["method"], rotation=35, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Main comparison under shared stale feedback", y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_main_baseline_no_facr.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_by_delay(by_delay: pd.DataFrame) -> None:
    set_style()
    methods = ["PriceEdge-QT-Bias", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    colors = dict(zip(methods, ["#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#D62728", "#8C564B"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, metric in zip(axes, ["average_latency", "cvar99"]):
        for method in methods:
            sub = by_delay[by_delay["method"].eq(method)].sort_values("feedback_delay")
            if sub.empty:
                continue
            ax.plot(sub["feedback_delay"], sub[f"{metric}_mean"], marker="o", lw=1.8, label=method, color=colors[method])
        ax.set_xlabel(r"Feedback delay $\delta$")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.25)
        ax.set_xticks([0, 1, 2, 4, 8])
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_main_by_delay_no_facr.{ext}", bbox_inches="tight")
    plt.close(fig)


def load_ablation() -> pd.DataFrame:
    return pd.read_csv(OUT / "paper_ablation_summary_no_priceonly_localcritic.csv")


def plot_ablation(summary: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.5))
    colors = ["#1F77B4", "#91BFDB", "#FC8D59", "#B2ABD2", "#A6D96A", "#FDD262", "#BDBDBD"]
    for ax, metric in zip(axes, METRICS):
        vals = summary[f"{metric}_mean"].to_numpy(float)
        bars = ax.bar(np.arange(len(summary)), vals, color=colors[: len(summary)], edgecolor="white", linewidth=0.6)
        add_bar_labels(ax, bars)
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary["Variant"], rotation=35, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PriceEdge-QT-Bias internal ablations", y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_ablation_no_priceonly_localcritic.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_interventions() -> None:
    path = ROOT / "results_priceedge_qt_bias_final_analysis" / "testtime_intervention_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = ["aligned", "shuffle", "zero", "constant"]
    df["order"] = df["intervention"].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("order")
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, metric in zip(axes, ["average_latency", "cvar99"]):
        bars = ax.bar(np.arange(len(df)), df[f"{metric}_mean"], color=["#1F77B4", "#FC8D59", "#BDBDBD", "#9E9AC8"])
        add_bar_labels(ax, bars)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df["intervention"], rotation=25, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Same-checkpoint price intervention", y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_testtime_price_intervention.{ext}", bbox_inches="tight")
    plt.close(fig)


def censoring_audit() -> pd.DataFrame:
    rows = []
    root = ROOT / "results_priceedge_monoqt_formal_5x5" / "formal" / "PriceEdgeBias-QT-P0p75-Final"
    for seed in [1, 2, 3, 4, 5]:
        for delay in [0, 1, 2, 4, 8]:
            path = root / f"seed_{seed}" / f"delay_{delay}" / "KAIST" / "30u" / "poisson_bursty" / "latency_samples_LCC-MAPPO.csv"
            df = pd.read_csv(path)
            if "phase" in df:
                df = df[df["phase"].eq("test")]
            lat = pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(float)
            rows.append(
                {
                    "method": "PriceEdge-QT-Bias",
                    "seed": seed,
                    "delay": delay,
                    "n_test_tasks": int(lat.size),
                    "tail99_sample_count": max(1, int(math.ceil(0.01 * lat.size))),
                    "avg_latency_uncapped": float(np.mean(lat)),
                    "cvar95_uncapped": empirical_cvar(lat, 0.95),
                    "cvar99_uncapped": empirical_cvar(lat, 0.99),
                    "avg_latency_capped_10s": float(np.mean(np.minimum(lat, 10.0))),
                    "cvar99_capped_10s": empirical_cvar(np.minimum(lat, 10.0), 0.99),
                    "max_latency": float(np.max(lat)),
                    "count_latency_ge_10s": int(np.sum(lat >= 10.0)),
                    "prob_latency_ge_10s": float(np.mean(lat >= 10.0)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "qtbias_latency_cvar_censoring_audit.csv", index=False)
    mean = mean_std(out, ["method"], ["n_test_tasks", "tail99_sample_count", "avg_latency_uncapped", "cvar99_uncapped", "cvar99_capped_10s", "max_latency", "prob_latency_ge_10s"])
    mean.to_csv(OUT / "qtbias_latency_cvar_censoring_audit_summary.csv", index=False)
    return out


def plot_ccdf() -> None:
    root = ROOT / "results_priceedge_monoqt_formal_5x5" / "formal" / "PriceEdgeBias-QT-P0p75-Final"
    arrays = []
    for seed in [1, 2, 3, 4, 5]:
        path = root / f"seed_{seed}" / "delay_8" / "KAIST" / "30u" / "poisson_bursty" / "latency_samples_LCC-MAPPO.csv"
        df = pd.read_csv(path)
        df = df[df["phase"].eq("test")]
        arrays.append(pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(float))
    arr = np.sort(np.concatenate(arrays))
    ccdf = 1.0 - np.arange(1, arr.size + 1, dtype=float) / arr.size
    ccdf = np.maximum(ccdf, 1.0 / arr.size)
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(arr, ccdf, lw=2.0, color="#1F77B4", label="PriceEdge-QT-Bias")
    ax.set_yscale("log")
    ax.set_xlabel("Task latency (s)")
    ax.set_ylabel(r"CCDF $P(T>x)$")
    ax.set_title(r"Tail latency distribution at $\delta=8$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_latency_ccdf_delay8.{ext}", bbox_inches="tight")
    plt.close(fig)


def load_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    pe = pd.read_csv(ROOT / "results_priceedge_qtbias_fixed_checkpoint_robustness" / "priceedge_qtbias_fixed_checkpoint_raw.csv")
    pe40 = pd.read_csv(ROOT / "results_priceedge_qtbias_fixed_checkpoint_stress_n40" / "priceedge_qtbias_fixed_checkpoint_raw.csv")
    base = pd.read_csv(ROOT / "results_fixed_checkpoint_robustness" / "fixed_checkpoint_robustness_raw.csv")
    base = base[~base["method"].eq("FACR-MAPPO")].copy()
    df = pd.concat([pe, pe40, base], ignore_index=True, sort=False)
    df = df[df["status"].astype(str).eq("ok")].copy()
    for metric in METRICS + ["global_demand_concentration", "inference_time_ms"]:
        if metric in df:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    robust = df[df["users"].astype(int).isin([15, 20, 25, 30])].copy()
    stress = df[df["users"].astype(int).eq(40)].copy()
    robust_summary = mean_std(robust, ["method"], METRICS)
    stress_summary = mean_std(stress, ["method", "distribution"], METRICS)
    robust.to_csv(OUT / "fixed_checkpoint_robustness_raw_with_qtbias_no_facr.csv", index=False)
    robust_summary.to_csv(OUT / "fixed_checkpoint_robustness_summary_with_qtbias_no_facr.csv", index=False)
    stress.to_csv(OUT / "n40_stress_raw_with_qtbias_no_facr.csv", index=False)
    stress_summary.to_csv(OUT / "n40_stress_summary_with_qtbias_no_facr.csv", index=False)
    return robust_summary, stress_summary


def plot_robustness(robust_summary: pd.DataFrame, stress_summary: pd.DataFrame) -> None:
    set_style()
    methods = ["PriceEdge-QT-Bias", "PPO-MV-A", "PPO-DSR-A", "QECO-A"]
    robust_summary["order"] = robust_summary["method"].apply(lambda x: methods.index(x) if x in methods else 99)
    robust_summary = robust_summary.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, metric in zip(axes, ["average_latency", "cvar99"]):
        bars = ax.bar(np.arange(len(robust_summary)), robust_summary[f"{metric}_mean"], color=["#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"])
        add_bar_labels(ax, bars)
        ax.set_xticks(np.arange(len(robust_summary)))
        ax.set_xticklabels(robust_summary["method"], rotation=30, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed-checkpoint cross-environment robustness", y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_fixed_checkpoint_robustness.{ext}", bbox_inches="tight")
    plt.close(fig)

    if stress_summary.empty:
        return
    stress_summary["label"] = stress_summary["method"] + "\n" + stress_summary["distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, metric in zip(axes, ["average_latency", "cvar99"]):
        bars = ax.bar(np.arange(len(stress_summary)), stress_summary[f"{metric}_mean"], color="#9ECAE1")
        add_bar_labels(ax, bars)
        ax.set_xticks(np.arange(len(stress_summary)))
        ax.set_xticklabels(stress_summary["label"], rotation=35, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("KAIST N=40 fixed-checkpoint stress test", y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG / f"fig_qtbias_n40_stress.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(main_summary: pd.DataFrame, ablation_summary: pd.DataFrame, robust_summary: pd.DataFrame, stress_summary: pd.DataFrame) -> None:
    lines = [
        "# PriceEdge-QT-Bias paper-ready experiment outputs",
        "",
        "Scope: shared stale-feedback multi-user cooperative portfolio offloading. FACR comparison, QT-PriceOnly, and Local-Critic are kept as internal audits, not main-paper claims.",
        "",
        "## Main adapted-baseline comparison",
        "",
        main_summary[["method", "runs", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "portfolio_migration_rate_mean"]].to_markdown(index=False),
        "",
        "## Main ablation set",
        "",
        ablation_summary[["Variant", "runs", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "portfolio_migration_rate_mean", "lcc_price_action_spearman_mean"]].to_markdown(index=False),
        "",
        "Interpretation: QT-NoBias and QT-Bias-Shuffle worsen CVaR99, supporting explicit price-to-logit bias and correct price-server alignment. Flat-Bias-QT is weaker on latency/CVaR, supporting edge-wise actor structure. The stale-margin and tail-state removals do not hurt under QT-only price, so they should not be claimed as core modules.",
        "",
        "## Fixed-checkpoint robustness",
        "",
        robust_summary[["method", "runs", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "portfolio_migration_rate_mean"]].to_markdown(index=False),
        "",
        "## KAIST N=40 stress",
        "",
        stress_summary[["method", "distribution", "runs", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "portfolio_migration_rate_mean"]].to_markdown(index=False),
        "",
        "## Generated figures",
        "",
        "- figures/fig_qtbias_main_baseline_no_facr.*",
        "- figures/fig_qtbias_main_by_delay_no_facr.*",
        "- figures/fig_qtbias_ablation_no_priceonly_localcritic.*",
        "- figures/fig_qtbias_testtime_price_intervention.*",
        "- figures/fig_qtbias_latency_ccdf_delay8.*",
        "- figures/fig_qtbias_fixed_checkpoint_robustness.*",
        "- figures/fig_qtbias_n40_stress.*",
    ]
    (OUT / "PRICEEDGE_QTBIAS_PAPER_READY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    main_summary, by_delay = load_main_baseline()
    plot_main_baseline(main_summary)
    plot_by_delay(by_delay)
    ablation_summary = load_ablation()
    plot_ablation(ablation_summary)
    plot_interventions()
    censoring_audit()
    plot_ccdf()
    robust_summary, stress_summary = load_robustness()
    plot_robustness(robust_summary, stress_summary)
    write_report(main_summary, ablation_summary, robust_summary, stress_summary)
    print(OUT / "PRICEEDGE_QTBIAS_PAPER_READY_REPORT.md")


if __name__ == "__main__":
    main()
