"""Generate v4 paper §6.10/§6.11 figures and TODO-EXP-2/3/4 analyses.

Inputs (all already on disk):
  - results_qla_simplified_rerun/main/QLA-MAPPO-NoMargin  (5 seeds × 5 delays, no-margin)
  - results_selected_baseline_comparison/selected_baseline_common_grid_raw.csv
  - results_recent_mec_baselines_strong_multiseed/{Lyapunov-Q-fresh, Lyapunov-Q-tail, TODAuction-MAB-tail}/...
  - results_recent_mec_baselines_strong_qeco_multiseed/QECO-load-topk/...
  - results_qla_simplified_rerun/inference_audit/priceedge_qtbias_inference_cpu.csv
  - results_qla_simplified_rerun/mi_audit/qla_wait_action_mutual_information.csv
  - results_qla_simplified_rerun/heterogeneous_age/qla_shared_vs_heterogeneous_age_summary.csv

Outputs (in results_qla_simplified_rerun/paper_figures/):
  - paired_bootstrap_vs_baselines.csv       (TODO-EXP-2)
  - stale_degradation_slope.csv             (TODO-EXP-3)
  - ccdf_delta8.csv + tail_audit_delta8.csv (TODO-EXP-4)
  - fig_inference_breakdown.png             (§6.10)
  - fig_mi_vs_delta.png                     (§6.11 left)
  - fig_shared_vs_heterogeneous.png         (§6.11 right)
  - fig_ccdf_delta8.png                     (§6 tail)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_qla_simplified_rerun" / "paper_figures"
SEEDS = [1, 2, 3, 4, 5]
DELAYS = [0, 1, 2, 4, 8]
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
RNG_SEED = 20260725


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_publication_figure(fig: plt.Figure, stem: str) -> None:
    """Export line art as both high-resolution raster and editable vectors."""
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def load_qla_main() -> pd.DataFrame:
    """Load 25 main cells of QLA-MAPPO-NoMargin."""
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / "main" / "QLA-MAPPO-NoMargin"
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
                "method": "QLA-MAPPO-NoMargin",
                "seed": seed,
                "feedback_delay": delay,
                "average_latency": float(rec["average_latency"]),
                "cvar99": float(rec["cvar99"]),
                "queue_oscillation": float(rec["queue_oscillation"]),
                "portfolio_migration_rate": float(rec["portfolio_migration_rate"]),
                "global_demand_concentration": float(rec["global_demand_concentration"]),
                "mean_portfolio_entropy": float(rec["mean_portfolio_entropy"]),
                "inference_time_ms": float(rec["inference_time_ms"]),
            })
    return pd.DataFrame(rows)


# The paper baselines. JSEQ-A was added after the citation audit as the direct
# delayed-information queue-estimation mechanism baseline.
# Display names match selected_baseline_common_grid_raw.csv columns.
BASELINE_DIRS = {
    "PPO-MV-A":   ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-MV",
    "PPO-DSR-A":  ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-DSR",
    "QECO-A":     ROOT / "results_recent_mec_baselines_strong_qeco_multiseed" / "QECO-load-topk",
    "TODA-MAB-A": ROOT / "results_recent_mec_baselines_strong_multiseed" / "TODAuction-MAB-tail",
    "LyFOTO-LC":  ROOT / "results_recent_mec_local_lyapunov_multiseed" / "Lyapunov-Q-local-stale",
    "JSEQ-A":     ROOT / "results_delayed_information_baselines_formal" / "JSEQ-A",
    # MFC-DI-A is a prescreen only (seed=1 × δ={0,4,8}); paired CI is degenerate but
    # the row enters the slope + tail audit for transparency.
    "MFC-DI-A":   ROOT / "results_delayed_information_baselines_prescreen" / "MFC-DI-A",
}
BASELINE_SUMMARY_NAME = {
    "PPO-MV-A":   "summary_Towards-PPO-MV.csv",
    "PPO-DSR-A":  "summary_Towards-PPO-DSR.csv",
    "QECO-A":     "summary_QECO-load-topk.csv",
    "TODA-MAB-A": "summary_TODAuction-MAB-tail.csv",
    "LyFOTO-LC":  "summary_Lyapunov-Q-local-stale.csv",
    "JSEQ-A":     "summary_JSEQ-A.csv",
    "MFC-DI-A":   "summary_MFC-DI-A.csv",
}
BASELINE_LATENCY_NAME = {
    "PPO-MV-A":   "latency_samples_Towards-PPO-MV.csv",
    "PPO-DSR-A":  "latency_samples_Towards-PPO-DSR.csv",
    "QECO-A":     "latency_samples_QECO-load-topk.csv",
    "TODA-MAB-A": "latency_samples_TODAuction-MAB-tail.csv",
    "LyFOTO-LC":  "latency_samples_Lyapunov-Q-local-stale.csv",
    "JSEQ-A":     "latency_samples_JSEQ-A.csv",
    "MFC-DI-A":   "latency_samples_MFC-DI-A.csv",
}


def _baseline_paths(method: str, seed: int, delay: int, kind: str) -> Path:
    """Resolve summary or latency CSV path for a baseline. Some Towards-* methods nest an extra dir."""
    base = BASELINE_DIRS[method]
    if method in {"PPO-MV-A", "PPO-DSR-A"}:
        # Results layout: base/seed_X/delay_Y/<method>_delay_Y/KAIST/30u/poisson_bursty/<csv>
        sub = f"{BASELINE_DIRS[method].name}_delay_{delay}"
        return (base / f"seed_{seed}" / f"delay_{delay}" / sub / "KAIST" / "30u" / "poisson_bursty"
                / ("summary_Towards-PPO-MV.csv" if method == "PPO-MV-A" else "summary_Towards-PPO-DSR.csv")
                if kind == "summary"
                else (base / f"seed_{seed}" / f"delay_{delay}" / sub / "KAIST" / "30u" / "poisson_bursty"
                      / ("latency_samples_Towards-PPO-MV.csv" if method == "PPO-MV-A"
                         else "latency_samples_Towards-PPO-DSR.csv")))
    csv_name = BASELINE_SUMMARY_NAME[method] if kind == "summary" else BASELINE_LATENCY_NAME[method]
    return base / f"seed_{seed}" / f"delay_{delay}" / "KAIST" / "30u" / "poisson_bursty" / csv_name


def load_baselines_common_grid() -> pd.DataFrame:
    """Load all formal baselines directly from per-cell 5×5 summaries."""
    rows = []
    for method in BASELINE_DIRS:
        for seed in SEEDS:
            for delay in DELAYS:
                csv = _baseline_paths(method, seed, delay, "summary")
                if not csv.exists():
                    continue
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    continue
                if len(df) == 0:
                    continue
                rec = df.iloc[0]
                rows.append({
                    "method": method,
                    "seed": seed,
                    "feedback_delay": delay,
                    "average_latency": float(rec.get("average_latency", np.nan)),
                    "cvar99": float(rec.get("cvar99", np.nan)),
                    "queue_oscillation": float(rec.get("queue_oscillation", np.nan)),
                    "portfolio_migration_rate": float(rec.get("portfolio_migration_rate", np.nan)),
                    "global_demand_concentration": float(rec.get("global_demand_concentration", np.nan)),
                    "inference_time_ms": float(rec.get("inference_time_ms", np.nan)),
                })
    return pd.DataFrame(rows)


def require_complete_common_grid(qla: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """Refuse to overwrite formal outputs unless every method has the exact 5x5 grid.

    MFC-DI-A is a documented prescreen (seed=1 × δ={0,4,8}) and is exempt from the
    common-grid requirement; it is filtered out before the check and re-added later
    so that paired bootstrap (which requires ≥2 seeds) naturally skips it.
    """
    expected = {(seed, delay) for seed in SEEDS for delay in DELAYS}
    problems: list[str] = []
    formal_baselines = baselines[~baselines["method"].eq("MFC-DI-A")]
    for method, frame in [("QLA-MAPPO-NoMargin", qla)] + list(formal_baselines.groupby("method")):
        keys = [(int(s), int(d)) for s, d in frame[["seed", "feedback_delay"]].itertuples(index=False, name=None)]
        unique = set(keys)
        if unique != expected or len(keys) != len(unique):
            problems.append(
                f"{method}: rows={len(keys)}, unique={len(unique)}, "
                f"missing={sorted(expected - unique)}, unexpected={sorted(unique - expected)}"
            )
        if not np.isfinite(frame[METRICS].to_numpy(dtype=float)).all():
            problems.append(f"{method}: non-finite required metrics")
    missing_methods = set(BASELINE_DIRS) - set(baselines["method"].unique())
    # MFC-DI-A may be absent from the formal grid (prescreen-only); not a hard error.
    if missing_methods:
        problems.append(f"missing baseline methods: {sorted(missing_methods)}")
    if problems:
        raise RuntimeError("Incomplete or invalid common grid; formal outputs not overwritten:\n" + "\n".join(problems))


def plot_main_baseline_metrics(all_methods: pd.DataFrame) -> None:
    order = ["QLA-MAPPO-NoMargin", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC", "JSEQ-A"]
    labels = ["QLA-MAPPO", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC", "JSEQ-A"]
    colors = ["#D55E00"] + ["#56B4E9"] * 5 + ["#009E73"]
    titles = ["Average latency (s)", "CVaR99 (s)", "Queue oscillation", "Migration rate"]
    aggregate = all_methods.groupby("method")[METRICS].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, metric, title in zip(axes.flat, METRICS, titles):
        means = np.array([aggregate.loc[m, (metric, "mean")] for m in order], dtype=float)
        stds = np.array([aggregate.loc[m, (metric, "std")] for m in order], dtype=float)
        x = np.arange(len(order))
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        offset = 0.02 * max(float(np.nanmax(means + stds)), 1e-6)
        for idx, value in enumerate(means):
            ax.text(idx, value + stds[idx] + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Main comparison on the common 5-seed × 5-delay grid (mean ± SD)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.91, hspace=0.50, wspace=0.28)
    save_publication_figure(fig, "fig_main_baseline_metrics")
    plt.close(fig)


def plot_main_by_delay(all_methods: pd.DataFrame) -> None:
    order = ["QLA-MAPPO-NoMargin", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC", "JSEQ-A"]
    labels = {"QLA-MAPPO-NoMargin": "QLA-MAPPO"}
    styles = ["o-", "s--", "^--", "D--", "v--", "P--", "X-."]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for method, style in zip(order, styles):
        sub = (all_methods[all_methods["method"] == method]
               .groupby("feedback_delay", as_index=False)[["average_latency", "cvar99"]].mean())
        for ax, metric in zip(axes, ("average_latency", "cvar99")):
            ax.plot(sub["feedback_delay"], sub[metric], style, linewidth=1.3,
                    markersize=4, label=labels.get(method, method))
    axes[0].set_ylabel("Average latency (s)")
    axes[1].set_ylabel("CVaR99 (s)")
    for ax in axes:
        ax.set_xlabel("Feedback delay δ (slots)")
        ax.set_xticks(DELAYS)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Performance by feedback delay (mean across 5 matched seeds)")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.83, wspace=0.28)
    save_publication_figure(fig, "fig_main_baseline_by_delay")
    plt.close(fig)


# -------- TODO-EXP-2: paired bootstrap CI vs each baseline --------

def paired_bootstrap(qla: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """For each baseline method × metric, compute seed-block paired bootstrap 95% CI of (QLA - baseline).

    Pairs on (seed, delay). QLA has 5 seeds × 5 delays; baselines may have 3-5 seeds × 5 delays.
    Uses intersection of seeds.
    """
    rng = np.random.default_rng(RNG_SEED)
    out_rows = []
    baseline_methods = sorted(baselines["method"].unique())
    for method in baseline_methods:
        bsub = baselines[baselines["method"] == method]
        common_seeds = sorted(set(bsub["seed"].unique()) & set(qla["seed"].unique()))
        common_delays = sorted(set(bsub["feedback_delay"].unique()) & set(qla["feedback_delay"].unique()))
        if len(common_seeds) < 2 or len(common_delays) < 2:
            continue
        qsub = qla[qla["seed"].isin(common_seeds) & qla["feedback_delay"].isin(common_delays)]
        bsub = bsub[bsub["seed"].isin(common_seeds) & bsub["feedback_delay"].isin(common_delays)]
        keys = ["seed", "feedback_delay"]
        merged = pd.merge(
            qsub[keys + METRICS].rename(columns={m: f"{m}_qla" for m in METRICS}),
            bsub[keys + METRICS].rename(columns={m: f"{m}_base" for m in METRICS}),
            on=keys, how="inner",
        )
        if len(merged) == 0:
            continue
        for metric in METRICS:
            diff = (merged[f"{metric}_qla"] - merged[f"{metric}_base"]).to_numpy()
            point = float(np.mean(diff))
            base_mean = float(np.mean(merged[f"{metric}_base"]))
            qla_mean = float(np.mean(merged[f"{metric}_qla"]))
            # Seed-block bootstrap: resample seeds with replacement
            boot_means = []
            for _ in range(10_000):
                idx = rng.choice(common_seeds, size=len(common_seeds), replace=True)
                block_means = []
                for s in idx:
                    block = merged[merged["seed"] == s]
                    block_means.append((block[f"{metric}_qla"] - block[f"{metric}_base"]).mean())
                boot_means.append(np.mean(block_means))
            boot = np.asarray(boot_means)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            out_rows.append({
                "baseline": method,
                "metric": metric,
                "n_paired": len(merged),
                "n_seeds": len(common_seeds),
                "n_delays": len(common_delays),
                "qla_mean": qla_mean,
                "baseline_mean": base_mean,
                "mean_diff_qla_minus_baseline": point,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "rel_pct_vs_baseline": float(point / max(abs(base_mean), 1e-9) * 100),
                "qla_wins": bool((point < 0) if metric != "mean_portfolio_entropy" else (point > 0)),
                "ci_excludes_zero": bool((ci_low < 0 and ci_high < 0) or (ci_low > 0 and ci_high > 0)),
            })
    return pd.DataFrame(out_rows)


# -------- TODO-EXP-3: D_stale slope per method --------

def stale_degradation(all_methods: pd.DataFrame) -> pd.DataFrame:
    """For each method, compute D_stale(CVaR99) = (CVaR99(δ=8) - CVaR99(δ=0)) / CVaR99(δ=0) × 100%.

    Also fit a linear slope via per-seed pairing.
    """
    out_rows = []
    for method, msub in all_methods.groupby("method"):
        for seed, ssub in msub.groupby("seed"):
            ssub_sorted = ssub.sort_values("feedback_delay")
            d = ssub_sorted["feedback_delay"].to_numpy()
            c = ssub_sorted["cvar99"].to_numpy()
            if len(d) < 2:
                continue
            slope = float(np.polyfit(d, c, 1)[0])
            d_stale = float((c[-1] - c[0]) / max(c[0], 1e-9) * 100) if d[0] == 0 and d[-1] == 8 else np.nan
            out_rows.append({
                "method": method,
                "seed": int(seed),
                "cvar99_d0": float(c[0]) if d[0] == 0 else np.nan,
                "cvar99_d8": float(c[-1]) if d[-1] == 8 else np.nan,
                "slope_per_slot": slope,
                "D_stale_pct": d_stale,
            })
    # Aggregate per method
    df = pd.DataFrame(out_rows)
    agg = df.groupby("method").agg(
        n_seeds=("seed", "count"),
        cvar99_d0_mean=("cvar99_d0", "mean"),
        cvar99_d0_std=("cvar99_d0", "std"),
        cvar99_d8_mean=("cvar99_d8", "mean"),
        cvar99_d8_std=("cvar99_d8", "std"),
        slope_mean=("slope_per_slot", "mean"),
        slope_std=("slope_per_slot", "std"),
        D_stale_pct_mean=("D_stale_pct", "mean"),
        D_stale_pct_std=("D_stale_pct", "std"),
    ).reset_index()
    agg["D_stale_ratio_of_means_pct"] = (
        (agg["cvar99_d8_mean"] - agg["cvar99_d0_mean"])
        / agg["cvar99_d0_mean"].clip(lower=1e-9) * 100.0
    )
    return df, agg


def plot_paired_forest(paired: pd.DataFrame) -> None:
    """Forest plot of paired QLA-minus-baseline effects with bootstrap CIs."""
    metrics = [("average_latency", "Average latency difference (s)"),
               ("cvar99", "CVaR99 difference (s)")]
    methods = ["PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for panel, (metric, xlabel) in zip(axes, metrics):
        sub = paired[paired["metric"] == metric].set_index("baseline").reindex(methods)
        y = np.arange(len(methods))
        point = sub["mean_diff_qla_minus_baseline"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        panel.errorbar(point, y, xerr=np.vstack([point - low, high - point]),
                       fmt="o", color="#0072B2", ecolor="#4D4D4D",
                       capsize=3, markersize=5, linewidth=1.2)
        panel.axvline(0, color="black", linewidth=0.8, linestyle="--")
        panel.set_xlabel(xlabel)
        panel.set_yticks(y)
        panel.set_yticklabels(methods)
        panel.invert_yaxis()
        panel.grid(axis="x", alpha=0.25)
        panel.set_title("QLA-MAPPO minus baseline")
    fig.suptitle("Paired seed-block bootstrap effects (5 seeds × 5 feedback delays; 95% CI)")
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.18, top=0.78, wspace=0.18)
    save_publication_figure(fig, "fig_paired_bootstrap_vs_baselines")
    plt.close(fig)


def plot_stale_degradation(agg: pd.DataFrame) -> None:
    """Show feedback-age degradation and fitted CVaR99 slope across methods."""
    order = ["QLA-MAPPO-NoMargin", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    sub = agg.set_index("method").reindex(order)
    labels = ["QLA-MAPPO", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    x = np.arange(len(labels))
    n = sub["n_seeds"].to_numpy(dtype=float)
    stale = sub["D_stale_ratio_of_means_pct"].to_numpy(dtype=float)
    t_crit_95_df4 = 2.776
    stale_ci = t_crit_95_df4 * sub["D_stale_pct_std"].to_numpy(dtype=float) / np.sqrt(n)
    slope = sub["slope_mean"].to_numpy(dtype=float)
    slope_ci = t_crit_95_df4 * sub["slope_std"].to_numpy(dtype=float) / np.sqrt(n)
    colors = ["#D55E00"] + ["#56B4E9"] * (len(labels) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    axes[0].bar(x, stale, yerr=stale_ci, color=colors, capsize=3, edgecolor="black", linewidth=0.4)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel(r"$D_{stale}$(CVaR99) (%)")
    axes[0].set_title(r"Relative degradation: $\delta=0 \rightarrow 8$")
    axes[1].bar(x, slope, yerr=slope_ci, color=colors, capsize=3, edgecolor="black", linewidth=0.4)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("CVaR99 slope (s per delay slot)")
    axes[1].set_title("Linear feedback-age sensitivity")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Feedback-age sensitivity (mean and 95% t interval across 5 training seeds)")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.30, top=0.78, wspace=0.32)
    save_publication_figure(fig, "fig_stale_degradation_slope")
    plt.close(fig)


# -------- TODO-EXP-4: CCDF at δ=8 + tail audit --------

def load_latency_samples(method: str, base: Path | None, seeds: list[int], delay: int = 8) -> np.ndarray:
    """Load 'test' phase latency samples from per-cell CSVs.

    For QLA-MAPPO-NoMargin, base is the main cell root.
    For the 5 paper baselines, base is None and we resolve via _baseline_paths.
    """
    samples = []
    for seed in seeds:
        if method == "QLA-MAPPO-NoMargin":
            csv = base / f"seed_{seed}" / f"delay_{delay}" / "KAIST" / "30u" / "poisson_bursty" / "latency_samples_LCC-MAPPO.csv"
        else:
            csv = _baseline_paths(method, seed, delay, "latency")
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "phase" in df.columns:
            df = df[df["phase"] == "test"]
        if "latency" in df.columns:
            samples.append(df["latency"].to_numpy(dtype=np.float64))
    if not samples:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(samples)


def compute_ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    xs = np.sort(x)
    n = xs.size
    ys = 1.0 - (np.arange(1, n + 1) / n)
    return xs, ys


def tail_audit_table(samples_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute tail audit: P(latency > k × median) and CVaR-style thresholds."""
    rows = []
    for method, x in samples_by_method.items():
        if x.size == 0:
            continue
        median = float(np.median(x))
        mean = float(np.mean(x))
        p95 = float(np.percentile(x, 95))
        p99 = float(np.percentile(x, 99))
        cvar99 = float(np.mean(x[x >= p99])) if (x >= p99).any() else np.nan
        capped = np.minimum(x, 10.0)
        capped_p99 = float(np.percentile(capped, 99))
        capped_cvar99 = float(np.mean(capped[capped >= capped_p99]))
        rows.append({
            "method": method,
            "dataset": "KAIST",
            "users": 30,
            "distribution": "poisson_bursty",
            "feedback_delay": 8,
            "n_samples": int(x.size),
            "n_tail_samples": int((x >= p99).sum()),
            "mean": mean,
            "median": median,
            "p95": p95,
            "p99": p99,
            "max_latency": float(np.max(x)),
            "uncapped_cvar99": cvar99,
            "capped_at_10s_cvar99": capped_cvar99,
            "frac_ge_10s": float((x >= 10.0).mean()),
            "ratio_p99_over_median": p99 / median if median > 0 else np.nan,
            "frac_gt_2x_median": float((x > 2 * median).mean()),
            "frac_gt_3x_median": float((x > 3 * median).mean()),
            "frac_gt_4x_median": float((x > 4 * median).mean()),
            "frac_gt_5": float((x > 5.0).mean()),
        })
    return pd.DataFrame(rows)


# -------- §6.10: inference breakdown figure --------

def plot_inference_breakdown() -> None:
    csv = ROOT / "results_qla_simplified_rerun" / "inference_audit" / "priceedge_qtbias_inference_cpu.csv"
    df = pd.read_csv(csv)
    components = ["base_observation", "risk_price", "actor_network_and_transfer",
                  "softmax", "numpy_postprocess"]
    labels = ["Base obs", "Risk price (QL)", "Actor fwd+transfer", "Softmax", "NP postproc"]
    colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2"]
    means = [float(df[df["component"] == c]["mean_ms"].iloc[0]) for c in components]
    p95s = [float(df[df["component"] == c]["p95_ms"].iloc[0]) for c in components]
    p99s = [float(df[df["component"] == c]["p99_ms"].iloc[0]) for c in components]
    e2e = float(df[df["component"] == "end_to_end"]["mean_ms"].iloc[0])
    e2e_p95 = float(df[df["component"] == "end_to_end"]["p95_ms"].iloc[0])
    e2e_p99 = float(df[df["component"] == "end_to_end"]["p99_ms"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    # Left: stacked bar of components (mean) with end-to-end P95/P99 markers
    cum = 0.0
    for label, m, color in zip(labels, means, colors):
        ax1.bar(["QLA-MAPPO"], [m], bottom=[cum], color=color, label=label, edgecolor="white", linewidth=0.6)
        cum += m
    ax1.axhline(e2e, color="black", linestyle="--", linewidth=1.0, label=f"E2E mean = {e2e:.2f} ms")
    ax1.axhline(e2e_p95, color="#444", linestyle=":", linewidth=1.0, label=f"E2E P95 = {e2e_p95:.2f} ms")
    ax1.axhline(e2e_p99, color="#888", linestyle=":", linewidth=1.0, label=f"E2E P99 = {e2e_p99:.2f} ms")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Per-component inference timing (CPU, N=30, δ=8)")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax1.set_ylim(0, max(e2e_p99 * 1.15, cum * 1.15))
    ax1.grid(axis="y", alpha=0.3)

    # Right: P95/P99 by component
    x = np.arange(len(components))
    w = 0.28
    ax2.bar(x - w, means, w, label="mean", color="#4C72B0")
    ax2.bar(x, p95s, w, label="P95", color="#DD8452")
    ax2.bar(x + w, p99s, w, label="P99", color="#C44E52")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Tail latency of component timing")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"QLA-MAPPO online inference cost (slot budget = 1000 ms)", fontsize=11)
    fig.tight_layout()
    save_publication_figure(fig, "fig_inference_breakdown")
    plt.close(fig)


# -------- §6.11 left: MI vs δ --------

def plot_mi_vs_delta() -> None:
    csv = ROOT / "results_qla_simplified_rerun" / "mi_audit" / "qla_wait_action_mutual_information.csv"
    df = pd.read_csv(csv)
    df = df.sort_values("feedback_delay")
    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    delays = df["feedback_delay"].to_numpy()
    mi = df["mutual_information_nats_wait_to_weight"].to_numpy()
    rho = df["spearman_neg_wait_vs_weight"].to_numpy()
    ax1.plot(delays, mi, "o-", color="#4C72B0", label="MI (nats)", linewidth=1.8, markersize=7)
    ax1.set_xlabel("Feedback delay δ (slots)")
    ax1.set_ylabel("Mutual information  I(B(t-δ).wait; w(t))  [nats]", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(delays, rho, "s--", color="#C44E52", label="Spearman ρ(-wait, w)", linewidth=1.5, markersize=6)
    ax2.set_ylabel("Spearman ρ(-wait_norm, weight)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(0, color="#C44E52", linewidth=0.5, alpha=0.3)
    fig.suptitle("Wait-to-action dependence vs feedback delay (QLA-MAPPO-NoMargin, seed=1)")
    fig.tight_layout()
    save_publication_figure(fig, "fig_mi_vs_delta")
    plt.close(fig)


# -------- §6.11 right: shared vs heterogeneous age --------

def plot_shared_vs_heterogeneous() -> None:
    csv = ROOT / "results_qla_simplified_rerun" / "heterogeneous_age" / "qla_shared_vs_heterogeneous_age_summary.csv"
    df = pd.read_csv(csv)
    df = df.set_index("condition")
    metrics = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
    labels = ["Avg latency", "CVaR99", "Queue osc.", "Migration rate"]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.6))
    conditions = ["shared_stale", "heterogeneous_age"]
    colors = {"shared_stale": "#4C72B0", "heterogeneous_age": "#55A467"}
    labels_pretty = {"shared_stale": "Shared stale\n(δ=8)", "heterogeneous_age": "Heterogeneous\nage"}
    for ax, metric, label in zip(axes, metrics, labels):
        means = [df.loc[c, f"{metric}_mean"] for c in conditions]
        stds = [df.loc[c, f"{metric}_std"] for c in conditions]
        x = np.arange(len(conditions))
        ax.bar(x, means, yerr=stds,
               color=[colors[c] for c in conditions], capsize=5, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([labels_pretty[c] for c in conditions], ha="center")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        # Annotate
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.02 * max(means), f"{m:.3f}", ha="center", fontsize=8)
    fig.suptitle("Shared stale vs heterogeneous feedback age (fixed checkpoint, 3 evaluation seeds, N=30)")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.78, wspace=0.55)
    save_publication_figure(fig, "fig_shared_vs_heterogeneous")
    plt.close(fig)


# -------- §6 tail: CCDF at δ=8 --------

def plot_ccdf(samples_by_method: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {
        "QLA-MAPPO-NoMargin": "#C44E52",
        "PPO-MV-A":   "#D55E00",
        "PPO-DSR-A":  "#CC79A7",
        "QECO-A":     "#56B4E9",
        "TODA-MAB-A": "#E69F00",
        "LyFOTO-LC":  "#009E73",
        "JSEQ-A":     "#0072B2",
        "MFC-DI-A":   "#999999",
    }
    labels = {
        "QLA-MAPPO-NoMargin": "QLA-MAPPO (ours)",
        "PPO-MV-A":   "PPO-MV-A",
        "PPO-DSR-A":  "PPO-DSR-A",
        "QECO-A":     "QECO-A",
        "TODA-MAB-A": "TODA-MAB-A",
        "LyFOTO-LC":  "LyFOTO-LC",
        "JSEQ-A":     "JSEQ-A",
        "MFC-DI-A":   "MFC-DI-A (1×3 prescreen)",
    }
    linewidths = {"QLA-MAPPO-NoMargin": 2.2}
    for method, x in samples_by_method.items():
        if x.size == 0:
            continue
        xs, ys = compute_ccdf(x)
        # Downsample for plotting (max 4000 points on log scale)
        if xs.size > 4000:
            idx = np.unique(np.linspace(0, xs.size - 1, 4000).astype(int))
            xs, ys = xs[idx], ys[idx]
        positive = ys > 0
        ax.plot(xs[positive], ys[positive], label=labels.get(method, method),
                color=colors.get(method, "gray"),
                linewidth=linewidths.get(method, 1.3))
    ax.set_xlabel("Per-step latency (s)")
    ax.set_ylabel("P(latency ≥ x)")
    ax.set_yscale("log")
    ax.set_xlim(0, 12)
    ax.set_title("Per-step latency CCDF at δ=8 (KAIST, N=30, poisson_bursty)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_publication_figure(fig, "fig_ccdf_delta8")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[load] QLA main 25 cells + baselines ...", flush=True)
    qla = load_qla_main()
    baselines = load_baselines_common_grid()
    require_complete_common_grid(qla, baselines)
    print(f"  QLA rows: {len(qla)}, baselines rows: {len(baselines)} ({sorted(baselines['method'].unique())})", flush=True)
    all_methods = pd.concat([qla, baselines], ignore_index=True)
    plot_main_baseline_metrics(all_methods)
    plot_main_by_delay(all_methods)

    # TODO-EXP-2: paired bootstrap
    print(f"[exp2] paired bootstrap CI vs each baseline ...", flush=True)
    paired = paired_bootstrap(qla, baselines)
    paired.to_csv(OUT / "paired_bootstrap_vs_baselines.csv", index=False)
    plot_paired_forest(paired)
    print(f"  wrote {len(paired)} rows")

    # TODO-EXP-3: D_stale slope
    print(f"[exp3] D_stale slope per method ...", flush=True)
    per_seed, agg = stale_degradation(all_methods)
    per_seed.to_csv(OUT / "stale_degradation_per_seed.csv", index=False)
    agg.to_csv(OUT / "stale_degradation_slope.csv", index=False)
    plot_stale_degradation(agg)
    print(f"  wrote {len(agg)} method rows")

    # TODO-EXP-4: CCDF at δ=8 + tail audit
    print(f"[exp4] CCDF + tail audit at δ=8 ...", flush=True)
    samples_by_method = {
        "QLA-MAPPO-NoMargin": load_latency_samples(
            "QLA-MAPPO-NoMargin",
            ROOT / "results_qla_simplified_rerun" / "main" / "QLA-MAPPO-NoMargin",
            SEEDS, delay=8,
        ),
    }
    for method in BASELINE_DIRS:
        samples_by_method[method] = load_latency_samples(method, None, SEEDS, delay=8)
    sizes = {m: len(s) for m, s in samples_by_method.items()}
    print(f"  sample counts: {sizes}")
    tail_df = tail_audit_table(samples_by_method)
    tail_df.to_csv(OUT / "tail_audit_delta8.csv", index=False)
    # Save CCDF curves as CSV (downsampled)
    ccdf_rows = []
    for method, x in samples_by_method.items():
        xs, ys = compute_ccdf(x)
        if xs.size == 0:
            continue
        idx = np.unique(np.linspace(0, xs.size - 1, min(xs.size, 1500)).astype(int))
        for xv, yv in zip(xs[idx], ys[idx]):
            ccdf_rows.append({"method": method, "dataset": "KAIST", "users": 30,
                              "distribution": "poisson_bursty", "feedback_delay": 8,
                              "latency_s": float(xv), "ccdf": float(yv)})
    pd.DataFrame(ccdf_rows).to_csv(OUT / "ccdf_delta8.csv", index=False)

    # §6.10 inference figure
    print(f"[fig] §6.10 inference breakdown ...", flush=True)
    plot_inference_breakdown()

    # §6.11 figures
    print(f"[fig] §6.11 MI vs δ + shared vs heterogeneous ...", flush=True)
    plot_mi_vs_delta()
    plot_shared_vs_heterogeneous()

    # §6 tail CCDF figure
    print(f"[fig] §6 CCDF at δ=8 ...", flush=True)
    plot_ccdf(samples_by_method)

    print(f"\nAll outputs in: {OUT}", flush=True)
    print("\n=== Paired bootstrap (avg_latency) ===")
    sub = paired[paired["metric"] == "average_latency"]
    print(sub[["baseline", "qla_mean", "baseline_mean", "mean_diff_qla_minus_baseline",
               "ci_low", "ci_high", "rel_pct_vs_baseline", "ci_excludes_zero"]].to_string(index=False))
    print("\n=== Paired bootstrap (cvar99) ===")
    sub = paired[paired["metric"] == "cvar99"]
    print(sub[["baseline", "qla_mean", "baseline_mean", "mean_diff_qla_minus_baseline",
               "ci_low", "ci_high", "rel_pct_vs_baseline", "ci_excludes_zero"]].to_string(index=False))
    print("\n=== D_stale slope per method ===")
    print(agg.to_string(index=False))
    print("\n=== Tail audit at δ=8 ===")
    print(tail_df.to_string(index=False))


if __name__ == "__main__":
    main()
