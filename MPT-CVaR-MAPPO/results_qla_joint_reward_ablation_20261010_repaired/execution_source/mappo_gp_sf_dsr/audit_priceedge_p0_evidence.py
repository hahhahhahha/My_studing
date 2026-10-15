from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_priceedge_p0_audit"
PRICEEDGE_RAW = ROOT / "results_lcc_priceedge_formal_5x5" / "all_completed_raw.csv"
FACR_RAW = ROOT / "results_selected_baseline_comparison" / "selected_baseline_common_grid_raw.csv"
PRICEEDGE_RUN_ROOT = ROOT / "results_lcc_priceedge_formal_5x5" / "prescreen" / "LCC-PriceEdge-Full"
FACR_RUN_ROOT = ROOT / "results_v4_agegate_lyapprior_formal" / "lyapprior_w20_smooth80"


METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "portfolio_migration_rate",
]


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil((1.0 - alpha) * arr.size)))
    return float(np.mean(np.sort(arr)[-k:]))


def paired_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 20260722) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    dz = float(mean / sd) if sd > 1e-12 else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), dz


def load_main_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    pe = pd.read_csv(PRICEEDGE_RAW)
    pe = pe[pe["variant"].eq("LCC-PriceEdge-Full")].copy()
    pe["method_label"] = "PriceEdge-Full"
    pe["seed"] = pe["seed"].astype(int)
    pe["delay"] = pe["delay"].astype(int)

    facr = pd.read_csv(FACR_RAW)
    facr = facr[facr["method"].eq("FACR-MAPPO")].copy()
    facr["method_label"] = "FACR-MAPPO"
    facr["seed"] = facr["seed"].astype(int)
    facr["delay"] = facr["feedback_delay"].astype(int)
    return pe, facr


def build_paired_stats(pe: pd.DataFrame, facr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for metric in METRICS:
        merged = pe[["seed", "delay", metric]].merge(
            facr[["seed", "delay", metric]], on=["seed", "delay"], suffixes=("_priceedge", "_facr")
        )
        merged["diff"] = merged[f"{metric}_priceedge"] - merged[f"{metric}_facr"]
        merged["rel_diff_pct"] = 100.0 * merged["diff"] / merged[f"{metric}_facr"].replace(0, np.nan)
        for delay, group in merged.groupby("delay"):
            mean, lo, hi, dz = paired_ci(group["diff"].to_numpy())
            rel_mean, rel_lo, rel_hi, _ = paired_ci(group["rel_diff_pct"].to_numpy())
            rows.append(
                {
                    "metric": metric,
                    "delay": int(delay),
                    "n_seed_pairs": int(group.shape[0]),
                    "paired_mean_diff": mean,
                    "paired_ci95_low": lo,
                    "paired_ci95_high": hi,
                    "paired_mean_rel_pct": rel_mean,
                    "paired_rel_ci95_low": rel_lo,
                    "paired_rel_ci95_high": rel_hi,
                    "paired_cohens_dz": dz,
                }
            )

    # Stratified bootstrap: keep delay as condition and resample seed pairs inside each delay.
    strat_rows = []
    rng = np.random.default_rng(20260722)
    for metric in METRICS:
        merged = pe[["seed", "delay", metric]].merge(
            facr[["seed", "delay", metric]], on=["seed", "delay"], suffixes=("_priceedge", "_facr")
        )
        merged["diff"] = merged[f"{metric}_priceedge"] - merged[f"{metric}_facr"]
        merged["rel_diff_pct"] = 100.0 * merged["diff"] / merged[f"{metric}_facr"].replace(0, np.nan)
        delay_groups = [g for _, g in merged.groupby("delay")]
        boot_abs = []
        boot_rel = []
        for _ in range(20000):
            sampled_abs = []
            sampled_rel = []
            for g in delay_groups:
                idx = rng.integers(0, len(g), size=len(g))
                sampled_abs.append(g["diff"].to_numpy()[idx].mean())
                sampled_rel.append(g["rel_diff_pct"].to_numpy()[idx].mean())
            boot_abs.append(float(np.mean(sampled_abs)))
            boot_rel.append(float(np.mean(sampled_rel)))
        abs_lo, abs_hi = np.percentile(boot_abs, [2.5, 97.5])
        rel_lo, rel_hi = np.percentile(boot_rel, [2.5, 97.5])
        diffs = merged["diff"].to_numpy()
        dz = float(diffs.mean() / np.std(diffs, ddof=1)) if diffs.size > 1 and np.std(diffs, ddof=1) > 1e-12 else float("nan")
        strat_rows.append(
            {
                "metric": metric,
                "n_seed_delay_pairs": int(merged.shape[0]),
                "stratified_mean_diff": float(np.mean(boot_abs)),
                "stratified_ci95_low": float(abs_lo),
                "stratified_ci95_high": float(abs_hi),
                "stratified_mean_rel_pct": float(np.mean(boot_rel)),
                "stratified_rel_ci95_low": float(rel_lo),
                "stratified_rel_ci95_high": float(rel_hi),
                "paired_cohens_dz_all_pairs": dz,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(strat_rows)


def latency_sample_paths(method: str, seed: int, delay: int) -> list[Path]:
    if method == "PriceEdge-Full":
        return list((PRICEEDGE_RUN_ROOT / f"seed_{seed}" / f"delay_{delay}").glob("KAIST/30u/poisson_bursty/latency_samples_*.csv"))
    if method == "FACR-MAPPO":
        return list((FACR_RUN_ROOT / f"seed_{seed}" / f"delay_{delay}").glob("KAIST/30u/poisson_bursty/latency_samples_*.csv"))
    return []


def build_latency_censoring_audit() -> pd.DataFrame:
    rows = []
    for method in ["PriceEdge-Full", "FACR-MAPPO"]:
        for seed in range(1, 6):
            for delay in [0, 1, 2, 4, 8]:
                paths = latency_sample_paths(method, seed, delay)
                if not paths:
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "delay": delay,
                            "status": "missing_latency_samples",
                        }
                    )
                    continue
                df = pd.read_csv(paths[0])
                if "phase" in df.columns:
                    df = df[df["phase"].astype(str).eq("test")]
                lat = pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(dtype=np.float64)
                n = int(lat.size)
                if n == 0:
                    rows.append({"method": method, "seed": seed, "delay": delay, "status": "empty_test_latency"})
                    continue
                queue_s = pd.to_numeric(df.get("queue_s", pd.Series([], dtype=float)), errors="coerce")
                tran_s = pd.to_numeric(df.get("tran_s", pd.Series([], dtype=float)), errors="coerce")
                comp_s = pd.to_numeric(df.get("comp_s", pd.Series([], dtype=float)), errors="coerce")
                tail95_n = max(1, int(math.ceil(0.05 * n)))
                tail99_n = max(1, int(math.ceil(0.01 * n)))
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "delay": delay,
                        "status": "ok",
                        "n_test_tasks": n,
                        "tail95_sample_count": tail95_n,
                        "tail99_sample_count": tail99_n,
                        "avg_latency_uncapped_recorded": float(np.mean(lat)),
                        "cvar95_uncapped_recorded": empirical_cvar(lat, 0.95),
                        "cvar99_uncapped_recorded": empirical_cvar(lat, 0.99),
                        "avg_latency_capped_10s": float(np.mean(np.minimum(lat, 10.0))),
                        "cvar95_capped_10s": empirical_cvar(np.minimum(lat, 10.0), 0.95),
                        "cvar99_capped_10s": empirical_cvar(np.minimum(lat, 10.0), 0.99),
                        "max_latency_recorded": float(np.max(lat)),
                        "count_latency_ge_10s": int(np.sum(lat >= 10.0)),
                        "prob_latency_ge_10s": float(np.mean(lat >= 10.0)),
                        "prob_latency_gt_3s": float(np.mean(lat > 3.0)),
                        "mean_queue_s": float(np.nanmean(queue_s)) if len(queue_s) else float("nan"),
                        "max_queue_s": float(np.nanmax(queue_s)) if len(queue_s) else float("nan"),
                        "mean_tran_s": float(np.nanmean(tran_s)) if len(tran_s) else float("nan"),
                        "mean_comp_s": float(np.nanmean(comp_s)) if len(comp_s) else float("nan"),
                        "latency_sample_path": str(paths[0]),
                    }
                )
    return pd.DataFrame(rows)


def plot_paired_ci(strat: pd.DataFrame, by_delay: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=180)
    metric_labels = {
        "average_latency": "Avg latency",
        "cvar99": "CVaR99",
        "queue_oscillation": "Queue osc.",
        "portfolio_migration_rate": "Migration",
    }

    ax = axes[0]
    data = strat.set_index("metric").loc[METRICS].reset_index()
    x = np.arange(len(data))
    y = data["stratified_mean_rel_pct"].to_numpy()
    lo = y - data["stratified_rel_ci95_low"].to_numpy()
    hi = data["stratified_rel_ci95_high"].to_numpy() - y
    colors = ["#4C78A8" if v <= 0 else "#F58518" for v in y]
    ax.bar(x, y, color=colors, alpha=0.85)
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in data["metric"]], rotation=20, ha="right")
    ax.set_ylabel("PriceEdge - FACR (%)")
    ax.set_title("Stratified paired bootstrap")

    ax = axes[1]
    cvar = by_delay[by_delay["metric"].eq("cvar99")].sort_values("delay")
    x = cvar["delay"].to_numpy()
    y = cvar["paired_mean_rel_pct"].to_numpy()
    lo = y - cvar["paired_rel_ci95_low"].to_numpy()
    hi = cvar["paired_rel_ci95_high"].to_numpy() - y
    ax.errorbar(x, y, yerr=[lo, hi], marker="o", color="#54A24B", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Feedback delay δ")
    ax.set_ylabel("CVaR99 difference (%)")
    ax.set_title("Per-delay paired CVaR99")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_priceedge_facr_paired_ci.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_priceedge_facr_paired_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pe, facr = load_main_tables()
    by_delay, strat = build_paired_stats(pe, facr)
    censoring = build_latency_censoring_audit()

    by_delay.to_csv(OUT_DIR / "priceedge_vs_facr_paired_by_delay_ci.csv", index=False)
    strat.to_csv(OUT_DIR / "priceedge_vs_facr_stratified_bootstrap_ci.csv", index=False)
    censoring.to_csv(OUT_DIR / "latency_cvar_censoring_audit.csv", index=False)
    plot_paired_ci(strat, by_delay)

    print("Wrote:")
    print(OUT_DIR / "priceedge_vs_facr_paired_by_delay_ci.csv")
    print(OUT_DIR / "priceedge_vs_facr_stratified_bootstrap_ci.csv")
    print(OUT_DIR / "latency_cvar_censoring_audit.csv")
    print(OUT_DIR / "fig_priceedge_facr_paired_ci.png")


if __name__ == "__main__":
    main()
