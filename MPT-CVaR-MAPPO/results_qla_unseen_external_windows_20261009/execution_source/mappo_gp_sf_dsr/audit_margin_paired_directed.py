"""Audit stale margin / tail-state Z / static reward ablations.

Outputs:
  - audit_flag_correctness.csv     per-variant flag means (verifies ablation actually disables)
  - audit_margin_price_magnitude.csv  z/h/nu queue + price-action alignment magnitudes
  - paired_bootstrap_extended.csv   paired difference vs PriceEdge-QT-Bias main, stratified by delay
  - audit_z_dormancy.csv            fraction of runs where Z/H are activated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results_priceedge_qtbias_paper_ready" / "paper_ablation_raw_no_priceonly_localcritic.csv"
OUT_DIR = ROOT / "results_priceedge_qt_bias_final_analysis"

MAIN = "PriceEdge-QT-Bias"
ABLATIONS = ["QT-NoBias", "QT-Bias-Shuffle", "Flat-Bias-QT", "w/o stale margin", "w/o tail state Z", "Static reward"]

METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
MARGIN_COLS = [
    "lcc_stale_margin", "lcc_z_queue", "lcc_h_queue", "lcc_var_nu",
    "lcc_cvar_violation", "lcc_concentration_violation",
    "lcc_tail_latency_norm", "lcc_concentration",
    "lcc_price_action_spearman",
]
FLAG_COLS = [
    "lcc_use_stale_margin", "lcc_use_z_queue", "lcc_use_h_queue",
    "lcc_static_reward",
]


def audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in df.groupby("paper_variant"):
        row = {"paper_variant": variant, "n_runs": int(len(g))}
        for col in FLAG_COLS:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def audit_margin_price(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in df.groupby("paper_variant"):
        row = {"paper_variant": variant}
        for col in MARGIN_COLS:
            vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def audit_dormancy(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of runs where Z/H queues are activated (violation > 0)."""
    rows = []
    for variant, g in df.groupby("paper_variant"):
        z_vio = pd.to_numeric(g["lcc_cvar_violation"], errors="coerce").dropna()
        h_vio = pd.to_numeric(g["lcc_concentration_violation"], errors="coerce").dropna()
        z_active = (z_vio > 0).sum() if len(z_vio) else 0
        h_active = (h_vio > 0).sum() if len(h_vio) else 0
        rows.append({
            "paper_variant": variant,
            "n_runs": int(len(g)),
            "z_violation_mean": float(z_vio.mean()) if len(z_vio) else float("nan"),
            "z_active_rate": float(z_active / max(len(z_vio), 1)),
            "h_violation_mean": float(h_vio.mean()) if len(h_vio) else float("nan"),
            "h_active_rate": float(h_active / max(len(h_vio), 1)),
        })
    return pd.DataFrame(rows)


def paired_bootstrap_stratified(
    main_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    metric: str,
    n_boot: int = 20000,
    seed: int = 20260725,
) -> dict:
    """Stratified bootstrap by delay, paired by (seed, delay)."""
    merged = main_df[["seed", "delay", metric]].merge(
        variant_df[["seed", "delay", metric]],
        on=["seed", "delay"],
        suffixes=("_main", "_variant"),
    )
    if merged.empty:
        return {}
    diffs = (merged[f"{metric}_variant"] - merged[f"{metric}_main"]).to_numpy()
    rel = (100.0 * diffs / merged[f"{metric}_main"].replace(0, np.nan).to_numpy()).tolist()
    rel = np.asarray([x for x in rel if np.isfinite(x)])
    n = len(diffs)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    rng = np.random.default_rng(seed)
    # Stratified bootstrap: resample within each delay group then average
    delay_groups = [g.copy() for _, g in merged.groupby("delay")]
    boot_abs, boot_rel = [], []
    for _ in range(n_boot):
        per_delay_abs, per_delay_rel = [], []
        for g in delay_groups:
            d = (g[f"{metric}_variant"] - g[f"{metric}_main"]).to_numpy()
            idx = rng.integers(0, len(d), size=len(d))
            per_delay_abs.append(float(np.mean(d[idx])))
            denom = g[f"{metric}_main"].to_numpy()[idx]
            r = 100.0 * d[idx] / np.where(denom == 0, np.nan, denom)
            r = r[np.isfinite(r)]
            if len(r):
                per_delay_rel.append(float(np.mean(r)))
        boot_abs.append(float(np.mean(per_delay_abs)))
        if per_delay_rel:
            boot_rel.append(float(np.mean(per_delay_rel)))
    abs_lo, abs_hi = np.percentile(boot_abs, [2.5, 97.5])
    rel_lo, rel_hi = (np.percentile(boot_rel, [2.5, 97.5]) if boot_rel else (float("nan"), float("nan")))
    win_rate = float(np.mean(diffs < 0))  # variant lower than main
    return {
        "metric": metric,
        "n_pairs": n,
        "mean_diff_variant_minus_main": mean,
        "std_diff": std,
        "relative_diff_pct": float(np.mean(rel)) if len(rel) else float("nan"),
        "win_rate_variant_lower": win_rate,
        "stratified_bootstrap_ci_low": float(abs_lo),
        "stratified_bootstrap_ci_high": float(abs_hi),
        "rel_ci_low": float(rel_lo),
        "rel_ci_high": float(rel_hi),
    }


def build_paired_table(df: pd.DataFrame) -> pd.DataFrame:
    main_df = df[df["paper_variant"].eq(MAIN)].copy()
    rows = []
    for variant in ABLATIONS:
        var_df = df[df["paper_variant"].eq(variant)].copy()
        if var_df.empty:
            continue
        for metric in METRICS:
            stats = paired_bootstrap_stratified(main_df, var_df, metric)
            if stats:
                stats["variant"] = variant
                rows.append(stats)
    cols = ["variant", "metric", "n_pairs", "mean_diff_variant_minus_main", "std_diff",
            "relative_diff_pct", "win_rate_variant_lower",
            "stratified_bootstrap_ci_low", "stratified_bootstrap_ci_high",
            "rel_ci_low", "rel_ci_high"]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce")

    flag_df = audit_flags(df)
    margin_df = audit_margin_price(df)
    dormancy_df = audit_dormancy(df)
    paired_df = build_paired_table(df)

    flag_df.to_csv(OUT_DIR / "audit_flag_correctness.csv", index=False)
    margin_df.to_csv(OUT_DIR / "audit_margin_price_magnitude.csv", index=False)
    dormancy_df.to_csv(OUT_DIR / "audit_z_dormancy.csv", index=False)
    paired_df.to_csv(OUT_DIR / "paired_bootstrap_extended.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=== Flag audit ===")
    print(flag_df.to_string(index=False))
    print("\n=== Margin / price magnitude ===")
    print(margin_df.round(4).to_string(index=False))
    print("\n=== Z / H dormancy ===")
    print(dormancy_df.round(4).to_string(index=False))
    print("\n=== Paired bootstrap (vs %s) ===" % MAIN)
    pivot = paired_df.pivot(index="variant", columns="metric", values="mean_diff_variant_minus_main")
    print(pivot.round(4).to_string())
    ci_low = paired_df.pivot(index="variant", columns="metric", values="stratified_bootstrap_ci_low")
    ci_high = paired_df.pivot(index="variant", columns="metric", values="stratified_bootstrap_ci_high")
    print("\n95% CI bands (CVaR99):")
    cva99 = paired_df[paired_df["metric"].eq("cvar99")][["variant", "mean_diff_variant_minus_main", "stratified_bootstrap_ci_low", "stratified_bootstrap_ci_high", "win_rate_variant_lower"]]
    print(cva99.round(4).to_string(index=False))
    print("\nWritten to:", OUT_DIR)


if __name__ == "__main__":
    main()
