"""Flat-Bias-QL feedback-age mechanism analysis (no retraining).

Computes four diagnostics on the existing 25-cell main + 25-cell Flat grids:

  (1) Per-delay paired mean difference (Flat - Main) with seed-block 95% CI.
  (2) Stale degradation D_stale = (M_delta8 - M_delta0) / M_delta0 for
      avg_latency and cvar99, per variant.
  (3) Per-delay slope of metric vs delta (linear regression on 5 points,
      reporting slope and Spearman).
  (4) Paired difference-in-differences:
      DiD = (Flat_8 - Flat_0) - (Main_8 - Main_0)
      computed seed-by-seed, then seed-block bootstrap 95% CI.

All bootstrap CIs use the same RNG seed and seed-block estimand as the
finalizer (seed=20260726, 10 000 resamples over 5 seeds). This is descriptive
analysis only; it does not change any number in the paper Tables.

Output: results_qla_simplified_rerun/final_analysis/flat_feedback_age_mechanism.csv
        results_qla_simplified_rerun/final_analysis/flat_feedback_age_mechanism.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_qla_simplified_rerun"
OUT = RESULTS / "final_analysis"
RNG_SEED = 20260726
N_BOOT = 10_000
METRICS = ["average_latency", "cvar99"]
DELAYS = [0, 1, 2, 4, 8]
SEEDS = [1, 2, 3, 4, 5]
EPS = 1e-12


def load_variant(phase: str, dirname: str) -> pd.DataFrame:
    rows = []
    base = RESULTS / phase / dirname
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
                "variant": dirname,
                "seed": seed,
                "feedback_delay": delay,
                "average_latency": float(rec["average_latency"]),
                "cvar99": float(rec["cvar99"]),
            })
    return pd.DataFrame(rows)


def seed_block_bootstrap_ci(values: np.ndarray, rng_seed: int = RNG_SEED, n_boot: int = N_BOOT) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def per_delay_paired_diff(main: pd.DataFrame, flat: pd.DataFrame) -> pd.DataFrame:
    """For each delay, compute per-seed Flat-Main diff, then seed-block CI."""
    rows = []
    for delay in DELAYS:
        m_sub = main[main["feedback_delay"] == delay].set_index("seed")
        f_sub = flat[flat["feedback_delay"] == delay].set_index("seed")
        common = sorted(set(m_sub.index) & set(f_sub.index))
        for metric in METRICS:
            diffs = (f_sub.loc[common, metric].to_numpy(dtype=float)
                     - m_sub.loc[common, metric].to_numpy(dtype=float))
            point = float(diffs.mean())
            lo, hi = seed_block_bootstrap_ci(diffs)
            rows.append({
                "analysis": "per_delay_paired_diff",
                "metric": metric,
                "feedback_delay": delay,
                "n_seeds": len(common),
                "mean_diff_flat_minus_main": point,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_excludes_zero": bool(hi < 0 or lo > 0),
            })
    return pd.DataFrame(rows)


def stale_degradation(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """D_stale = (M_delta8 - M_delta0) / M_delta0 per variant, per metric.

    Also reports per-seed D_stale so we can bootstrap.
    """
    rows = []
    for metric in METRICS:
        per_seed = []
        for seed in SEEDS:
            sub = df[(df["seed"] == seed)]
            m0 = sub[sub["feedback_delay"] == 0][metric].to_numpy(dtype=float)
            m8 = sub[sub["feedback_delay"] == 8][metric].to_numpy(dtype=float)
            if len(m0) == 0 or len(m8) == 0:
                continue
            ratio = float((m8[0] - m0[0]) / max(abs(m0[0]), EPS))
            per_seed.append(ratio)
        arr = np.asarray(per_seed, dtype=float)
        if len(arr) == 0:
            continue
        point = float(arr.mean())
        lo, hi = seed_block_bootstrap_ci(arr)
        rows.append({
            "analysis": "stale_degradation_D_stale",
            "variant": label,
            "metric": metric,
            "n_seeds": len(arr),
            "D_stale_mean": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
        })
    return pd.DataFrame(rows)


def slope_vs_delay(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-seed OLS slope of metric vs delta, then seed-block CI."""
    rows = []
    x = np.asarray(DELAYS, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    for metric in METRICS:
        per_seed = []
        for seed in SEEDS:
            sub = df[df["seed"] == seed].sort_values("feedback_delay")
            if len(sub) != len(DELAYS):
                continue
            y = sub[metric].to_numpy(dtype=float)
            slope = float((x_centered * (y - y.mean())).sum() / denom)
            per_seed.append(slope)
        arr = np.asarray(per_seed, dtype=float)
        if len(arr) == 0:
            continue
        point = float(arr.mean())
        lo, hi = seed_block_bootstrap_ci(arr)
        rows.append({
            "analysis": "slope_vs_delay",
            "variant": label,
            "metric": metric,
            "n_seeds": len(arr),
            "slope_mean_per_delta": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
        })
    return pd.DataFrame(rows)


def paired_did(main: pd.DataFrame, flat: pd.DataFrame) -> pd.DataFrame:
    """DiD = (Flat_8 - Flat_0) - (Main_8 - Main_0) per seed, then bootstrap."""
    rows = []
    for metric in METRICS:
        per_seed = []
        for seed in SEEDS:
            m_sub = main[main["seed"] == seed]
            f_sub = flat[flat["seed"] == seed]
            m0 = m_sub[m_sub["feedback_delay"] == 0][metric].to_numpy(dtype=float)
            m8 = m_sub[m_sub["feedback_delay"] == 8][metric].to_numpy(dtype=float)
            f0 = f_sub[f_sub["feedback_delay"] == 0][metric].to_numpy(dtype=float)
            f8 = f_sub[f_sub["feedback_delay"] == 8][metric].to_numpy(dtype=float)
            if min(len(m0), len(m8), len(f0), len(f8)) == 0:
                continue
            main_delta = float(m8[0] - m0[0])
            flat_delta = float(f8[0] - f0[0])
            per_seed.append(flat_delta - main_delta)
        arr = np.asarray(per_seed, dtype=float)
        if len(arr) == 0:
            continue
        point = float(arr.mean())
        lo, hi = seed_block_bootstrap_ci(arr)
        rows.append({
            "analysis": "paired_did_flat_minus_main",
            "metric": metric,
            "n_seeds": len(arr),
            "did_mean": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "interpretation": "positive => Flat degrades more from delta=0 to delta=8 than Main does",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_df = load_variant("main", "QLA-MAPPO-NoMargin")
    flat_df = load_variant("ablation", "Flat-Bias-QL-NoMargin")
    if len(main_df) != 25 or len(flat_df) != 25:
        raise SystemExit(f"Expected 25 cells each, got main={len(main_df)} flat={len(flat_df)}")

    per_delay = per_delay_paired_diff(main_df, flat_df)
    d_stale = pd.concat([stale_degradation(main_df, "Main"), stale_degradation(flat_df, "Flat")], ignore_index=True)
    slopes = pd.concat([slope_vs_delay(main_df, "Main"), slope_vs_delay(flat_df, "Flat")], ignore_index=True)
    did = paired_did(main_df, flat_df)

    per_delay.to_csv(OUT / "flat_per_delay_paired.csv", index=False)
    d_stale.to_csv(OUT / "flat_stale_degradation.csv", index=False)
    slopes.to_csv(OUT / "flat_slope_vs_delay.csv", index=False)
    did.to_csv(OUT / "flat_paired_did.csv", index=False)

    print("=== Per-delay paired diff (Flat - Main), seed-block 95% CI ===")
    print(per_delay.to_string(index=False))
    print("\n=== D_stale = (M_8 - M_0) / M_0 per variant ===")
    print(d_stale.to_string(index=False))
    print("\n=== Slope of metric vs delta (per-seed OLS, then seed-block CI) ===")
    print(slopes.to_string(index=False))
    print("\n=== Paired DiD: (Flat_8-Flat_0) - (Main_8-Main_0) ===")
    print(did.to_string(index=False))


if __name__ == "__main__":
    main()
