"""Generate ablation Table 5 + cross-env Table 7 + N=40 Table 8 (simplified rerun).

Table 6 (paired bootstrap CI) is *not* computed here. The authoritative source
is ``finalize_qla_simplified_results.py``, which writes
``results_qla_simplified_rerun/final_analysis/core_ablation_paired_bootstrap.csv``
using a single module-level RNG (seed=20260726) and a single seed-block
estimand. Any script or prose that consumes Table 6 numbers must read that
CSV; do not recompute bootstrap intervals locally.

Outputs (under results_qla_simplified_rerun/paper_figures/):
  - ablation_summary_simplified.csv : 4-method × 5-metric 25-cell means and sample SDs (Table 5)
  - crossenv_breakdown.csv          : 48-cell breakdown by dataset × dist × N × δ (Table 7a)
  - crossenv_aggregates.csv         : dataset × dist / N / δ aggregates (Table 7b)
  - n40_breakdown.csv               : 10-cell breakdown by dist × seed (Table 8 raw)
  - n40_aggregates.csv              : dist-level aggregates (Table 8 summary)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_qla_simplified_rerun" / "paper_figures"
AUTHORITATIVE_PAIRED_CSV = (
    ROOT / "results_qla_simplified_rerun" / "final_analysis" / "core_ablation_paired_bootstrap.csv"
)

SEEDS = [1, 2, 3, 4, 5]
DELAYS = [0, 1, 2, 4, 8]
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]

ABLATION_VARIANTS = {
    "QLA-MAPPO":      ("main",     "QLA-MAPPO-NoMargin"),
    "QL-NoBias":      ("ablation", "QL-NoBias-NoMargin"),
    "QL-Bias-Shuffle":("ablation", "QL-Bias-Shuffle-NoMargin"),
    "Flat-Bias-QL":   ("ablation", "Flat-Bias-QL-NoMargin"),
}


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
                "lcc_price_action_spearman": float(rec.get("lcc_price_action_spearman", float("nan"))),
            })
    return pd.DataFrame(rows)


def table5_ablation_summary() -> pd.DataFrame:
    """25-cell mean ± sample SD for each variant × metric (paper Table 5)."""
    rows = []
    for label, (phase, dirname) in ABLATION_VARIANTS.items():
        df = load_variant(phase, dirname)
        for metric in METRICS + ["lcc_price_action_spearman"]:
            vals = df[metric].dropna().to_numpy(dtype=float)
            rows.append({
                "variant": label,
                "metric": metric,
                "n": len(vals),
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def assert_authoritative_paired_csv_exists() -> None:
    """Block Table 6 consumption if the finalizer has not run."""
    if not AUTHORITATIVE_PAIRED_CSV.exists():
        raise SystemExit(
            f"Authoritative Table 6 CSV not found at {AUTHORITATIVE_PAIRED_CSV}.\n"
            "Run `python -m mappo_gp_sf_dsr.finalize_qla_simplified_results` first. "
            "Do not recompute paired bootstrap CIs in this script."
        )


def load_authoritative_paired() -> pd.DataFrame:
    """Read Table 6 from the finalizer output (single source of truth)."""
    assert_authoritative_paired_csv_exists()
    return pd.read_csv(AUTHORITATIVE_PAIRED_CSV)


def load_crossenv() -> pd.DataFrame:
    """Load 48 cross-env cells: dataset × dist × N × δ."""
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / "crossenv" / "QLA-MAPPO-NoMargin"
    datasets = ["KAIST", "NCSU", "NewYork"]
    distributions = ["uniform", "gaussian"]
    user_counts = [15, 20, 25, 30]
    delays = [0, 8]
    for ds in datasets:
        for dist in distributions:
            for n in user_counts:
                for delay in delays:
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


def table7a_crossenv_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot by dataset × dist × N × δ for each metric."""
    out = df.copy()
    return out.sort_values(["dataset", "distribution", "users", "feedback_delay"])


def table7b_crossenv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by dataset × dist and by N, δ separately."""
    rows = []
    for (ds, dist), g in df.groupby(["dataset", "distribution"]):
        for metric in METRICS:
            vals = g[metric].dropna().to_numpy(dtype=float)
            rows.append({"breakdown": f"{ds}/{dist}", "n": len(vals), "metric": metric,
                         "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")})
    for n, g in df.groupby("users"):
        for metric in METRICS:
            vals = g[metric].dropna().to_numpy(dtype=float)
            rows.append({"breakdown": f"N={n}", "n": len(vals), "metric": metric,
                         "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")})
    for d, g in df.groupby("feedback_delay"):
        for metric in METRICS:
            vals = g[metric].dropna().to_numpy(dtype=float)
            rows.append({"breakdown": f"δ={d}", "n": len(vals), "metric": metric,
                         "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")})
    return pd.DataFrame(rows)


def load_n40() -> pd.DataFrame:
    rows = []
    base = ROOT / "results_qla_simplified_rerun" / "n40" / "QLA-MAPPO-NoMargin"
    distributions = ["uniform", "gaussian"]
    for dist in distributions:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main_df = load_variant("main", "QLA-MAPPO-NoMargin")
    print(f"[main] {len(main_df)} cells", flush=True)

    t5 = table5_ablation_summary()
    t5.to_csv(OUT_DIR / "ablation_summary_simplified.csv", index=False)
    print(f"[Table 5] {len(t5)} rows -> ablation_summary_simplified.csv", flush=True)

    # Table 6: authoritative source is the finalizer. Do not recompute here.
    paired = load_authoritative_paired()
    print(f"[Table 6] {len(paired)} rows read from {AUTHORITATIVE_PAIRED_CSV.name}", flush=True)
    print("  (authoritative — finalize_qla_simplified_results.py writes this CSV)", flush=True)

    ce = load_crossenv()
    print(f"[cross-env] {len(ce)} cells", flush=True)
    if len(ce) != 48:
        print(f"  WARN: expected 48 cells, got {len(ce)}", flush=True)
    t7a = table7a_crossenv_breakdown(ce)
    t7a.to_csv(OUT_DIR / "crossenv_breakdown.csv", index=False)
    t7b = table7b_crossenv_summary(ce)
    t7b.to_csv(OUT_DIR / "crossenv_aggregates.csv", index=False)
    print(f"[Table 7a] {len(t7a)} rows -> crossenv_breakdown.csv", flush=True)
    print(f"[Table 7b] {len(t7b)} rows -> crossenv_aggregates.csv", flush=True)

    n40 = load_n40()
    print(f"[N=40] {len(n40)} cells", flush=True)
    n40.to_csv(OUT_DIR / "n40_breakdown.csv", index=False)
    n40_agg = []
    for dist, g in n40.groupby("distribution"):
        for metric in METRICS:
            vals = g[metric].dropna().to_numpy(dtype=float)
            n40_agg.append({"distribution": dist, "metric": metric, "n": len(vals),
                            "mean": float(vals.mean()) if len(vals) else float("nan"),
                            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")})
    pd.DataFrame(n40_agg).to_csv(OUT_DIR / "n40_aggregates.csv", index=False)
    print(f"[Table 8] {len(n40_agg)} rows -> n40_breakdown.csv + n40_aggregates.csv", flush=True)


if __name__ == "__main__":
    main()
