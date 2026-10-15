"""N=40 paired bootstrap of QLA-MAPPO vs each baseline.

Pairs by (seed, distribution) — 5 seeds × 2 distributions = 10 paired units.
Bootstraps 10000 resamples over the 10 units (with replacement) and reports
mean QLA−baseline diff plus 95% percentile CI for each metric.

Inputs (all complete):
  - QLA-MAPPO:    results_qla_simplified_rerun/n40/QLA-MAPPO-NoMargin/{dist}/seed_{1-5}/KAIST/40u/{dist}/summary_LCC-MAPPO.csv
  - PPO-MV-A:     results_qla_simplified_rerun/n40_baselines/PPO-MV-A/{dist}/seed_{1-5}/Towards-PPO-MV_delay_4/KAIST/40u/{dist}/summary_Towards-PPO-MV.csv
  - PPO-DSR-A:    ...PPO-DSR-A/.../Towards-PPO-DSR_delay_4/.../summary_Towards-PPO-DSR.csv
  - QECO-A:       ...QECO-A/.../KAIST/40u/{dist}/summary_QECO-load-topk.csv
  - JSEQ-A:       ...JSEQ-A/.../KAIST/40u/{dist}/summary_JSEQ-A.csv

Output:
  - results_qla_simplified_rerun/paper_figures/n40_paired_bootstrap.csv
  - .../n40_breakdown_by_method.csv   (mean ± SD per method × dist × metric)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QLA_BASE = ROOT / "results_qla_simplified_rerun" / "n40" / "QLA-MAPPO-NoMargin"
BL_BASE = ROOT / "results_qla_simplified_rerun" / "n40_baselines"
OUT_DIR = ROOT / "results_qla_simplified_rerun" / "paper_figures"

SEEDS = [1, 2, 3, 4, 5]
DISTRIBUTIONS = ["uniform", "gaussian"]
DELAY = 4
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
N_BOOT = 10_000
RNG_SEED = 20260727


def qla_csv(dist: str, seed: int) -> Path:
    return QLA_BASE / dist / f"seed_{seed}" / "KAIST" / "40u" / dist / "summary_LCC-MAPPO.csv"


def baseline_csv(method: str, dist: str, seed: int) -> Path:
    seed_dir = BL_BASE / method / dist / f"seed_{seed}"
    sub = f"KAIST/40u/{dist}"
    if method == "PPO-MV-A":
        return seed_dir / f"Towards-PPO-MV_delay_{DELAY}" / sub / "summary_Towards-PPO-MV.csv"
    if method == "PPO-DSR-A":
        return seed_dir / f"Towards-PPO-DSR_delay_{DELAY}" / sub / "summary_Towards-PPO-DSR.csv"
    if method == "QECO-A":
        return seed_dir / sub / "summary_QECO-load-topk.csv"
    if method == "JSEQ-A":
        return seed_dir / sub / "summary_JSEQ-A.csv"
    raise ValueError(method)


def load_cell(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"empty: {csv_path}")
    rec = df.iloc[0]
    return {m: float(rec[m]) for m in METRICS}


def collect(method: str) -> pd.DataFrame:
    rows = []
    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            csv = qla_csv(dist, seed) if method == "QLA-MAPPO" else baseline_csv(method, dist, seed)
            if not csv.exists():
                raise FileNotFoundError(csv)
            cells = load_cell(csv)
            cells.update({"method": method, "distribution": dist, "seed": seed})
            rows.append(cells)
    return pd.DataFrame(rows)


def paired_bootstrap(
    qla: pd.DataFrame, baseline: pd.DataFrame, metric: str, n_boot: int = N_BOOT, rng_seed: int = RNG_SEED,
) -> dict[str, float]:
    """Pair by (seed, distribution). Returns mean diff, CI low, CI high."""
    merged = qla.merge(baseline, on=["seed", "distribution"], suffixes=("_qla", "_bl"))
    diffs = (merged[f"{metric}_qla"] - merged[f"{metric}_bl"]).to_numpy(dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(rng_seed)
    boot_means = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(diffs[idx]))
    return {
        "metric": metric,
        "n_pairs": n,
        "mean_diff": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
        "frac_negative": float(np.mean(boot_means < 0)),
        "qla_mean": float(merged[f"{metric}_qla"].mean()),
        "baseline_mean": float(merged[f"{metric}_bl"].mean()),
    }


def main() -> None:
    methods = ["QLA-MAPPO", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "JSEQ-A"]
    data = {m: collect(m) for m in methods}

    # Breakdown: mean ± SD per method × distribution
    breakdown_rows = []
    for m in methods:
        for dist in DISTRIBUTIONS:
            sub = data[m][data[m]["distribution"].eq(dist)]
            for metric in METRICS:
                vals = sub[metric].to_numpy(dtype=float)
                breakdown_rows.append({
                    "method": m, "distribution": dist, "metric": metric,
                    "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                    "n": len(vals),
                })
    pd.DataFrame(breakdown_rows).to_csv(
        OUT_DIR / "n40_breakdown_by_method.csv", index=False,
    )

    # Paired bootstrap QLA vs each baseline (per metric, pooled over distributions)
    paired_rows = []
    for bl in ["PPO-MV-A", "PPO-DSR-A", "QECO-A", "JSEQ-A"]:
        for metric in METRICS:
            row = paired_bootstrap(data["QLA-MAPPO"], data[bl], metric)
            row["baseline"] = bl
            paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)[
        ["baseline", "metric", "n_pairs", "qla_mean", "baseline_mean", "mean_diff", "ci95_low", "ci95_high", "frac_negative"]
    ]
    paired_df.to_csv(OUT_DIR / "n40_paired_bootstrap.csv", index=False)

    print("=== N=40 paired bootstrap (QLA-MAPPO - baseline; negative = QLA better) ===")
    print(paired_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
