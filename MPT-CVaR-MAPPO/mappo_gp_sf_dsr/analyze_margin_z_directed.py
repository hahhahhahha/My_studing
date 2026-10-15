"""Summarize the bug-fixed stale-margin and tight-budget Z experiments.

The script treats seed as the independent repetition and delay as an
experimental condition.  With only three seeds, intervals are descriptive;
the raw paired differences are always emitted next to the bootstrap interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "average_latency": "Avg latency",
    "cvar99": "CVaR99",
    "queue_oscillation": "Queue osc.",
    "portfolio_migration_rate": "Migration",
    "lcc_z_queue": "Z pressure",
    "lcc_cvar_violation": "CVaR violation",
}


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_scenario(df: pd.DataFrame, scenario: str, n_boot: int,
                       seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    for (variant, delay), g in df.groupby(["variant", "delay"], sort=False):
        for col, label in METRICS.items():
            if col not in g or g[col].isna().all():
                continue
            vals = g[col].astype(float).to_numpy()
            lo, hi = bootstrap_mean_ci(vals, rng, n_boot)
            summary_rows.append({
                "scenario": scenario,
                "variant": variant,
                "delay": int(delay),
                "metric": label,
                "n_seeds": len(vals),
                "mean": vals.mean(),
                "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            })

    if scenario == "A":
        reference = "Main"
        comparisons = ["NoStaleMargin-BugFixed"]
    else:
        reference = "Main-TightBudget"
        comparisons = ["NoZ-TightBudget", "StaticReward-TightBudget"]

    paired_rows: list[dict[str, object]] = []
    for candidate in comparisons:
        for delay in sorted(df["delay"].astype(int).unique()):
            ref = df[(df.variant == reference) & (df.delay.astype(int) == delay)]
            alt = df[(df.variant == candidate) & (df.delay.astype(int) == delay)]
            for col, label in METRICS.items():
                if col not in df:
                    continue
                pair = ref[["seed", col]].merge(
                    alt[["seed", col]], on="seed", suffixes=("_ref", "_candidate")
                ).dropna()
                if pair.empty:
                    continue
                # Positive means the ablated/alternative variant is worse.
                diff = (pair[f"{col}_candidate"].astype(float)
                        - pair[f"{col}_ref"].astype(float)).to_numpy()
                lo, hi = bootstrap_mean_ci(diff, rng, n_boot)
                paired_rows.append({
                    "scenario": scenario,
                    "reference": reference,
                    "candidate": candidate,
                    "delay": delay,
                    "metric": label,
                    "n_pairs": len(diff),
                    "paired_diff_candidate_minus_reference": diff.mean(),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "paired_differences": ";".join(f"{x:.9g}" for x in diff),
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("results_margin_z_directed"))
    p.add_argument("--bootstrap", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20260725)
    args = p.parse_args()

    all_summary: list[pd.DataFrame] = []
    all_paired: list[pd.DataFrame] = []
    for scenario in ("A", "B"):
        path = args.input_dir / f"{scenario}_raw.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        df = pd.read_csv(path)
        summary, paired = summarize_scenario(
            df, scenario, args.bootstrap, args.seed + ord(scenario)
        )
        all_summary.append(summary)
        all_paired.append(paired)

    if not all_summary:
        raise SystemExit("No scenario raw files were found")
    summary_out = pd.concat(all_summary, ignore_index=True)
    paired_out = pd.concat(all_paired, ignore_index=True)
    summary_out.to_csv(args.input_dir / "directed_summary_bootstrap.csv", index=False)
    paired_out.to_csv(args.input_dir / "directed_paired_differences.csv", index=False)
    print(summary_out.to_string(index=False))
    print("\nPaired differences (candidate - reference):")
    print(paired_out.to_string(index=False))


if __name__ == "__main__":
    main()
