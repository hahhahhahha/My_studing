"""Directed experiment audit: Scenario A (bug-fixed w/o stale margin) and Scenario B (tight-budget Z activation).

Outputs (in results_margin_z_directed/audit/):
  - directed_summary.csv          per (scenario, variant, seed, delay) test metrics + audit magnitudes
  - directed_paired.csv           paired diff vs scenario main, with seed-block bootstrap 95% CI
  - directed_z_dormancy.csv       Z/H activation rates per variant
  - directed_flag_check.csv       ablation flags per variant (from config JSON)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIRECTED_ROOT = ROOT / "results_margin_z_directed"
OUT_DIR = DIRECTED_ROOT / "audit"

# Columns to read from the LAST N rows of learning CSV (post-training steady state)
LEARNING_TAIL = 100  # use last 100 rows for steady-state estimate
LEARNING_COLS = [
    "lcc_phi", "lcc_reward", "lcc_q_term",
    "lcc_tail_latency_norm", "lcc_var_nu",
    "lcc_cvar_violation", "lcc_concentration_violation",
    "lcc_z_queue", "lcc_h_queue",
    "lcc_migration",
]

# Columns to read from summary CSV (test phase)
SUMMARY_COLS = [
    "average_latency", "cvar99", "queue_oscillation",
    "portfolio_migration_rate", "global_demand_concentration",
    "mean_portfolio_entropy", "inference_time_ms",
]

# Flags to verify from config JSON
FLAG_KEYS_NEG = [
    "lcc_no_z_queue", "lcc_no_h_queue", "lcc_no_stale_margin", "lcc_static_reward",
]
FLAG_KEYS_POS = [
    "lcc_use_z_queue", "lcc_use_h_queue", "lcc_use_stale_margin",
]


def find_run_dirs() -> list[Path]:
    runs = []
    for summary_path in DIRECTED_ROOT.rglob("summary_LCC-MAPPO.csv"):
        runs.append(summary_path.parent)
    return runs


def parse_run_path(run_dir: Path) -> dict[str, object]:
    """Extract scenario / variant / seed / delay from path."""
    parts = run_dir.parts
    rel = Path(*parts[parts.index("results_margin_z_directed") + 1:])
    # rel = scenario/variant_name/seed_X/delay_Y/dataset/.../...
    scenario = rel.parts[0]
    variant = rel.parts[1]
    seed = int(rel.parts[2].split("_")[1])
    delay = int(rel.parts[3].split("_")[1])
    return {"scenario": scenario, "variant": variant, "seed": seed, "delay": delay}


def read_learning_tail(run_dir: Path) -> dict[str, float]:
    learning_csv = run_dir / "learning_LCC-MAPPO.csv"
    if not learning_csv.exists():
        return {}
    try:
        df = pd.read_csv(learning_csv)
    except Exception:
        return {}
    df_tail = df.tail(LEARNING_TAIL)
    out = {}
    for col in LEARNING_COLS:
        if col in df_tail.columns:
            vals = pd.to_numeric(df_tail[col], errors="coerce").dropna()
            out[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    # active rates
    for col in ["lcc_cvar_violation", "lcc_concentration_violation"]:
        if col in df_tail.columns:
            vals = pd.to_numeric(df_tail[col], errors="coerce").dropna()
            active = (vals > 0).sum()
            out[f"{col}_active_rate"] = float(active / max(len(vals), 1))
    return out


def read_summary(run_dir: Path) -> dict[str, float]:
    summary_csv = run_dir / "summary_LCC-MAPPO.csv"
    if not summary_csv.exists():
        return {}
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return {}
    if len(df) == 0:
        return {}
    rec = df.iloc[0].to_dict()
    out = {}
    for col in SUMMARY_COLS:
        if col in rec:
            try:
                out[col] = float(rec[col])
            except (TypeError, ValueError):
                out[col] = float("nan")
    return out


def read_flags(run_dir: Path) -> dict[str, object]:
    config_json = run_dir / "config_LCC-MAPPO.json"
    if not config_json.exists():
        return {}
    try:
        with config_json.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return {}
    out = {}
    for k in FLAG_KEYS_NEG + FLAG_KEYS_POS:
        out[k] = bool(cfg.get(k, False)) if k in FLAG_KEYS_NEG else cfg.get(k, None)
    out["lcc_cvar_budget"] = cfg.get("lcc_cvar_budget", None)
    return out


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for run_dir in find_run_dirs():
        meta = parse_run_path(run_dir)
        row = {**meta, "run_dir": str(run_dir)}
        row.update(read_summary(run_dir))
        row.update(read_learning_tail(run_dir))
        row.update(read_flags(run_dir))
        rows.append(row)
    return rows


def paired_bootstrap(
    rows: pd.DataFrame,
    main_variant: str,
    other_variant: str,
    scenario: str,
    metrics: list[str],
    n_boot: int = 10_000,
    rng_seed: int = 20260726,
) -> list[dict[str, object]]:
    """Seed-block paired bootstrap: for each (seed, delay), compute main-other diff.

    Resample seeds with replacement, then compute mean over (seed, delay) blocks.
    """
    sub_main = rows[(rows["scenario"] == scenario) & (rows["variant"] == main_variant)]
    sub_other = rows[(rows["scenario"] == scenario) & (rows["variant"] == other_variant)]
    if len(sub_main) == 0 or len(sub_other) == 0:
        return []

    keys = ["seed", "delay"]
    merged = pd.merge(
        sub_main[keys + metrics].rename(columns={m: f"{m}_main" for m in metrics}),
        sub_other[keys + metrics].rename(columns={m: f"{m}_other" for m in metrics}),
        on=keys,
        how="inner",
    )
    if len(merged) == 0:
        return []

    out_rows = []
    seeds_unique = sorted(merged["seed"].unique())
    rng = np.random.default_rng(rng_seed)
    for metric in metrics:
        diff = (merged[f"{metric}_other"] - merged[f"{metric}_main"]).to_numpy()
        point = float(np.mean(diff))
        # seed-block bootstrap: resample seeds, average diff over (seed,delay) within resample
        boot_means = []
        for _ in range(n_boot):
            idx = rng.choice(seeds_unique, size=len(seeds_unique), replace=True)
            block_means = []
            for s in idx:
                block = merged[merged["seed"] == s][f"{metric}_other"] - merged[merged["seed"] == s][f"{metric}_main"]
                block_means.append(block.mean())
            boot_means.append(np.mean(block_means))
        boot = np.asarray(boot_means)
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        # win rate: fraction of pairs where other is better (lower) than main
        win_rate = float((diff < 0).mean())
        out_rows.append({
            "scenario": scenario,
            "main_variant": main_variant,
            "other_variant": other_variant,
            "metric": metric,
            "n_paired": len(merged),
            "mean_diff_other_minus_main": point,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "win_rate_other_better": win_rate,
            "rel_pct": float(point / max(abs(np.mean(merged[f"{metric}_main"])), 1e-9) * 100),
        })
    return out_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    if not rows:
        print(f"[error] no runs found under {DIRECTED_ROOT}")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "directed_summary.csv", index=False)
    print(f"[ok] wrote directed_summary.csv with {len(df)} rows")

    # Flag check
    flag_cols = FLAG_KEYS_NEG + FLAG_KEYS_POS + ["lcc_cvar_budget"]
    flag_view = df.groupby(["scenario", "variant"])[flag_cols].agg(lambda s: s.iloc[0] if len(s) else None).reset_index()
    flag_view.to_csv(OUT_DIR / "directed_flag_check.csv", index=False)
    print(f"[ok] wrote directed_flag_check.csv")

    # Z/H dormancy
    dorm_cols = [
        "scenario", "variant",
        "lcc_cvar_violation_mean", "lcc_cvar_violation_active_rate",
        "lcc_concentration_violation_mean", "lcc_concentration_violation_active_rate",
        "lcc_z_queue_mean", "lcc_h_queue_mean", "lcc_var_nu_mean",
    ]
    available = [c for c in dorm_cols if c in df.columns]
    dorm = df.groupby(["scenario", "variant"])[available[2:]].agg(["mean", "std"]).reset_index()
    dorm.to_csv(OUT_DIR / "directed_z_dormancy.csv", index=False)
    print(f"[ok] wrote directed_z_dormancy.csv")

    # Also write a flattened version
    flat = df.groupby(["scenario", "variant"]).agg(
        n_runs=("seed", "count"),
        avg_latency_mean=("average_latency", "mean"),
        avg_latency_std=("average_latency", "std"),
        cvar99_mean=("cvar99", "mean"),
        cvar99_std=("cvar99", "std"),
        queue_osc_mean=("queue_oscillation", "mean"),
        z_violation_mean=("lcc_cvar_violation_mean", "mean"),
        z_active_rate_mean=("lcc_cvar_violation_active_rate", "mean"),
        h_active_rate_mean=("lcc_concentration_violation_active_rate", "mean"),
        z_queue_mean=("lcc_z_queue_mean", "mean"),
        var_nu_mean=("lcc_var_nu_mean", "mean"),
    ).reset_index()
    flat.to_csv(OUT_DIR / "directed_aggregate.csv", index=False)
    print(f"[ok] wrote directed_aggregate.csv")

    # Paired bootstrap
    paired_rows = []
    metrics = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
    # Scenario A: Main vs NoStaleMargin-BugFixed
    paired_rows.extend(paired_bootstrap(df, "Main", "NoStaleMargin-BugFixed", "A", metrics))
    # Scenario B: Main-TightBudget vs NoZ-TightBudget and StaticReward-TightBudget
    paired_rows.extend(paired_bootstrap(df, "Main-TightBudget", "NoZ-TightBudget", "B", metrics))
    paired_rows.extend(paired_bootstrap(df, "Main-TightBudget", "StaticReward-TightBudget", "B", metrics))
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(OUT_DIR / "directed_paired.csv", index=False)
        print(f"[ok] wrote directed_paired.csv with {len(paired_rows)} rows")
    else:
        print("[warn] no paired rows produced")

    # Print summary to console
    print("\n=== Aggregate (mean per variant) ===")
    print(flat.to_string(index=False))

    print("\n=== Z/H activation summary ===")
    z_view = flat[["scenario", "variant", "n_runs", "z_violation_mean", "z_active_rate_mean", "h_active_rate_mean", "z_queue_mean"]]
    print(z_view.to_string(index=False))

    if paired_rows:
        print("\n=== Paired bootstrap (vs scenario main) ===")
        print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
