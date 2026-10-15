"""Dedicated lcc_v sweep to defend the lcc_v=0.5 choice.

Audit (§6.0 factorial) found V's main effect is inert in {0.25, 0.50, 1.00},
but a dedicated sweep at finer grain including V=2.0 has not been run. This
script trains QLA-MAPPO Full-noZ at lcc_v in {0.25, 0.5, 1.0, 2.0}, holding
lambda_pi=0.75 and H_target=0.99 fixed, over 3 seeds × 3 delays (0, 4, 8).

The 9 cells at lcc_v=0.5 are reused from the factorial center if available;
the other 27 cells are newly trained.

Output: results_qla_simplified_rerun/lcc_v_sweep/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_qla_simplified_rerun" / "lcc_v_sweep"
FACTORIAL_ROOT = ROOT / "results_priceedge_qtbias_factorial_2p3_center"

LCC_V_LEVELS = [0.25, 0.5, 1.0, 2.0]
SEEDS_DEFAULT = [1, 2, 3]
DELAYS_DEFAULT = [0, 4, 8]


def common_main_args(lcc_v: float) -> dict[str, object]:
    """Full-noZ QLA-MAPPO with lcc_v varied; all other knobs at center."""
    return {
        "dataset": "KAIST",
        "users": 30,
        "distribution": "poisson_bursty",
        "broadcast-period": 2,
        "slot-s": 1.0,
        "n-edges": 35,
        "edge-compute-mb-s": 63.0,
        "queue-upper-mb": 630.0,
        "task-compute-mb": 15.0,
        "task-package-mb": 270.0 / 1024.0,
        "max-requests-per-user": 7,
        "poisson-lambda": 3.5,
        "burst-prob": 0.1,
        "burst-multiplier": 2.0,
        "background-load": 0.0,
        "capacity-mode": "homogeneous",
        "edge-compute-scale": 1.0,
        "task-compute-scale": 1.0,
        "max-supported-users": 50,
        "score-resource-weight": 1.0,
        "score-capacity-weight": 0.5,
        "score-task-count-weight": 0.5,
        "score-wait-weight": 1.0,
        "learning-rate": 1e-3,
        "rollout-size": 1_024,
        "batch-size": 128,
        "target-portfolio-entropy": 0.99,
        "log-interval": 100,
        "train-steps": 10_000,
        "test-steps": 1_000,
        "lcc-cvar-alpha": 0.99,
        "lcc-cvar-budget": 1.30,
        "lcc-concentration-factor": 2.0,
        "lcc-stale-margin": 1.0,
        "lcc-price-observation": True,
        "lcc-price-scale-observation": True,
        "lcc-edgewise-price-actor": True,
        "lcc-price-components": "QT",
        "lcc-price-logit-bias": 0.75,
        "lcc-v": lcc_v,
        "lcc-migration-weight": 0.05,
        "lcc-no-stale-margin": True,
        "lcc-no-z-queue": True,
        "evaluation-split-role": "validation",
    }


def build_command(lcc_v: float, seed: int, delay: int, output_dir: Path, device: str | None) -> list[str]:
    base = common_main_args(lcc_v)
    base["seed"] = seed
    base["feedback-delay"] = delay
    base["output-dir"] = str(output_dir)
    cmd = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train", "--method", "LCC-MAPPO"]
    for k, v in base.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        elif v is not None:
            cmd.extend([f"--{k}", str(v)])
    if device:
        cmd.extend(["--device", device])
    return cmd


def factorial_center_path(seed: int, delay: int, factorial_root: Path = FACTORIAL_ROOT) -> Path | None:
    """Reuse lcc_v=0.5 cells from factorial center if present."""
    pattern = factorial_root / "factorial" / "F09_center_*" / f"seed_{seed}" / f"delay_{delay}"
    matches = list(pattern.parent.rglob(f"*/seed_{seed}/delay_{delay}/KAIST/30u/poisson_bursty/summary_LCC-MAPPO.csv"))
    # Filter for center cells only
    for m in matches:
        if "F09_center" in str(m):
            return m
    return None


def find_summary(out_dir: Path) -> Path | None:
    for p in out_dir.rglob("summary_LCC-MAPPO.csv"):
        return p
    return None


def run_cell(lcc_v: float, seed: int, delay: int, output_root: Path, device: str | None, force: bool) -> Path:
    out_dir = output_root / f"lcc_v_{lcc_v:.2f}" / f"seed_{seed}" / f"delay_{delay}"
    summary = find_summary(out_dir)
    if summary and summary.exists() and not force:
        print(f"[skip] lcc_v={lcc_v} seed={seed} delay={delay}", flush=True)
        return summary
    cmd = build_command(lcc_v, seed, delay, out_dir, device)
    print(f"[run] lcc_v={lcc_v} seed={seed} delay={delay}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    summary = find_summary(out_dir)
    if not summary or not summary.exists():
        raise RuntimeError(f"Missing summary for lcc_v={lcc_v} seed={seed} delay={delay}")
    return summary


def read_summary(summary_path: Path, lcc_v: float, seed: int, delay: int, source: str) -> dict[str, object]:
    df = pd.read_csv(summary_path)
    rec = df.iloc[0].to_dict()
    rec.update({
        "lcc_v": lcc_v,
        "seed": seed,
        "feedback_delay": delay,
        "source": source,
        "summary_path": str(summary_path),
    })
    return rec


def collect_all(
    output_root: Path,
    *,
    factorial_root: Path = FACTORIAL_ROOT,
    reuse_factorial: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lcc_v in LCC_V_LEVELS:
        for seed in SEEDS_DEFAULT:
            for delay in DELAYS_DEFAULT:
                summary = find_summary(output_root / f"lcc_v_{lcc_v:.2f}" / f"seed_{seed}" / f"delay_{delay}")
                if summary and summary.exists():
                    rows.append(read_summary(summary, lcc_v, seed, delay, "sweep"))
                else:
                    # Try factorial center for lcc_v=0.5
                    if lcc_v == 0.5 and reuse_factorial:
                        fpath = factorial_center_path(seed, delay, factorial_root)
                        if fpath and fpath.exists():
                            rows.append(read_summary(fpath, lcc_v, seed, delay, "factorial_center_reuse"))
    return rows


def summarize(rows: list[dict[str, object]], out_root: Path) -> None:
    if not rows:
        print("[warn] no rows to summarize", flush=True)
        return
    df = pd.DataFrame(rows)
    metrics = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate", "lcc_price_action_spearman"]
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Delay is a fixed within-seed condition.  Average over delays first so
    # uncertainty is estimated from the three independent training seeds.
    seed_blocks = df.groupby(["lcc_v", "seed"], as_index=False)[metrics].mean()
    summary_rows = []
    for lcc_v, group in seed_blocks.groupby("lcc_v"):
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            t_critical = float(student_t.ppf(0.975, len(values) - 1)) if len(values) > 1 else float("nan")
            ci_half = t_critical * std / np.sqrt(len(values)) if np.isfinite(t_critical) else float("nan")
            summary_rows.append({
                "lcc_v": float(lcc_v),
                "metric": metric,
                "n_training_seeds": len(values),
                "n_delay_cells": int(len(df[df["lcc_v"] == lcc_v])),
                "mean": mean,
                "seed_block_std": std,
                "ci95_low": mean - ci_half if np.isfinite(ci_half) else float("nan"),
                "ci95_high": mean + ci_half if np.isfinite(ci_half) else float("nan"),
            })
    summary_df = pd.DataFrame(summary_rows)

    paired_rows = []
    reference = seed_blocks[seed_blocks["lcc_v"] == 0.5].set_index("seed")
    for lcc_v, group in seed_blocks.groupby("lcc_v"):
        candidate = group.set_index("seed")
        common_seeds = sorted(set(reference.index) & set(candidate.index))
        for metric in metrics:
            differences = (
                candidate.loc[common_seeds, metric].to_numpy(dtype=float)
                - reference.loc[common_seeds, metric].to_numpy(dtype=float)
            )
            point = float(np.mean(differences))
            std = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
            t_critical = (
                float(student_t.ppf(0.975, len(differences) - 1))
                if len(differences) > 1 else float("nan")
            )
            ci_half = (
                t_critical * std / np.sqrt(len(differences))
                if np.isfinite(t_critical) else float("nan")
            )
            paired_rows.append({
                "lcc_v": float(lcc_v),
                "reference_lcc_v": 0.5,
                "metric": metric,
                "n_training_seeds": len(differences),
                "mean_diff_candidate_minus_reference": point,
                "ci95_low": point - ci_half if np.isfinite(ci_half) else float("nan"),
                "ci95_high": point + ci_half if np.isfinite(ci_half) else float("nan"),
                "ci_excludes_zero": bool(
                    np.isfinite(ci_half)
                    and (point + ci_half < 0.0 or point - ci_half > 0.0)
                ),
            })
    paired_df = pd.DataFrame(paired_rows)
    summary_df.to_csv(out_root / "lcc_v_sweep_summary.csv", index=False)
    paired_df.to_csv(out_root / "lcc_v_sweep_paired_vs_0p5.csv", index=False)
    df.to_csv(out_root / "lcc_v_sweep_raw.csv", index=False)
    print(f"[summary] -> {out_root / 'lcc_v_sweep_summary.csv'}", flush=True)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--delays", default=",".join(str(d) for d in DELAYS_DEFAULT))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--factorial-root", type=Path, default=FACTORIAL_ROOT)
    p.add_argument("--no-reuse-factorial", action="store_true")
    p.add_argument("--summary-only", action="store_true",
                   help="only collect existing rows and write summary, no training")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    delays = [int(d) for d in args.delays.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        total = len(LCC_V_LEVELS) * len(seeds) * len(delays)
        done = 0
        for lcc_v in LCC_V_LEVELS:
            for seed in seeds:
                for delay in delays:
                    # Reuse factorial center cells for lcc_v=0.5 if available
                    if lcc_v == 0.5 and not args.no_reuse_factorial:
                        fpath = factorial_center_path(seed, delay, args.factorial_root)
                        if fpath and fpath.exists():
                            done += 1
                            print(f"[reuse-factorial] lcc_v=0.5 seed={seed} delay={delay} ({done}/{total})", flush=True)
                            continue
                    run_cell(lcc_v, seed, delay, args.output_dir, args.device, args.force)
                    done += 1
                    print(f"[done] {done}/{total}", flush=True)

    rows = collect_all(
        args.output_dir,
        factorial_root=args.factorial_root,
        reuse_factorial=not args.no_reuse_factorial,
    )
    summarize(rows, args.output_dir)


if __name__ == "__main__":
    main()
