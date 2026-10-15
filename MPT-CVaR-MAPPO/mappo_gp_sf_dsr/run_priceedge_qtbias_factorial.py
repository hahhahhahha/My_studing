from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np

from .run_lcc_mappo_study import (
    ROOT,
    Variant,
    aggregate_rows,
    build_command,
    read_summary,
    summary_path,
    write_rows,
)


DEFAULT_OUTPUT = ROOT / "results_priceedge_qtbias_factorial_2p3_center"
CENTER_SOURCE = ROOT / "results_priceedge_monoqt_formal_5x5"
NO_MARGIN_CENTER_SOURCE = (
    ROOT / "results_qla_simplified_rerun" / "main" / "QLA-MAPPO-NoMargin"
)


@dataclass(frozen=True)
class FactorialConfig:
    name: str
    A: int
    B: int
    C: int
    lambda_pi: float
    h_target: float
    c_value: float
    c_factor_name: str
    is_center: bool = False


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_design(factor_c: str, include_center: bool = True) -> list[FactorialConfig]:
    a_levels = {-1: 0.50, 1: 1.00}
    b_levels = {-1: 0.970, 1: 0.995}
    if factor_c == "stale_margin":
        c_levels = {-1: 0.50, 1: 1.50}
        c_center = 1.00
    elif factor_c == "lcc_v":
        c_levels = {-1: 0.25, 1: 1.00}
        c_center = 0.50
    else:
        raise ValueError(f"Unknown factor C: {factor_c}")
    design: list[FactorialConfig] = []
    run = 1
    for A in (-1, 1):
        for B in (-1, 1):
            for C in (-1, 1):
                design.append(
                    FactorialConfig(
                        name=f"F{run:02d}_A{'m' if A < 0 else 'p'}_B{'m' if B < 0 else 'p'}_C{'m' if C < 0 else 'p'}",
                        A=A,
                        B=B,
                        C=C,
                        lambda_pi=a_levels[A],
                        h_target=b_levels[B],
                        c_value=c_levels[C],
                        c_factor_name=factor_c,
                    )
                )
                run += 1
    if include_center:
        design.append(
            FactorialConfig(
                name="F09_center_A0_B0_C0",
                A=0,
                B=0,
                C=0,
                lambda_pi=0.75,
                h_target=0.990,
                c_value=c_center,
                c_factor_name=factor_c,
                is_center=True,
            )
        )
    return design


def runner_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        train_steps=args.train_steps,
        test_steps=args.test_steps,
        learning_rate=args.learning_rate,
        target_portfolio_entropy=0.99,
        rollout_size=args.rollout_size,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        lcc_cvar_budget=args.lcc_cvar_budget,
        lcc_concentration_factor=args.lcc_concentration_factor,
        lcc_stale_margin=args.lcc_stale_margin,
        lcc_v=args.lcc_v,
        lcc_migration_weight=args.lcc_migration_weight,
        device=args.device,
        force=args.force,
    )


def variant_from_config(cfg: FactorialConfig, args: argparse.Namespace) -> Variant:
    settings = {
            # Factor selection is validation-only.  The held-out test split is
            # reserved for the frozen final configuration and baselines.
            "evaluation-split-role": "validation",
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": cfg.lambda_pi,
            "target-portfolio-entropy": cfg.h_target,
            "lcc-stale-margin": args.lcc_stale_margin,
            "lcc-v": args.lcc_v,
            "lcc-migration-weight": args.lcc_migration_weight,
        }
    if cfg.c_factor_name == "stale_margin":
        settings["lcc-stale-margin"] = cfg.c_value
    elif cfg.c_factor_name == "lcc_v":
        settings["lcc-v"] = cfg.c_value
    # The evidence-checked final structure removes stale margin and disables
    # the dormant Z path in newly launched cells. Keep the switches explicit
    # so historical diagnostic designs remain reproducible.
    if args.no_stale_margin:
        settings["lcc-no-stale-margin"] = True
    if args.no_z_queue:
        settings["lcc-no-z-queue"] = True
    return Variant(cfg.name, settings)


def center_source_path(seed: int, delay: int, no_stale_margin: bool) -> Path:
    if no_stale_margin:
        root = NO_MARGIN_CENTER_SOURCE / f"seed_{seed}" / f"delay_{delay}"
        matches = list(root.rglob("summary_LCC-MAPPO.csv"))
        return matches[0] if matches else root / "summary_LCC-MAPPO.csv"
    return (
        CENTER_SOURCE
        / "formal"
        / "PriceEdgeBias-QT-P0p75-Final"
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "summary_LCC-MAPPO.csv"
    )


def add_factor_metadata(row: dict[str, object], cfg: FactorialConfig) -> dict[str, object]:
    row = dict(row)
    row.update(
        {
            "factor_A_level": cfg.A,
            "factor_B_level": cfg.B,
            "factor_C_level": cfg.C,
            "lambda_pi": cfg.lambda_pi,
            "target_portfolio_entropy": cfg.h_target,
            "factor_C_name": cfg.c_factor_name,
            "factor_C_value": cfg.c_value,
            "kappa_delta": cfg.c_value if cfg.c_factor_name == "stale_margin" else float("nan"),
            "lcc_v_factor": cfg.c_value if cfg.c_factor_name == "lcc_v" else float("nan"),
            "is_center": cfg.is_center,
            "design_name": cfg.name,
        }
    )
    return row


def run_or_reuse(args: argparse.Namespace) -> list[dict[str, object]]:
    out_root = args.output_dir.resolve()
    base_args = runner_args(args)
    rows: list[dict[str, object]] = []
    design = make_design(args.factor_c, include_center=not args.no_center)
    if args.center_only:
        design = [cfg for cfg in make_design(args.factor_c, include_center=True) if cfg.is_center]
    seeds = parse_int_list(args.seeds)
    delays = parse_int_list(args.delays)
    total = len(design) * len(seeds) * len(delays)
    done = 0
    for cfg in design:
        variant = variant_from_config(cfg, args)
        for seed in seeds:
            for delay in delays:
                # A Z-on compatibility checkpoint is not a clean center for
                # the no-Z final design; train that center instead of reusing.
                if cfg.is_center and args.reuse_center and not args.no_z_queue:
                    src = center_source_path(seed, delay, args.no_stale_margin)
                    if src.exists():
                        row = read_summary(src, "factorial", cfg.name, seed, delay)
                        row["summary_path"] = str(src)
                        rows.append(add_factor_metadata(row, cfg))
                        done += 1
                        print(f"[reuse-center] {cfg.name} seed={seed} delay={delay} ({done}/{total})", flush=True)
                        continue
                out = summary_path(out_root, "factorial", cfg.name, seed, delay)
                if out.exists() and not args.force:
                    print(f"[skip] {cfg.name} seed={seed} delay={delay}", flush=True)
                else:
                    print(
                        f"[run] {cfg.name} A={cfg.lambda_pi} B={cfg.h_target} "
                        f"C({cfg.c_factor_name})={cfg.c_value} "
                        f"seed={seed} delay={delay}",
                        flush=True,
                    )
                    subprocess.run(build_command(base_args, out_root, "factorial", variant, seed, delay), cwd=str(ROOT), check=True)
                if not out.exists():
                    raise RuntimeError(f"Expected summary not found: {out}")
                rows.append(add_factor_metadata(read_summary(out, "factorial", cfg.name, seed, delay), cfg))
                done += 1
                print(f"[done] {done}/{total}", flush=True)
    write_rows(out_root / "factorial_raw.csv", rows)
    write_rows(out_root / "factorial_summary.csv", aggregate_rows(rows))
    write_design(out_root, design)
    analyze_factorial(out_root, rows)
    return rows


def write_design(out_root: Path, design: list[FactorialConfig]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "factorial_design.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["design_name", "A", "B", "C", "lambda_pi", "target_portfolio_entropy", "factor_C_name", "factor_C_value", "is_center"],
        )
        writer.writeheader()
        for cfg in design:
            writer.writerow(
                {
                    "design_name": cfg.name,
                    "A": cfg.A,
                    "B": cfg.B,
                    "C": cfg.C,
                    "lambda_pi": cfg.lambda_pi,
                    "target_portfolio_entropy": cfg.h_target,
                    "factor_C_name": cfg.c_factor_name,
                    "factor_C_value": cfg.c_value,
                    "is_center": cfg.is_center,
                }
            )


def analyze_factorial(out_root: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    metrics = [
        "average_latency",
        "cvar99",
        "queue_oscillation",
        "portfolio_migration_rate",
        "global_demand_concentration",
        "mean_portfolio_entropy",
        "lcc_price_action_spearman",
    ]
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    for f in ["factor_A_level", "factor_B_level", "factor_C_level", "delay", "seed"]:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    corners = df[df["is_center"].astype(str).str.lower().isin(["false", "0"])]
    effect_rows: list[dict[str, object]] = []
    terms = {
        "A_lambda_pi": corners["factor_A_level"],
        "B_Htarget": corners["factor_B_level"],
        "C_" + str(corners["factor_C_name"].iloc[0]): corners["factor_C_level"],
        "AB": corners["factor_A_level"] * corners["factor_B_level"],
        "AC": corners["factor_A_level"] * corners["factor_C_level"],
        "BC": corners["factor_B_level"] * corners["factor_C_level"],
    }
    rng = np.random.default_rng(20260725)
    seeds = sorted(int(value) for value in corners["seed"].dropna().unique())
    for metric in metrics:
        if metric not in corners:
            continue
        for term, coded in terms.items():
            plus = corners.loc[coded == 1, metric].dropna()
            minus = corners.loc[coded == -1, metric].dropna()
            if len(plus) and len(minus):
                effect = float(plus.mean() - minus.mean())
                seed_effects = []
                for seed in seeds:
                    block = corners[corners["seed"].astype(int).eq(seed)]
                    if term == "A_lambda_pi":
                        block_code = block["factor_A_level"]
                    elif term == "B_Htarget":
                        block_code = block["factor_B_level"]
                    elif term.startswith("C_"):
                        block_code = block["factor_C_level"]
                    elif term == "AB":
                        block_code = block["factor_A_level"] * block["factor_B_level"]
                    elif term == "AC":
                        block_code = block["factor_A_level"] * block["factor_C_level"]
                    elif term == "BC":
                        block_code = block["factor_B_level"] * block["factor_C_level"]
                    else:
                        raise ValueError(term)
                    seed_effects.append(
                        float(block.loc[block_code.eq(1), metric].mean() - block.loc[block_code.eq(-1), metric].mean())
                    )
                seed_effects_arr = np.asarray(seed_effects, dtype=float)
                idx = rng.integers(0, len(seed_effects_arr), size=(10_000, len(seed_effects_arr)))
                boot = seed_effects_arr[idx].mean(axis=1)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                response_sd = float(corners[metric].std(ddof=1))
                effect_rows.append(
                    {
                        "metric": metric,
                        "term": term,
                        "effect_plus_minus": effect,
                        "seed_block_bootstrap95_low": float(ci_low),
                        "seed_block_bootstrap95_high": float(ci_high),
                        "standardized_effect_response_sd": effect / response_sd if response_sd > 0 else float("nan"),
                        "plus_mean": plus.mean(),
                        "minus_mean": minus.mean(),
                        "n_plus": len(plus),
                        "n_minus": len(minus),
                    }
                )
    pd.DataFrame(effect_rows).to_csv(out_root / "factorial_effects.csv", index=False)

    delay_rows: list[dict[str, object]] = []
    for delay, group in corners.groupby("delay"):
        for metric in metrics:
            high = group.loc[group["factor_C_level"] == 1, metric].dropna()
            low = group.loc[group["factor_C_level"] == -1, metric].dropna()
            if len(high) and len(low):
                delay_seeds = sorted(int(value) for value in group["seed"].dropna().unique())
                seed_effects = []
                for seed in delay_seeds:
                    block = group[group["seed"].astype(int).eq(seed)]
                    seed_effects.append(
                        float(
                            block.loc[block["factor_C_level"] == 1, metric].mean()
                            - block.loc[block["factor_C_level"] == -1, metric].mean()
                        )
                    )
                seed_effects_arr = np.asarray(seed_effects, dtype=float)
                idx = rng.integers(0, len(seed_effects_arr), size=(10_000, len(seed_effects_arr)))
                boot = seed_effects_arr[idx].mean(axis=1)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                delay_rows.append(
                    {
                        "delay": delay,
                        "metric": metric,
                        "C_high_minus_low": high.mean() - low.mean(),
                        "seed_block_bootstrap95_low": float(ci_low),
                        "seed_block_bootstrap95_high": float(ci_high),
                        "C_high_mean": high.mean(),
                        "C_low_mean": low.mean(),
                        "n_high": len(high),
                        "n_low": len(low),
                    }
                )
    pd.DataFrame(delay_rows).to_csv(out_root / "factorial_C_by_delay.csv", index=False)

    corner_mean = corners.groupby(["factor_A_level", "factor_B_level", "factor_C_level"], as_index=False)[metrics].mean(numeric_only=True)
    center = df[df["is_center"].astype(str).str.lower().isin(["true", "1"])]
    center_mean = center[metrics].mean(numeric_only=True).to_dict()
    curvature_rows = []
    for metric in metrics:
        if metric in corner_mean.columns and metric in center_mean:
            curvature_rows.append(
                {
                    "metric": metric,
                    "center_mean": center_mean[metric],
                    "corner_grand_mean": corner_mean[metric].mean(),
                    "center_minus_corner_mean": center_mean[metric] - corner_mean[metric].mean(),
                }
            )
    pd.DataFrame(curvature_rows).to_csv(out_root / "factorial_center_curvature.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2^3 + center factorial experiment for PriceEdge-QT-Bias MAPPO.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rollout-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lcc-v", type=float, default=0.5)
    parser.add_argument("--lcc-stale-margin", type=float, default=1.0)
    parser.add_argument("--factor-c", choices=["stale_margin", "lcc_v"], default="lcc_v")
    parser.add_argument("--lcc-migration-weight", type=float, default=0.05)
    parser.add_argument("--lcc-cvar-budget", type=float, default=1.30)
    parser.add_argument("--lcc-concentration-factor", type=float, default=2.0)
    parser.add_argument("--no-stale-margin", action="store_true")
    parser.add_argument("--no-z-queue", dest="no_z_queue", action="store_true", default=True)
    parser.add_argument("--keep-z-queue", dest="no_z_queue", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-center", dest="reuse_center", action="store_true", default=True)
    parser.add_argument("--no-reuse-center", dest="reuse_center", action="store_false")
    parser.add_argument("--no-center", action="store_true",
                        help="Run only the eight factorial corners (no center point).")
    parser.add_argument("--center-only", action="store_true",
                        help="Run only the center configuration; useful for final-structure confirmation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_stale_margin and args.factor_c == "stale_margin":
        raise SystemExit(
            "Invalid factorial: --no-stale-margin makes the stale_margin "
            "factor inert. Use --factor-c lcc_v for the final no-margin design."
        )
    rows = run_or_reuse(args)
    print(f"Factorial experiment completed/updated: {args.output_dir.resolve()} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
