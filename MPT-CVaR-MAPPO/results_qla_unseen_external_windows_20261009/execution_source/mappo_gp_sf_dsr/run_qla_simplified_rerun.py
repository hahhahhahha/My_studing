"""Re-run QLA-MAPPO experiments with the final Full-noZ config.

Scope (all QLA-MAPPO cells):
  Phase 1: main 5 seeds x 5 delays = 25 training cells
  Phase 2: ablation 3 variants x 5 seeds x 5 delays = 75 training cells
  Phase 3: N40 stress 2 dist x 5 seeds = 10 training cells
  Phase 4: cross-env fixed-checkpoint test = 48 test-only cells (use phase-1 checkpoint)

All cells use --lcc-no-stale-margin and --lcc-no-z-queue. The legacy Z-on
main grid has been archived under main_legacy_zon/ and is no longer cited;
see ALGORITHM_FREEZE_FullNoZ.md for the freeze rationale. Per-variant
structural flags (edgewise vs flat actor, shuffle, logit bias, price
components) are validated by finalize_qla_simplified_results.py.

Output: results_qla_simplified_rerun/{main,ablation,n40,crossenv}/...
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_qla_simplified_rerun"


@dataclass(frozen=True)
class SimplifiedVariant:
    name: str
    extra_args: dict[str, object]


# Phase 1: main
MAIN_VARIANT = SimplifiedVariant(
    "QLA-MAPPO-NoMargin",
    {},
)

# Phase 2: ablations (all with --lcc-no-stale-margin added uniformly)
ABLATION_VARIANTS = [
    SimplifiedVariant("QL-NoBias-NoMargin", {"lcc-price-logit-bias": 0.0}),
    SimplifiedVariant("QL-Bias-Shuffle-NoMargin", {"lcc-price-shuffle": True}),
    # Actor modes are mutually exclusive.  Leaving the edge-wise flag enabled
    # makes MAPPOPolicy select EdgeWisePriceGaussianActor before it ever
    # reaches the flat-actor branch, turning this ablation into a no-op.
    SimplifiedVariant(
        "Flat-Bias-QL-NoMargin",
        {
            "lcc-edgewise-price-actor": False,
            "lcc-flat-price-bias-actor": True,
        },
    ),
]


def common_main_args() -> dict[str, object]:
    """Match run_lcc_mappo_study.py formal phase defaults + directed experiment env."""
    return {
        # Environment
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
        # Training
        "learning-rate": 1e-3,
        "rollout-size": 1_024,
        "batch-size": 128,
        "target-portfolio-entropy": 0.99,  # formal phase default
        "log-interval": 100,
        "train-steps": 10_000,  # formal protocol
        "test-steps": 1_000,
        # LCC base
        "lcc-cvar-alpha": 0.99,
        "lcc-cvar-budget": 1.30,
        "lcc-concentration-factor": 2.0,
        "lcc-stale-margin": 1.0,
        # QLA-MAPPO price-edge QT-bias selected on canonical validation
        "lcc-price-observation": True,
        "lcc-price-scale-observation": True,
        "lcc-edgewise-price-actor": True,
        "lcc-price-components": "QT",
        "lcc-price-logit-bias": 0.50,
        "lcc-v": 0.5,
        "lcc-migration-weight": 0.05,
        # The bug-fixed flag that disables stale margin in price path
        "lcc-no-stale-margin": True,
        # Directed audit: Z is dormant at budget=1.30 and harmful when forced
        # active under a tight budget, so new simplified cells disable it.
        "lcc-no-z-queue": True,
    }


def build_command(
    variant: SimplifiedVariant,
    seed: int,
    delay: int,
    output_dir: Path,
    device: str | None,
    extra_overrides: dict[str, object] | None = None,
) -> list[str]:
    base = common_main_args()
    base.update(variant.extra_args)
    if extra_overrides:
        base.update(extra_overrides)
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


def find_summary(out_dir: Path) -> Path | None:
    for p in out_dir.rglob("summary_LCC-MAPPO.csv"):
        return p
    return None


def run_cell(
    variant: SimplifiedVariant,
    seed: int,
    delay: int,
    output_root: Path,
    phase: str,
    device: str | None,
    force: bool,
    extra_overrides: dict[str, object] | None = None,
) -> Path:
    out_dir = output_root / phase / variant.name / f"seed_{seed}" / f"delay_{delay}"
    summary = find_summary(out_dir)
    if summary and summary.exists() and not force:
        print(f"[skip] {phase} {variant.name} seed={seed} delay={delay}", flush=True)
        return summary
    cmd = build_command(variant, seed, delay, out_dir, device, extra_overrides)
    print(f"[run] {phase} {variant.name} seed={seed} delay={delay}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    summary = find_summary(out_dir)
    if not summary or not summary.exists():
        raise RuntimeError(f"Missing summary for {variant.name} seed={seed} delay={delay}")
    return summary


def phase_main(args: argparse.Namespace, output_root: Path) -> None:
    seeds = parse_int_list(args.formal_seeds)
    delays = parse_int_list(args.formal_delays)
    total = len(seeds) * len(delays)
    done = 0
    for seed in seeds:
        for delay in delays:
            run_cell(MAIN_VARIANT, seed, delay, output_root, "main", args.device, args.force)
            done += 1
            print(f"[done] main {done}/{total}", flush=True)


def phase_ablation(args: argparse.Namespace, output_root: Path) -> None:
    requested = {
        value.strip() for value in args.ablation_variants.split(",") if value.strip()
    }
    variants = (
        ABLATION_VARIANTS
        if not requested
        else [variant for variant in ABLATION_VARIANTS if variant.name in requested]
    )
    unknown = requested - {variant.name for variant in ABLATION_VARIANTS}
    if unknown:
        raise ValueError(f"Unknown ablation variants: {sorted(unknown)}")
    seeds = parse_int_list(args.formal_seeds)
    delays = parse_int_list(args.formal_delays)
    total = len(variants) * len(seeds) * len(delays)
    done = 0
    for variant in variants:
        for seed in seeds:
            for delay in delays:
                run_cell(variant, seed, delay, output_root, "ablation", args.device, args.force)
                done += 1
                print(f"[done] ablation {done}/{total}", flush=True)


def phase_n40(args: argparse.Namespace, output_root: Path) -> None:
    seeds = parse_int_list(args.n40_seeds)
    distributions = args.n40_distributions.split(",")
    total = len(seeds) * len(distributions)
    done = 0
    for dist in distributions:
        for seed in seeds:
            extra = {"distribution": dist.strip(), "users": 40, "train-steps": 10_000, "test-steps": 1_000}
            out_dir = output_root / "n40" / MAIN_VARIANT.name / f"{dist.strip()}" / f"seed_{seed}"
            summary = find_summary(out_dir)
            if summary and summary.exists() and not args.force:
                print(f"[skip] n40 {dist} seed={seed}", flush=True)
            else:
                cmd = build_command(MAIN_VARIANT, seed, 4, out_dir, args.device, extra_overrides=extra)
                print(f"[run] n40 {dist} seed={seed}", flush=True)
                subprocess.run(cmd, cwd=str(ROOT), check=True)
            done += 1
            print(f"[done] n40 {done}/{total}", flush=True)


def phase_crossenv(args: argparse.Namespace, output_root: Path) -> None:
    """Cross-env uses fixed checkpoint: take KAIST main seed=1 delay=0 (already trained in phase 1).

    For each (dataset, distribution, N, delay), copy or load the checkpoint and run test-only.
    """
    # Locate checkpoint from phase 1
    checkpoint_dir = output_root / "main" / MAIN_VARIANT.name / "seed_1" / "delay_0"
    checkpoint = None
    for p in checkpoint_dir.rglob("checkpoint_LCC-MAPPO.pt"):
        checkpoint = p
        break
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}; run phase main first")

    datasets = args.crossenv_datasets.split(",")
    distributions = args.crossenv_distributions.split(",")
    user_counts = parse_int_list(args.crossenv_users)
    delays = parse_int_list(args.crossenv_delays)
    total = len(datasets) * len(distributions) * len(user_counts) * len(delays)
    done = 0
    for dataset in datasets:
        for dist in distributions:
            for n in user_counts:
                for delay in delays:
                    out_dir = (
                        output_root / "crossenv" / MAIN_VARIANT.name
                        / dataset.strip() / dist.strip() / f"n{n}" / f"delay_{delay}"
                    )
                    summary = find_summary(out_dir)
                    if summary and summary.exists() and not args.force:
                        print(f"[skip] crossenv {dataset}/{dist}/N{n}/d{delay}", flush=True)
                    else:
                        extra = {
                            "dataset": dataset.strip(),
                            "distribution": dist.strip(),
                            "users": n,
                            "train-steps": 0,
                            "test-steps": 1_000,
                            "load-checkpoint": str(checkpoint),
                        }
                        cmd = build_command(MAIN_VARIANT, 1, delay, out_dir, args.device, extra_overrides=extra)
                        print(f"[run] crossenv {dataset}/{dist}/N{n}/d{delay}", flush=True)
                        subprocess.run(cmd, cwd=str(ROOT), check=True)
                    done += 1
                    print(f"[done] crossenv {done}/{total}", flush=True)


def parse_int_list(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--phase",
        choices=["main", "ablation", "n40", "crossenv", "all"],
        default="all",
    )
    p.add_argument("--formal-seeds", default="1,2,3,4,5")
    p.add_argument("--formal-delays", default="0,1,2,4,8")
    p.add_argument(
        "--ablation-variants",
        default="",
        help="comma-separated ablation names; empty runs every ablation",
    )
    p.add_argument("--n40-seeds", default="1,2,3,4,5")
    p.add_argument("--n40-distributions", default="gaussian,uniform")
    p.add_argument("--crossenv-datasets", default="KAIST,NCSU,NewYork")
    p.add_argument("--crossenv-distributions", default="uniform,gaussian")
    p.add_argument("--crossenv-users", default="15,20,25,30")
    p.add_argument("--crossenv-delays", default="0,8")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"main", "all"}:
        print(f"=== Phase 1: main 5 seeds x 5 delays = 25 cells (no stale margin) ===", flush=True)
        phase_main(args, args.output_dir)
    if args.phase in {"ablation", "all"}:
        print(f"=== Phase 2: ablation 3 variants x 5 seeds x 5 delays = 75 cells ===", flush=True)
        phase_ablation(args, args.output_dir)
    if args.phase in {"n40", "all"}:
        print(f"=== Phase 3: N40 stress 2 dist x 5 seeds = 10 cells ===", flush=True)
        phase_n40(args, args.output_dir)
    if args.phase in {"crossenv", "all"}:
        print(f"=== Phase 4: cross-env fixed-checkpoint test = 48 cells ===", flush=True)
        phase_crossenv(args, args.output_dir)
    print(f"\nAll outputs in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
