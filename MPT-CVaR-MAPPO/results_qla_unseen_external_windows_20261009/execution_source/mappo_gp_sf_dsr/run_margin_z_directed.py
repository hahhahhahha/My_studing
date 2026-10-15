"""Directed experiments for bug-fixed stale margin and dormant-Z activation.

Two scenarios:
  A) high-stale: delay in {0, 8}, cvar_budget=1.30 (default), variants: main, w/o stale margin
  B) tight-budget: delay=4, cvar_budget=1.05, variants: main, w/o tail state Z, Static reward

Small scale: 3 seeds × 1-2 delays × 2-3 variants per scenario.
Output: results_margin_z_directed/{A,B}/{raw,summary}.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_margin_z_directed"


@dataclass(frozen=True)
class DirectedVariant:
    name: str
    extra_args: dict[str, object]


def common_args() -> dict[str, object]:
    return {
        # Environment (must match run_lcc_mappo_study.common_args defaults)
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
        "target-portfolio-entropy": 0.95,
        "log-interval": 100,
        "train-steps": 80_000,
        "test-steps": 10_000,
        # LCC base
        "lcc-cvar-alpha": 0.99,
        "lcc-concentration-factor": 2.0,
        "lcc-stale-margin": 1.0,
        # Algorithm (price-edge QT bias P0p75)
        "lcc-price-observation": True,
        "lcc-price-scale-observation": True,
        "lcc-edgewise-price-actor": True,
        "lcc-price-components": "QT",
        "lcc-price-logit-bias": 0.75,
        "lcc-v": 0.5,
        "lcc-migration-weight": 0.05,
    }


SCENARIO_A_VARIANTS = [
    DirectedVariant("Main", {}),
    DirectedVariant("NoStaleMargin-BugFixed", {"lcc-no-stale-margin": True}),
]

SCENARIO_B_VARIANTS = [
    DirectedVariant("Main-TightBudget", {}),
    DirectedVariant("NoZ-TightBudget", {"lcc-no-z-queue": True}),
    DirectedVariant("StaticReward-TightBudget", {
        "lcc-static-reward": True,
        "lcc-no-z-queue": True,
        "lcc-no-h-queue": True,
    }),
]


def build_command(scenario: str, variant: DirectedVariant, seed: int, delay: int,
                  cvar_budget: float, output_root: Path, device: str | None) -> list[str]:
    base = common_args()
    base.update(variant.extra_args)
    base["lcc-cvar-budget"] = cvar_budget
    base["seed"] = seed
    base["feedback-delay"] = delay
    out_dir = output_root / scenario / variant.name / f"seed_{seed}" / f"delay_{delay}"
    base["output-dir"] = str(out_dir)
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


def run_scenario(scenario: str, variants: list[DirectedVariant], seeds: list[int],
                 delays: list[int], cvar_budget: float, output_root: Path,
                 device: str | None, force: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(variants) * len(seeds) * len(delays)
    done = 0
    for variant in variants:
        for seed in seeds:
            for delay in delays:
                out_dir = output_root / scenario / variant.name / f"seed_{seed}" / f"delay_{delay}"
                summary = find_summary(out_dir)
                if summary and summary.exists() and not force:
                    print(f"[skip] {scenario} {variant.name} seed={seed} delay={delay}", flush=True)
                else:
                    cmd = build_command(scenario, variant, seed, delay, cvar_budget, output_root, device)
                    print(f"[run] {scenario} {variant.name} seed={seed} delay={delay}", flush=True)
                    subprocess.run(cmd, cwd=str(ROOT), check=True)
                    summary = find_summary(out_dir)
                if not summary or not summary.exists():
                    raise RuntimeError(f"Missing summary for {variant.name} seed={seed} delay={delay}")
                with summary.open("r", newline="", encoding="utf-8") as fh:
                    rec = next(csv.DictReader(fh))
                rec.update({
                    "scenario": scenario, "variant": variant.name,
                    "seed": seed, "delay": delay,
                    "summary_path": str(summary),
                })
                rows.append(rec)
                done += 1
                print(f"[done] {done}/{total}", flush=True)
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--delays-A", type=str, default="0,8")
    p.add_argument("--delays-B", type=str, default="4")
    p.add_argument("--budget-A", type=float, default=1.30)
    p.add_argument("--budget-B", type=float, default=1.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    delays_A = [int(d) for d in args.delays_A.split(",") if d.strip()]
    delays_B = [int(d) for d in args.delays_B.split(",") if d.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    if args.scenario in {"A", "both"}:
        print(f"=== Scenario A: high-stale, cvar_budget={args.budget_A} ===", flush=True)
        rows_A = run_scenario("A", SCENARIO_A_VARIANTS, seeds, delays_A,
                              args.budget_A, args.output_dir, args.device, args.force)
        write_rows(args.output_dir / "A_raw.csv", rows_A)
        all_rows.extend(rows_A)
    if args.scenario in {"B", "both"}:
        print(f"=== Scenario B: tight-budget, cvar_budget={args.budget_B} ===", flush=True)
        rows_B = run_scenario("B", SCENARIO_B_VARIANTS, seeds, delays_B,
                              args.budget_B, args.output_dir, args.device, args.force)
        write_rows(args.output_dir / "B_raw.csv", rows_B)
        all_rows.extend(rows_B)
    write_rows(args.output_dir / "all_raw.csv", all_rows)
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
