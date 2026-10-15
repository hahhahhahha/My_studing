from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results_priceedge_monoqt_formal_5x5" / "formal" / "PriceEdgeBias-QT-P0p75-Final"
OUTPUT_ROOT = ROOT / "results_priceedge_monoqt_testtime_interventions"
INTERVENTIONS = ("aligned", "shuffle", "zero", "constant")
SEEDS = (1, 2, 3, 4, 5)
DELAYS = (0, 1, 2, 4, 8)


def summary_path(out_dir: Path) -> Path:
    return out_dir / "KAIST" / "30u" / "poisson_bursty" / "summary_LCC-MAPPO.csv"


def checkpoint_path(seed: int, delay: int) -> Path:
    return (
        SOURCE_ROOT
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "checkpoint_LCC-MAPPO.pt"
    )


def build_command(out_dir: Path, checkpoint: Path, seed: int, delay: int, intervention: str) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "mappo_gp_sf_dsr.train",
        "--method",
        "LCC-MAPPO",
        "--output-dir",
        str(out_dir),
        "--dataset",
        "KAIST",
        "--users",
        "30",
        "--distribution",
        "poisson_bursty",
        "--feedback-delay",
        str(delay),
        "--train-steps",
        "0",
        "--test-steps",
        "1000",
        "--seed",
        str(seed),
        "--slot-s",
        "1.0",
        "--task-compute-mb",
        "15.0",
        "--edge-compute-mb-s",
        "63.0",
        "--queue-upper-mb",
        "630.0",
        "--max-supported-users",
        "50",
        "--max-requests-per-user",
        "7",
        "--poisson-lambda",
        "3.5",
        "--burst-prob",
        "0.1",
        "--burst-multiplier",
        "2.0",
        "--broadcast-period",
        "2",
        "--auto-portfolio-entropy",
        "--target-portfolio-entropy",
        "0.99",
        "--lcc-v",
        "0.5",
        "--lcc-migration-weight",
        "0.05",
        "--lcc-cvar-budget",
        "1.3",
        "--lcc-concentration-factor",
        "2.0",
        "--lcc-stale-margin",
        "1.0",
        "--lcc-price-observation",
        "--lcc-price-scale-observation",
        "--lcc-edgewise-price-actor",
        "--lcc-price-components",
        "QT",
        "--lcc-price-logit-bias",
        "0.75",
        "--lcc-price-intervention",
        intervention,
        "--load-checkpoint",
        str(checkpoint),
        "--device",
        "cuda",
    ]


def read_summary(path: Path, intervention: str, seed: int, delay: int) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row.update({"intervention": intervention, "seed": seed, "delay": delay, "summary_path": str(path)})
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "average_latency",
        "cvar99",
        "queue_oscillation",
        "portfolio_migration_rate",
        "global_demand_concentration",
        "lcc_price_action_spearman",
    )
    out: list[dict[str, object]] = []
    for intervention in INTERVENTIONS:
        group = [row for row in rows if row["intervention"] == intervention]
        item: dict[str, object] = {"intervention": intervention, "runs": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = var**0.5
        out.append(item)
    return out


def main() -> None:
    rows: list[dict[str, object]] = []
    total = len(INTERVENTIONS) * len(SEEDS) * len(DELAYS)
    done = 0
    for intervention in INTERVENTIONS:
        for seed in SEEDS:
            for delay in DELAYS:
                ckpt = checkpoint_path(seed, delay)
                if not ckpt.exists():
                    raise FileNotFoundError(ckpt)
                out_dir = OUTPUT_ROOT / intervention / f"seed_{seed}" / f"delay_{delay}"
                summary = summary_path(out_dir)
                if not summary.exists():
                    print(f"[run] {intervention} seed={seed} delay={delay}", flush=True)
                    subprocess.run(build_command(out_dir, ckpt, seed, delay, intervention), cwd=str(ROOT), check=True)
                else:
                    print(f"[skip] {intervention} seed={seed} delay={delay}", flush=True)
                rows.append(read_summary(summary, intervention, seed, delay))
                done += 1
                print(f"[done] {done}/{total}", flush=True)
    write_rows(OUTPUT_ROOT / "testtime_interventions_raw.csv", rows)
    write_rows(OUTPUT_ROOT / "testtime_interventions_summary.csv", aggregate(rows))


if __name__ == "__main__":
    main()
