"""Canonical QLA-signal composition ablation.

The four variants isolate the queue-pressure and predicted-latency terms in
the action-aligned QL signal.  Training seed is the random replicate and
feedback age is an experimental condition; all variants share the canonical
schedule for a given seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .run_qla_simplified_rerun import SimplifiedVariant, build_command, find_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_qla_signal_components_v1"


@dataclass(frozen=True)
class SignalVariant:
    name: str
    components: str
    question: str


VARIANTS = (
    SignalVariant("QLA-Full-QL", "QT", "Are queue pressure and predicted latency jointly useful?"),
    SignalVariant("QLA-Q-only", "Q", "What is contributed by stale queue-pressure direction alone?"),
    SignalVariant("QLA-L-only", "T", "What is contributed by predicted latency direction alone?"),
    SignalVariant("QLA-No-QL", "NoPrice", "Does the structured QL signal add value beyond the same edge-wise actor?"),
)


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def summary_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def valid_existing(path: Path, *, components: str, seed: int, delay: int) -> bool:
    if not path.exists():
        return False
    try:
        row = summary_row(path)
        return (
            row.get("lcc_price_components", "").upper() == components.upper()
            and int(float(row.get("feedback_delay", -1))) == delay
            and row.get("evaluation_split_role") == "test"
            and bool(row.get("train_schedule_sha256"))
            and bool(row.get("test_schedule_sha256"))
            and row.get("finite_buffer_protocol", "").lower() in {"true", "1"}
            and float(row.get("episode_mass_conservation_error", "inf")) <= 1e-6
        )
    except (OSError, StopIteration, ValueError):
        return False


def write_design(root: Path, seeds: list[int], delays: list[int], train_steps: int, test_steps: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": "QLA signal composition ablation",
        "variants": [variant.__dict__ for variant in VARIANTS],
        "random_replicate": "training_seed",
        "experimental_condition": "feedback_age",
        "seeds": seeds,
        "feedback_ages": delays,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "environment": "KAIST/N=30/poisson_bursty/M=35",
        "schedule_blocking": "same canonical exogenous schedule within seed",
        "checkpoint_rule": "final requested training budget; no test-set early stopping",
        "primary_metrics": ["average_latency", "cvar99"],
        "mechanism_metrics": ["queue_oscillation", "portfolio_migration_rate"],
    }
    (root / "DESIGN.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.output_dir.resolve()
    seeds = parse_ints(args.seeds)
    delays = parse_ints(args.delays)
    write_design(root, seeds, delays, args.train_steps, args.test_steps)

    jobs = [(variant, seed, delay) for variant in VARIANTS for seed in seeds for delay in delays]
    completed = 0
    for variant, seed, delay in jobs:
        out = root / "cells" / variant.name / f"seed_{seed}" / f"delay_{delay}"
        existing = find_summary(out)
        if existing and not args.force and valid_existing(
            existing, components=variant.components, seed=seed, delay=delay
        ):
            completed += 1
            print(f"[skip] {completed}/{len(jobs)} {variant.name} seed={seed} age={delay}", flush=True)
            continue

        simple = SimplifiedVariant(
            variant.name,
            {
                "lcc-price-components": variant.components,
                "lcc-price-logit-bias": 0.5,
                "lcc-price-shuffle": False,
                "lcc-edgewise-price-actor": True,
                "lcc-flat-price-bias-actor": False,
            },
        )
        command = build_command(
            simple,
            seed,
            delay,
            out,
            args.device,
            extra_overrides={"train-steps": args.train_steps, "test-steps": args.test_steps},
        )
        log = root / "logs" / f"{variant.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        started = perf_counter()
        print(f"[run] {completed + 1}/{len(jobs)} {variant.name} seed={seed} age={delay}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n$ " + subprocess.list2cmdline(command) + "\n")
            handle.flush()
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            handle.write(f"[elapsed_s] {perf_counter() - started:.3f}\n")
        summary = find_summary(out)
        if summary is None or not valid_existing(
            summary, components=variant.components, seed=seed, delay=delay
        ):
            raise SystemExit(f"Invalid or missing result: {variant.name} seed={seed} age={delay}")
        completed += 1
        print(f"[done] {completed}/{len(jobs)}", flush=True)

    print(f"Completed {completed}/{len(jobs)} cells in {root}", flush=True)


if __name__ == "__main__":
    main()
