"""Run delayed-information baselines on the common seed/delay grid.

The runner is resumable: a cell is skipped only when its summary and config
exist and match the requested seed/delay.  Use the three-delay one-seed grid as
the mandatory prescreen before launching the formal 5 x 5 grid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT


def parse_csv_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def complete(output: Path, method: str, seed: int, delay: int) -> bool:
    leaf = output / "KAIST" / "30u" / "poisson_bursty"
    summary = leaf / f"summary_{method}.csv"
    config = leaf / f"config_{method}.json"
    if not summary.exists() or not config.exists():
        return False
    try:
        frame = pd.read_csv(summary)
        payload = json.loads(config.read_text(encoding="utf-8"))
        cli = payload["cli"]
        return (
            not frame.empty
            and int(cli["seed"]) == seed
            and int(cli["feedback_delay"]) == delay
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "results_delayed_information_baselines_prescreen"))
    parser.add_argument("--methods", default="JSEQ-A,MFC-DI-A")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path(args.output_dir)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    seeds = parse_csv_ints(args.seeds)
    delays = parse_csv_ints(args.delays)
    total = len(methods) * len(seeds) * len(delays)
    index = 0
    for method in methods:
        for seed in seeds:
            for delay in delays:
                index += 1
                output = base / method / f"seed_{seed}" / f"delay_{delay}"
                if complete(output, method, seed, delay):
                    print(f"[{index}/{total}] skip complete {method} seed={seed} delay={delay}", flush=True)
                    continue
                command = [
                    sys.executable, "-B", "-m", "mappo_gp_sf_dsr.delayed_information_baselines",
                    "--method", method,
                    "--output-dir", str(output),
                    "--seed", str(seed),
                    "--feedback-delay", str(delay),
                    "--train-steps", str(args.train_steps if method == "MFC-DI-A" else 0),
                    "--test-steps", str(args.test_steps),
                    "--device", args.device if method == "MFC-DI-A" else "cpu",
                ]
                print(f"[{index}/{total}] run {method} seed={seed} delay={delay}", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
