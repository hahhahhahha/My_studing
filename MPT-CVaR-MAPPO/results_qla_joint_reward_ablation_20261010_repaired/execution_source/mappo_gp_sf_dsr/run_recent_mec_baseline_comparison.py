"""Run the three recent MEC baselines without touching prior result trees."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .recent_mec_baselines import (
    LYAPUNOV,
    LYAPUNOV_FRESH,
    LYAPUNOV_LOCAL_STALE,
    LYAPUNOV_STALE_TUNED,
    LYAPUNOV_TAIL,
    MAB,
    MAB_TAIL,
    QECO,
    QECO_LOAD_TOPK,
    SAFETAIL,
)
from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
DELAYS = [0, 1, 2, 4, 8]
SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class RecentMethod:
    name: str
    extra_args: dict[str, object] = field(default_factory=dict)


METHODS = [
    # Validation seed 0, delay 0 selected 1e-3 lexicographically by CVaR99
    # then average latency from {3e-4, 1e-3, 3e-3}. Architecture is unchanged.
    RecentMethod(QECO, {"learning-rate": 0.001}),
    RecentMethod(
        QECO_LOAD_TOPK,
        {
            "learning-rate": 0.001,
            "qeco-load-mask": True,
            "qeco-topk": 4,
            "qeco-topk-temperature": 0.7,
            "qeco-mask-quantile": 0.70,
        },
    ),
    RecentMethod(LYAPUNOV),
    RecentMethod(LYAPUNOV_LOCAL_STALE),
    RecentMethod(LYAPUNOV_FRESH),
    RecentMethod(
        LYAPUNOV_STALE_TUNED,
        {"lyapunov-tuned-v": 1.0, "lyapunov-tuned-temperature": 2.5},
    ),
    RecentMethod(
        LYAPUNOV_TAIL,
        {"lyapunov-v": 1.0, "lyapunov-temperature": 2.5, "lyapunov-tail-weight": 1.0},
    ),
    RecentMethod(MAB),
    RecentMethod(
        MAB_TAIL,
        {"mab-tail-weight": 1.0, "mab-stale-weight": 1.0, "mab-capacity-weight": 0.2},
    ),
    RecentMethod(SAFETAIL),
]


def build_command(
    method: RecentMethod,
    seed: int,
    delay: int,
    output_dir: Path,
    device: str | None,
    train_steps: int,
    test_steps: int,
) -> list[str]:
    out_base = output_dir / method.name / f"seed_{seed}" / f"delay_{delay}"
    values: dict[str, object] = {
        **COMMON_ARGS,
        **method.extra_args,
        "method": method.name,
        "seed": seed,
        "feedback-delay": delay,
        "train-steps": train_steps,
        "test-steps": test_steps,
        "output-dir": str(out_base),
    }
    if device:
        values["device"] = device
    command = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.recent_mec_baselines"]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def summary_path(root: Path, method: str, seed: int, delay: int) -> Path:
    return (
        root
        / method
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{method}.csv"
    )


def run_job(
    method: RecentMethod,
    seed: int,
    delay: int,
    root: Path,
    device: str | None,
    train_steps: int,
    test_steps: int,
    force: bool,
) -> str:
    summary = summary_path(root, method.name, seed, delay)
    if summary.exists() and not force:
        return f"[skip] {method.name} seed={seed} delay={delay}"
    log_dir = root / method.name / f"seed_{seed}" / f"delay_{delay}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        method,
        seed,
        delay,
        root,
        device,
        train_steps,
        test_steps,
    )
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    return (
        f"[done] {method.name} seed={seed} delay={delay} "
        f"elapsed_min={(time.perf_counter() - started) / 60.0:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recent MEC baseline adapters.")
    parser.add_argument("--output-dir", default="results_recent_mec_baselines_multiseed")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--delays", type=int, nargs="+", default=DELAYS)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = METHODS
    if args.methods:
        unknown = set(args.methods) - {method.name for method in METHODS}
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        selected = [method for method in METHODS if method.name in set(args.methods)]

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    total = len(selected) * len(args.seeds) * len(args.delays)
    print(f"Running {total} isolated recent-baseline experiments under {root.resolve()}")
    for method in selected:
        jobs = [(seed, delay) for seed in args.seeds for delay in args.delays]
        print(f"[method] {method.name}: {len(jobs)} jobs, workers={args.workers}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    run_job,
                    method,
                    seed,
                    delay,
                    root,
                    args.device,
                    args.train_steps,
                    args.test_steps,
                    args.force,
                ): (seed, delay)
                for seed, delay in jobs
            }
            for future in as_completed(futures):
                print(future.result(), flush=True)


if __name__ == "__main__":
    main()
