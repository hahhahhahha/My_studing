from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PaperBaseline:
    name: str
    module: str
    summary_name: str


BASELINES = [
    PaperBaseline("M2PPO-paper", "mappo_gp_sf_dsr.m2ppo_paper", "M2PPO-paper"),
    PaperBaseline("MCM-FDRL-paper", "mappo_gp_sf_dsr.mcm_fdrl_paper", "MCM-FDRL-paper"),
]


def common_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset": "KAIST",
        "users": 30,
        "distribution": "poisson_bursty",
        "test-steps": args.test_steps,
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
        "log-interval": args.log_interval,
    }


def command_for(
    baseline: PaperBaseline,
    args: argparse.Namespace,
    seed: int,
    delay: int,
    output_root: Path,
) -> list[str]:
    output = output_root / baseline.name / f"seed_{seed}" / f"delay_{delay}"
    values = {
        **common_arguments(args),
        "seed": seed,
        "feedback-delay": delay,
        "output-dir": str(output),
    }
    if baseline.name == "M2PPO-paper":
        values.update(
            {
                "train-steps": args.m2ppo_train_steps,
                "rollout-steps": args.m2ppo_rollout_steps,
                "actor-learning-rate": args.m2ppo_actor_lr,
                "critic-learning-rate": args.m2ppo_critic_lr,
            }
        )
    else:
        values.update(
            {
                "episodes": args.mcm_episodes,
                "steps-per-episode": 25,
                "fed-every-episodes": 10,
                "adjust-delta": 0.04,
                "tau": 0.03,
                "learning-rate": 5e-4,
                "batch-size": 32,
                "replay-size": args.mcm_replay_size,
                "local-compute-ratio": args.mcm_local_compute_ratio,
                "density-radius": args.mcm_density_radius,
            }
        )
    if args.device:
        values["device"] = args.device
    command = [sys.executable, "-B", "-m", baseline.module]
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])
    return command


def summary_path(
    baseline: PaperBaseline,
    output_root: Path,
    seed: int,
    delay: int,
) -> Path:
    return (
        output_root
        / baseline.name
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{baseline.summary_name}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-paper-structure baseline adaptations.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_paper_structure_baselines")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--m2ppo-train-steps", type=int, default=50_000)
    parser.add_argument("--m2ppo-rollout-steps", type=int, default=128)
    parser.add_argument("--m2ppo-actor-lr", type=float, default=3e-4)
    parser.add_argument("--m2ppo-critic-lr", type=float, default=3e-4)
    parser.add_argument("--mcm-episodes", type=int, default=7_200)
    parser.add_argument("--mcm-replay-size", type=int, default=100_000)
    parser.add_argument("--mcm-local-compute-ratio", type=float, default=0.25)
    parser.add_argument("--mcm-density-radius", type=float, default=1500.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    protected = (ROOT / "results_paper_ready_outputs").resolve()
    if output_root == protected or protected in output_root.parents:
        raise ValueError("Refusing to write source-paper runs into paper-ready results.")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = BASELINES
    if args.methods:
        requested = set(args.methods)
        known = {baseline.name for baseline in BASELINES}
        if requested - known:
            raise ValueError(f"Unknown methods: {sorted(requested - known)}")
        selected = [baseline for baseline in BASELINES if baseline.name in requested]

    total = len(selected) * len(args.seeds) * len(args.delays)
    print(f"Running {total} source-paper-structure experiments", flush=True)
    for baseline in selected:
        for seed in args.seeds:
            for delay in args.delays:
                summary = summary_path(baseline, output_root, seed, delay)
                if summary.exists() and not args.force:
                    print(f"[skip] {baseline.name} seed={seed} delay={delay}", flush=True)
                    continue
                log_dir = output_root / baseline.name / f"seed_{seed}" / f"delay_{delay}" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                command = command_for(baseline, args, seed, delay, output_root)
                print(f"[run] {baseline.name} seed={seed} delay={delay}", flush=True)
                started = time.perf_counter()
                with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        check=True,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                print(
                    f"[done] {baseline.name} seed={seed} delay={delay} "
                    f"elapsed_min={(time.perf_counter() - started) / 60.0:.2f}",
                    flush=True,
                )
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
