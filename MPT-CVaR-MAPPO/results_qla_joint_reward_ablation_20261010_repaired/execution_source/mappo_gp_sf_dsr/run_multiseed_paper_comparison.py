from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DELAYS = [0, 1, 2, 4, 8]
SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for one paper baseline / method."""

    display_name: str
    module: str
    method_arg: str | None
    summary_name: str
    extra_args: dict[str, object] = field(default_factory=dict)


# Common environment / training arguments used for all methods.
COMMON_ARGS = {
    "dataset": "KAIST",
    "users": 30,
    "distribution": "poisson_bursty",
    "train-steps": 10_000,
    "test-steps": 1_000,
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
    "score-resource-weight": 1.0,
    "score-capacity-weight": 0.5,
    "score-task-count-weight": 0.5,
    "score-wait-weight": 1.0,
    "log-interval": 1000,
}


METHODS: list[MethodConfig] = [
    # V4-AgeGate with the final coordinate-tuned hyperparameters.
    MethodConfig(
        display_name="V4-AgeGate",
        module="mappo_gp_sf_dsr.train",
        method_arg="DSR-MTG-Qpen-v4-agegate",
        summary_name="DSR-MTG-Qpen-v4-agegate",
        extra_args={
            "alpha": 0.30,
            "q-penalty-weight": 0.08,
            "kappa-i": 0.02,
            "kappa-g": 0.02,
            "target-portfolio-entropy": 0.99,
            "feedback-gate-eta": 0.10,
            "max-supported-users": 50,
        },
    ),
    # M2PPO: MAPPO with a 5-critic ensemble.
    MethodConfig(
        display_name="M2PPO",
        module="mappo_gp_sf_dsr.train",
        method_arg="M2PPO",
        summary_name="M2PPO",
        extra_args={
            "alpha": 0.30,
            "q-penalty-weight": 0.05,
            "kappa-i": 0.01,
            "kappa-g": 0.01,
            "target-portfolio-entropy": 0.95,
            "feedback-gate-eta": 0.25,
            "critic-ensemble-count": 5,
            "auto-portfolio-entropy": False,
            "twin-critic": False,
            "max-supported-users": 50,
        },
    ),
    # MCM-FDRL: federated DQN-style discrete adjustment.
    MethodConfig(
        display_name="MCM-FDRL",
        module="mappo_gp_sf_dsr.mcm_fdrl",
        method_arg=None,
        summary_name="MCM-FDRL",
        extra_args={
            "max-supported-users": 80,
            "hidden": 128,
            "replay-size": 100_000,
            "batch-size": 32,
            "learning-rate": 0.0005,
            "gamma": 0.99,
            "fed-interval": 250,
            "target-update-interval": 250,
            "adjust-delta": 0.1,
            "reward-mode": "improvement",
            "latency-norm": 3.0,
            "epsilon-start": 1.0,
            "epsilon-end": 0.05,
            "epsilon-decay": 0.9995,
        },
    ),
    # Towards-PPO-DSR: PPO + DSR reward from the risk-averse edge paper.
    MethodConfig(
        display_name="Towards-PPO-DSR",
        module="mappo_gp_sf_dsr.paper_policy_baselines",
        method_arg="Towards-PPO-DSR",
        summary_name="Towards-PPO-DSR",
        extra_args={
            "max-supported-users": 50,
            "buffer-size": 20_000,
            "rollout-size": 1024,
            "batch-size": 128,
            "learning-rate": 0.001,
            "gamma": 0.99,
            "gae-lambda": 0.95,
            "clip-eps": 0.2,
            "update-epochs": 4,
            "entropy-coef": 0.005,
            "latency-norm": 3.0,
            "dsr-mu": 1.5,
            "dsr-eta": 0.01,
            "dsr-temperature": 5.0,
            "log-std-min": -1.5,
            "fed-interval": 250,
            "update-every": 4,
            "tau": 0.01,
            "noise-start": 0.08,
            "noise-final": 0.01,
        },
    ),
    # Towards-PPO-MV: PPO + MV reward from the risk-averse edge paper.
    MethodConfig(
        display_name="Towards-PPO-MV",
        module="mappo_gp_sf_dsr.paper_policy_baselines",
        method_arg="Towards-PPO-MV",
        summary_name="Towards-PPO-MV",
        extra_args={
            "max-supported-users": 50,
            "buffer-size": 20_000,
            "rollout-size": 1024,
            "batch-size": 128,
            "learning-rate": 0.001,
            "gamma": 0.99,
            "gae-lambda": 0.95,
            "clip-eps": 0.2,
            "update-epochs": 4,
            "entropy-coef": 0.005,
            "latency-norm": 3.0,
            "dsr-mu": 1.5,
            "dsr-eta": 0.01,
            "dsr-temperature": 5.0,
            "log-std-min": -1.5,
            "fed-interval": 250,
            "update-every": 4,
            "tau": 0.01,
            "noise-start": 0.08,
            "noise-final": 0.01,
        },
    ),
    # Fed-MADDPG: federated actor aggregation + centralized critics.
    MethodConfig(
        display_name="Fed-MADDPG",
        module="mappo_gp_sf_dsr.paper_policy_baselines",
        method_arg="Fed-MADDPG",
        summary_name="Fed-MADDPG",
        extra_args={
            "max-supported-users": 50,
            "buffer-size": 20_000,
            "rollout-size": 1024,
            "batch-size": 128,
            "learning-rate": 0.001,
            "gamma": 0.99,
            "gae-lambda": 0.95,
            "clip-eps": 0.2,
            "update-epochs": 4,
            "entropy-coef": 0.005,
            "latency-norm": 3.0,
            "dsr-mu": 1.5,
            "dsr-eta": 0.01,
            "dsr-temperature": 5.0,
            "log-std-min": -1.5,
            "fed-interval": 250,
            "update-every": 4,
            "tau": 0.01,
            "noise-start": 0.08,
            "noise-final": 0.01,
        },
    ),
]


def build_command(
    method: MethodConfig,
    seed: int,
    delay: int,
    output_dir: Path,
    device: str | None,
) -> list[str]:
    """Construct the CLI command for one run."""
    out_base = output_dir / method.display_name / f"seed_{seed}" / f"delay_{delay}"
    args_dict: dict[str, object] = {
        **COMMON_ARGS,
        **method.extra_args,
        "seed": seed,
        "feedback-delay": delay,
        "output-dir": str(out_base),
    }
    if device is not None:
        args_dict["device"] = device

    cmd = [sys.executable, "-B", "-m", method.module]
    if method.method_arg is not None:
        cmd.extend(["--method", method.method_arg])

    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def summary_path(method: MethodConfig, output_dir: Path, seed: int, delay: int) -> Path:
    """Location of the summary CSV produced by a run."""
    out_base = output_dir / method.display_name / f"seed_{seed}" / f"delay_{delay}"
    # paper_policy_baselines.py creates an extra {method}_delay_{delay} subdir
    # inside the provided output directory; account for that here.
    if method.module == "mappo_gp_sf_dsr.paper_policy_baselines":
        out_base = out_base / f"{method.summary_name}_delay_{delay}"
    return (
        out_base
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{method.summary_name}.csv"
    )


def run_one(
    method: MethodConfig,
    seed: int,
    delay: int,
    output_dir: Path,
    device: str | None,
    force: bool,
) -> None:
    summary = summary_path(method, output_dir, seed, delay)
    if summary.exists() and not force:
        print(
            f"[skip] method={method.display_name} seed={seed} delay={delay} summary={summary}",
            flush=True,
        )
        return

    cmd = build_command(method, seed, delay, output_dir, device)
    log_dir = output_dir / method.display_name / f"seed_{seed}" / f"delay_{delay}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    print(
        f"[run] method={method.display_name} seed={seed} delay={delay} device={device or 'auto'} log={log_path}",
        flush=True,
    )
    started = time.perf_counter()
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    print(
        f"[done] method={method.display_name} seed={seed} delay={delay} "
        f"elapsed_min={elapsed / 60.0:.2f} summary={summary}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 6 methods x 5 delays x 5 seeds for the paper comparison."
    )
    parser.add_argument(
        "--output-dir",
        default="results_paper_algorithms_v4scenario_multiseed",
        help="Root directory for all multi-seed results.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Training seeds to use. Eval seed is always train_seed + 10000.",
    )
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=DELAYS,
        help="Feedback delays to scan.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of methods to run (display names). Default: all.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda or cpu. Default: auto.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if summary CSV already exists.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = METHODS
    if args.methods is not None:
        name_set = {m.display_name for m in METHODS}
        for name in args.methods:
            if name not in name_set:
                raise ValueError(f"Unknown method: {name}. Choose from {sorted(name_set)}")
        methods = [m for m in METHODS if m.display_name in args.methods]

    total = len(methods) * len(args.seeds) * len(args.delays)
    print(f"Running {total} experiments: {len(methods)} methods x {len(args.seeds)} seeds x {len(args.delays)} delays")
    print(f"Output root: {output_dir.resolve()}")
    print(f"Eval seed = train seed + 10000 for every run")
    print("-" * 60)

    for method in methods:
        for seed in args.seeds:
            for delay in args.delays:
                run_one(method, seed, delay, output_dir, args.device, args.force)

    print("-" * 60)
    print(f"All runs finished. Next step: python -m mappo_gp_sf_dsr.summarize_multiseed_paper_comparison --root {output_dir}")


if __name__ == "__main__":
    main()
