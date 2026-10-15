from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_convergence_stability_ablation"


@dataclass(frozen=True)
class Variant:
    name: str
    module: str
    method: str | None
    summary_name: str
    extra_args: dict[str, object] = field(default_factory=dict)


# These variants preserve each paper's reward and algorithm family.  They
# change only paper-compatible optimization settings or the mapping of the
# paper's fixed adjustment amount into this environment.  They are diagnostic
# adaptations, not substitutes for a full original-environment reproduction.
VARIANTS = [
    Variant(
        name="M2PPO-original-lr1e-3",
        module="mappo_gp_sf_dsr.train",
        method="M2PPO",
        summary_name="M2PPO",
        extra_args={
            "learning-rate": 1e-3,
            "critic-ensemble-count": 5,
            "auto-portfolio-entropy": False,
            "twin-critic": False,
            "max-supported-users": 50,
        },
    ),
    Variant(
        name="M2PPO-paper-clip0.1-lr1e-3",
        module="mappo_gp_sf_dsr.train_m2ppo_clip01",
        method="M2PPO",
        summary_name="M2PPO",
        extra_args={
            "learning-rate": 1e-3,
            "critic-ensemble-count": 5,
            "auto-portfolio-entropy": False,
            "twin-critic": False,
            "max-supported-users": 50,
        },
    ),
    Variant(
        name="M2PPO-paper-clip0.1-lr3e-4",
        module="mappo_gp_sf_dsr.train_m2ppo_clip01",
        method="M2PPO",
        summary_name="M2PPO",
        extra_args={
            "learning-rate": 3e-4,
            "critic-ensemble-count": 5,
            "auto-portfolio-entropy": False,
            "twin-critic": False,
            "max-supported-users": 50,
        },
    ),
    Variant(
        name="MCM-original-improvement-delta0.1",
        module="mappo_gp_sf_dsr.mcm_fdrl",
        method=None,
        summary_name="MCM-FDRL",
        extra_args={
            "max-supported-users": 80,
            "hidden": 128,
            "replay-size": 100_000,
            "batch-size": 32,
            "learning-rate": 5e-4,
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
    Variant(
        name="MCM-literal-volume-number-delta0.028",
        module="mappo_gp_sf_dsr.mcm_fdrl",
        method=None,
        summary_name="MCM-FDRL",
        extra_args={
            "max-supported-users": 80,
            "hidden": 128,
            "replay-size": 100_000,
            "batch-size": 32,
            "learning-rate": 5e-4,
            "gamma": 0.99,
            "fed-interval": 250,
            "target-update-interval": 250,
            "adjust-delta": 0.028,
            "reward-mode": "improvement",
            "latency-norm": 3.0,
            "epsilon-start": 1.0,
            "epsilon-end": 0.05,
            "epsilon-decay": 0.9995,
        },
    ),
]


def common_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset": "KAIST",
        "users": 30,
        "distribution": "poisson_bursty",
        "train-steps": args.train_steps,
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
        "score-resource-weight": 1.0,
        "score-capacity-weight": 0.5,
        "score-task-count-weight": 0.5,
        "score-wait-weight": 1.0,
        "log-interval": args.log_interval,
    }


def build_command(
    variant: Variant,
    args: argparse.Namespace,
    seed: int,
    delay: int,
    output_root: Path,
) -> list[str]:
    out_base = output_root / variant.name / f"seed_{seed}" / f"delay_{delay}"
    values = {
        **common_args(args),
        **variant.extra_args,
        "seed": seed,
        "feedback-delay": delay,
        "output-dir": str(out_base),
    }
    if args.device:
        values["device"] = args.device

    command = [sys.executable, "-B", "-m", variant.module]
    if variant.method:
        command.extend(["--method", variant.method])
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def summary_path(variant: Variant, output_root: Path, seed: int, delay: int) -> Path:
    return (
        output_root
        / variant.name
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{variant.summary_name}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated one-factor convergence stability ablations."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--delays", type=int, nargs="+", default=[8])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--train-steps", type=int, default=3_000)
    parser.add_argument("--test-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    protected = (ROOT / "results_paper_ready_outputs").resolve()
    if output_root == protected or protected in output_root.parents:
        raise ValueError("Refusing to write stability ablations into paper-ready results.")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = VARIANTS
    if args.variants:
        requested = set(args.variants)
        known = {variant.name for variant in VARIANTS}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}; choose from {sorted(known)}")
        selected = [variant for variant in VARIANTS if variant.name in requested]

    for variant in selected:
        for seed in args.seeds:
            for delay in args.delays:
                summary = summary_path(variant, output_root, seed, delay)
                log_dir = output_root / variant.name / f"seed_{seed}" / f"delay_{delay}" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                command = build_command(variant, args, seed, delay, output_root)
                manifest = {
                    "purpose": "diagnostic adaptation; not a replacement for full paper reproduction",
                    "variant": variant.name,
                    "module": variant.module,
                    "method": variant.method,
                    "seed": seed,
                    "feedback_delay": delay,
                    "extra_args": variant.extra_args,
                    "common_args": common_args(args),
                    "command": command,
                }
                with (log_dir / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=False)
                if summary.exists() and not args.force:
                    print(f"[skip] {variant.name} seed={seed} delay={delay}", flush=True)
                    continue
                print(f"[run] {variant.name} seed={seed} delay={delay}", flush=True)
                started = time.perf_counter()
                with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        check=True,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                elapsed = time.perf_counter() - started
                print(f"[done] {variant.name} elapsed_s={elapsed:.1f}", flush=True)

    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
