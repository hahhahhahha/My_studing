from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_facr_ablation"


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    summary_name: str
    extra_args: dict[str, object] = field(default_factory=dict)


FACR_BASE_ARGS: dict[str, object] = {
    "alpha": 0.20,
    "q-penalty-weight": 0.08,
    "kappa-i": 0.02,
    "kappa-g": 0.02,
    "target-portfolio-entropy": 0.99,
    "feedback-gate-eta": 0.25,
    "lyapunov-prior-weight": 0.20,
    "lyapunov-prior-v": 1.0,
    "lyapunov-prior-temperature": 2.5,
    "lyapunov-prior-smoothing": 0.80,
    "lyapunov-prior-max-step": 0.03,
    "max-supported-users": 50,
}


VARIANTS = [
    Variant(
        name="FACR-MAPPO",
        method="DSR-MTG-Qpen-v4-agegate-lyapprior",
        summary_name="DSR-MTG-Qpen-v4-agegate-lyapprior",
    ),
    Variant(
        name="w-o-LyapPrior",
        method="DSR-MTG-Qpen-v4-agegate",
        summary_name="DSR-MTG-Qpen-v4-agegate",
        extra_args={"lyapunov-prior-weight": 0.0},
    ),
    Variant(
        name="w-o-AgeGate",
        method="DSR-MTG-Qpen-v4",
        summary_name="DSR-MTG-Qpen-v4",
    ),
    Variant(
        name="w-o-AgeGate-LyapPrior",
        method="DSR-MTG-Qpen-v4",
        summary_name="DSR-MTG-Qpen-v4",
        extra_args={"lyapunov-prior-weight": 0.0},
    ),
    Variant(
        name="w-o-QPenalty",
        method="DSR-MTG-Qpen-v4-agegate-noQpen",
        summary_name="DSR-MTG-Qpen-v4-agegate-noQpen",
        extra_args={"q-penalty-weight": 0.0},
    ),
    Variant(
        name="w-o-SafeProjection",
        method="DSR-MTG-Qpen-v4-agegate-lyapprior",
        summary_name="DSR-MTG-Qpen-v4-agegate-lyapprior",
        extra_args={"lyapunov-prior-max-step": 0.0},
    ),
    Variant(
        name="w-o-ConservativeFusion",
        method="DSR-MTG-Qpen-v4-agegate-lyapprior",
        summary_name="DSR-MTG-Qpen-v4-agegate-lyapprior",
        extra_args={"lyapunov-prior-smoothing": 0.0, "lyapunov-prior-max-step": 0.0},
    ),
    Variant(
        name="w-o-Concentration",
        method="DSR-MTG-Qpen-v4-agegate-noConc",
        summary_name="DSR-MTG-Qpen-v4-agegate-noConc",
        extra_args={"kappa-i": 0.0, "kappa-g": 0.0},
    ),
    Variant(
        name="w-o-AuxRegularizers",
        method="DSR-MTG-Qpen-v4-agegate-noConc",
        summary_name="DSR-MTG-Qpen-v4-agegate-noConc",
        extra_args={"q-penalty-weight": 0.0, "kappa-i": 0.0, "kappa-g": 0.0},
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


def merged_args(variant: Variant, args: argparse.Namespace, seed: int, delay: int, output_root: Path) -> dict[str, object]:
    return {
        **common_args(args),
        **FACR_BASE_ARGS,
        **variant.extra_args,
        "seed": seed,
        "feedback-delay": delay,
        "output-dir": str(output_root / variant.name / f"seed_{seed}" / f"delay_{delay}"),
    }


def build_command(variant: Variant, args: argparse.Namespace, seed: int, delay: int, output_root: Path) -> list[str]:
    values = merged_args(variant, args, seed, delay, output_root)
    if args.device:
        values["device"] = args.device
    command = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train", "--method", variant.method]
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
    parser = argparse.ArgumentParser(description="FACR-MAPPO module ablations.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    protected = (ROOT / "results_paper_ready_outputs").resolve()
    if output_root == protected or protected in output_root.parents:
        raise ValueError("Refusing to write FACR ablations into paper-ready results.")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = VARIANTS
    if args.variants:
        requested = set(args.variants)
        known = {variant.name for variant in VARIANTS}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}; choose from {sorted(known)}")
        selected = [variant for variant in VARIANTS if variant.name in requested]

    total = len(selected) * len(args.seeds) * len(args.delays)
    done = 0
    print(f"Running {total} FACR ablation jobs under {output_root}", flush=True)
    for variant in selected:
        for seed in args.seeds:
            for delay in args.delays:
                summary = summary_path(variant, output_root, seed, delay)
                log_dir = output_root / variant.name / f"seed_{seed}" / f"delay_{delay}" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                command = build_command(variant, args, seed, delay, output_root)
                manifest = {
                    "purpose": "FACR-MAPPO module ablation",
                    "variant": variant.name,
                    "method": variant.method,
                    "seed": seed,
                    "feedback_delay": delay,
                    "base_args": FACR_BASE_ARGS,
                    "variant_extra_args": variant.extra_args,
                    "common_args": common_args(args),
                    "command": command,
                }
                with (log_dir / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=False)
                if summary.exists() and not args.force:
                    print(f"[skip] {variant.name} seed={seed} delay={delay}", flush=True)
                    done += 1
                    continue
                print(f"[run] {variant.name} seed={seed} delay={delay}", flush=True)
                started = time.perf_counter()
                with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
                    subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - started
                done += 1
                print(f"[done] {done}/{total} {variant.name} seed={seed} delay={delay} elapsed_min={elapsed/60:.2f}", flush=True)
    print(f"Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
