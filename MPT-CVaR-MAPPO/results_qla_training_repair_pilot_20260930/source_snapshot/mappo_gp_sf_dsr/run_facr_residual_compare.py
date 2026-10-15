from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_facr_residual_policy"


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
        extra_args={"lyapunov-prior-weight": 0.20},
    ),
    Variant(
        name="FACR-Residual",
        method="DSR-MTG-Qpen-v4-agegate-lyapresidual",
        summary_name="DSR-MTG-Qpen-v4-agegate-lyapresidual",
        extra_args={
            "lyapunov-prior-weight": 0.0,
        },
    ),
    Variant(
        name="FACR-Residual-v2",
        method="DSR-MTG-Qpen-v4-lyapresidual-v2",
        summary_name="DSR-MTG-Qpen-v4-lyapresidual-v2",
        extra_args={
            "lyapunov-prior-weight": 0.0,
        },
    ),
    Variant(
        name="FACR-Residual-v3",
        method="DSR-MTG-Qpen-v4-lyapresidual-v3",
        summary_name="DSR-MTG-Qpen-v4-lyapresidual-v3",
        extra_args={
            "lyapunov-prior-weight": 0.0,
        },
    ),
    Variant(
        name="FACR-Residual-v3.1",
        method="DSR-MTG-Qpen-v4-lyapresidual-v3p1",
        summary_name="DSR-MTG-Qpen-v4-lyapresidual-v3p1",
        extra_args={
            "lyapunov-prior-weight": 0.0,
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


def build_command(variant: Variant, args: argparse.Namespace, seed: int, delay: int, output_root: Path) -> list[str]:
    values = {
        **common_args(args),
        **FACR_BASE_ARGS,
        **variant.extra_args,
        "lyapunov-prior-smoothing": args.lyapunov_prior_smoothing,
        "lyapunov-prior-max-step": args.lyapunov_prior_max_step,
        "residual-radius-min": args.residual_radius_min,
        "residual-radius-max": args.residual_radius_max,
        "residual-direction-l1-cap": args.residual_direction_l1_cap,
        "residual-amplitude-init-bias": args.residual_amplitude_init_bias,
        "residual-amplitude-penalty": args.residual_amplitude_penalty,
        "residual-proposal-temperature": args.residual_proposal_temperature,
        "residual-beta-max": args.residual_beta_max,
        "residual-beta-gamma-delta": args.residual_beta_gamma_delta,
        "residual-beta-gamma-volatility": args.residual_beta_gamma_volatility,
        "residual-aux-kl-weight": args.residual_aux_kl_weight,
        "residual-aux-proposal-kl-weight": args.residual_aux_proposal_kl_weight,
        "residual-aux-amplitude-weight": args.residual_aux_amplitude_weight,
        "residual-beta-penalty": args.residual_beta_penalty,
        "residual-confidence-eta-delta": args.residual_confidence_eta_delta,
        "residual-confidence-eta-volatility": args.residual_confidence_eta_volatility,
        "seed": seed,
        "feedback-delay": delay,
        "mixed-feedback-delays": args.mixed_feedback_delays,
        "output-dir": str(output_root / variant.name / f"seed_{seed}" / f"delay_{delay}"),
    }
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
    parser = argparse.ArgumentParser(description="Run FACR residual-prior comparison.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--variants", nargs="+", default=["FACR-Residual"])
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lyapunov-prior-smoothing", type=float, default=0.80)
    parser.add_argument("--lyapunov-prior-max-step", type=float, default=0.03)
    parser.add_argument("--residual-radius-min", type=float, default=0.0)
    parser.add_argument("--residual-radius-max", type=float, default=0.25)
    parser.add_argument("--residual-direction-l1-cap", type=float, default=1.0)
    parser.add_argument("--residual-amplitude-init-bias", type=float, default=0.0)
    parser.add_argument("--residual-amplitude-penalty", type=float, default=0.0)
    parser.add_argument("--residual-proposal-temperature", type=float, default=1.0)
    parser.add_argument("--residual-beta-max", type=float, default=0.05)
    parser.add_argument("--residual-beta-gamma-delta", type=float, default=1.0)
    parser.add_argument("--residual-beta-gamma-volatility", type=float, default=1.0)
    parser.add_argument("--residual-aux-kl-weight", type=float, default=0.0)
    parser.add_argument("--residual-aux-proposal-kl-weight", type=float, default=0.0)
    parser.add_argument("--residual-aux-amplitude-weight", type=float, default=0.0)
    parser.add_argument("--residual-beta-penalty", type=float, default=0.0)
    parser.add_argument("--residual-confidence-eta-delta", type=float, default=0.10)
    parser.add_argument("--residual-confidence-eta-volatility", type=float, default=0.10)
    parser.add_argument("--mixed-feedback-delays", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    variants = [variant for variant in VARIANTS if variant.name in set(args.variants)]
    if not variants:
        raise SystemExit(f"No selected variants. Available: {', '.join(v.name for v in VARIANTS)}")
    total = len(variants) * len(args.seeds) * len(args.delays)
    completed = 0
    print(f"Running {total} FACR residual comparison jobs under {output_root}", flush=True)
    for variant in variants:
        for seed in args.seeds:
            for delay in args.delays:
                out = summary_path(variant, output_root, seed, delay)
                if out.exists() and not args.force:
                    print(f"[skip] {variant.name} seed={seed} delay={delay}", flush=True)
                    completed += 1
                    continue
                command = build_command(variant, args, seed, delay, output_root)
                print(f"[run] {variant.name} seed={seed} delay={delay}", flush=True)
                start = __import__("time").perf_counter()
                subprocess.run(command, cwd=str(ROOT), check=True)
                elapsed = __import__("time").perf_counter() - start
                if not out.exists():
                    raise RuntimeError(f"Expected summary not found: {out}")
                completed += 1
                print(f"[done] {variant.name} seed={seed} delay={delay} in {elapsed:.1f}s ({completed}/{total})", flush=True)
    print("All selected FACR residual comparison jobs completed.", flush=True)


if __name__ == "__main__":
    main()
