"""Complete the best V4+LyapPrior setting to a full 5-seed x 5-delay grid.

This runner is deliberately narrow: it only runs the factorial-selected best
setting AL_BH_CH_DH_EH and skips any existing summaries in the target directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .run_multiseed_paper_comparison import COMMON_ARGS


ROOT = Path(__file__).resolve().parents[1]
METHOD = "DSR-MTG-Qpen-v4-agegate-lyapprior"
BEST_ARGS: dict[str, object] = {
    "alpha": 0.20,
    "q-penalty-weight": 0.08,
    "kappa-i": 0.02,
    "kappa-g": 0.02,
    "target-portfolio-entropy": 0.99,
    "feedback-gate-eta": 0.25,
    "max-supported-users": 50,
    "lyapunov-prior-weight": 0.20,
    "lyapunov-prior-v": 1.0,
    "lyapunov-prior-temperature": 2.5,
    "lyapunov-prior-smoothing": 0.80,
    "lyapunov-prior-max-step": 0.03,
}


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def out_base(output_dir: Path, seed: int, delay: int) -> Path:
    return output_dir / f"seed_{seed}" / f"delay_{delay}"


def summary_path(output_dir: Path, seed: int, delay: int) -> Path:
    return (
        out_base(output_dir, seed, delay)
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"summary_{METHOD}.csv"
    )


def build_command(args: argparse.Namespace, seed: int, delay: int) -> list[str]:
    base = out_base(Path(args.output_dir), seed, delay)
    values: dict[str, object] = {
        **COMMON_ARGS,
        **BEST_ARGS,
        "method": METHOD,
        "seed": seed,
        "feedback-delay": delay,
        "train-steps": args.train_steps,
        "test-steps": args.test_steps,
        "output-dir": str(base),
        "log-interval": args.log_interval,
    }
    if args.device:
        values["device"] = args.device
    cmd = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train"]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def run_one(args: argparse.Namespace, seed: int, delay: int) -> None:
    output_dir = Path(args.output_dir)
    summary = summary_path(output_dir, seed, delay)
    if summary.exists() and not args.force:
        print(f"[skip] best seed={seed} delay={delay}", flush=True)
        return
    log_dir = out_base(output_dir, seed, delay) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] best seed={seed} delay={delay}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(build_command(args, seed, delay), cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results_v4_lyapprior_factorial_formal/AL_BH_CH_DH_EH",
        help="Existing best-candidate directory; existing summaries are skipped.",
    )
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--delays", default="0,1,2,4,8")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for seed in parse_ints(args.seeds):
        for delay in parse_ints(args.delays):
            run_one(args, seed, delay)


if __name__ == "__main__":
    main()
