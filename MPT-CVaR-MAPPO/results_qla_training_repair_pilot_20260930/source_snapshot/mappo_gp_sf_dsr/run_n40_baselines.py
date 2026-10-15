"""Run TODO-EXP-6: N=40 baselines (PPO-MV-A, PPO-DSR-A, QECO-A, JSEQ-A).

5 seeds × 2 distributions (uniform, gaussian) × 4 methods = 40 cells.
Matches QLA-MAPPO N=40 protocol: KAIST, 40u, delay=4, 10k train / 1k test.

Output: results_qla_simplified_rerun/n40_baselines/{METHOD}/{dist}/seed_{seed}/KAIST/40u/{dist}/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_qla_simplified_rerun" / "n40_baselines"

N40_SEEDS = [1, 2, 3, 4, 5]
N40_DISTRIBUTIONS = ["uniform", "gaussian"]
N40_DELAY = 4
N40_TRAIN = 10_000
N40_TEST = 1_000


@dataclass(frozen=True)
class MethodConfig:
    name: str
    module: str
    method_arg: str
    summary_filename: str
    extra_args: dict[str, object] = field(default_factory=dict)


METHODS: list[MethodConfig] = [
    MethodConfig(
        name="PPO-MV-A",
        module="mappo_gp_sf_dsr.paper_policy_baselines",
        method_arg="Towards-PPO-MV",
        summary_filename="summary_Towards-PPO-MV.csv",
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
            "update-every": 4,
            "tau": 0.01,
            "noise-start": 0.08,
            "noise-final": 0.01,
        },
    ),
    MethodConfig(
        name="PPO-DSR-A",
        module="mappo_gp_sf_dsr.paper_policy_baselines",
        method_arg="Towards-PPO-DSR",
        summary_filename="summary_Towards-PPO-DSR.csv",
        extra_args={  # same as PPO-MV-A; the method_arg drives the reward form
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
            "update-every": 4,
            "tau": 0.01,
            "noise-start": 0.08,
            "noise-final": 0.01,
        },
    ),
    MethodConfig(
        name="QECO-A",
        module="mappo_gp_sf_dsr.recent_mec_baselines",
        method_arg="QECO-load-topk",
        summary_filename="summary_QECO-load-topk.csv",
        extra_args={},
    ),
    MethodConfig(
        name="JSEQ-A",
        module="mappo_gp_sf_dsr.delayed_information_baselines",
        method_arg="JSEQ-A",
        summary_filename="summary_JSEQ-A.csv",
        extra_args={},
    ),
]


COMMON_ENV_ARGS = {
    "dataset": "KAIST",
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


def summary_leaf(out_dir: Path, method: MethodConfig, dist: str) -> Path:
    """Resolve the summary CSV path for one cell.

    PPO-MV-A / PPO-DSR-A insert an extra `{method_arg}_delay_{delay}/` subdir
    between out_dir and the dataset/users/distribution leaf. QECO-A and JSEQ-A
    write directly under {out_dir}/KAIST/{users}u/{dist}/.
    """
    if method.name in {"PPO-MV-A", "PPO-DSR-A"}:
        return out_dir / f"{method.method_arg}_delay_{N40_DELAY}" / "KAIST" / "40u" / dist / method.summary_filename
    return out_dir / "KAIST" / "40u" / dist / method.summary_filename


def find_summary(out_dir: Path, method: MethodConfig) -> Path | None:
    for dist in N40_DISTRIBUTIONS:
        leaf = summary_leaf(out_dir, method, dist)
        if leaf.exists():
            return leaf
    return None


def build_command(
    method: MethodConfig,
    seed: int,
    dist: str,
    output_root: Path,
    device: str | None,
    force: bool,
) -> tuple[list[str], Path]:
    out_dir = output_root / method.name / dist / f"seed_{seed}"
    args_dict: dict[str, object] = {
        **COMMON_ENV_ARGS,
        **method.extra_args,
        "users": 40,
        "distribution": dist,
        "feedback-delay": N40_DELAY,
        "seed": seed,
        "output-dir": str(out_dir),
    }
    # JSEQ-A is mechanism-only (no training); the existing grid runner uses 0.
    args_dict["train-steps"] = 0 if method.name == "JSEQ-A" else N40_TRAIN
    args_dict["test-steps"] = N40_TEST
    if device is not None:
        args_dict["device"] = device
    cmd = [sys.executable, "-B", "-m", method.module, "--method", method.method_arg]
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif value is not None:
            cmd.extend([f"--{key}", str(value)])
    return cmd, out_dir


def cell_complete(out_dir: Path, method: MethodConfig, seed: int, dist: str) -> bool:
    return summary_leaf(out_dir, method, dist).exists()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--methods", default=",".join(m.name for m in METHODS))
    p.add_argument("--distributions", default=",".join(N40_DISTRIBUTIONS))
    p.add_argument("--seeds", default=",".join(str(s) for s in N40_SEEDS))
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    distributions = [d.strip() for d in args.distributions.split(",") if d.strip()]
    selected = [m for m in METHODS if m.name in requested_methods]
    missing = set(requested_methods) - {m.name for m in selected}
    if missing:
        raise SystemExit(f"Unknown method(s): {sorted(missing)}")

    total = len(selected) * len(seeds) * len(distributions)
    done = 0
    for method in selected:
        for dist in distributions:
            for seed in seeds:
                cmd, out_dir = build_command(method, seed, dist, output_root, args.device, args.force)
                summary_path = summary_leaf(out_dir, method, dist)
                if summary_path.exists() and not args.force:
                    print(f"[skip] {method.name} {dist} seed={seed} ({done+1}/{total})", flush=True)
                else:
                    print(
                        f"[run] {method.name} {dist} seed={seed} delay={N40_DELAY} "
                        f"({done+1}/{total}) -> {out_dir.relative_to(ROOT)}",
                        flush=True,
                    )
                    subprocess.run(cmd, cwd=str(ROOT), check=True)
                if not summary_path.exists():
                    raise RuntimeError(f"Expected summary not found: {summary_path}")
                done += 1
                print(f"[done] {done}/{total}", flush=True)

    print(f"\nTODO-EXP-6 complete: {done} cells in {output_root}", flush=True)


if __name__ == "__main__":
    main()
