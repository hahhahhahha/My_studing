"""Resumable canonical rerun suite for the QLA-MAPPO paper.

Every formal method uses the same finite-buffer simulator, chronological
train/validation/test split policy, exogenous request schedule and common
assignment random numbers.  Existing legacy result trees are never reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .run_multiseed_paper_comparison import COMMON_ARGS
from .run_qla_simplified_rerun import common_main_args


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results_qla_canonical_v1"
SEEDS = (1, 2, 3, 4, 5)
DELAYS = (0, 1, 2, 4, 8)
PPO_METHODS = ("Towards-PPO-MV", "Towards-PPO-DSR")
RECENT_METHODS = ("QECO-load-topk", "Lyapunov-Q-local-stale")
# JSEQ-A and MFC-DI-A are the formal delayed-information baselines.
DELAYED_METHODS = ("JSEQ-A", "MFC-DI-A")


def _require_frozen_qla_config(root: Path) -> dict[str, object]:
    path = root / "frozen_config.json"
    if not path.exists():
        raise SystemExit(
            f"Formal QLA stage is blocked until validation selection is frozen: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "lambda_pi": float(common_main_args()["lcc-price-logit-bias"]),
        "target_portfolio_entropy": float(common_main_args()["target-portfolio-entropy"]),
        "lcc_v": float(common_main_args()["lcc-v"]),
        "lcc_migration_weight": float(common_main_args()["lcc-migration-weight"]),
        "no_stale_margin": bool(common_main_args()["lcc-no-stale-margin"]),
        "no_z_queue": bool(common_main_args()["lcc-no-z-queue"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"Frozen config mismatch for {key}: file={payload.get(key)!r}, runner={value!r}"
            )
    if payload.get("selection_split") != "validation":
        raise SystemExit("frozen_config.json must record selection_split='validation'")
    return payload


def _run(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    started = perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
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


def _summary_matches(path: Path, *, seed: int, delay: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        return (
            int(float(row.get("feedback_delay", delay))) == delay
            and row.get("finite_buffer_protocol", "").lower() in {"true", "1"}
            and bool(row.get("test_schedule_sha256"))
        )
    except (OSError, StopIteration, ValueError):
        return False


def run_qla(root: Path, device: str, force: bool) -> None:
    _require_frozen_qla_config(root)
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_qla_simplified_rerun",
        "--output-dir", str(root / "qla"),
        "--phase", "all",
        "--device", device,
    ]
    if force:
        command.append("--force")
    _run(command, log_path=root / "logs" / "qla_all.log")


def run_ppo(root: Path, device: str, force: bool) -> None:
    schedule_root = root / "shared_schedules"
    for method in PPO_METHODS:
        for seed in SEEDS:
            for delay in DELAYS:
                out = root / "baselines" / method / f"seed_{seed}" / f"delay_{delay}"
                summary_candidates = list(out.rglob(f"summary_{method}.csv"))
                if summary_candidates and not force and _summary_matches(
                    summary_candidates[0], seed=seed, delay=delay
                ):
                    continue
                values = {
                    **COMMON_ARGS,
                    "method": method,
                    "seed": seed,
                    "feedback-delay": delay,
                    "output-dir": str(out),
                    "canonical-schedule-root": str(schedule_root),
                    "canonical-max-feedback-delay": 8,
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
                    "device": device,
                }
                command = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.paper_policy_baselines"]
                for key, value in values.items():
                    command.extend([f"--{key}", str(value)])
                _run(command, log_path=root / "logs" / f"ppo_{method}.log")


def run_recent(root: Path, device: str, force: bool) -> None:
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_recent_mec_baseline_comparison",
        "--output-dir", str(root / "baselines"),
        "--methods", *RECENT_METHODS,
        "--seeds", *[str(v) for v in SEEDS],
        "--delays", *[str(v) for v in DELAYS],
        "--train-steps", "10000",
        "--test-steps", "1000",
        "--device", device,
        "--workers", "1",
    ]
    if force:
        command.append("--force")
    _run(command, log_path=root / "logs" / "recent_baselines.log")


def run_delayed(root: Path, device: str) -> None:
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_delayed_information_baseline_grid",
        "--output-dir", str(root / "baselines"),
        "--methods", ",".join(DELAYED_METHODS),
        "--seeds", ",".join(str(v) for v in SEEDS),
        "--delays", ",".join(str(v) for v in DELAYS),
        "--train-steps", "10000",
        "--test-steps", "1000",
        "--device", device,
    ]
    _run(command, log_path=root / "logs" / "delayed_baselines.log")


def run_mfc_qualification(root: Path, device: str, force: bool) -> None:
    # Directly relevant learning baseline, but kept outside the formal grid
    # until this predeclared 3-seed x 3-age qualification is reviewed.
    output = root / "baseline_qualification" / "MFC-DI-A"
    if force and output.exists():
        raise SystemExit(
            "Refusing destructive force cleanup for MFC qualification; choose a new output root."
        )
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_delayed_information_baseline_grid",
        "--output-dir", str(root / "baseline_qualification"),
        "--methods", "MFC-DI-A",
        "--seeds", "1,2,3",
        "--delays", "0,4,8",
        "--train-steps", "10000",
        "--test-steps", "1000",
        "--device", device,
    ]
    _run(command, log_path=root / "logs" / "mfc_qualification.log")


def run_factorial(root: Path, device: str, force: bool) -> None:
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_priceedge_qtbias_factorial",
        "--output-dir", str(root / "validation" / "factorial_2p3_center"),
        "--seeds", "1,2,3",
        "--delays", "0,4,8",
        "--train-steps", "10000",
        "--test-steps", "1000",
        "--factor-c", "lcc_v",
        "--no-stale-margin",
        "--no-z-queue",
        "--no-reuse-center",
        "--device", device,
    ]
    if force:
        command.append("--force")
    _run(command, log_path=root / "logs" / "factorial_validation.log")


def run_v_sweep(root: Path, device: str, force: bool) -> None:
    command = [
        sys.executable, "-B", "-m", "mappo_gp_sf_dsr.run_qla_lcc_v_sweep",
        "--output-dir", str(root / "validation" / "lcc_v_sweep"),
        "--factorial-root", str(root / "validation" / "factorial_2p3_center"),
        "--device", device,
    ]
    if force:
        command.append("--force")
    _run(command, log_path=root / "logs" / "lcc_v_validation.log")


STAGES = {
    "factorial": run_factorial,
    "v_sweep": run_v_sweep,
    "qla": run_qla,
    "ppo": run_ppo,
    "recent": run_recent,
    "delayed": run_delayed,
    "mfc_qualification": run_mfc_qualification,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stages",
        default="factorial,v_sweep",
        help=(
            "Comma-separated ordered stages. The safe default runs validation only; "
            "freeze the selected configuration before explicitly launching qla/ppo/recent/delayed."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = set(selected) - set(STAGES)
    if unknown:
        raise SystemExit(f"Unknown stages: {sorted(unknown)}")
    (root / "suite_protocol.json").write_text(
        json.dumps(
            {
                "stages": selected,
                "seeds": SEEDS,
                "delays": DELAYS,
                "finite_buffer_protocol": True,
                "atomic_task_admission": True,
                "train_validation_test": "chronological_60_20_20_with_guards",
                "checkpoint_rule": "final_requested_training_budget_no_test_early_stopping",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for name in selected:
        print(f"=== canonical stage: {name} ===", flush=True)
        fn = STAGES[name]
        if name == "delayed":
            fn(root, args.device)  # type: ignore[misc]
        else:
            fn(root, args.device, args.force)  # type: ignore[misc]


if __name__ == "__main__":
    main()
