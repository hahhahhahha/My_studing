from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import PPOConfig, SystemConfig, TRACE_ROOT


EXPECTED_SYSTEM = {
    "n_edges": 35,
    "slot_s": 10.0,
    "edge_compute_mb_s": 63.0,
    "queue_upper_mb": 630.0,
    "max_requests_per_user": 7,
    "task_package_mb": 270.0 / 1024.0,
    "task_compute_mb": 10.0,
    "bandwidth_hz": 2e6,
    "noise_w": 1e-10,
    "antenna_gain": 4.11,
    "carrier_hz": 915e6,
    "tx_power_w": 0.25,
}

EXPECTED_PPO = {
    "hidden": 128,
    "buffer_size": 20_000,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "batch_size": 128,
}


def trace_interval_report(trace_root: Path) -> dict[str, dict[str, float | str]]:
    report = {}
    for dataset in ("KAIST", "NCSU", "NewYork"):
        files = sorted((trace_root / dataset).glob("*.txt"))
        if not files:
            report[dataset] = {"status": "missing"}
            continue
        data = np.loadtxt(files[0], dtype=np.float64)
        intervals = np.diff(data[:, 0])
        report[dataset] = {
            "status": "ok",
            "file": files[0].name,
            "median_raw_interval_s": float(np.median(intervals)),
            "min_raw_interval_s": float(np.min(intervals)),
            "max_raw_interval_s": float(np.max(intervals)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reproduction parameters against the paper table")
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    args = parser.parse_args()

    sys_cfg = SystemConfig()
    ppo_cfg = PPOConfig()
    result = {
        "system_parameters": {
            name: {"expected": expected, "actual": getattr(sys_cfg, name), "match": getattr(sys_cfg, name) == expected}
            for name, expected in EXPECTED_SYSTEM.items()
        },
        "ppo_parameters": {
            name: {"expected": expected, "actual": getattr(ppo_cfg, name), "match": getattr(ppo_cfg, name) == expected}
            for name, expected in EXPECTED_PPO.items()
        },
        "trace_intervals": trace_interval_report(Path(args.trace_root)),
        "known_caveats": [
            "Bundled Traces_TimeXY_30sec_txt files are raw 30-second traces; data.py interpolates them to 10-second slots.",
            "Paper edge-server coordinates are not present in the bundled trace files; data.py samples them uniformly inside each dataset bounding box with a fixed seed.",
            "PPO-L/PG/SAC baselines are local adaptations to the same MEC environment, not full external paper reimplementations.",
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
