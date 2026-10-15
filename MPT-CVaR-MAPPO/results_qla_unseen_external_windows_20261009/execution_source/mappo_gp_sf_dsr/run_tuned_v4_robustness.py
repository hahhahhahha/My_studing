from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    name: str
    args: dict[str, object] = field(default_factory=dict)


COMMON = {
    "dataset": "KAIST",
    "users": 30,
    "distribution": "poisson_bursty",
    "method": "DSR-MTG-Qpen-v4-agegate",
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
    "max-supported-users": 80,
    # Coordinate-tuned V4-AgeGate.
    "alpha": 0.30,
    "q-penalty-weight": 0.08,
    "kappa-i": 0.02,
    "kappa-g": 0.02,
    "target-portfolio-entropy": 0.99,
    "feedback-gate-eta": 0.10,
}


SCENARIOS = [
    Scenario("main_task15_bg0_homo", {}),
    Scenario("bg02_task15_homo", {"background-load": 0.2}),
    Scenario("bg04_task15_homo", {"background-load": 0.4}),
    Scenario("task10_bg0_homo", {"task-compute-mb": 10.0}),
    Scenario("task20_bg0_homo", {"task-compute-mb": 20.0}),
    Scenario("task15_bg0_heterogeneous", {"capacity-mode": "heterogeneous"}),
]


METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "portfolio_migration_rate",
    "p90_queue_lifetime",
    "inference_time_ms",
    "queue_overflow_rate",
    "mean_utilization",
]


def build_cmd(args: dict[str, object]) -> list[str]:
    cmd = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train"]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def summary_path(out_dir: Path) -> Path:
    return out_dir / "KAIST" / "30u" / "poisson_bursty" / "summary_DSR-MTG-Qpen-v4-agegate.csv"


def run_one(scenario: Scenario, seed: int, delay: int, out_root: Path, device: str | None, force: bool) -> Path:
    out_dir = out_root / scenario.name / f"seed_{seed}" / f"delay_{delay}"
    summary = summary_path(out_dir)
    if summary.exists() and not force:
        print(f"[skip] scenario={scenario.name} seed={seed} delay={delay} summary={summary}", flush=True)
        return summary
    args = dict(COMMON)
    args.update(scenario.args)
    args.update({"seed": seed, "feedback-delay": delay, "output-dir": str(out_dir)})
    if device:
        args["device"] = device
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    cmd = build_cmd(args)
    print(f"[run] scenario={scenario.name} seed={seed} delay={delay} device={device or 'auto'}", flush=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    elapsed = (time.time() - start) / 60.0
    print(f"[done] scenario={scenario.name} seed={seed} delay={delay} elapsed_min={elapsed:.2f}", flush=True)
    return summary


def summarize(out_root: Path, scenarios: list[Scenario], seeds: list[int], delays: list[int]) -> None:
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            for delay in delays:
                path = summary_path(out_root / scenario.name / f"seed_{seed}" / f"delay_{delay}")
                if not path.exists():
                    continue
                raw = pd.read_csv(path).iloc[0].to_dict()
                row = {"scenario": scenario.name, "seed": seed, "feedback_delay": delay, "summary_path": str(path)}
                for metric in METRICS:
                    row[metric] = raw.get(metric, np.nan)
                rows.append(row)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_root / "tuned_v4_robustness_raw.csv", index=False)
    if raw_df.empty:
        return
    by_delay = (
        raw_df.groupby(["scenario", "feedback_delay"])[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    by_delay.columns = [
        "_".join([str(part) for part in col if part != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in by_delay.columns
    ]
    overall = raw_df.groupby("scenario")[METRICS].agg(["mean", "std"]).reset_index()
    overall.columns = [
        "_".join([str(part) for part in col if part != ""]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in overall.columns
    ]
    by_delay.to_csv(out_root / "tuned_v4_robustness_by_delay.csv", index=False)
    overall.to_csv(out_root / "tuned_v4_robustness_overall.csv", index=False)
    write_report(out_root, by_delay, overall)
    plot(out_root, by_delay)


def write_report(out_root: Path, by_delay: pd.DataFrame, overall: pd.DataFrame) -> None:
    lines = [
        "# Tuned V4-AgeGate Learned-Policy Robustness",
        "",
        "This is not calibration-only: each row is trained and evaluated with tuned V4-AgeGate in the target scenario.",
        "Compact setting: seeds {0,1,2}, delays {0,8}, train_steps=10000, test_steps=1000.",
        "",
        "## Overall By Scenario",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## By Delay",
        "",
        by_delay.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_root / "TUNED_V4_ROBUSTNESS.md").write_text("\n".join(lines), encoding="utf-8")


def plot(out_root: Path, by_delay: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("average_latency_mean", "Average latency (s)"),
        ("cvar99_mean", "CVaR99 (s)"),
        ("queue_oscillation_mean", "Queue oscillation"),
        ("global_demand_concentration_mean", "Global concentration"),
        ("queue_overflow_rate_mean", "Overflow ratio"),
        ("inference_time_ms_mean", "Inference time (ms)"),
    ]
    scenarios = list(by_delay["scenario"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for scenario in scenarios:
            sub = by_delay[by_delay["scenario"] == scenario].sort_values("feedback_delay")
            ax.plot(sub["feedback_delay"], sub[metric], marker="o", linewidth=2.0, label=scenario)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("feedback delay δ", fontsize=12)
        ax.grid(alpha=0.28)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Tuned V4-AgeGate learned-policy robustness", fontsize=18, y=1.08)
    fig.tight_layout()
    fig.savefig(out_root / "tuned_v4_robustness_key_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results_tuned_v4_robustness")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--delays", default="0,8")
    parser.add_argument("--scenarios", default=",".join(s.name for s in SCENARIOS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    delays = [int(x.strip()) for x in args.delays.split(",") if x.strip()]
    wanted = {x.strip() for x in args.scenarios.split(",") if x.strip()}
    scenarios = [scenario for scenario in SCENARIOS if scenario.name in wanted]
    for scenario in scenarios:
        for seed in seeds:
            for delay in delays:
                run_one(scenario, seed, delay, out_root, args.device, args.force)
                summarize(out_root, scenarios, seeds, delays)
    summarize(out_root, scenarios, seeds, delays)
    print(f"Wrote robustness outputs to {out_root}")


if __name__ == "__main__":
    main()
