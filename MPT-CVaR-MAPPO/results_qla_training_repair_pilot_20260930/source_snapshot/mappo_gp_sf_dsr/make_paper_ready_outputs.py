from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from repro_risk_averse_edge.data import load_trace_set

from .calibrate_environment import _edge_capacities, _policy_portfolios
from .config import MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv


LEARNED_METHODS = [
    "V4-AgeGate",
    "Towards-PPO-DSR",
    "Towards-PPO-MV",
    "M2PPO",
    "MCM-FDRL",
]

SUMMARY_NAMES = {
    "V4-AgeGate": "DSR-MTG-Qpen-v4-agegate",
    "Towards-PPO-DSR": "Towards-PPO-DSR",
    "Towards-PPO-MV": "Towards-PPO-MV",
    "M2PPO": "M2PPO",
    "MCM-FDRL": "MCM-FDRL",
}

PAPER_METHODS = [
    "V4-AgeGate",
    "Towards-PPO-DSR",
    "Towards-PPO-MV",
    "score-softmax",
    "delayed-JSQ",
]

COLORS = {
    "V4-AgeGate": "#1B9E77",
    "Towards-PPO-DSR": "#7570B3",
    "Towards-PPO-MV": "#8C564B",
    "M2PPO": "#2CA02C",
    "MCM-FDRL": "#D62728",
    "score-softmax": "#D95F02",
    "delayed-JSQ": "#4C78A8",
}

MARKERS = {
    "V4-AgeGate": "o",
    "Towards-PPO-DSR": "D",
    "Towards-PPO-MV": "P",
    "M2PPO": "s",
    "MCM-FDRL": "^",
    "score-softmax": "X",
    "delayed-JSQ": "v",
}


def _set_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "---"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def make_final_table(final_dir: Path, out_dir: Path) -> None:
    overall = pd.read_csv(final_dir / "final_main_overall_mean_std.csv")
    metrics = [
        ("average_latency", "Avg latency (s)"),
        ("cvar95", "CVaR95 (s)"),
        ("cvar99", "CVaR99 (s)"),
        ("queue_oscillation", "Queue oscillation"),
        ("global_demand_concentration", "Global concentration"),
        ("inference_time_ms", "Inference time (ms)"),
    ]
    rows = []
    for method in LEARNED_METHODS:
        sub = overall[overall["policy"] == method]
        if sub.empty:
            continue
        item = {"Method": method}
        row = sub.iloc[0]
        for metric, label in metrics:
            item[label] = _fmt_pm(float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]))
        rows.append(item)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "paper_main_table_mean_std.csv", index=False)
    lines = [
        "# Paper-Ready Main Table",
        "",
        "Setting: KAIST, N=30, poisson-bursty workload, task=15MB, background=0, homogeneous servers, broadcast K=2, delays={0,1,2,4,8}.",
        "",
        "Values are mean +/- std over the completed multi-seed, multi-delay grid. Lower is better for all metrics.",
        "",
        table.to_markdown(index=False),
        "",
        "Note: heuristic mechanism baselines are reported separately because their role is to validate stale-feedback crowding, not to reproduce learned paper algorithms.",
    ]
    (out_dir / "PAPER_MAIN_TABLE_MEAN_STD.md").write_text("\n".join(lines), encoding="utf-8")


def plot_mechanism(scale_csv: Path, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_style()
    df = pd.read_csv(scale_csv)
    df = df[df["users"] == 30].copy()
    df["method"] = df["method"].replace({"V4-AgeGate-Tuned": "V4-AgeGate"})
    metrics = [
        ("average_latency_mean", "Average latency (s)"),
        ("cvar99_mean", "CVaR99 (s)"),
        ("queue_oscillation_mean", "Queue oscillation"),
        ("p90_queue_lifetime_mean", "P90 queue lifetime"),
        ("global_demand_concentration_mean", "Global concentration"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.6), constrained_layout=False)
    for ax, (metric, title) in zip(axes, metrics):
        for method in ["score-softmax", "delayed-JSQ", "V4-AgeGate"]:
            sub = df[df["method"] == method].sort_values("feedback_delay")
            if sub.empty:
                continue
            ax.plot(
                sub["feedback_delay"],
                sub[metric],
                marker=MARKERS[method],
                color=COLORS[method],
                linewidth=2.1,
                markersize=5.5,
                label=method,
            )
        ax.set_title(title, pad=7)
        ax.set_xlabel(r"Feedback delay $\delta$")
        ax.grid(alpha=0.25)
        ax.set_xticks([0, 1, 2, 4, 8])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Mechanism baselines under stale feedback", fontsize=14, y=1.19)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.2, top=0.77, wspace=0.36)
    fig.savefig(out_dir / "mechanism_baselines_delay_curves.png", bbox_inches="tight")
    plt.close(fig)


def _heuristic_latency_path(out_dir: Path, method: str, seed: int, delay: int) -> Path:
    return out_dir / "heuristic_latency_samples" / method / f"seed_{seed}" / f"delay_{delay}" / f"latency_samples_{method}.csv"


def _write_latency_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["phase", "slot", "user", "task", "edge", "latency", "tran_s", "queue_s", "comp_s"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sample_heuristic_latencies(out_dir: Path, method: str, seed: int, delay: int, steps: int) -> Path:
    path = _heuristic_latency_path(out_dir, method, seed, delay)
    if path.exists():
        return path
    sys_cfg = MAPPOSystemConfig(
        n_users=30,
        n_edges=35,
        slot_s=1.0,
        edge_compute_mb_s=63.0,
        queue_upper_mb=630.0,
        max_requests_per_user=7,
        task_compute_mb=15.0,
        task_package_mb=270.0 / 1024.0,
        max_supported_users=50,
    )
    traces = load_trace_set("KAIST", 30, sys_cfg.n_edges, sys_cfg.slot_s, seed, trace_root=TRACE_ROOT)
    capacities = _edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, "homogeneous", seed)
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution="poisson_bursty",
        seed=seed,
        queue_constraint="resample",
        score_config=ServerScoreConfig(),
        feedback_delay=delay,
        broadcast_period=2,
        poisson_lambda=3.5,
        burst_prob=0.1,
        burst_multiplier=2.0,
        edge_capacities_mb_s=capacities,
        background_load=0.0,
    )
    policy_name = "score-softmax" if method == "score-softmax" else "delayed-jsq"
    rows: list[dict] = []
    env.reset()
    for step in range(steps):
        portfolios = _policy_portfolios(env, policy_name, 5.0, 5)
        result = env.joint_step(portfolios, stats=env.portfolio_stats(portfolios), decision_latency_s=0.0)
        for idx, record in enumerate(result.records):
            rows.append(
                {
                    "phase": "test",
                    "slot": step,
                    "user": record.user,
                    "task": idx,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                }
            )
    _write_latency_rows(path, rows)
    return path


def _collect_learned_latency(root: Path, method: str, delay: int, seeds: list[int], max_points: int = 220_000) -> np.ndarray:
    summary_name = SUMMARY_NAMES[method]
    arrays = []
    for seed in seeds:
        pattern = root / method / f"seed_{seed}" / f"delay_{delay}" / "**" / f"latency_samples_{summary_name}.csv"
        for path in sorted(root.glob(str(pattern.relative_to(root)))):
            df = pd.read_csv(path)
            if "phase" in df.columns:
                df = df[df["phase"] == "test"]
            if "latency" in df.columns:
                arrays.append(pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(float))
    if not arrays:
        return np.asarray([], dtype=float)
    arr = np.concatenate(arrays)
    if arr.size > max_points:
        rng = np.random.default_rng(7)
        arr = rng.choice(arr, size=max_points, replace=False)
    return np.sort(arr)


def _collect_heuristic_latency(out_dir: Path, method: str, delay: int, seeds: list[int], steps: int) -> np.ndarray:
    arrays = []
    for seed in seeds:
        path = sample_heuristic_latencies(out_dir, method, seed, delay, steps)
        df = pd.read_csv(path)
        arrays.append(pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(float))
    return np.sort(np.concatenate(arrays)) if arrays else np.asarray([], dtype=float)


def plot_ccdf(root: Path, out_dir: Path, delay: int, seeds: list[int], steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_style()
    samples: dict[str, np.ndarray] = {}
    for method in LEARNED_METHODS:
        samples[method] = _collect_learned_latency(root, method, delay, seeds)
    for method in ["score-softmax", "delayed-JSQ"]:
        samples[method] = _collect_heuristic_latency(out_dir, method, delay, seeds, steps)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in PAPER_METHODS:
        arr = samples.get(method, np.asarray([], dtype=float))
        if arr.size == 0:
            continue
        arr = np.sort(arr)
        ccdf = 1.0 - np.arange(1, arr.size + 1, dtype=float) / arr.size
        ccdf = np.maximum(ccdf, 1.0 / arr.size)
        ax.plot(arr, ccdf, label=method, color=COLORS[method], linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Task latency T (s)")
    ax.set_ylabel(r"CCDF  $P(T > x)$")
    ax.set_title(rf"Tail latency distribution at $\delta={delay}$")
    ax.grid(alpha=0.28, which="both")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8.8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / f"latency_ccdf_delay{delay}.png", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for method, arr in samples.items():
        if arr.size == 0:
            continue
        summary_rows.append(
            {
                "method": method,
                "delay": delay,
                "n_samples": int(arr.size),
                "mean": float(np.mean(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / f"latency_ccdf_delay{delay}_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_paper_algorithms_v4scenario_multiseed")
    parser.add_argument("--final-dir", default="results_final_main_tuned_v4")
    parser.add_argument("--scale-csv", default="results_learned_policy_scale_n20_30_40/learned_policy_scale_by_delay.csv")
    parser.add_argument("--out-dir", default="results_paper_ready_outputs")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--heuristic-steps", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    make_final_table(Path(args.final_dir), out_dir)
    plot_mechanism(Path(args.scale_csv), out_dir)
    plot_ccdf(Path(args.root), out_dir, args.delay, seeds, args.heuristic_steps)
    lines = [
        "# Paper-Ready Output Index",
        "",
        "- `paper_main_table_mean_std.csv` / `.md`: final scalar main table with mean +/- std.",
        "- `mechanism_baselines_delay_curves.png`: score-softmax, delayed-JSQ, and V4-AgeGate over feedback delay.",
        f"- `latency_ccdf_delay{args.delay}.png`: tail latency CCDF including learned baselines and mechanism heuristics.",
        "",
        "These outputs are presentation-only; no learning runs are executed. Heuristic latency samples are generated by evaluation-only simulation when missing.",
    ]
    (out_dir / "PAPER_READY_OUTPUTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote paper-ready outputs to {out_dir}")


if __name__ == "__main__":
    main()
