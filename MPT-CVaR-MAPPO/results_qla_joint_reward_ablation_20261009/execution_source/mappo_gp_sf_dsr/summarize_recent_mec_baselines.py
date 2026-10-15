"""Merge fixed V4 results with isolated recent-baseline results and gate drafting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECENT_GATE_METHODS = [
    "V4-AgeGate",
    "QECO-adapted",
    "QECO-load-topk",
    "Lyapunov-Q-stale",
    "Lyapunov-Q-fresh",
    "Lyapunov-Q-stale-tuned",
    "Lyapunov-Q-tail",
    "TODAuction-MAB-adapted",
    "TODAuction-MAB-tail",
]
PAPER_BASELINE_METHODS = [
    "Towards-PPO-MV",
    "Towards-PPO-DSR",
]
COMBINED_METHODS = [
    "V4-AgeGate",
    *PAPER_BASELINE_METHODS,
    "QECO-adapted",
    "QECO-load-topk",
    "Lyapunov-Q-stale",
    "Lyapunov-Q-fresh",
    "Lyapunov-Q-stale-tuned",
    "Lyapunov-Q-tail",
    "TODAuction-MAB-adapted",
    "TODAuction-MAB-tail",
]
METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "portfolio_migration_rate",
    "inference_time_ms",
]
COLORS = {
    "V4-AgeGate": "#1f77b4",
    "Towards-PPO-MV": "#8c564b",
    "Towards-PPO-DSR": "#9467bd",
    "QECO-adapted": "#d62728",
    "QECO-load-topk": "#e377c2",
    "Lyapunov-Q-stale": "#2ca02c",
    "Lyapunov-Q-fresh": "#98df8a",
    "Lyapunov-Q-stale-tuned": "#bcbd22",
    "Lyapunov-Q-tail": "#17becf",
    "TODAuction-MAB-adapted": "#ff7f0e",
    "TODAuction-MAB-tail": "#ffbb78",
    "SafeTail-redundancy-adapted": "#17becf",
}
MARKERS = {
    "V4-AgeGate": "o",
    "Towards-PPO-MV": "P",
    "Towards-PPO-DSR": "D",
    "QECO-adapted": "s",
    "QECO-load-topk": "h",
    "Lyapunov-Q-stale": "^",
    "Lyapunov-Q-fresh": "v",
    "Lyapunov-Q-stale-tuned": "<",
    "Lyapunov-Q-tail": ">",
    "TODAuction-MAB-adapted": "X",
    "TODAuction-MAB-tail": "*",
    "SafeTail-redundancy-adapted": "v",
}
SUMMARY_NAMES = {
    "V4-AgeGate": "DSR-MTG-Qpen-v4-agegate",
    "Towards-PPO-MV": "Towards-PPO-MV",
    "Towards-PPO-DSR": "Towards-PPO-DSR",
    "QECO-adapted": "QECO-adapted",
    "QECO-load-topk": "QECO-load-topk",
    "Lyapunov-Q-stale": "Lyapunov-Q-stale",
    "Lyapunov-Q-fresh": "Lyapunov-Q-fresh",
    "Lyapunov-Q-stale-tuned": "Lyapunov-Q-stale-tuned",
    "Lyapunov-Q-tail": "Lyapunov-Q-tail",
    "TODAuction-MAB-adapted": "TODAuction-MAB-adapted",
    "TODAuction-MAB-tail": "TODAuction-MAB-tail",
    "SafeTail-redundancy-adapted": "SafeTail-redundancy-adapted",
}


def _seed_delay(path: Path) -> tuple[int, int]:
    seed = next(int(part.removeprefix("seed_")) for part in path.parts if part.startswith("seed_"))
    delay = next(int(part.removeprefix("delay_")) for part in path.parts if part.startswith("delay_"))
    return seed, delay


def _read(path: Path, display_name: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    seed, delay = _seed_delay(path)
    output: dict[str, object] = {"method": display_name, "seed": seed, "feedback_delay": delay}
    for metric in METRICS:
        try:
            output[metric] = float(row.get(metric, "nan"))
        except ValueError:
            output[metric] = float("nan")
    return output


def _collect(v4_root: Path, recent_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(_collect_paper_methods(v4_root, ["V4-AgeGate"]))
    for method in RECENT_GATE_METHODS[1:]:
        for path in sorted((recent_root / method).rglob(f"summary_{method}.csv")):
            rows.append(_read(path, method))
    if not rows:
        raise SystemExit("No matching summaries found")
    return pd.DataFrame(rows)


def _collect_paper_methods(root: Path, methods: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in methods:
        summary_name = SUMMARY_NAMES[method]
        for path in sorted((root / method).rglob(f"summary_{summary_name}.csv")):
            rows.append(_read(path, method))
    return rows


def _draft_gate(raw: pd.DataFrame, expected_seeds: int, expected_delays: int) -> dict[str, object]:
    coverage = raw.groupby("method").size().to_dict()
    expected = expected_seeds * expected_delays
    grids = {
        method: set(
            raw.loc[raw["method"] == method, ["seed", "feedback_delay"]]
            .itertuples(index=False, name=None)
        )
        for method in RECENT_GATE_METHODS
    }
    common_grid = set.intersection(*(grids[method] for method in RECENT_GATE_METHODS))
    common = raw[
        raw.apply(lambda row: (row["seed"], row["feedback_delay"]) in common_grid, axis=1)
    ].copy()
    complete = (
        len(common_grid) == expected
        and all(int(coverage.get(method, 0)) >= expected for method in RECENT_GATE_METHODS)
    )
    overall = common.groupby("method")[["average_latency", "cvar99"]].mean()
    by_delay = common.groupby(["method", "feedback_delay"])[["average_latency", "cvar99"]].mean()
    checks: dict[str, object] = {}
    all_worse = True
    if "V4-AgeGate" not in overall.index:
        all_worse = False
    else:
        for method in RECENT_GATE_METHODS[1:]:
            if method not in overall.index:
                checks[method] = {"available": False}
                all_worse = False
                continue
            overall_worse = bool(
                overall.loc[method, "average_latency"] > overall.loc["V4-AgeGate", "average_latency"]
                and overall.loc[method, "cvar99"] > overall.loc["V4-AgeGate", "cvar99"]
            )
            per_delay_wins: list[dict[str, object]] = []
            for delay in sorted(common["feedback_delay"].unique()):
                key_method = (method, delay)
                key_v4 = ("V4-AgeGate", delay)
                if key_method not in by_delay.index or key_v4 not in by_delay.index:
                    continue
                for metric in ["average_latency", "cvar99"]:
                    if by_delay.loc[key_method, metric] <= by_delay.loc[key_v4, metric]:
                        per_delay_wins.append(
                            {
                                "delay": int(delay),
                                "metric": metric,
                                "baseline": float(by_delay.loc[key_method, metric]),
                                "v4": float(by_delay.loc[key_v4, metric]),
                            }
                        )
            checks[method] = {
                "available": True,
                "overall_both_primary_metrics_worse": overall_worse,
                "per_delay_primary_wins_or_ties": per_delay_wins,
            }
            all_worse = all_worse and overall_worse and not per_delay_wins
    return {
        "coverage": {method: int(coverage.get(method, 0)) for method in RECENT_GATE_METHODS},
        "common_grid_runs_per_method": len(common_grid),
        "common_grid": [
            {"seed": int(seed), "feedback_delay": int(delay)}
            for seed, delay in sorted(common_grid)
        ],
        "expected_runs_per_method": expected,
        "coverage_complete": complete,
        "all_new_baselines_strictly_worse": bool(all_worse and complete),
        "paper_update_allowed": bool(all_worse and complete),
        "checks": checks,
    }


def _plot(raw: pd.DataFrame, path: Path, methods: list[str], legend_cols: int = 4) -> None:
    plot_metrics = [
        ("average_latency", "Average latency (s)"),
        ("cvar99", "CVaR99 (s)"),
        ("queue_oscillation", "Queue oscillation"),
        ("global_demand_concentration", "Global demand concentration"),
    ]
    grouped = raw.groupby(["method", "feedback_delay"])[[key for key, _ in plot_metrics]].agg(
        ["mean", "std"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, label) in zip(axes.ravel(), plot_metrics):
        for method in methods:
            if method not in grouped.index.get_level_values(0):
                continue
            sub = grouped.loc[method].sort_index()
            x = sub.index.to_numpy(dtype=float)
            mean = sub[(metric, "mean")].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.1,
                label=method,
            )
        axis.set_xlabel("Feedback delay")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=legend_cols, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", default="results_paper_algorithms_v4scenario_multiseed")
    parser.add_argument("--recent-root", default="results_recent_mec_baselines_multiseed_tuned")
    parser.add_argument("--out-dir", default="results_recent_mec_baselines_comparison")
    parser.add_argument(
        "--paper-baselines",
        default="Towards-PPO-MV,Towards-PPO-DSR",
        help="Old paper-baseline methods to include in the combined figure. M2PPO and MCM-FDRL are intentionally omitted by default.",
    )
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--expected-delays", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = _collect(Path(args.v4_root), Path(args.recent_root))
    paper_methods = [m.strip() for m in args.paper_baselines.split(",") if m.strip()]
    combined_raw = pd.concat(
        [raw, pd.DataFrame(_collect_paper_methods(Path(args.v4_root), paper_methods))],
        ignore_index=True,
    )
    by_delay = raw.groupby(["method", "feedback_delay"], as_index=False)[METRICS].agg(["mean", "std"])
    overall = raw.groupby("method", as_index=False)[METRICS].agg(["mean", "std"])
    combined_by_delay = combined_raw.groupby(["method", "feedback_delay"], as_index=False)[METRICS].agg(["mean", "std"])
    combined_overall = combined_raw.groupby("method", as_index=False)[METRICS].agg(["mean", "std"])
    gate = _draft_gate(raw, args.expected_seeds, args.expected_delays)
    raw.to_csv(output / "recent_baselines_raw.csv", index=False)
    by_delay.to_csv(output / "recent_baselines_by_delay_mean_std.csv", index=False)
    overall.to_csv(output / "recent_baselines_overall_mean_std.csv", index=False)
    combined_raw.to_csv(output / "combined_baselines_raw.csv", index=False)
    combined_by_delay.to_csv(output / "combined_baselines_by_delay_mean_std.csv", index=False)
    combined_overall.to_csv(output / "combined_baselines_overall_mean_std.csv", index=False)
    _plot(raw, output / "recent_baselines_key_metrics.png", RECENT_GATE_METHODS, legend_cols=4)
    _plot(combined_raw, output / "combined_baselines_key_metrics_no_std.png", COMBINED_METHODS, legend_cols=3)
    with (output / "paper_update_gate.json").open("w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, ensure_ascii=False)
    lines = [
        "# Recent MEC Baseline Comparison Gate",
        "",
        f"- Coverage complete: **{gate['coverage_complete']}**",
        f"- All new baselines strictly worse: **{gate['all_new_baselines_strictly_worse']}**",
        f"- Paper update allowed: **{gate['paper_update_allowed']}**",
        "",
        "The gate requires complete coverage, worse overall mean latency and CVaR99 for every new baseline, and no per-delay win or tie on either primary metric.",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "```",
    ]
    (output / "PAPER_UPDATE_GATE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
