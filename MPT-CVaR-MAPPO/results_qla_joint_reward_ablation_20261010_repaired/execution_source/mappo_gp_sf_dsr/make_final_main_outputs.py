from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
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

COLORS = {
    "V4-AgeGate": "#264653",
    "Towards-PPO-DSR": "#457B9D",
    "Towards-PPO-MV": "#2A9D8F",
    "M2PPO": "#E9C46A",
    "MCM-FDRL": "#E76F51",
}


def _read_summary(path: Path) -> dict[str, float | str | int]:
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    return row


def _collect_summaries(root: Path, delays: set[int], seeds: set[int]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        summary_name = SUMMARY_NAMES[method]
        for summary_path in sorted((root / method).glob(f"seed_*/delay_*/**/summary_{summary_name}.csv")):
            seed_match = re.search(r"seed_(\d+)", str(summary_path))
            delay_match = re.search(r"delay_(\d+)", str(summary_path))
            if not seed_match or not delay_match:
                continue
            seed = int(seed_match.group(1))
            delay = int(delay_match.group(1))
            if seed not in seeds or delay not in delays:
                continue
            raw = _read_summary(summary_path)
            rows.append(
                {
                    "policy": method,
                    "seed": seed,
                    "feedback_delay": delay,
                    "average_latency": raw.get("average_latency", np.nan),
                    "cvar95": raw.get("cvar95", np.nan),
                    "cvar99": raw.get("cvar99", np.nan),
                    "queue_oscillation": raw.get("queue_oscillation", np.nan),
                    "global_demand_concentration": raw.get("global_demand_concentration", np.nan),
                    "portfolio_migration_rate": raw.get("portfolio_migration_rate", np.nan),
                    "p90_queue_lifetime": raw.get("p90_queue_lifetime", np.nan),
                    "inference_time_ms": raw.get("inference_time_ms", np.nan),
                    "summary_path": str(summary_path),
                }
            )
    return pd.DataFrame(rows)


def _mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["runs"] = len(sub)
        for metric in metrics:
            values = pd.to_numeric(sub[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_main_table_bars(overall: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("average_latency", "Average latency (s)"),
        ("cvar95", "CVaR95 (s)"),
        ("cvar99", "CVaR99 (s)"),
        ("queue_oscillation", "Queue oscillation"),
        ("global_demand_concentration", "Global concentration"),
        ("inference_time_ms", "Inference time (ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    axes = axes.ravel()
    policies = [p for p in METHODS if p in set(overall["policy"])]
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        errs = []
        for policy in policies:
            row = overall[overall["policy"] == policy].iloc[0]
            means.append(row[f"{metric}_mean"])
            errs.append(row[f"{metric}_std"])
        ax.bar(
            np.arange(len(policies)),
            means,
            yerr=errs,
            capsize=4,
            color=[COLORS[p] for p in policies],
            alpha=0.88,
        )
        ax.set_title(title, fontsize=15)
        ax.set_xticks(np.arange(len(policies)), policies, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=11)
    fig.suptitle("Final multi-seed comparison under the V4 scenario", fontsize=18, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _collect_latency_samples(root: Path, method: str, delay: int, seeds: set[int], max_points: int = 250_000) -> np.ndarray:
    summary_name = SUMMARY_NAMES[method]
    samples = []
    for path in sorted((root / method).glob(f"seed_*/delay_{delay}/**/latency_samples_{summary_name}.csv")):
        seed_match = re.search(r"seed_(\d+)", str(path))
        if not seed_match or int(seed_match.group(1)) not in seeds:
            continue
        df = pd.read_csv(path)
        col = "latency" if "latency" in df.columns else df.columns[0]
        samples.append(pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float))
    if not samples:
        return np.asarray([], dtype=float)
    arr = np.concatenate(samples)
    if arr.size > max_points:
        rng = np.random.default_rng(0)
        arr = rng.choice(arr, size=max_points, replace=False)
    return np.sort(arr)


def _plot_cdf(root: Path, out_path: Path, seeds: set[int], delays: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(delays), figsize=(8.2 * len(delays), 5.6), squeeze=False)
    for ax, delay in zip(axes.ravel(), delays):
        for method in METHODS:
            arr = _collect_latency_samples(root, method, delay, seeds)
            if arr.size == 0:
                continue
            y = np.linspace(0.0, 1.0, arr.size, endpoint=False)
            ax.plot(arr, y, label=method, color=COLORS[method], linewidth=2.3)
        ax.set_title(f"Delay δ={delay}", fontsize=15)
        ax.set_xlabel("task latency (s)", fontsize=13)
        ax.set_ylabel("CDF", fontsize=13)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.28)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Latency CDF: V4-AgeGate vs reproduced baselines", fontsize=18, y=1.10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(by_delay: pd.DataFrame, overall: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Final Main Outputs: Tuned V4-AgeGate",
        "",
        "All rows are multi-seed same-environment summaries under KAIST, 30 users, poisson-bursty workload, task compute 15 MB, background load 0, homogeneous servers, K=2, train_steps=10000, test_steps=1000.",
        "",
        "## Overall Mean Across Delays",
        "",
        overall.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## By Delay Mean +/- Std",
        "",
        by_delay.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "- V4-AgeGate here uses the coordinate-tuned parameters from `run_multiseed_paper_comparison.py`.",
        "- Fed-MADDPG is intentionally excluded because its multiseed grid was incomplete.",
        "- This table adds CVaR95 and inference time columns that were missing from the earlier completed-method summary.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_paper_algorithms_v4scenario_multiseed")
    parser.add_argument("--out-dir", default="results_final_main_tuned_v4")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--delays", default="0,1,2,4,8")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    seeds = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
    delays = {int(x.strip()) for x in args.delays.split(",") if x.strip()}
    metrics = [
        "average_latency",
        "cvar95",
        "cvar99",
        "queue_oscillation",
        "global_demand_concentration",
        "portfolio_migration_rate",
        "p90_queue_lifetime",
        "inference_time_ms",
    ]
    raw = _collect_summaries(root, delays, seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "final_main_raw.csv", index=False)
    by_delay = _mean_std(raw, ["policy", "feedback_delay"], metrics)
    overall = _mean_std(raw, ["policy"], metrics)
    order = {policy: idx for idx, policy in enumerate(METHODS)}
    by_delay = by_delay.sort_values(["feedback_delay", "policy"], key=lambda s: s.map(order).fillna(s) if s.name == "policy" else s)
    overall = overall.sort_values("policy", key=lambda s: s.map(order).fillna(999))
    by_delay.to_csv(out_dir / "final_main_by_delay_mean_std.csv", index=False)
    overall.to_csv(out_dir / "final_main_overall_mean_std.csv", index=False)
    _plot_main_table_bars(overall, out_dir / "final_main_overall_bars.png")
    _plot_cdf(root, out_dir / "latency_cdf_delay0_delay8.png", seeds, [0, 8])
    _write_report(by_delay, overall, out_dir / "FINAL_MAIN_TABLE_AND_CDF.md")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
