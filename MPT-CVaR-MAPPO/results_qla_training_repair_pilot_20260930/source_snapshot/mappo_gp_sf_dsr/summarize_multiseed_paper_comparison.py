from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Metrics used in the paper comparison table and figure.
KEY_METRICS = [
    ("average_latency", "平均时延 (s)"),
    ("cvar99", "CVaR$_{99}$ (s)"),
    ("queue_oscillation", "队列振荡"),
    ("global_demand_concentration", "全局需求集中度"),
    ("portfolio_migration_rate", "迁移率"),
    ("p90_queue_lifetime", "P90 队列寿命"),
]

PLOT_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("global_demand_concentration", "Global demand concentration"),
]

METHOD_ORDER = [
    "V4-AgeGate",
    "Towards-PPO-MV",
    "Towards-PPO-DSR",
    "M2PPO",
    "MCM-FDRL",
    "Fed-MADDPG",
]

PLOT_METHOD_ORDER = [
    "V4-AgeGate",
    "Towards-PPO-MV",
    "Towards-PPO-DSR",
]

PLOT_COLORS = {
    "V4-AgeGate": "#1f77b4",
    "M2PPO": "#2ca02c",
    "MCM-FDRL": "#d62728",
    "Towards-PPO-DSR": "#9467bd",
    "Towards-PPO-MV": "#8c564b",
    "Fed-MADDPG": "#ff7f0e",
}

PLOT_MARKERS = {
    "V4-AgeGate": "o",
    "M2PPO": "s",
    "MCM-FDRL": "^",
    "Towards-PPO-DSR": "D",
    "Towards-PPO-MV": "P",
    "Fed-MADDPG": "X",
}

STALE_DELAYS = {2, 4, 8}


def _parse_seed_delay_from_path(summary_path: Path, root: Path) -> tuple[int | None, int | None]:
    rel = summary_path.relative_to(root)
    seed = None
    delay = None
    for part in rel.parts:
        if part.startswith("seed_"):
            seed = int(part.replace("seed_", ""))
        if part.startswith("delay_"):
            delay = int(part.replace("delay_", ""))
    return seed, delay


def _read_one_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows[0]


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_summary(path: Path, display_name: str, root: Path) -> dict[str, object]:
    row = _read_one_row_csv(path)
    seed, path_delay = _parse_seed_delay_from_path(path, root)
    csv_delay = int(_to_float(row.get("feedback_delay", row.get("delay", path_delay if path_delay is not None else 0))))
    if path_delay is not None and csv_delay != path_delay:
        raise RuntimeError(f"Delay mismatch: path delay={path_delay}, csv delay={csv_delay}, file={path}")
    out: dict[str, object] = {
        "policy": display_name,
        "feedback_delay": csv_delay,
        "seed": seed,
    }
    for key, _ in KEY_METRICS:
        out[key] = _to_float(row.get(key))
    out["method"] = row.get("method", display_name)
    return out


def _collect_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary_*.csv")):
        rel = summary_path.relative_to(root)
        display_name = rel.parts[0]
        rows.append(_read_summary(summary_path, display_name, root))
    if not rows:
        raise SystemExit(f"No summary CSVs found under {root}")
    return pd.DataFrame(rows)


def _sort_policy(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    df = df.copy()
    df["_policy_order"] = df["policy"].map(order).fillna(99)
    df = df.sort_values(["_policy_order", "feedback_delay", "seed"]).drop(columns=["_policy_order"])
    return df


def _check_data_quality(df: pd.DataFrame) -> None:
    # Check for duplicate (policy, delay, seed) triples.
    dup = df.groupby(["policy", "feedback_delay", "seed"]).size().reset_index(name="count")
    dup = dup[dup["count"] > 1]
    if not dup.empty:
        raise RuntimeError(f"Duplicate summaries found for {len(dup)} (policy, delay, seed) groups:\n{dup.to_string(index=False)}")


def _check_seed_coverage(df: pd.DataFrame) -> None:
    grouped = df.groupby(["policy", "feedback_delay"])["seed"].apply(
        lambda s: tuple(int(x) for x in sorted(s.dropna().astype(int).unique()))
    )
    seed_sets = grouped.unique()
    if len(seed_sets) == 1:
        print(f"[OK] all policy-delay groups use seeds {list(seed_sets[0])}")
    else:
        print("[WARN] policy-delay groups use inconsistent seed sets:")
        for (policy, delay), seeds in grouped.items():
            print(f"[WARN] {policy} delay={delay}: seeds={list(seeds)}")


def _aggregate_by_delay(df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [key for key, _ in KEY_METRICS]
    by_delay = df.groupby(["policy", "feedback_delay"], as_index=False)[metric_keys].agg(["mean", "std"])
    by_delay.columns = ["_".join(col).strip("_") for col in by_delay.columns.values]
    return by_delay


def _format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    if not np.isfinite(mean):
        return "---"
    if not np.isfinite(std) or std < 1e-12:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _build_by_delay_table(by_delay: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in METHOD_ORDER:
        for delay in sorted(by_delay["feedback_delay"].unique()):
            sub = by_delay[(by_delay["policy"] == policy) & (by_delay["feedback_delay"] == delay)]
            if sub.empty:
                continue
            row = {"方法": policy, "延迟": int(delay)}
            for key, label_cn in KEY_METRICS:
                mean = float(sub[f"{key}_mean"].iloc[0])
                std = float(sub[f"{key}_std"].iloc[0])
                row[label_cn] = _format_mean_std(mean, std)
            rows.append(row)
    return pd.DataFrame(rows)


def _stale_summary_from_by_delay(by_delay: pd.DataFrame) -> pd.DataFrame:
    """Compute stale-feedback summaries from the per-delay mean values."""
    rows = []
    for policy in METHOD_ORDER:
        sub = by_delay[by_delay["policy"] == policy]
        if sub.empty:
            continue
        stale = sub[sub["feedback_delay"].isin(STALE_DELAYS)]
        all_delay = sub
        row = {"方法": policy}
        row["AvgStaleLatency"] = float(stale["average_latency_mean"].mean())
        row["AvgStaleCVaR99"] = float(stale["cvar99_mean"].mean())
        row["AvgStaleQueueOsc"] = float(stale["queue_oscillation_mean"].mean())
        row["WorstCVaR99"] = float(all_delay["cvar99_mean"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_metrics(by_delay: pd.DataFrame, out_path: Path) -> None:
    available = [(k, l) for k, l in PLOT_METRICS if f"{k}_mean" in by_delay.columns]
    n_metrics = len(available)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.0 * n_rows))
    if n_metrics == 1:
        axes = np.asarray([axes])
    else:
        axes = axes.ravel()

    for ax, (metric, label) in zip(axes, available):
        for policy in PLOT_METHOD_ORDER:
            sub = by_delay[by_delay["policy"] == policy].sort_values("feedback_delay")
            if sub.empty:
                continue
            x = sub["feedback_delay"].to_numpy()
            y = sub[f"{metric}_mean"].to_numpy()
            ax.plot(
                x, y,
                marker=PLOT_MARKERS.get(policy, "o"),
                linewidth=2.2,
                markersize=6,
                color=PLOT_COLORS.get(policy),
                label=policy,
            )
        ax.set_xlabel("Feedback delay", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.grid(True, alpha=0.28)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, labels = [], []
    for policy in PLOT_METHOD_ORDER:
        if policy not in set(by_delay["policy"]):
            continue
        handle = plt.Line2D(
            [0], [0],
            color=PLOT_COLORS.get(policy),
            marker=PLOT_MARKERS.get(policy, "o"),
            linewidth=2.2,
            markersize=6,
            label=policy,
        )
        handles.append(handle)
        labels.append(policy)

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        fontsize=11,
        frameon=True,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.86, hspace=0.42, wspace=0.30)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_stale_summary_report(
    df: pd.DataFrame,
    by_delay: pd.DataFrame,
    out_path: Path,
) -> None:
    by_delay_table = _build_by_delay_table(by_delay)
    stale_summary = _stale_summary_from_by_delay(by_delay)

    # Seed counts per method.
    seed_counts = df.groupby("policy")["seed"].apply(lambda s: len(sorted(s.dropna().astype(int).unique())))

    lines = [
        "# Multi-Seed Baseline Comparison Summary",
        "",
        "Scenario: KAIST, N=30, poisson_bursty, task=15MB, K=2, delays={0,1,2,4,8}.",
        "Each reported value is mean ± std over the available seeds for that (method, delay) pair.",
        "",
        "## Valid Seed Counts per Method",
        "",
    ]
    for policy in METHOD_ORDER:
        count = int(seed_counts.get(policy, 0))
        lines.append(f"- {policy}: {count} seeds")
    lines.append("")

    lines.extend([
        "## Per-Delay Results (mean ± std)",
        "",
        by_delay_table.to_markdown(index=False),
        "",
        "## V4-AgeGate vs Baselines at Key Delays",
        "",
    ])

    for delay in [0, 4, 8]:
        sub = by_delay[by_delay["feedback_delay"] == delay]
        if sub.empty:
            continue
        lines.append(f"### delay = {delay}")
        lines.append("")
        v4 = sub[sub["policy"] == "V4-AgeGate"]
        if v4.empty:
            lines.append("- V4-AgeGate data missing.")
        else:
            v4_lat = float(v4["average_latency_mean"].iloc[0])
            v4_cvar = float(v4["cvar99_mean"].iloc[0])
            lines.append(
                f"- **V4-AgeGate**: latency {v4_lat:.3f}s, CVaR99 {v4_cvar:.3f}s"
            )
        for policy in [p for p in METHOD_ORDER if p != "V4-AgeGate"]:
            row = sub[sub["policy"] == policy]
            if row.empty or v4.empty:
                continue
            lat = float(row["average_latency_mean"].iloc[0])
            cvar = float(row["cvar99_mean"].iloc[0])
            lat_delta = (lat / v4_lat - 1.0) * 100.0 if v4_lat > 1e-12 else float("nan")
            cvar_delta = (cvar / v4_cvar - 1.0) * 100.0 if v4_cvar > 1e-12 else float("nan")
            lines.append(
                f"- {policy}: latency {lat:.3f}s ({lat_delta:+.1f}%), CVaR99 {cvar:.3f}s ({cvar_delta:+.1f}%)"
            )
        lines.append("")

    lines.extend([
        "## Stale-Feedback Summary (δ ∈ {2,4,8})",
        "",
        "*Stale-feedback summary is computed from the per-delay mean values. It reflects the overall trend, but is not a rigorous per-seed statistical summary.*",
        "",
        stale_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "**Interpretation:** AvgStaleLatency / AvgStaleCVaR99 are averages over the stale delays; WorstCVaR99 is the maximum CVaR99 across all delays.",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed paper comparison results."
    )
    parser.add_argument(
        "--root",
        default="results_paper_algorithms_v4scenario_multiseed",
        help="Root directory produced by run_multiseed_paper_comparison.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="results_paper_algorithms_v4scenario_multiseed",
        help="Directory to write aggregated outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _sort_policy(_collect_rows(root))
    _check_data_quality(df)
    _check_seed_coverage(df)

    by_delay = _aggregate_by_delay(df)

    df.to_csv(out_dir / "multiseed_raw.csv", index=False)
    by_delay.to_csv(out_dir / "multiseed_by_delay_mean_std.csv", index=False)
    _plot_metrics(by_delay, out_dir / "multiseed_key_metrics.png")
    _write_stale_summary_report(df, by_delay, out_dir / "multiseed_stale_summary.md")

    print(f"Wrote {out_dir / 'multiseed_raw.csv'}")
    print(f"Wrote {out_dir / 'multiseed_by_delay_mean_std.csv'}")
    print(f"Wrote {out_dir / 'multiseed_key_metrics.png'}")
    print(f"Wrote {out_dir / 'multiseed_stale_summary.md'}")
    print("\nPer-delay preview:")
    print(_build_by_delay_table(by_delay).to_string(index=False))
    print("\nStale summary preview:")
    print(
        _stale_summary_from_by_delay(by_delay).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )


if __name__ == "__main__":
    main()
