from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("median_queue_lifetime", "Median queue lifetime"),
    ("p90_queue_lifetime", "P90 queue lifetime"),
    ("queue_overflow_rate", "Queue overflow ratio"),
    ("mean_utilization", "Mean utilization"),
]

CROWDING_METRICS = [
    ("global_demand_concentration", "Global demand concentration"),
    ("server_selection_gini", "Server selection Gini"),
    ("portfolio_migration_rate", "Portfolio migration rate"),
    ("mean_max_server_pressure", "Mean max server pressure"),
]

NINE_PANEL_METRICS = CORE_METRICS[:5] + CROWDING_METRICS


def _load_agegate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"] == "DSR-MTG-Qpen-v4-agegate"].copy()
    df["policy"] = "MAPPO-V4-AgeGate"
    df = df.rename(columns={"delay": "feedback_delay"})
    return df


def _load_score_softmax(path: Path, users: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["users"].astype(int) == int(users)].copy()
    mapping = {
        "average_latency_mean": "average_latency",
        "cvar99_mean": "cvar99",
        "queue_oscillation_mean": "queue_oscillation",
        "median_queue_lifetime_mean": "median_queue_lifetime",
        "p90_queue_lifetime_mean": "p90_queue_lifetime",
        "queue_overflow_rate_mean": "queue_overflow_rate",
        "mean_utilization_mean": "mean_utilization",
        "global_demand_concentration_mean": "global_demand_concentration",
        "server_selection_gini_mean": "server_selection_gini",
        "portfolio_migration_rate_mean": "portfolio_migration_rate",
        "mean_max_server_pressure_mean": "mean_max_server_pressure",
    }
    std_mapping = {
        "average_latency_std": "average_latency_std",
        "cvar99_std": "cvar99_std",
        "queue_oscillation_std": "queue_oscillation_std",
        "median_queue_lifetime_std": "median_queue_lifetime_std",
        "p90_queue_lifetime_std": "p90_queue_lifetime_std",
        "queue_overflow_rate_std": "queue_overflow_rate_std",
        "mean_utilization_std": "mean_utilization_std",
        "global_demand_concentration_std": "global_demand_concentration_std",
        "server_selection_gini_std": "server_selection_gini_std",
        "portfolio_migration_rate_std": "portfolio_migration_rate_std",
        "mean_max_server_pressure_std": "mean_max_server_pressure_std",
    }
    keep = ["feedback_delay", *mapping.keys(), *[k for k in std_mapping if k in df.columns]]
    df = df[keep].rename(columns={**mapping, **std_mapping})
    df["policy"] = "score-softmax"
    return df


def _combined(agegate_path: Path, score_path: Path, users: int) -> pd.DataFrame:
    agegate = _load_agegate(agegate_path)
    score = _load_score_softmax(score_path, users)
    metrics = {name for name, _ in CORE_METRICS + CROWDING_METRICS}
    cols = ["policy", "feedback_delay", *sorted(metrics)]
    std_cols = [f"{m}_std" for m in sorted(metrics) if f"{m}_std" in score.columns]
    return pd.concat([score, agegate], ignore_index=True, sort=False)[cols + std_cols].sort_values(
        ["policy", "feedback_delay"]
    )


def _plot_lines(df: pd.DataFrame, metrics: list[tuple[str, str]], path: Path, title: str, ncols: int = 2) -> None:
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.7 * nrows), squeeze=False)
    policies = [
        ("score-softmax", "#d95f02", "o"),
        ("MAPPO-V4-AgeGate", "#1f77b4", "s"),
    ]
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for policy, color, marker in policies:
            sub = df[df["policy"] == policy].sort_values("feedback_delay")
            x = sub["feedback_delay"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            yerr = None
            if policy == "score-softmax" and f"{metric}_std" in sub.columns:
                yerr = sub[f"{metric}_std"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=color,
                marker=marker,
                linewidth=2.0,
                capsize=3 if yerr is not None else 0,
                label=policy,
            )
        ax.set_title(label)
        ax.set_xlabel("feedback delay")
        ax.grid(True, alpha=0.28)
        if metric in {"queue_overflow_rate", "mean_utilization"}:
            ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.988, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_nine_panel(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(19.0, 15.0), squeeze=False)
    policies = [
        ("score-softmax", "#d95f02", "o"),
        ("MAPPO-V4-AgeGate", "#1f77b4", "s"),
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    for ax, panel, (metric, label) in zip(axes.ravel(), panel_labels, NINE_PANEL_METRICS):
        for policy, color, marker in policies:
            sub = df[df["policy"] == policy].sort_values("feedback_delay")
            x = sub["feedback_delay"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            yerr = None
            if policy == "score-softmax" and f"{metric}_std" in sub.columns:
                yerr = sub[f"{metric}_std"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=color,
                marker=marker,
                markersize=7.5,
                linewidth=2.8,
                capsize=4 if yerr is not None else 0,
                elinewidth=1.6,
                label=policy,
            )
        ax.set_title(f"{panel} {label}", fontsize=17, pad=10)
        ax.set_xlabel("Feedback delay", fontsize=15)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.30)
        ax.ticklabel_format(axis="y", useOffset=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("MAPPO-V4-AgeGate vs score-softmax", y=0.992, fontsize=22)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=2,
        frameon=False,
        fontsize=17,
        handlelength=3.2,
        columnspacing=2.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.4, w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _pct_reduction(baseline: float, value: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return 100.0 * (baseline - value) / baseline


def _write_summary_tables(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    score = df[df["policy"] == "score-softmax"].set_index("feedback_delay")
    agegate = df[df["policy"] == "MAPPO-V4-AgeGate"].set_index("feedback_delay")
    for delay in sorted(set(score.index).intersection(set(agegate.index))):
        s = score.loc[delay]
        a = agegate.loc[delay]
        row: dict[str, float | int | str] = {"feedback_delay": int(delay)}
        for metric, _ in CORE_METRICS + CROWDING_METRICS:
            row[f"score_{metric}"] = float(s[metric])
            row[f"agegate_{metric}"] = float(a[metric])
            row[f"{metric}_reduction_pct"] = _pct_reduction(float(s[metric]), float(a[metric]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "agegate_vs_scoresoftmax_summary_table.csv", index=False)

    markdown_path = out_dir / "AGEGATE_VS_SCORESOFTMAX_SUMMARY.md"
    selected = table[
        [
            "feedback_delay",
            "score_average_latency",
            "agegate_average_latency",
            "average_latency_reduction_pct",
            "score_cvar99",
            "agegate_cvar99",
            "cvar99_reduction_pct",
            "score_queue_oscillation",
            "agegate_queue_oscillation",
            "queue_oscillation_reduction_pct",
            "score_p90_queue_lifetime",
            "agegate_p90_queue_lifetime",
            "p90_queue_lifetime_reduction_pct",
        ]
    ].copy()
    for col in selected.columns:
        if col != "feedback_delay":
            selected[col] = selected[col].map(lambda x: f"{x:.3f}")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# MAPPO-V4-AgeGate vs score-softmax\n\n")
        handle.write(
            "Aligned setting: KAIST, N=30, poisson_bursty workload, task=15MB, bg=0, K=2. "
            "score-softmax is a 5-seed calibration mean; AgeGate is the 10k single-seed run.\n\n"
        )
        handle.write(selected.to_markdown(index=False))
        handle.write("\n\n")
        avg_reductions = table[
            [
                "average_latency_reduction_pct",
                "cvar99_reduction_pct",
                "queue_oscillation_reduction_pct",
                "p90_queue_lifetime_reduction_pct",
                "server_selection_gini_reduction_pct",
                "global_demand_concentration_reduction_pct",
            ]
        ].mean()
        handle.write("## Mean reduction across delays\n\n")
        handle.write(avg_reductions.to_frame("mean_reduction_pct").to_markdown())
        handle.write("\n")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPPO-V4-AgeGate vs score-softmax calibration baseline.")
    parser.add_argument("--agegate-csv", default="results_v4_agegate_10k_compare/v4_vs_agegate_10k_by_delay.csv")
    parser.add_argument(
        "--score-csv",
        default="results_environment_calibration_clean_grid/story_plots/user_delay_task15_bg0_K2.csv",
    )
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--out-dir", default="results_v4_agegate_10k_compare/agegate_vs_scoresoftmax_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = _combined(Path(args.agegate_csv), Path(args.score_csv), args.users)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "agegate_vs_scoresoftmax_key_metrics.csv", index=False)
    _plot_lines(
        df,
        CORE_METRICS,
        out_dir / "agegate_vs_scoresoftmax_core_metrics.png",
        "MAPPO-V4-AgeGate vs score-softmax: performance and queue metrics",
        ncols=2,
    )
    _plot_lines(
        df,
        CROWDING_METRICS,
        out_dir / "agegate_vs_scoresoftmax_crowding_metrics.png",
        "MAPPO-V4-AgeGate vs score-softmax: crowding metrics",
        ncols=2,
    )
    _plot_nine_panel(df, out_dir / "agegate_vs_scoresoftmax_nine_panel.png")
    _write_summary_tables(df, out_dir)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
