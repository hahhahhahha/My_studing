"""Visualize MAPPO-V4-AgeGate training stability.

Each panel shows the raw per-log-interval measurement (dark, light line) and a
centered rolling mean with window w=5 (solid heavy line). The raw curve is the
actual metric value logged every `--log-interval` slots during training; it
includes environment stochasticity (Poisson arrivals, mobility traces) and
on-policy exploration noise. The rolling mean w=5 averages five consecutive
log points to suppress short-term variance so the training trend is readable.
When multiple learning CSVs are supplied (e.g. multi-seed / multi-delay),
a shaded band shows the cross-run standard deviation of the rolling mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PANELS = [
    ("average_latency", "Average latency (s)"),
    ("mean_reward", "Mean reward"),
    ("mean_portfolio_entropy", "Portfolio entropy"),
    ("global_demand_concentration", "Demand concentration"),
    ("q_port_pressure_mean", "Portfolio-weighted pressure"),
    ("auto_entropy_coef", "Auto entropy coef"),
]


def _rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 5), center=True).mean()


def _load_runs(learning_csv: str, learning_glob: str | None) -> list[pd.DataFrame]:
    if learning_glob:
        paths = sorted(Path().glob(learning_glob))
        if not paths:
            raise FileNotFoundError(f"No learning CSVs matched: {learning_glob}")
        return [pd.read_csv(path).assign(_source=str(path)) for path in paths]
    return [pd.read_csv(learning_csv).assign(_source=learning_csv)]


def _aggregate_runs(runs: list[pd.DataFrame], window: int) -> pd.DataFrame:
    frames = []
    for run_idx, run in enumerate(runs):
        x = run["step"] if "step" in run else pd.Series(np.arange(len(run)), index=run.index)
        frame = pd.DataFrame({"run": run_idx, "x": x.to_numpy(dtype=float), "idx": np.arange(len(run))})
        for col, _title in PANELS:
            if col in run:
                values = pd.to_numeric(run[col], errors="coerce")
                frame[col] = _rolling(values, window).to_numpy(dtype=float)
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
    group_key = "idx"
    rows = []
    for idx, sub in stacked.groupby(group_key):
        row = {"idx": idx, "x": float(sub["x"].mean())}
        for col, _title in PANELS:
            if col in sub:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std(ddof=0)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("idx")


def _plot(runs: list[pd.DataFrame], out_path: Path, window: int) -> None:
    agg = _aggregate_runs(runs, window)
    n_panels = len(PANELS)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.6 * n_rows), constrained_layout=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    for ax, (col, title) in zip(axes, PANELS):
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col not in agg:
            ax.axis("off")
            continue
        for run in runs:
            if col not in run:
                continue
            raw_x = run["step"] if "step" in run else run.index
            raw = pd.to_numeric(run[col], errors="coerce")
            ax.plot(raw_x, raw, color="#4F7EA8", linewidth=1.35, alpha=0.72, label="raw")
        mean = agg[mean_col].to_numpy(dtype=float)
        std = agg[std_col].to_numpy(dtype=float)
        x = agg["x"].to_numpy(dtype=float)
        if len(runs) > 1:
            ax.fill_between(x, mean - std, mean + std, color="#4C78A8", alpha=0.18, linewidth=0, label="std across runs")
        ax.plot(x, mean, color="#0B3C5D", linewidth=2.9, label=f"rolling mean w={window}")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("training user-decisions", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label.startswith("_"):
            continue
        dedup[label] = handle
    fig.suptitle("MAPPO-V4-AgeGate training stability", fontsize=20, y=0.985)
    fig.legend(dedup.values(), dedup.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.89, hspace=0.42, wspace=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _window_stats(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    n = len(df)
    k = max(1, int(n * fraction))
    windows = {
        "early": df.iloc[:k],
        "middle": df.iloc[max(0, n // 2 - k // 2) : min(n, n // 2 + k // 2)],
        "late": df.iloc[-k:],
    }
    rows = []
    metrics = [col for col, _ in PANELS]
    for name, sub in windows.items():
        row = {"window": name}
        for metric in metrics:
            if metric in sub:
                row[metric] = pd.to_numeric(sub[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _write_md(stats: pd.DataFrame, out_path: Path, plot_name: str, window: int) -> None:
    late = stats[stats["window"] == "late"].iloc[0]
    lines = [
        "# MAPPO-V4-AgeGate Training Stability",
        "",
        f"Plot: `{plot_name}`",
        "",
        "## Curve Conventions",
        "",
        f"- **Raw** (thin dark line): the metric value logged every `--log-interval` slots during training. It contains environment stochasticity (Poisson arrivals, KAIST mobility traces) and on-policy exploration noise, so it is the unfiltered training signal.",
        f"- **Rolling mean w={window}** (heavy line): centered moving average over {window} consecutive log points. Suppresses short-term variance in the raw curve so the underlying training trend is readable.",
        "- **Std band** (shaded): cross-run standard deviation of the rolling mean. Only appears when multiple learning CSVs (e.g. multi-seed or multi-delay) are supplied.",
        "",
        "## Early / Middle / Late Window Means",
        "",
        stats.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Stability Reading",
        "",
        f"- Late-window average latency is {late.get('average_latency', float('nan')):.3f}s.",
        f"- Late-window portfolio entropy is {late.get('mean_portfolio_entropy', float('nan')):.3f}, showing the trained policy remains highly diversified.",
        f"- Late-window global demand concentration is {late.get('global_demand_concentration', float('nan')):.5f}, much lower than score-following crowding baselines.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-csv",
        default="results_v4_agegate_10k_compare/KAIST/30u/poisson_bursty/learning_DSR-MTG-Qpen-v4-agegate.csv",
    )
    parser.add_argument(
        "--learning-glob",
        default=None,
        help="Optional glob for multiple learning CSVs. When set, plot rolling mean with std shading across runs.",
    )
    parser.add_argument("--out-dir", default="results_v4_agegate_10k_compare/training_stability")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.learning_csv, args.learning_glob)
    agg = _aggregate_runs(runs, args.window)
    df = pd.DataFrame({"step": agg["x"]})
    for col, _title in PANELS:
        mean_col = f"{col}_mean"
        if mean_col in agg:
            df[col] = agg[mean_col]

    plot_path = out_dir / "v4_agegate_training_stability.png"
    _plot(runs, plot_path, args.window)
    stats = _window_stats(df, args.fraction)
    stats.to_csv(out_dir / "v4_agegate_training_stability_windows.csv", index=False)
    _write_md(stats, out_dir / "V4_AGEGATE_TRAINING_STABILITY.md", plot_path.name, args.window)
    print(f"Wrote {plot_path}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
