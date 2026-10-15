from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "task10_bg0_homo": "Task 10, bg 0",
    "bg02_task15_homo": "Task 15, bg 0.2",
    "task15_bg0_heterogeneous": "Task 15, hetero",
    "task20_bg0_homo": "Task 20, bg 0",
    "bg04_task15_homo": "Task 15, bg 0.4",
}

SCENARIO_ORDER = [
    "task10_bg0_homo",
    "bg02_task15_homo",
    "task15_bg0_heterogeneous",
    "task20_bg0_homo",
    "bg04_task15_homo",
]

METRICS = [
    ("average_latency_mean", "Average latency", "s", "{:.2f}"),
    ("cvar99_mean", "CVaR99", "s", "{:.2f}"),
    ("queue_oscillation_mean", "Queue oscillation", "", "{:.2f}"),
    ("p90_queue_lifetime_mean", "P90 queue lifetime", "slots", "{:.0f}"),
    ("queue_overflow_rate_mean", "Overflow ratio", "", "{:.2f}"),
    ("global_demand_concentration_mean", "Global concentration", "", "{:.4f}"),
]


def _load_by_delay(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = set(SCENARIO_ORDER)
    df = df[df["scenario"].isin(needed)].copy()
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["scenario_rank"] = df["scenario"].map({name: i for i, name in enumerate(SCENARIO_ORDER)})
    return df.sort_values(["scenario_rank", "feedback_delay"])


def _nature_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def plot_dumbbell(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _nature_style()
    color0 = "#4C78A8"
    color8 = "#D95F02"
    line_color = "#B8B8B8"

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=False)
    axes = axes.ravel()
    y = np.arange(len(SCENARIO_ORDER))
    y_labels = [SCENARIO_LABELS[name] for name in SCENARIO_ORDER]

    for ax, (metric, title, unit, fmt) in zip(axes, METRICS):
        sub0 = df[df["feedback_delay"] == 0].set_index("scenario").reindex(SCENARIO_ORDER)
        sub8 = df[df["feedback_delay"] == 8].set_index("scenario").reindex(SCENARIO_ORDER)
        x0 = sub0[metric].to_numpy(dtype=float)
        x8 = sub8[metric].to_numpy(dtype=float)
        for yi, a, b in zip(y, x0, x8):
            ax.plot([a, b], [yi, yi], color=line_color, linewidth=1.4, zorder=1)
        ax.scatter(x0, y, s=44, facecolor="white", edgecolor=color0, linewidth=1.6, label=r"$\delta=0$", zorder=3)
        ax.scatter(x8, y, s=44, facecolor=color8, edgecolor="white", linewidth=0.7, label=r"$\delta=8$", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels if ax in axes[::3] else [])
        ax.invert_yaxis()
        suffix = f" ({unit})" if unit else ""
        ax.set_xlabel(title + suffix)
        ax.set_title(title, pad=7)
        xmin = float(np.nanmin([np.nanmin(x0), np.nanmin(x8)]))
        xmax = float(np.nanmax([np.nanmax(x0), np.nanmax(x8)]))
        span = max(xmax - xmin, 1e-9)
        ax.set_xlim(xmin - 0.08 * span, xmax + 0.16 * span)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.986))
    fig.suptitle("Tuned V4-AgeGate robustness under fresh vs stale feedback", fontsize=14, y=1.02)
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.085, top=0.875, wspace=0.38, hspace=0.58)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _nature_style()
    rows = []
    for scenario in SCENARIO_ORDER:
        sub = df[df["scenario"] == scenario].set_index("feedback_delay")
        if 0 not in sub.index or 8 not in sub.index:
            continue
        item = {"scenario": SCENARIO_LABELS[scenario]}
        for metric, title, _unit, _fmt in METRICS:
            base = float(sub.loc[0, metric])
            stale = float(sub.loc[8, metric])
            item[title] = 100.0 * (stale - base) / (abs(base) + 1e-12)
        rows.append(item)
    mat_df = pd.DataFrame(rows).set_index("scenario")
    matrix = mat_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    vmax = float(np.nanmax(np.abs(matrix)))
    vmax = max(vmax, 1.0)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(mat_df.shape[1]), labels=mat_df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat_df.shape[0]), labels=mat_df.index)
    for i in range(mat_df.shape[0]):
        for j in range(mat_df.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=8.5, color="#222222")
    ax.set_title(r"Relative change from $\delta=0$ to $\delta=8$", pad=10, fontsize=13)
    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("Change (%)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-delay", default="results_tuned_v4_robustness/tuned_v4_robustness_by_delay.csv")
    parser.add_argument("--out-dir", default="results_tuned_v4_robustness")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_by_delay(Path(args.by_delay))
    plot_dumbbell(df, out_dir / "tuned_v4_robustness_key_metrics.png")
    plot_dumbbell(df, out_dir / "tuned_v4_robustness_dumbbell_nature.png")
    plot_delta_heatmap(df, out_dir / "tuned_v4_robustness_delta_heatmap_nature.png")
    print(f"Wrote {out_dir / 'tuned_v4_robustness_key_metrics.png'}")
    print(f"Wrote {out_dir / 'tuned_v4_robustness_delta_heatmap_nature.png'}")


if __name__ == "__main__":
    main()
