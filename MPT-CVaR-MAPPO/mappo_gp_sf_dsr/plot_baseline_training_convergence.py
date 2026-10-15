from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "V4-AgeGate",
    "Towards-PPO-DSR",
    "Towards-PPO-MV",
]

COLORS = {
    "V4-AgeGate": "#1B9E77",
    "Towards-PPO-DSR": "#7570B3",
    "Towards-PPO-MV": "#8C564B",
    "M2PPO": "#2CA02C",
    "MCM-FDRL": "#D62728",
}

MARKERS = {
    "V4-AgeGate": "o",
    "Towards-PPO-DSR": "D",
    "Towards-PPO-MV": "P",
    "M2PPO": "s",
    "MCM-FDRL": "^",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def find_learning_csv(root: Path, method: str, seed: int, delay: int) -> Path | None:
    base = root / method / f"seed_{seed}" / f"delay_{delay}"
    if not base.exists():
        return None
    files = sorted(base.rglob("learning_*.csv"))
    return files[0] if files else None


def read_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"step", "average_latency", "mean_reward"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["step", "average_latency", "mean_reward"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("step")
    df = df.drop_duplicates("step", keep="last")
    return df


def interpolate_run(df: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(df) < 2:
        x = np.array([0.0, 1.0])
        lat = np.repeat(float(df["average_latency"].iloc[0]), 2)
        rew = np.repeat(float(df["mean_reward"].iloc[0]), 2)
    else:
        steps = df["step"].to_numpy(dtype=np.float64)
        span = max(float(steps[-1] - steps[0]), 1.0)
        x = (steps - steps[0]) / span
        lat = df["average_latency"].to_numpy(dtype=np.float64)
        rew = df["mean_reward"].to_numpy(dtype=np.float64)
    return np.interp(grid, x, lat), np.interp(grid, x, rew)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def collect_curves(
    root: Path,
    delay: int,
    seeds: list[int],
    points: int,
    smooth_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = np.linspace(0.0, 1.0, points)
    rows = []
    summary_rows = []

    for method in METHOD_ORDER:
        lat_runs: list[np.ndarray] = []
        rew_runs: list[np.ndarray] = []
        found: list[int] = []
        for seed in seeds:
            path = find_learning_csv(root, method, seed, delay)
            if path is None:
                continue
            try:
                df = read_curve(path)
            except ValueError:
                continue
            lat, rew = interpolate_run(df, grid)
            lat_runs.append(lat)
            rew_runs.append(rew)
            found.append(seed)

        if not lat_runs:
            continue

        lat_arr = np.vstack(lat_runs)
        rew_arr = np.vstack(rew_runs)
        lat_mean = smooth(lat_arr.mean(axis=0), smooth_window)
        rew_mean = smooth(rew_arr.mean(axis=0), smooth_window)
        rew_norm = minmax(rew_mean)

        for progress, lat, rew, rnorm in zip(grid, lat_mean, rew_mean, rew_norm):
            rows.append(
                {
                    "delay": delay,
                    "method": method,
                    "progress": progress,
                    "average_latency": lat,
                    "mean_reward": rew,
                    "normalized_reward": rnorm,
                    "n_seeds": len(found),
                }
            )

        tail_n = max(1, int(0.2 * len(grid)))
        summary_rows.append(
            {
                "delay": delay,
                "method": method,
                "n_seeds": len(found),
                "seeds": ",".join(map(str, found)),
                "initial_latency": float(lat_mean[0]),
                "final_latency": float(lat_mean[-1]),
                "best_latency": float(np.nanmin(lat_mean)),
                "tail_mean_latency": float(np.nanmean(lat_mean[-tail_n:])),
                "initial_reward": float(rew_mean[0]),
                "final_reward": float(rew_mean[-1]),
                "tail_mean_reward": float(np.nanmean(rew_mean[-tail_n:])),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def plot_delay(curves: pd.DataFrame, delay: int, out_dir: Path) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    for method in METHOD_ORDER:
        sub = curves[(curves["delay"] == delay) & (curves["method"] == method)]
        if sub.empty:
            continue
        x = sub["progress"].to_numpy() * 100.0
        axes[0].plot(
            x,
            sub["average_latency"],
            label=method,
            color=COLORS.get(method),
            marker=MARKERS.get(method, "o"),
            markevery=max(1, len(sub) // 5),
            linewidth=2.0,
            markersize=4.2,
        )
        axes[1].plot(
            x,
            sub["normalized_reward"],
            label=method,
            color=COLORS.get(method),
            marker=MARKERS.get(method, "o"),
            markevery=max(1, len(sub) // 5),
            linewidth=2.0,
            markersize=4.2,
        )

    axes[0].set_title("(a) Training latency")
    axes[0].set_xlabel("Training progress (%)")
    axes[0].set_ylabel("Average latency per task (s)")
    axes[0].grid(True)

    axes[1].set_title("(b) Method-normalized reward")
    axes[1].set_xlabel("Training progress (%)")
    axes[1].set_ylabel("Normalized reward")
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Baseline training convergence under feedback delay $\\delta={delay}$", y=1.18)
    fig.savefig(out_dir / f"baseline_training_convergence_delay{delay}.png", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, delays: list[int], out_dir: Path) -> None:
    report = out_dir / "BASELINE_TRAINING_CONVERGENCE.md"
    lines = [
        "# Baseline Training Convergence",
        "",
        "This report summarizes multi-baseline learning curves generated from existing `learning_*.csv` logs.",
        "No additional training was run.",
        "",
        "The latency panel is directly comparable across methods. The reward panel is normalized within each method because V4-AgeGate, PPO-DSR, PPO-MV, M2PPO and MCM-FDRL use different reward definitions.",
        "",
        "## Outputs",
        "",
    ]
    for delay in delays:
        lines.append(f"- `baseline_training_convergence_delay{delay}.png`")
    lines.extend(
        [
            "- `baseline_training_convergence_curves.csv`",
            "- `baseline_training_convergence_summary.csv`",
            "",
            "## Tail-Window Summary",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Interpretation",
            "",
            "These curves are intended to play the same diagnostic role as the learning curves in the Towards paper: they verify that each reproduced model has a visible training trajectory and reaches a late-stage regime before final evaluation. They should not be used as a direct reward-ranking plot because reward scales differ across algorithms.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_paper_algorithms_v4scenario_multiseed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_paper_ready_outputs/training_convergence"))
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--smooth-window", type=int, default=9)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_curves = []
    all_summaries = []
    for delay in args.delays:
        curves, summary = collect_curves(args.root, delay, args.seeds, args.points, args.smooth_window)
        if curves.empty:
            print(f"[warn] no learning curves found for delay={delay}")
            continue
        all_curves.append(curves)
        all_summaries.append(summary)
        plot_delay(curves, delay, args.out_dir)
        print(f"[done] plotted delay={delay}: {args.out_dir / f'baseline_training_convergence_delay{delay}.png'}")

    if not all_curves:
        raise SystemExit("No convergence data found.")

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    curves_df.to_csv(args.out_dir / "baseline_training_convergence_curves.csv", index=False)
    summary_df.to_csv(args.out_dir / "baseline_training_convergence_summary.csv", index=False)
    write_report(summary_df, args.delays, args.out_dir)
    print(f"[done] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
