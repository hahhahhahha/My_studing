"""Aggregate delayed-information baselines and compare them with QLA-MAPPO.

The unit of random replication is the training seed; feedback delay is treated as
an experimental condition.  The script therefore reports per-delay paired
differences and a seed-block bootstrap over the five-dimensional delay vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "average_latency": "Avg latency (s)",
    "cvar99": "CVaR99 (s)",
    "queue_oscillation": "Queue oscillation",
    "portfolio_migration_rate": "Migration",
}


def _read_grid(root: Path, method_dir: str, summary_name: str, method: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((root / method_dir).glob(f"seed_*/delay_*/KAIST/30u/poisson_bursty/{summary_name}")):
        seed = int(next(p for p in path.parts if p.startswith("seed_")).split("_")[1])
        delay = int(next(p for p in path.parts if p.startswith("delay_")).split("_")[1])
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one summary row in {path}, got {len(frame)}")
        row = frame.iloc[0].to_dict()
        row.update(method=method, seed=seed, feedback_delay=delay, source_path=str(path))
        rows.append(row)
    out = pd.DataFrame(rows)
    expected = {(s, d) for s in range(1, 6) for d in (0, 1, 2, 4, 8)}
    actual = set(zip(out.seed, out.feedback_delay)) if not out.empty else set()
    if actual != expected:
        raise ValueError(f"Incomplete {method} grid: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return out


def _seed_block_ci(diff: pd.DataFrame, metric: str, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    pivot = diff.pivot(index="seed", columns="feedback_delay", values=metric).sort_index()
    values = pivot.to_numpy(dtype=float)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = values[rng.integers(0, values.shape[0], size=values.shape[0]), :]
        boot[b] = sampled.mean()
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _annotate_bars(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jseq-root", type=Path, default=Path("results_delayed_information_baselines_formal"))
    parser.add_argument("--qla-root", type=Path, default=Path("results_qla_simplified_rerun/main"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_delayed_information_baselines_formal/analysis"))
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jseq = _read_grid(args.jseq_root, "JSEQ-A", "summary_JSEQ-A.csv", "JSEQ-A")
    qla = _read_grid(args.qla_root, "QLA-MAPPO-NoMargin", "summary_LCC-MAPPO.csv", "QLA-MAPPO")
    raw = pd.concat([qla, jseq], ignore_index=True)
    raw.to_csv(args.output_dir / "qla_jseq_common_grid_raw.csv", index=False)

    summary_rows = []
    for method, frame in raw.groupby("method", sort=False):
        row = {"method": method, "n_cells": len(frame)}
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "qla_jseq_summary_mean_std.csv", index=False)

    by_delay = raw.groupby(["method", "feedback_delay"], as_index=False)[list(METRICS)].agg(["mean", "std"])
    by_delay.columns = ["_".join(c).rstrip("_") for c in by_delay.columns]
    by_delay.to_csv(args.output_dir / "qla_jseq_by_delay.csv", index=False)

    merged = qla.merge(jseq, on=["seed", "feedback_delay"], suffixes=("_qla", "_jseq"))
    diff = merged[["seed", "feedback_delay"]].copy()
    for metric in METRICS:
        diff[metric] = merged[f"{metric}_qla"] - merged[f"{metric}_jseq"]
    diff.to_csv(args.output_dir / "qla_minus_jseq_paired_raw.csv", index=False)

    rng = np.random.default_rng(20260726)
    ci_rows = []
    for metric in METRICS:
        mean, low, high = _seed_block_ci(diff, metric, rng, args.bootstrap)
        ci_rows.append({"metric": metric, "paired_mean_qla_minus_jseq": mean, "ci95_low": low, "ci95_high": high})
        for delay, frame in diff.groupby("feedback_delay"):
            values = frame[metric].to_numpy(dtype=float)
            boot = np.mean(values[rng.integers(0, len(values), size=(args.bootstrap, len(values)))], axis=1)
            ci_rows.append({"metric": metric, "feedback_delay": delay,
                            "paired_mean_qla_minus_jseq": values.mean(),
                            "ci95_low": np.quantile(boot, 0.025), "ci95_high": np.quantile(boot, 0.975)})
    pd.DataFrame(ci_rows).to_csv(args.output_dir / "qla_vs_jseq_paired_bootstrap.csv", index=False)

    colors = {"QLA-MAPPO": "#1f77b4", "JSEQ-A": "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), METRICS.items()):
        means = [raw.loc[raw.method == m, metric].mean() for m in colors]
        stds = [raw.loc[raw.method == m, metric].std(ddof=1) for m in colors]
        bars = ax.bar(list(colors), means, yerr=stds, capsize=4, color=list(colors.values()), width=0.62)
        _annotate_bars(ax, bars, "{:.3f}" if metric != "portfolio_migration_rate" else "{:.4f}")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Delayed-information mechanism baseline on the common 5×5 grid", fontsize=13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"fig_qla_vs_jseq_metrics.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for ax, metric in zip(axes, ("average_latency", "cvar99")):
        for method, color in colors.items():
            frame = raw[raw.method == method].groupby("feedback_delay")[metric].mean().reindex([0, 1, 2, 4, 8])
            ax.plot(frame.index, frame.values, marker="o", linewidth=2, color=color, label=method)
            for x, y in frame.items():
                ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xlabel("Feedback age δ")
        ax.set_ylabel(METRICS[metric])
        ax.set_xticks([0, 1, 2, 4, 8])
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Delay-conditioned performance (no uncertainty shading)", fontsize=13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output_dir / f"fig_qla_vs_jseq_by_delay.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # MFC-DI-A is a training-based candidate and is intentionally screened on
    # one seed and three representative delays before any 5×5 commitment.
    mfc_root = Path("results_delayed_information_baselines_prescreen/MFC-DI-A")
    mfc_rows = []
    for path in sorted(mfc_root.glob("seed_1/delay_*/KAIST/30u/poisson_bursty/summary_MFC-DI-A.csv")):
        delay = int(next(p for p in path.parts if p.startswith("delay_")).split("_")[1])
        if delay not in (0, 4, 8):
            continue
        record = pd.read_csv(path).iloc[0].to_dict()
        record.update(method="MFC-DI-A", seed=1, feedback_delay=delay, source_path=str(path))
        mfc_rows.append(record)
    if len(mfc_rows) == 3:
        mfc = pd.DataFrame(mfc_rows)
        qla_screen = qla[(qla.seed == 1) & qla.feedback_delay.isin([0, 4, 8])].copy()
        screen = pd.concat([qla_screen, mfc], ignore_index=True)
        screen.to_csv(args.output_dir / "qla_mfc_prescreen_raw.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
        for ax, metric in zip(axes, ("average_latency", "cvar99")):
            for method, color, marker in (("QLA-MAPPO", "#1f77b4", "o"), ("MFC-DI-A", "#9467bd", "s")):
                frame = screen[screen.method == method].sort_values("feedback_delay")
                ax.plot(frame.feedback_delay, frame[metric], marker=marker, linewidth=2, color=color, label=method)
                for x, y in zip(frame.feedback_delay, frame[metric]):
                    ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xlabel("Feedback age δ")
            ax.set_ylabel(METRICS[metric])
            ax.set_xticks([0, 4, 8])
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle("MFC-DI-A one-seed structural prescreen (not a formal result)", fontsize=12)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(args.output_dir / f"fig_mfc_di_prescreen.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
