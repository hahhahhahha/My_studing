from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def make_training_plots(out_dir: Path, method: str) -> None:
    """Create learning curve, latency PDF, and entropy-latency plots.

    Plotting is best-effort so headless systems without matplotlib can still run
    training and write all numeric CSV outputs.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    method_file = method.replace("/", "_")
    learning_path = out_dir / f"learning_{method_file}.csv"
    latency_path = out_dir / f"latency_samples_{method_file}.csv"
    if learning_path.exists():
        rows = _read_csv(learning_path)
        step = _column(rows, "step")
        reward = _column(rows, "mean_reward")
        latency = _column(rows, "average_latency")
        entropy = _column(rows, "mean_portfolio_entropy")
        concentration = _column(rows, "mean_portfolio_concentration")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(step, reward, label="mean reward", color="#1f77b4")
        axes[0].set_ylabel("reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(step, latency, label="average latency", color="#ff7f0e")
        axes[1].set_ylabel("latency (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[2].plot(step, entropy, label="entropy", color="#2ca02c")
        axes[2].plot(step, concentration, label="concentration", color="#d62728")
        axes[2].set_xlabel("training user-decisions")
        axes[2].set_ylabel("portfolio")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"learning_{method_file}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(entropy, latency, s=12, alpha=0.7)
        ax.set_xlabel("mean portfolio entropy")
        ax.set_ylabel("average latency (s)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"entropy_vs_latency_{method_file}.png", dpi=180)
        plt.close(fig)

    if latency_path.exists():
        rows = [row for row in _read_csv(latency_path) if row.get("phase") == "test"]
        latencies = _column(rows, "latency")
        latencies = latencies[np.isfinite(latencies)]
        if latencies.size:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(latencies, bins=40, density=True, alpha=0.75, color="#1f77b4")
            ax.set_xlabel("latency (s)")
            ax.set_ylabel("PDF")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"latency_pdf_{method_file}.png", dpi=180)
            plt.close(fig)
