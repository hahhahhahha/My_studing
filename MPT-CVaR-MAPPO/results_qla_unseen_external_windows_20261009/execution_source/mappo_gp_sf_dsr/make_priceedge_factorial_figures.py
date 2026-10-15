from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def effect_figure(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "factorial_effects.csv")
    metrics = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
    labels = ["Avg latency", "CVaR99", "Queue osc.", "Migration"]
    terms = list(dict.fromkeys(df["term"].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        sub = df[df["metric"].eq(metric)].set_index("term").reindex(terms).reset_index()
        y = np.arange(len(sub))
        value = sub["standardized_effect_response_sd"].to_numpy(float)
        bars = ax.barh(y, value, color=[COLORS[i % len(COLORS)] for i in range(len(sub))])
        ax.axvline(0, color="0.35", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["term"])
        ax.invert_yaxis()
        ax.set_xlabel("Standardized high − low effect")
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.2)
        for bar, val in zip(bars, value):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=6.5)
    fig.suptitle(r"$2^3$ factorial main and interaction effects", y=1.01)
    fig.tight_layout()
    save(fig, out, "fig_factorial_standardized_effects")


def interaction_figure(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "factorial_raw.csv")
    corners = df[~df["is_center"].astype(str).str.lower().isin(["true", "1"])].copy()
    metrics = ["average_latency", "cvar99"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, metric, title in zip(axes, metrics, ["Average latency (s)", "CVaR99 (s)"]):
        for index, b in enumerate(sorted(corners["target_portfolio_entropy"].astype(float).unique())):
            sub = corners[np.isclose(corners["target_portfolio_entropy"].astype(float), b)]
            grouped = sub.groupby("lambda_pi")[metric]
            x = np.asarray(sorted(grouped.groups), dtype=float)
            mean = np.asarray([grouped.get_group(v).mean() for v in x])
            sd = np.asarray([grouped.get_group(v).std(ddof=1) for v in x])
            ax.errorbar(x, mean, yerr=sd, marker=["o", "s"][index], capsize=3, lw=1.6, color=COLORS[index], label=rf"$H_{{target}}={b:g}$")
        ax.set_xlabel(r"Price-logit bias $\lambda_\pi$")
        ax.set_ylabel(title)
        ax.set_xticks(sorted(corners["lambda_pi"].astype(float).unique()))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(r"Priority interaction: $\lambda_\pi \times H_{target}$", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_factorial_AB_interaction")


def c_by_delay_figure(root: Path, out: Path) -> None:
    df = pd.read_csv(root / "factorial_C_by_delay.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, metric, title in zip(axes, ["average_latency", "cvar99"], ["Average latency effect (s)", "CVaR99 effect (s)"]):
        sub = df[df["metric"].eq(metric)].sort_values("delay")
        x = sub["delay"].to_numpy(float)
        y = sub["C_high_minus_low"].to_numpy(float)
        low = sub["seed_block_bootstrap95_low"].to_numpy(float)
        high = sub["seed_block_bootstrap95_high"].to_numpy(float)
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=3, color=COLORS[2], lw=1.7)
        ax.axhline(0, color="0.35", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xlabel(r"Feedback delay $\delta$")
        ax.set_ylabel(title)
        ax.grid(alpha=0.2)
    c_name = str(pd.read_csv(root / "factorial_raw.csv")["factor_C_name"].dropna().iloc[0])
    fig.suptitle(f"Factor C ({c_name}) by feedback delay", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig_factorial_C_by_delay")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results_priceedge_qtbias_factorial_lambda_entropy_v")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.is_absolute():
        root = ROOT / root
    out = Path(args.output_dir) if args.output_dir else root / "figures"
    if not out.is_absolute():
        out = ROOT / out
    set_style()
    effect_figure(root, out)
    interaction_figure(root, out)
    c_by_delay_figure(root, out)
    print(f"Wrote factorial figures to {out}")


if __name__ == "__main__":
    main()
