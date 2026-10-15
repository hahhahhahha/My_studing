"""Aggregate paper-inspired baselines against MAPPO-V4-AgeGate.

This script is intentionally read-only with respect to trained policies. It
collects already-produced summary CSVs for M2PPO and MCM-FDRL, appends the
existing V4-AgeGate by-delay results, and writes comparison tables/figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("median_queue_lifetime", "Median queue lifetime"),
    ("p90_queue_lifetime", "P90 queue lifetime"),
    ("queue_overflow_rate", "Overflow ratio"),
    ("global_demand_concentration", "Global demand concentration"),
    ("server_selection_gini", "Server selection Gini"),
    ("portfolio_migration_rate", "Portfolio migration rate"),
]

PLOT_COLORS = {
    "MAPPO-V4-AgeGate": "#1f77b4",
    "M2PPO": "#2ca02c",
    "MCM-FDRL": "#d62728",
    "Towards-PPO-DSR": "#9467bd",
    "Towards-PPO-MV": "#8c564b",
    "Fed-MADDPG": "#ff7f0e",
}

PLOT_MARKERS = {
    "MAPPO-V4-AgeGate": "o",
    "M2PPO": "s",
    "MCM-FDRL": "^",
    "Towards-PPO-DSR": "D",
    "Towards-PPO-MV": "P",
    "Fed-MADDPG": "X",
}


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


def _normalize_row(row: dict[str, object], policy: str) -> dict[str, object]:
    out: dict[str, object] = {
        "policy": policy,
        "method": row.get("method", policy),
        "feedback_delay": int(_to_float(row.get("feedback_delay", row.get("delay", 0)))),
        "train_steps": int(_to_float(row.get("train_steps", 0))),
        "test_steps": int(_to_float(row.get("test_steps", 0))),
    }
    for key, _ in METRICS:
        out[key] = _to_float(row.get(key))
    out["inference_time_ms"] = _to_float(row.get("inference_time_ms"))
    out["mean_portfolio_entropy"] = _to_float(row.get("mean_portfolio_entropy"))
    out["portfolio_overlap"] = _to_float(row.get("portfolio_overlap"))
    return out


def _collect_paper_baseline_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.rglob("summary_*.csv")):
        name = summary_path.name
        if name == "summary_M2PPO.csv":
            rows.append(_normalize_row(_read_one_row_csv(summary_path), "M2PPO"))
        elif name == "summary_MCM-FDRL.csv":
            rows.append(_normalize_row(_read_one_row_csv(summary_path), "MCM-FDRL"))
        elif name == "summary_Towards-PPO-DSR.csv":
            rows.append(_normalize_row(_read_one_row_csv(summary_path), "Towards-PPO-DSR"))
        elif name == "summary_Towards-PPO-MV.csv":
            rows.append(_normalize_row(_read_one_row_csv(summary_path), "Towards-PPO-MV"))
        elif name == "summary_Fed-MADDPG.csv":
            rows.append(_normalize_row(_read_one_row_csv(summary_path), "Fed-MADDPG"))
    return rows


def _collect_agegate_rows(agegate_csv: Path) -> list[dict[str, object]]:
    if not agegate_csv.exists():
        return []
    df = pd.read_csv(agegate_csv)
    df = df[df["method"].astype(str).str.contains("agegate", case=False, na=False)]
    return [_normalize_row(row.to_dict(), "MAPPO-V4-AgeGate") for _, row in df.iterrows()]


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    order = {
        "MAPPO-V4-AgeGate": 0,
        "Towards-PPO-DSR": 1,
        "Towards-PPO-MV": 2,
        "M2PPO": 3,
        "MCM-FDRL": 4,
        "Fed-MADDPG": 5,
    }
    df = df.copy()
    df["_policy_order"] = df["policy"].map(order).fillna(99)
    df = df.sort_values(["_policy_order", "feedback_delay"]).drop(columns=["_policy_order"])
    return df


def _plot_metrics(df: pd.DataFrame, out_path: Path) -> None:
    available_metrics = [(key, label) for key, label in METRICS if key in df.columns and df[key].notna().any()]
    fig, axes = plt.subplots(3, 3, figsize=(19, 15.5), constrained_layout=False)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, available_metrics):
        missing_policies = []
        for policy, sub in df.groupby("policy", sort=False):
            sub = sub.sort_values("feedback_delay")
            if sub[metric].notna().sum() == 0:
                missing_policies.append(policy)
                continue
            ax.plot(
                sub["feedback_delay"],
                sub[metric],
                marker=PLOT_MARKERS.get(policy, "o"),
                linewidth=2.4,
                markersize=7,
                color=PLOT_COLORS.get(policy),
                label=policy,
            )
        ax.set_title(label, fontsize=15)
        ax.set_xlabel("feedback delay", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.28)
        if missing_policies:
            ax.text(
                0.98,
                0.04,
                "n/a: " + ", ".join(missing_policies),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color="#555555",
            )
    for ax in axes[len(available_metrics) :]:
        ax.axis("off")
    handles, labels = [], []
    for policy in ["MAPPO-V4-AgeGate", "Towards-PPO-DSR", "Towards-PPO-MV", "M2PPO", "MCM-FDRL", "Fed-MADDPG"]:
        if policy not in set(df["policy"]):
            continue
        handle = plt.Line2D(
            [0],
            [0],
            color=PLOT_COLORS.get(policy),
            marker=PLOT_MARKERS.get(policy, "o"),
            linewidth=2.4,
            markersize=7,
            label=policy,
        )
        handles.append(handle)
        labels.append(policy)
    fig.suptitle("Paper-inspired baselines vs MAPPO-V4-AgeGate", fontsize=21, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        fontsize=12,
        frameon=True,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.885, hspace=0.42, wspace=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mean_table(df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [key for key, _ in METRICS if key in df.columns]
    return (
        df.groupby("policy", as_index=False)[metric_keys]
        .mean(numeric_only=True)
        .sort_values("average_latency")
    )


def _best_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in METRICS:
        if key not in df.columns or not df[key].notna().any():
            continue
        idx = df[key].idxmin()
        row = df.loc[idx]
        rows.append(
            {
                "metric": key,
                "label": label,
                "best_policy": row["policy"],
                "best_delay": int(row["feedback_delay"]),
                "best_value": row[key],
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(df: pd.DataFrame, mean_df: pd.DataFrame, out_path: Path) -> None:
    completed = df.groupby("policy")["feedback_delay"].apply(lambda x: ",".join(str(int(v)) for v in sorted(x))).to_dict()
    agegate = mean_df[mean_df["policy"] == "MAPPO-V4-AgeGate"]
    lines = [
        "# Paper-Inspired Baselines vs MAPPO-V4-AgeGate",
        "",
        "Aligned scenario: KAIST, N=30, poisson_bursty workload, task=15MB, background=0, broadcast K=2, slot=1s.",
        "The paper baselines here are algorithm-core reproductions adapted to the same V4-AgeGate evaluation environment, not full recreations of the original papers' datasets and simulators.",
        "",
        "## Completed Delay Coverage",
        "",
    ]
    for policy, delays in completed.items():
        lines.append(f"- {policy}: delay {{{delays}}}")
    lines.extend(["", "## Mean Across Completed Delays", ""])
    lines.append(mean_df.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Main Reading", ""])
    if not agegate.empty:
        ag = agegate.iloc[0]
        lines.append(
            f"- MAPPO-V4-AgeGate remains the most stable completed policy so far: "
            f"mean latency {ag['average_latency']:.3f}s and CVaR99 {ag['cvar99']:.3f}s."
        )
    for policy in ["Towards-PPO-DSR", "Towards-PPO-MV", "M2PPO", "MCM-FDRL", "Fed-MADDPG"]:
        row = mean_df[mean_df["policy"] == policy]
        if row.empty or agegate.empty:
            continue
        base = agegate.iloc[0]
        cur = row.iloc[0]
        lat_delta = (cur["average_latency"] / base["average_latency"] - 1.0) * 100.0
        cvar_delta = (cur["cvar99"] / base["cvar99"] - 1.0) * 100.0
        qosc_delta = (cur["queue_oscillation"] / base["queue_oscillation"] - 1.0) * 100.0
        lines.append(
            f"- {policy}: latency is {lat_delta:+.1f}%, CVaR99 is {cvar_delta:+.1f}%, "
            f"and queue oscillation is {qosc_delta:+.1f}% relative to V4-AgeGate "
            f"over the currently completed delays."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Towards-PPO-DSR and Towards-PPO-MV are the two original PPO methods from `Towards Risk-Averse Edge Computing With DRL`; both are adapted to the same stale-feedback V4 scenario here.",
            "- M2PPO reflects the MARL Crowd-Edge idea of MAPPO with multiple critics. In this stale-feedback RSU-MEC setting, critic ensembling alone does not address systemic crowding because it does not directly encode feedback age, queue pressure penalty, or portfolio concentration control.",
            "- MCM-FDRL reflects federated DQN-style per-user adjustment. It is much less stable in the current continuous portfolio setting because discrete add/decrease adjustments create sparse, high-variance control over a 35-server portfolio.",
            "- Fed-MADDPG reflects federated actor aggregation with centralized critics. It is expected to help privacy-preserving collaboration, but it does not directly solve stale broadcast trust unless additional age/pressure mechanisms are added.",
            "- V4-AgeGate is better matched to the research problem because it explicitly combines queue-pressure shaping, concentration control, and feedback-age-aware score usage.",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_paper_algorithms_v4scenario")
    parser.add_argument("--agegate-csv", default="results_v4_agegate_10k_compare/v4_vs_agegate_10k_by_delay.csv")
    parser.add_argument("--out-dir", default="results_paper_algorithms_v4scenario")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_agegate_rows(Path(args.agegate_csv))
    rows.extend(_collect_paper_baseline_rows(root))
    if not rows:
        raise SystemExit("No rows found.")

    df = _sort_df(pd.DataFrame(rows))
    full_csv = out_dir / "paper_baselines_vs_v4agegate_by_delay.csv"
    df.to_csv(full_csv, index=False)

    mean_df = _mean_table(df)
    mean_df.to_csv(out_dir / "paper_baselines_vs_v4agegate_mean.csv", index=False)
    _best_by_metric(df).to_csv(out_dir / "paper_baselines_vs_v4agegate_best_by_metric.csv", index=False)
    _plot_metrics(df, out_dir / "paper_baselines_vs_v4agegate_key_metrics.png")
    _write_markdown(df, mean_df, out_dir / "PAPER_BASELINE_V4AGEGATE_COMPARISON.md")

    print(f"Wrote {full_csv}")
    print(f"Wrote {out_dir / 'paper_baselines_vs_v4agegate_key_metrics.png'}")


if __name__ == "__main__":
    main()
