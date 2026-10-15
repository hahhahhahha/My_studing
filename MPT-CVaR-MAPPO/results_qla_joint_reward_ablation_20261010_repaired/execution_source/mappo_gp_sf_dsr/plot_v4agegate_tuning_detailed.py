from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE1_METRICS = [
    ("average_latency", "Avg latency"),
    ("cvar99", "CVaR99"),
    ("queue_oscillation", "Queue osc."),
    ("global_demand_concentration", "Demand conc."),
    ("p90_queue_lifetime", "P90 queue life"),
]


CONFIRM_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("global_demand_concentration", "Global demand concentration"),
]


DESIGN_ROWS = [
    ("baseline", "default", "alpha=.30, q=.05, kappa=.01, H*=.95, eta=.25"),
    ("alpha020", "alpha", ".30 -> .20"),
    ("qpen003", "q_penalty", ".05 -> .03"),
    ("qpen010", "q_penalty", ".05 -> .10"),
    ("conc005", "kappa_i/g", ".01 -> .005"),
    ("entropy090", "target entropy", ".95 -> .90"),
    ("gateeta040", "age gate eta", ".25 -> .40"),
]


def stage1_figure(stage_dir: Path, out_path: Path) -> None:
    ranking = pd.read_csv(stage_dir / "stage1_candidate_ranking.csv")
    scored = pd.read_csv(stage_dir / "stage1_scored_by_delay.csv")
    ranking = ranking.sort_values("score_mean", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], wspace=0.28, hspace=0.36)
    ax_table = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_delay = fig.add_subplot(gs[1, 1])

    # Control-variable design table.
    ax_table.axis("off")
    table = ax_table.table(
        cellText=[list(row) for row in DESIGN_ROWS],
        colLabels=["Candidate", "Changed knob", "Control-variable change"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.24, 0.54],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.42)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2166ac")
        elif DESIGN_ROWS[row - 1][0] == "gateeta040":
            cell.set_facecolor("#d1e5f0")
        elif DESIGN_ROWS[row - 1][0] == "baseline":
            cell.set_facecolor("#f7f7f7")
    ax_table.set_title("Stage-1 Control-Variable Design", fontsize=15, pad=12)

    # Composite ranking.
    colors = ["#2166ac" if c == "gateeta040" else "#92c5de" for c in ranking["candidate"]]
    ax_rank.barh(ranking["candidate"], ranking["score_mean"], color=colors, edgecolor="#404040", linewidth=0.4)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Weighted composite score (lower is better)", fontsize=12)
    ax_rank.set_title("Stage-1 Candidate Ranking", fontsize=15, pad=12)
    ax_rank.grid(axis="x", alpha=0.25)
    for i, row in ranking.iterrows():
        ax_rank.text(row["score_mean"] + 0.015, i, f"{row['score_mean']:.3f}", va="center", fontsize=10)

    # Relative metric heatmap vs baseline.
    baseline = ranking[ranking["candidate"] == "baseline"].iloc[0]
    heat_rows = ranking["candidate"].tolist()
    rel = []
    for _, row in ranking.iterrows():
        vals = []
        for metric, _ in STAGE1_METRICS:
            b = float(baseline[metric])
            vals.append((float(row[metric]) / b - 1.0) * 100.0 if abs(b) > 1e-12 else np.nan)
        rel.append(vals)
    rel_arr = np.asarray(rel, dtype=float)
    vmax = float(np.nanmax(np.abs(rel_arr)))
    im = ax_heat.imshow(rel_arr, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(np.arange(len(STAGE1_METRICS)), [label for _, label in STAGE1_METRICS], rotation=25, ha="right", fontsize=10.5)
    ax_heat.set_yticks(np.arange(len(heat_rows)), heat_rows, fontsize=10.5)
    ax_heat.set_title("Mean Metric Change vs Baseline (%)", fontsize=15, pad=12)
    for i in range(rel_arr.shape[0]):
        for j in range(rel_arr.shape[1]):
            ax_heat.text(j, i, f"{rel_arr[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.set_label("Relative change (%)", fontsize=10.5)

    # Per-delay composite score.
    pivot = scored.pivot(index="candidate", columns="feedback_delay", values="score").reindex(heat_rows)
    x = np.arange(len(heat_rows))
    width = 0.34
    ax_delay.bar(x - width / 2, pivot[0], width=width, label="delay=0", color="#67a9cf", edgecolor="#404040", linewidth=0.3)
    ax_delay.bar(x + width / 2, pivot[8], width=width, label="delay=8", color="#ef8a62", edgecolor="#404040", linewidth=0.3)
    ax_delay.set_xticks(x, heat_rows, rotation=30, ha="right", fontsize=10.5)
    ax_delay.set_ylabel("Composite score", fontsize=12)
    ax_delay.set_title("Fresh vs Stale Feedback Screening Scores", fontsize=15, pad=12)
    ax_delay.legend(fontsize=10.5, frameon=True)
    ax_delay.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "V4-AgeGate Hyperparameter Tuning: Stage-1 Screening\n"
        "delay={0,8}, train_steps=2000, test_steps=300; one knob changed per candidate",
        fontsize=17,
        y=0.995,
    )
    fig.subplots_adjust(top=0.88, left=0.06, right=0.985, bottom=0.08)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def stage2_figure(confirm_dir: Path, out_path: Path) -> None:
    df = pd.read_csv(confirm_dir / "v4_agegate_eta_tuning_full_delay.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.ravel()
    colors = {"baseline_eta025": "#2166ac", "gateeta040": "#b2182b"}
    labels = {"baseline_eta025": "Default eta=0.25", "gateeta040": "Stage-1 best eta=0.40"}
    for ax, (metric, ylabel) in zip(axes, CONFIRM_METRICS):
        for policy in ["baseline_eta025", "gateeta040"]:
            sub = df[df["policy"] == policy].sort_values("feedback_delay")
            ax.plot(
                sub["feedback_delay"],
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=colors[policy],
                label=labels[policy],
            )
        ax.set_xlabel("Feedback delay", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=14, pad=10)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=10.5)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Stage-2 Confirmation: Full Delay Grid, 10k Training\n"
        "Short-run winner eta=0.40 is checked against the default eta=0.25",
        fontsize=17,
        y=0.995,
    )
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.08, hspace=0.36, wspace=0.25)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", default="results_v4_agegate_tuning_stage1")
    parser.add_argument("--confirm-dir", default="results_v4_agegate_tuning_confirm")
    args = parser.parse_args()
    stage1_dir = Path(args.stage1_dir)
    confirm_dir = Path(args.confirm_dir)
    stage1_figure(stage1_dir, stage1_dir / "v4_agegate_stage1_control_variable_visualization.png")
    stage2_figure(confirm_dir, confirm_dir / "v4_agegate_stage2_confirmation_visualization.png")
    print(f"Wrote {stage1_dir / 'v4_agegate_stage1_control_variable_visualization.png'}")
    print(f"Wrote {confirm_dir / 'v4_agegate_stage2_confirmation_visualization.png'}")


if __name__ == "__main__":
    main()
