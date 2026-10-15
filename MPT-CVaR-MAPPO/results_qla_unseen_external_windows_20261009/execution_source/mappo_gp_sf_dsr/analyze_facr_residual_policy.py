from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_facr_residual_policy_analysis"
FIG = ROOT / "figures"


METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "portfolio_migration_rate",
    "lyapunov_prior_kl_w_to_prior",
    "residual_prior_confidence",
    "residual_actor_amplitude",
    "residual_beta",
    "residual_beta_p10",
    "residual_beta_p50",
    "residual_beta_p90",
    "residual_authority_envelope",
    "residual_radius",
    "residual_proposal_kl_q_to_prior",
    "residual_kl_fresh_to_prior",
    "residual_kl_fresh_to_q",
    "residual_proposal_improvement_ratio",
    "residual_correction_l1",
    "residual_fresh_stale_prior_l1",
    "residual_fresh_proposal_l1",
    "residual_fresh_presmooth_l1",
    "residual_fresh_final_l1",
    "residual_oracle_gap_improvement_ratio",
    "residual_beta_oracle_gap_corr",
    "residual_amplitude_init_bias",
    "residual_amplitude_penalty",
    "residual_aux_proposal_kl_weight",
    "residual_aux_amplitude_weight",
    "residual_beta_penalty",
    "residual_amplitude_mean_update",
    "residual_amplitude_penalty_update",
    "residual_aux_proposal_loss_update",
    "residual_aux_amplitude_loss_update",
    "residual_beta_penalty_update",
    "residual_direction_l1_raw",
    "residual_direction_l1",
    "projection_activation_rate",
]


SPECS = [
    ("FACR-MAPPO", ROOT / "results_facr_ablation_clean" / "FACR-MAPPO"),
    ("Residual-r0.25-rho0.80", ROOT / "results_facr_residual_policy" / "FACR-Residual"),
    ("Residual-r0.10-rho0.90", ROOT / "results_facr_residual_policy_r010_s90" / "FACR-Residual"),
    ("Residual-r0.05-rho0.90", ROOT / "results_facr_residual_policy_r005_s90" / "FACR-Residual"),
    ("Residual-v2-r0.05-rho0.90", ROOT / "results_facr_residual_v2_r005_s90" / "FACR-Residual-v2"),
    (
        "Residual-v2-r0.03-cap0.5-bias-2-pen0.01",
        ROOT / "results_facr_residual_v2_r003_cap05_bias2_pen001_s90" / "FACR-Residual-v2",
    ),
    (
        "Residual-v2-r0.03-cap0.5-bias-2-pen0.01-s95",
        ROOT / "results_facr_residual_v2_r003_cap05_bias2_pen001_s95" / "FACR-Residual-v2",
    ),
    ("Residual-v3-mixed-aux0.01", ROOT / "results_facr_residual_v3_prescreen" / "FACR-Residual-v3"),
]


def read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect() -> list[dict]:
    rows: list[dict] = []
    for label, root in SPECS:
        for seed in [1]:
            for delay in [0, 4, 8]:
                files = sorted(root.glob(f"seed_{seed}/delay_{delay}/KAIST/30u/poisson_bursty/summary_*.csv"))
                if not files:
                    continue
                row = read_summary(files[0])
                item = {
                    "variant": label,
                    "seed": seed,
                    "feedback_delay": delay,
                    "source": str(files[0].relative_to(ROOT)),
                }
                for metric in METRICS:
                    item[metric] = row.get(metric, "")
                rows.append(item)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for variant in sorted({r["variant"] for r in rows}):
        group = [r for r in rows if r["variant"] == variant]
        item = {"variant": variant, "runs": len(group)}
        for metric in METRICS:
            vals = []
            for row in group:
                value = row.get(metric, "")
                if value not in ("", "nan", "NaN"):
                    vals.append(float(value))
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{metric}_mean"] = float(np.mean(arr))
                item[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        out.append(item)
    return out


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [
        "FACR-MAPPO",
        "Residual-r0.25-rho0.80",
        "Residual-r0.10-rho0.90",
        "Residual-r0.05-rho0.90",
        "Residual-v2-r0.05-rho0.90",
        "Residual-v2-r0.03-cap0.5-bias-2-pen0.01",
        "Residual-v2-r0.03-cap0.5-bias-2-pen0.01-s95",
        "Residual-v3-mixed-aux0.01",
    ]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "cvar99", "CVaR99 (s)"),
        (axes[1], "average_latency", "Avg latency (s)"),
        (axes[2], "portfolio_migration_rate", "Migration"),
    ]:
        for variant, color in zip(variants, colors):
            group = [r for r in rows if r["variant"] == variant]
            xs = [int(r["feedback_delay"]) for r in group]
            ys = [float(r[metric]) for r in group]
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", linewidth=2.0, color=color, label=variant)
        ax.set_xlabel("Feedback delay δ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("FACR residual-prior route-B prescreen (seed=1)")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"fig_facr_residual_policy_prescreen.{ext}", dpi=240)
    plt.close(fig)


def main() -> None:
    rows = collect()
    summary = summarize(rows)
    write_csv(OUT / "facr_residual_policy_prescreen_raw.csv", rows)
    write_csv(OUT / "facr_residual_policy_prescreen_summary.csv", summary)
    plot(rows)
    print(OUT / "facr_residual_policy_prescreen_summary.csv")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
