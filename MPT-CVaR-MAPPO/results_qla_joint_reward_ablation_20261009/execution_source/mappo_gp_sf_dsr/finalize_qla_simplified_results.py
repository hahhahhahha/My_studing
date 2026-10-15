"""Finalize simplified QLA evidence only after every planned cell is complete."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_qla_simplified_rerun"
OUT = RESULTS / "final_analysis"
SEEDS = [1, 2, 3, 4, 5]
DELAYS = [0, 1, 2, 4, 8]
RNG = np.random.default_rng(20260726)
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
EXPECTED = {"main": 25, "ablation": 75, "n40": 10, "crossenv": 48}
EXPECTED_VARIANTS = {
    "main": {"QLA-MAPPO-NoMargin"},
    "ablation": {"QL-NoBias-NoMargin", "QL-Bias-Shuffle-NoMargin", "Flat-Bias-QL-NoMargin"},
    "n40": {"QLA-MAPPO-NoMargin"},
    "crossenv": {"QLA-MAPPO-NoMargin"},
}

# Per-variant structural config. Every cell of a given variant must match.
# These fields exist precisely so that a future Flat-no-op regression (both
# edgewise and flat actor flags true) or a stale margin / Z queue leak is
# caught at finalize time rather than leaking into the paper.
EXPECTED_VARIANT_CONFIG = {
    "QLA-MAPPO-NoMargin": {
        "lcc_edgewise_price_actor": True,
        "lcc_flat_price_bias_actor": False,
        "lcc_price_shuffle": False,
        "lcc_price_logit_bias": 0.75,
        "lcc_price_components": "QT",
        "lcc_no_stale_margin": True,
        "lcc_no_z_queue": True,
    },
    "QL-NoBias-NoMargin": {
        "lcc_edgewise_price_actor": True,
        "lcc_flat_price_bias_actor": False,
        "lcc_price_shuffle": False,
        "lcc_price_logit_bias": 0.0,
        "lcc_price_components": "QT",
        "lcc_no_stale_margin": True,
        "lcc_no_z_queue": True,
    },
    "QL-Bias-Shuffle-NoMargin": {
        "lcc_edgewise_price_actor": True,
        "lcc_flat_price_bias_actor": False,
        "lcc_price_shuffle": True,
        "lcc_price_logit_bias": 0.75,
        "lcc_price_components": "QT",
        "lcc_no_stale_margin": True,
        "lcc_no_z_queue": True,
    },
    "Flat-Bias-QL-NoMargin": {
        "lcc_edgewise_price_actor": False,
        "lcc_flat_price_bias_actor": True,
        "lcc_price_shuffle": False,
        "lcc_price_logit_bias": 0.75,
        "lcc_price_components": "QT",
        "lcc_no_stale_margin": True,
        "lcc_no_z_queue": True,
    },
}
STRUCTURAL_FIELDS = [
    "lcc_edgewise_price_actor",
    "lcc_flat_price_bias_actor",
    "lcc_price_shuffle",
    "lcc_price_logit_bias",
    "lcc_price_components",
    "lcc_no_stale_margin",
    "lcc_no_z_queue",
]


def read_cell(summary_path: Path, phase: str) -> dict[str, object]:
    record = pd.read_csv(summary_path).iloc[0].to_dict()
    config_path = summary_path.with_name("config_LCC-MAPPO.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts = summary_path.parts
    phase_index = parts.index(phase)
    variant = parts[phase_index + 1]
    record.update({
        "phase": phase,
        "variant": variant,
        "seed": int(config["seed"]),
        "feedback_delay": int(config["feedback_delay"]),
        "dataset": config["dataset"],
        "distribution": config["distribution"],
        "users": int(config["users"]),
        "no_stale_margin": bool(config.get("lcc_no_stale_margin", False)),
        "no_z_queue": bool(config.get("lcc_no_z_queue", False)),
        "edgewise_price_actor": bool(config.get("lcc_edgewise_price_actor", False)),
        "flat_price_bias_actor": bool(config.get("lcc_flat_price_bias_actor", False)),
        "price_shuffle": bool(config.get("lcc_price_shuffle", False)),
        "price_logit_bias": float(config.get("lcc_price_logit_bias", float("nan"))),
        "price_components": str(config.get("lcc_price_components", "")),
        "summary_path": str(summary_path),
    })
    return record


def load_phase(phase: str) -> pd.DataFrame:
    path = RESULTS / phase
    rows = [read_cell(p, phase) for p in sorted(path.rglob("summary_LCC-MAPPO.csv"))]
    return pd.DataFrame(rows)


def expected_grid(phase: str) -> set[tuple[object, ...]]:
    seeds = (1, 2, 3, 4, 5)
    delays = (0, 1, 2, 4, 8)
    if phase == "main":
        return {("QLA-MAPPO-NoMargin", s, d, "KAIST", "poisson_bursty", 30)
                for s, d in product(seeds, delays)}
    if phase == "ablation":
        return {(v, s, d, "KAIST", "poisson_bursty", 30)
                for v, s, d in product(sorted(EXPECTED_VARIANTS[phase]), seeds, delays)}
    if phase == "n40":
        return {("QLA-MAPPO-NoMargin", s, 4, "KAIST", dist, 40)
                for s, dist in product(seeds, ("uniform", "gaussian"))}
    if phase == "crossenv":
        return {("QLA-MAPPO-NoMargin", 1, d, dataset, dist, users)
                for dataset, dist, users, d in product(
                    ("KAIST", "NCSU", "NewYork"), ("uniform", "gaussian"),
                    (15, 20, 25, 30), (0, 8))}
    raise ValueError(phase)


def validate_variant_config(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Per-cell structural check against EXPECTED_VARIANT_CONFIG.

    Catches the exact failure modes that broke the original Flat rerun
    (both edgewise and flat actor flags true -> flat actor never instantiated),
    plus stale-margin / Z-queue leaks and logit-bias drift.
    """
    structural_rows = []
    required_fields = STRUCTURAL_FIELDS + ["price_logit_bias"]
    actual_fields = [
        "edgewise_price_actor", "flat_price_bias_actor", "price_shuffle",
        "price_logit_bias", "price_components", "no_stale_margin", "no_z_queue",
    ]
    for rec in frame.to_dict(orient="records"):
        variant = rec["variant"]
        expected = EXPECTED_VARIANT_CONFIG.get(variant)
        if expected is None:
            structural_rows.append({
                "cell": f"{variant}/seed_{rec['seed']}/delay_{rec['feedback_delay']}",
                "issue": "no EXPECTED_VARIANT_CONFIG entry for variant",
                "ok": False,
            })
            continue
        actual = {
            "lcc_edgewise_price_actor": bool(rec["edgewise_price_actor"]),
            "lcc_flat_price_bias_actor": bool(rec["flat_price_bias_actor"]),
            "lcc_price_shuffle": bool(rec["price_shuffle"]),
            "lcc_price_logit_bias": float(rec["price_logit_bias"]),
            "lcc_price_components": str(rec["price_components"]),
            "lcc_no_stale_margin": bool(rec["no_stale_margin"]),
            "lcc_no_z_queue": bool(rec["no_z_queue"]),
        }
        mismatches = []
        for field in STRUCTURAL_FIELDS:
            exp = expected[field]
            got = actual[field]
            if isinstance(exp, float):
                if abs(exp - got) > 1e-9:
                    mismatches.append(f"{field}={got!r} (expected {exp!r})")
            elif exp != got:
                mismatches.append(f"{field}={got!r} (expected {exp!r})")
        structural_rows.append({
            "cell": f"{variant}/seed_{rec['seed']}/delay_{rec['feedback_delay']}",
            "ok": len(mismatches) == 0,
            "mismatches": "; ".join(mismatches),
        })
    return structural_rows


def validate_phase_grid(phase: str, frame: pd.DataFrame) -> dict[str, object]:
    required = set(METRICS) | {
        "variant", "seed", "feedback_delay", "dataset", "distribution", "users",
        "no_stale_margin", "no_z_queue",
        "edgewise_price_actor", "flat_price_bias_actor", "price_shuffle",
        "price_logit_bias", "price_components",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {"valid_unique": 0, "target": len(expected_grid(phase)), "missing": len(expected_grid(phase)),
                "duplicates": 0, "unexpected": len(frame), "protocol_mismatches": len(frame),
                "structural_mismatches": len(frame),
                "complete": False, "reason": f"missing columns: {missing_columns}"}
    key_columns = ["variant", "seed", "feedback_delay", "dataset", "distribution", "users"]
    keys = [tuple(row) for row in frame[key_columns].itertuples(index=False, name=None)]
    expected = expected_grid(phase)
    unique = set(keys)
    protocol_ok = frame["no_stale_margin"].astype(bool)
    # All QLA-MAPPO cells (main, ablation, n40, crossenv) must now use the
    # final no-Z structure. See ALGORITHM_FREEZE_FullNoZ.md for the freeze
    # rationale and main_legacy_zon/ for the pre-freeze main cells.
    protocol_ok &= frame["no_z_queue"].astype(bool)
    numeric_ok = np.isfinite(frame[METRICS].to_numpy(dtype=float)).all(axis=1)

    structural_rows = validate_variant_config(frame)
    structural_ok = pd.Series([r["ok"] for r in structural_rows])

    return {
        "valid_unique": len(unique & expected),
        "target": len(expected),
        "missing": len(expected - unique),
        "duplicates": len(keys) - len(unique),
        "unexpected": len(unique - expected),
        "protocol_mismatches": int((~protocol_ok | ~numeric_ok).sum()),
        "structural_mismatches": int((~structural_ok).sum()),
        "complete": (unique == expected and len(keys) == len(unique)
                     and bool(protocol_ok.all()) and bool(numeric_ok.all())
                     and bool(structural_ok.all())),
        "reason": "",
    }


def completion_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([
        {"phase": phase, **validate_phase_grid(phase, frames[phase])}
        for phase in EXPECTED
    ])


def aggregate_seed_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Treat seed as repetition and delay as a fixed within-seed condition."""
    seed_means = data.groupby(["variant", "seed"], as_index=False)[METRICS].mean()
    rows = []
    for variant, group in seed_means.groupby("variant"):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            ci_half = 2.776 * std / np.sqrt(len(values)) if len(values) == 5 else np.nan
            rows.append({
                "variant": variant,
                "metric": metric,
                "n_training_seeds": len(values),
                "n_delay_cells": len(data[data["variant"] == variant]),
                "mean": mean,
                "seed_block_std": std,
                "ci95_low": mean - ci_half if np.isfinite(ci_half) else np.nan,
                "ci95_high": mean + ci_half if np.isfinite(ci_half) else np.nan,
            })
    return pd.DataFrame(rows)


def paired_ablation(main: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, subset in ablation.groupby("variant"):
        merged = pd.merge(
            main[["seed", "feedback_delay"] + METRICS],
            subset[["seed", "feedback_delay"] + METRICS],
            on=["seed", "feedback_delay"], suffixes=("_main", "_variant"),
        )
        seeds = sorted(merged["seed"].unique())
        for metric in METRICS:
            per_seed = merged.assign(
                diff=merged[f"{metric}_variant"] - merged[f"{metric}_main"]
            ).groupby("seed")["diff"].mean()
            point = float(per_seed.mean())
            boot = np.empty(10_000, dtype=float)
            values = per_seed.reindex(seeds).to_numpy(dtype=float)
            for index in range(len(boot)):
                boot[index] = RNG.choice(values, size=len(values), replace=True).mean()
            low, high = np.percentile(boot, [2.5, 97.5])
            main_mean = float(merged[f"{metric}_main"].mean())
            rows.append({
                "variant": variant,
                "metric": metric,
                "n_pairs": len(merged),
                "n_training_seeds": len(seeds),
                "mean_diff_variant_minus_main": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "relative_pct_vs_main": 100.0 * point / max(abs(main_mean), 1e-12),
                "ci_excludes_zero": bool(high < 0 or low > 0),
            })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")


def plot_ablation(aggregate: pd.DataFrame) -> None:
    order = ["QLA-MAPPO-NoMargin", "QL-NoBias-NoMargin",
             "QL-Bias-Shuffle-NoMargin", "Flat-Bias-QL-NoMargin"]
    labels = ["QLA-MAPPO", "QL-NoBias", "QL-Bias-Shuffle", "Flat-Bias-QL"]
    colors = ["#D55E00", "#56B4E9", "#009E73", "#CC79A7"]
    titles = ["Average latency (s)", "CVaR99 (s)", "Queue oscillation", "Migration rate"]
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.4))
    for ax, metric, title in zip(axes, METRICS, titles):
        sub = aggregate[aggregate["metric"] == metric].set_index("variant").reindex(order)
        means = sub["mean"].to_numpy(dtype=float)
        low = sub["ci95_low"].to_numpy(dtype=float)
        high = sub["ci95_high"].to_numpy(dtype=float)
        x = np.arange(len(order))
        ax.bar(x, means, yerr=np.vstack([means - low, high - means]),
               color=colors, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(means):
            ax.text(index, high[index] + 0.02 * max(means), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Simplified QLA-MAPPO core ablation (5 training seeds × 5 delays; 95% t interval)")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.31, top=0.77, wspace=0.38)
    save_figure(fig, "fig_simplified_core_ablation")
    plt.close(fig)


def write_report(completion: pd.DataFrame, aggregate: pd.DataFrame,
                 paired: pd.DataFrame, complete: bool) -> None:
    lines = ["# Simplified QLA-MAPPO Result Audit", "", "## Completion", "",
             completion.to_markdown(index=False), ""]
    if not complete:
        lines.extend([
            "Formal aggregation is blocked because at least one planned phase is incomplete.",
            "Partial cells are retained only for run-health monitoring and must not be cited.",
        ])
    else:
        lines.extend([
            "## Seed-Block Aggregates", "", aggregate.to_markdown(index=False), "",
            "## Paired Core Ablation", "", paired.to_markdown(index=False), "",
            "All ablation differences are variant minus main. Confidence intervals use seed-block bootstrap.",
        ])
    (OUT / "SIMPLIFIED_RERUN_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_structural_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for phase, frame in frames.items():
        for r in validate_variant_config(frame):
            r["phase"] = phase
            rows.append(r)
    return pd.DataFrame(rows)[["phase", "cell", "ok", "mismatches"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Write a progress report, but never formal partial aggregates.")
    parser.add_argument("--allow-structural-mismatch", action="store_true",
                        help=("Write the audit CSV but still refuse to publish aggregates. "
                              "Use only when intentionally diagnosing a known regression."))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {phase: load_phase(phase) for phase in EXPECTED}
    completion = completion_table(frames)
    completion.to_csv(OUT / "completion.csv", index=False)
    structural_audit = collect_structural_audit(frames)
    structural_audit.to_csv(OUT / "structural_audit.csv", index=False)

    structural_failures = structural_audit[~structural_audit["ok"].astype(bool)]
    if len(structural_failures) > 0:
        print(structural_failures.to_string(index=False))
        msg = (f"\n{len(structural_failures)} cell(s) failed structural validation. "
               "Aggregates are refused because published numbers would not match the "
               "claimed variant. Inspect structural_audit.csv and re-train the offending "
               "cell(s) before re-running the finalizer.")
        if not args.allow_structural_mismatch:
            raise SystemExit(msg)

    is_complete = bool(completion["complete"].all())
    if not is_complete:
        write_report(completion, pd.DataFrame(), pd.DataFrame(), False)
        print(completion.to_string(index=False))
        if not args.allow_partial:
            raise SystemExit("Incomplete grid: formal aggregation refused")
        return

    main_df = frames["main"].assign(variant="QLA-MAPPO-NoMargin")
    ablation_df = frames["ablation"]
    combined = pd.concat([main_df, ablation_df], ignore_index=True)
    aggregate = aggregate_seed_blocks(combined)
    paired = paired_ablation(main_df, ablation_df)
    aggregate.to_csv(OUT / "core_ablation_seed_block_summary.csv", index=False)
    paired.to_csv(OUT / "core_ablation_paired_bootstrap.csv", index=False)
    frames["crossenv"].to_csv(OUT / "crossenv_raw_48.csv", index=False)
    frames["n40"].to_csv(OUT / "n40_raw_10.csv", index=False)
    crossenv_pivot = frames["crossenv"].pivot_table(
        index=["dataset", "distribution", "users", "feedback_delay"],
        values=METRICS, aggfunc="mean",
    ).reset_index()
    crossenv_pivot.to_csv(OUT / "crossenv_breakdown.csv", index=False)
    n40_summary = frames["n40"].groupby("distribution")[METRICS].agg(["mean", "std"])
    n40_summary.to_csv(OUT / "n40_by_distribution.csv")
    plot_ablation(aggregate)
    write_report(completion, aggregate, paired, True)
    print("Formal simplified analysis complete:", OUT)


if __name__ == "__main__":
    main()
