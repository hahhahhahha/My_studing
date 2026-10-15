"""Audit QLA experiment progress and protocol flags without loading large samples."""

from __future__ import annotations

import json
from itertools import product
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SIMPLIFIED = ROOT / "results_qla_simplified_rerun"
FACTORIAL = ROOT / "results_priceedge_qtbias_factorial_2p3_center"
OUT = SIMPLIFIED / "progress"


PHASES = [
    ("main", SIMPLIFIED / "main", 25),
    ("ablation", SIMPLIFIED / "ablation", 75),
    ("n40", SIMPLIFIED / "n40", 10),
    ("crossenv", SIMPLIFIED / "crossenv", 48),
    # Eight 2^3 corners plus one center, each on 3 seeds x 3 delays.
    ("factorial", FACTORIAL, 81),
]


def summary_files(path: Path) -> list[Path]:
    return sorted(path.rglob("summary_*.csv")) if path.exists() else []


def expected_keys(phase: str) -> set[tuple[object, ...]]:
    seeds = (1, 2, 3, 4, 5)
    delays = (0, 1, 2, 4, 8)
    if phase == "main":
        return {("QLA-MAPPO-NoMargin", seed, delay, "KAIST", "poisson_bursty", 30)
                for seed, delay in product(seeds, delays)}
    if phase == "ablation":
        variants = ("QL-NoBias-NoMargin", "QL-Bias-Shuffle-NoMargin", "Flat-Bias-QL-NoMargin")
        return {(variant, seed, delay, "KAIST", "poisson_bursty", 30)
                for variant, seed, delay in product(variants, seeds, delays)}
    if phase == "n40":
        return {("QLA-MAPPO-NoMargin", seed, 4, "KAIST", distribution, 40)
                for seed, distribution in product(seeds, ("uniform", "gaussian"))}
    if phase == "crossenv":
        return {("QLA-MAPPO-NoMargin", 1, delay, dataset, distribution, users)
                for dataset, distribution, users, delay in product(
                    ("KAIST", "NCSU", "NewYork"), ("uniform", "gaussian"),
                    (15, 20, 25, 30), (0, 8))}
    if phase == "factorial":
        factorial_seeds = (1, 2, 3)
        corners = {(lambda_pi, entropy, lcc_v, seed, delay)
                   for lambda_pi, entropy, lcc_v, seed, delay in product(
                       (0.50, 1.00), (0.970, 0.995), (0.25, 1.00), factorial_seeds, (0, 4, 8))}
        center = {(0.75, 0.990, 0.50, seed, delay)
                  for seed, delay in product(factorial_seeds, (0, 4, 8))}
        return corners | center
    raise ValueError(phase)


def inspect_summaries(phase: str, path: Path) -> dict[str, object]:
    expected = expected_keys(phase)
    valid_keys: list[tuple[object, ...]] = []
    invalid_files: list[str] = []
    for summary in summary_files(path):
        config_path = summary.with_name("config_LCC-MAPPO.json")
        try:
            frame = pd.read_csv(summary)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            required = {"average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"}
            if frame.empty or not required.issubset(frame.columns):
                raise ValueError("empty summary or missing required metric columns")
            if phase == "factorial":
                key = (round(float(config["lcc_price_logit_bias"]), 6),
                       round(float(config["target_portfolio_entropy"]), 6),
                       round(float(config["lcc_v"]), 6),
                       int(config["seed"]), int(config["feedback_delay"]))
            else:
                parts = summary.parts
                phase_index = parts.index(phase)
                variant = parts[phase_index + 1]
                key = (variant, int(config["seed"]), int(config["feedback_delay"]),
                       str(config["dataset"]), str(config["distribution"]), int(config["users"]))
            if key not in expected:
                raise ValueError(f"unexpected cell key: {key}")
            valid_keys.append(key)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as exc:
            invalid_files.append(f"{summary}: {exc}")
    unique = set(valid_keys)
    duplicates = len(valid_keys) - len(unique)
    return {
        "done": len(unique),
        "missing": len(expected - unique),
        "duplicates": duplicates,
        "invalid": len(invalid_files),
        "invalid_details": " | ".join(invalid_files),
        "complete": unique == expected and duplicates == 0 and not invalid_files,
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")


def build_progress() -> pd.DataFrame:
    rows = []
    for phase, path, target in PHASES:
        files = summary_files(path)
        inspection = inspect_summaries(phase, path)
        latest = max((p.stat().st_mtime for p in files), default=None)
        rows.append({
            "phase": phase,
            "done": inspection["done"],
            "target": target,
            "remaining": inspection["missing"],
            "completion_pct": 100.0 * inspection["done"] / target,
            "duplicates": inspection["duplicates"],
            "invalid": inspection["invalid"],
            "grid_complete": inspection["complete"],
            "invalid_details": inspection["invalid_details"],
            "latest_result_time": datetime.fromtimestamp(latest).isoformat(timespec="seconds") if latest else "",
            "result_root": str(path),
        })
    return pd.DataFrame(rows)


def build_protocol_audit() -> pd.DataFrame:
    rows = []
    for phase, path, _ in PHASES:
        if phase == "factorial":
            configs = sorted(path.rglob("config_LCC-MAPPO.json")) if path.exists() else []
        else:
            configs = sorted(path.rglob("config_LCC-MAPPO.json")) if path.exists() else []
        for config_path in configs:
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rows.append({"phase": phase, "config": str(config_path), "status": "unreadable"})
                continue
            no_margin = bool(cfg.get("lcc_no_stale_margin", False))
            no_z = bool(cfg.get("lcc_no_z_queue", False))
            expected_no_z = phase != "main"
            status = "ok" if no_margin and (no_z == expected_no_z) else "mismatch"
            rows.append({
                "phase": phase,
                "config": str(config_path),
                "seed": cfg.get("seed"),
                "feedback_delay": cfg.get("feedback_delay"),
                "no_stale_margin": no_margin,
                "no_z_queue": no_z,
                "expected_no_z_queue": expected_no_z,
                "price_components": cfg.get("lcc_price_components"),
                "price_logit_bias": cfg.get("lcc_price_logit_bias"),
                "target_entropy": cfg.get("target_portfolio_entropy"),
                "lcc_v": cfg.get("lcc_v"),
                "status": status,
            })
    return pd.DataFrame(rows)


def plot_progress(progress: pd.DataFrame) -> None:
    colors = ["#009E73" if done >= target else "#56B4E9"
              for done, target in zip(progress["done"], progress["target"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = range(len(progress))
    ax.barh(list(y), progress["completion_pct"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(progress["phase"])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Completion (%)")
    ax.set_title("QLA-MAPPO evidence generation progress")
    ax.grid(axis="x", alpha=0.25)
    for index, row in progress.iterrows():
        ax.text(min(row["completion_pct"] + 1.5, 102), index,
                f'{int(row["done"])}/{int(row["target"])}', va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_experiment_progress")
    plt.close(fig)


def write_report(progress: pd.DataFrame, audit: pd.DataFrame) -> None:
    mismatch = audit[audit["status"] != "ok"] if not audit.empty else audit
    lines = [
        "# QLA-MAPPO Experiment Progress",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Phase | Valid unique cells | Target | Completion | Missing | Duplicates | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in progress.iterrows():
        lines.append(
            f'| {row["phase"]} | {int(row["done"])} | {int(row["target"])} | '
            f'{row["completion_pct"]:.1f}% | {int(row["remaining"])} | '
            f'{int(row["duplicates"])} | {int(row["invalid"])} |'
        )
    lines.extend([
        "",
        "## Protocol Audit",
        "",
        f"Audited configuration files: {len(audit)}.",
        f"Mismatches: {len(mismatch)}.",
        "",
        "The completed main compatibility grid uses no stale margin and retains the dormant Z state. "
        "All newly started ablation, N=40, cross-environment, and factorial cells are expected to disable both stale margin and Z.",
        "",
        "This is a progress artifact, not a final statistical result table. Partial ablation or factorial aggregates must not be used as manuscript claims.",
    ])
    (OUT / "EXPERIMENT_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    progress = build_progress()
    audit = build_protocol_audit()
    progress.to_csv(OUT / "experiment_progress.csv", index=False)
    audit.to_csv(OUT / "protocol_audit.csv", index=False)
    plot_progress(progress)
    write_report(progress, audit)
    print(progress.to_string(index=False))
    if not audit.empty:
        print(f"\nProtocol configs: {len(audit)}; mismatches: {(audit['status'] != 'ok').sum()}")


if __name__ == "__main__":
    main()
