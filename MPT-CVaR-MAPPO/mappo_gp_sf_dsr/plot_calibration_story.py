from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


METRICS = [
    ("average_latency_mean", "Average latency (s)", "avg_latency"),
    ("cvar99_mean", "CVaR99 (s)", "cvar99"),
    ("queue_oscillation_mean", "Queue oscillation", "queue_osc"),
    ("median_queue_lifetime_mean", "Median queue lifetime", "med_queue_life"),
    ("p90_queue_lifetime_mean", "P90 queue lifetime", "p90_queue_life"),
    ("queue_overflow_rate_mean", "Overflow ratio", "overflow"),
    ("mean_utilization_mean", "Mean utilization", "mean_util"),
]

CROWDING_METRICS = [
    ("global_demand_concentration_mean", "Global demand concentration", "demand_conc"),
    ("server_selection_gini_mean", "Server selection Gini", "server_gini"),
    ("mean_max_server_pressure_mean", "Mean max server pressure", "max_pressure"),
    ("p90_max_server_pressure_mean", "P90 max server pressure", "p90_max_pressure"),
    ("portfolio_migration_rate_mean", "Portfolio migration rate", "migration"),
    ("topk_flip_d4_mean", "Top-K score flip d=4", "topk_flip_d4"),
    ("score_err_d4_mean", "Score stale error d=4", "score_err_d4"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _to_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object) -> int:
    return int(round(_to_float(value, 0.0)))


def _numeric_key(value: object) -> float:
    return _to_float(value, 0.0)


def _load_rows(roots: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in roots:
        for summary_path in root.rglob("calibration_summary.csv"):
            cfg_path = summary_path.with_name("calibration_config.json")
            cfg: dict[str, object] = {}
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as handle:
                    cfg = json.load(handle)
            for row in _read_csv(summary_path):
                item: dict[str, object] = dict(row)
                item["source_dir"] = str(summary_path.parent)
                item["task_compute_mb"] = _to_float(cfg.get("task_compute_mb"))
                item["background_load"] = _to_float(cfg.get("background_load"))
                item["capacity_mode"] = str(cfg.get("capacity_mode", "unknown"))
                item["workload"] = str(cfg.get("workload", "unknown"))
                item["policy"] = str(cfg.get("policy", "unknown"))
                item["slot_s"] = _to_float(cfg.get("slot_s"))
                rows.append(item)
    return rows


def _mean_by_cell(rows: list[dict[str, object]], row_key: str, col_key: str, metric: str) -> dict[tuple[float, float], float]:
    cells: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        value = _to_float(row.get(metric))
        if np.isfinite(value):
            cells[(_numeric_key(row[row_key]), _numeric_key(row[col_key]))].append(value)
    return {key: float(np.mean(values)) for key, values in cells.items() if values}


def _filter_rows(
    rows: list[dict[str, object]],
    *,
    task_compute_mb: float,
    background_load: float | None = None,
    capacity_mode: str = "homogeneous",
    broadcast_period: int,
    users: int | None = None,
) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        if abs(_to_float(row.get("task_compute_mb")) - task_compute_mb) > 1e-6:
            continue
        if background_load is not None and abs(_to_float(row.get("background_load")) - background_load) > 1e-6:
            continue
        if str(row.get("capacity_mode")) != capacity_mode:
            continue
        if _to_int(row.get("broadcast_period")) != broadcast_period:
            continue
        if users is not None and _to_int(row.get("users")) != users:
            continue
        selected.append(row)
    return selected


def _sorted_unique(rows: list[dict[str, object]], key: str) -> list[object]:
    values = sorted({_to_float(row[key]) for row in rows})
    if all(abs(value - round(value)) < 1e-9 for value in values):
        return [int(round(value)) for value in values]
    return values


def _plot_heatmaps(
    rows: list[dict[str, object]],
    *,
    row_key: str,
    col_key: str,
    metrics: list[tuple[str, str, str]] = METRICS,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_values = _sorted_unique(rows, row_key)
    col_values = _sorted_unique(rows, col_key)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, label, _) in zip(axes_flat, metrics):
        cells = _mean_by_cell(rows, row_key, col_key, metric)
        matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=np.float64)
        for i, rv in enumerate(row_values):
            for j, cv in enumerate(col_values):
                matrix[i, j] = cells.get((float(rv), float(cv)), np.nan)
        image = ax.imshow(matrix, aspect="auto", cmap="YlOrBr", origin="lower")
        ax.set_title(label)
        ax.set_xticks(np.arange(len(col_values)), labels=[str(v) for v in col_values])
        ax.set_yticks(np.arange(len(row_values)), labels=[str(v) for v in row_values])
        ax.set_xlabel(col_key)
        ax.set_ylabel(row_key)
        for i in range(len(row_values)):
            for j in range(len(col_values)):
                value = matrix[i, j]
                if np.isfinite(value):
                    finite = matrix[np.isfinite(matrix)]
                    threshold = np.nanmin(finite) + 0.55 * (np.nanmax(finite) - np.nanmin(finite)) if finite.size else value
                    text_color = "white" if value >= threshold else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.8)

    axes_flat[-1].axis("off")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_lines(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    line_key: str,
    metrics: list[tuple[str, str, str]] = METRICS,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = _sorted_unique(rows, x_key)
    line_values = _sorted_unique(rows, line_key)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (metric, label, _) in zip(axes_flat, metrics):
        for lv in line_values:
            ys = []
            yerrs = []
            for xv in x_values:
                matching = [
                    row
                    for row in rows
                    if _to_float(row.get(x_key)) == float(xv) and _to_float(row.get(line_key)) == float(lv)
                ]
                values = [_to_float(row.get(metric)) for row in matching]
                values = [value for value in values if np.isfinite(value)]
                ys.append(float(np.mean(values)) if values else np.nan)
                std_key = metric.replace("_mean", "_std")
                stds = [_to_float(row.get(std_key)) for row in matching]
                stds = [value for value in stds if np.isfinite(value)]
                yerrs.append(float(np.mean(stds)) if stds else 0.0)
            ax.errorbar(x_values, ys, yerr=yerrs, marker="o", capsize=3, label=f"{line_key}={lv}")
        ax.set_title(label)
        ax.set_xlabel(x_key)
        if metric in {"mean_utilization_mean", "queue_overflow_rate_mean"}:
            values = [_to_float(row.get(metric)) for row in rows]
            values = [value for value in values if np.isfinite(value)]
            upper = max(1.05, max(values, default=1.0) * 1.08)
            ax.set_ylim(0.0, upper)
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes_flat[-1].axis("off")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _export_plot_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keep = [
        "users",
        "feedback_delay",
        "broadcast_period",
        "task_compute_mb",
        "background_load",
        "capacity_mode",
    ]
    metric_keys = [metric for metric, _, _ in METRICS + CROWDING_METRICS]
    std_keys = [metric.replace("_mean", "_std") for metric in metric_keys]
    out_rows = []
    for row in rows:
        out = {key: row.get(key) for key in keep}
        out.update({key: row.get(key) for key in metric_keys})
        out.update({key: row.get(key) for key in std_keys if key in row})
        out_rows.append(out)
    _write_csv(path, out_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create narrative calibration visualizations.")
    parser.add_argument("--roots", nargs="+", default=["results_environment_calibration_main", "results_environment_calibration_task10"])
    parser.add_argument("--out-dir", default="results_environment_calibration_main/story_plots")
    parser.add_argument("--capacity-mode", default="homogeneous")
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    out_dir = Path(args.out_dir)
    rows = _load_rows(roots)
    if not rows:
        raise SystemExit("No calibration_summary.csv files found.")

    plot_specs = [
        {
            "name": "user_delay_task15_bg0_K2",
            "task": 15.0,
            "bg": 0.0,
            "k": 2,
            "title": "User count x feedback delay, task=15MB, background=0, K=2",
        },
        {
            "name": "user_delay_task10_bg0_K1",
            "task": 10.0,
            "bg": 0.0,
            "k": 1,
            "title": "User count x feedback delay, task=10MB, background=0, K=1",
        },
    ]

    for spec in plot_specs:
        selected = _filter_rows(
            rows,
            task_compute_mb=spec["task"],
            background_load=spec["bg"],
            capacity_mode=args.capacity_mode,
            broadcast_period=spec["k"],
        )
        if not selected:
            continue
        _export_plot_rows(out_dir / f"{spec['name']}.csv", selected)
        _plot_heatmaps(
            selected,
            row_key="users",
            col_key="feedback_delay",
            title=spec["title"],
            output_path=out_dir / f"{spec['name']}_heatmaps.png",
        )
        _plot_lines(
            selected,
            x_key="feedback_delay",
            line_key="users",
            title=spec["title"],
            output_path=out_dir / f"{spec['name']}_lines.png",
        )
        _plot_heatmaps(
            selected,
            row_key="users",
            col_key="feedback_delay",
            metrics=CROWDING_METRICS,
            title=f"Crowding indicators, {spec['title']}",
            output_path=out_dir / f"{spec['name']}_crowding_heatmaps.png",
        )
        _plot_lines(
            selected,
            x_key="feedback_delay",
            line_key="users",
            metrics=CROWDING_METRICS,
            title=f"Crowding indicators, {spec['title']}",
            output_path=out_dir / f"{spec['name']}_crowding_lines.png",
        )

    bg_selected = _filter_rows(
        rows,
        task_compute_mb=15.0,
        background_load=None,
        capacity_mode=args.capacity_mode,
        broadcast_period=1,
        users=30,
    )
    bg_selected = [row for row in bg_selected if _to_float(row.get("background_load")) in {0.0, 0.2, 0.4}]
    if bg_selected:
        _export_plot_rows(out_dir / "background_delay_30u_task15_K1.csv", bg_selected)
        _plot_heatmaps(
            bg_selected,
            row_key="background_load",
            col_key="feedback_delay",
            title="Background load x feedback delay, N=30, task=15MB, K=1",
            output_path=out_dir / "background_delay_30u_task15_K1_heatmaps.png",
        )
        _plot_lines(
            bg_selected,
            x_key="feedback_delay",
            line_key="background_load",
            title="Background load x feedback delay, N=30, task=15MB, K=1",
            output_path=out_dir / "background_delay_30u_task15_K1_lines.png",
        )
        _plot_heatmaps(
            bg_selected,
            row_key="background_load",
            col_key="feedback_delay",
            metrics=CROWDING_METRICS,
            title="Crowding indicators, background x delay, N=30, task=15MB, K=1",
            output_path=out_dir / "background_delay_30u_task15_K1_crowding_heatmaps.png",
        )
        _plot_lines(
            bg_selected,
            x_key="feedback_delay",
            line_key="background_load",
            metrics=CROWDING_METRICS,
            title="Crowding indicators, background x delay, N=30, task=15MB, K=1",
            output_path=out_dir / "background_delay_30u_task15_K1_crowding_lines.png",
        )

    hetero_selected = _filter_rows(
        rows,
        task_compute_mb=15.0,
        background_load=0.2,
        capacity_mode="heterogeneous",
        broadcast_period=1,
        users=30,
    )
    if hetero_selected:
        _export_plot_rows(out_dir / "heterogeneous_delay_30u_task15_bg02_K1.csv", hetero_selected)
        _plot_lines(
            hetero_selected,
            x_key="feedback_delay",
            line_key="users",
            title="Heterogeneous robustness, N=30, task=15MB, background=0.2, K=1",
            output_path=out_dir / "heterogeneous_delay_30u_task15_bg02_K1_lines.png",
        )
        _plot_lines(
            hetero_selected,
            x_key="feedback_delay",
            line_key="users",
            metrics=CROWDING_METRICS,
            title="Crowding indicators, heterogeneous, N=30, task=15MB, background=0.2, K=1",
            output_path=out_dir / "heterogeneous_delay_30u_task15_bg02_K1_crowding_lines.png",
        )


if __name__ == "__main__":
    main()
