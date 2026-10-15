from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "system_risk_latency",
    "dsr_input_x",
    "t_mean_norm",
    "t_topk_norm",
    "gap_norm",
    "q_pressure",
    "q_pressure_for_dsr",
    "q_port_pressure_mean",
    "q_penalty_mean",
    "auto_entropy_coef",
    "portfolio_entropy_update",
    "dsr_norm",
    "risk_tail_latency",
    "risk_gap_latency",
    "per_user_latency_jain",
    "mean_portfolio_entropy",
    "mean_portfolio_concentration",
    "global_demand_concentration",
    "server_selection_gini",
    "inference_time_ms",
]


LOWER_IS_BETTER = {
    "average_latency",
    "cvar95",
    "cvar99",
    "system_risk_latency",
    "dsr_input_x",
    "t_mean_norm",
    "t_topk_norm",
    "gap_norm",
    "q_pressure",
    "q_pressure_for_dsr",
    "q_port_pressure_mean",
    "q_penalty_mean",
    "risk_tail_latency",
    "risk_gap_latency",
    "mean_portfolio_concentration",
    "global_demand_concentration",
    "server_selection_gini",
    "inference_time_ms",
}


def read_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def parse_summary_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Summary label cannot be empty")
    return label, Path(path)


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def write_metric_table(labels: list[str], rows_by_label: dict[str, dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for metric in METRICS:
        metric_values = {label: as_float(rows_by_label[label], metric) for label in labels}
        if all(value is None for value in metric_values.values()):
            continue
        numeric = {label: value for label, value in metric_values.items() if value is not None}
        if numeric:
            reverse = metric not in LOWER_IS_BETTER
            best_label = sorted(numeric, key=lambda label: numeric[label], reverse=reverse)[0]
        else:
            best_label = ""
        row = {"metric": metric, "best": best_label}
        for label in labels:
            row[label] = metric_values[label] if metric_values[label] is not None else ""
        table_rows.append(row)

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "best", *labels])
        writer.writeheader()
        writer.writerows(table_rows)


def write_method_table(labels: list[str], rows_by_label: dict[str, dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in labels:
        row = {"method_label": label}
        for metric in METRICS:
            value = as_float(rows_by_label[label], metric)
            if value is not None:
                row[metric] = value
        rows.append(row)
    fieldnames = ["method_label", *METRICS]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(labels: list[str], rows_by_label: dict[str, dict[str, str]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plot_metrics = [
        "average_latency",
        "cvar95",
        "cvar99",
        "mean_portfolio_concentration",
        "global_demand_concentration",
        "server_selection_gini",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, plot_metrics):
        values = [as_float(rows_by_label[label], metric) for label in labels]
        ax.bar(labels, [value if value is not None else 0.0 for value in values])
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple MAPPO summary CSV files.")
    parser.add_argument("--summary", action="append", required=True, type=parse_summary_arg, help="LABEL=PATH")
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--out-methods", required=True)
    parser.add_argument("--plot", default="")
    args = parser.parse_args()

    labels = [label for label, _ in args.summary]
    rows_by_label = {label: read_one(path) for label, path in args.summary}
    write_metric_table(labels, rows_by_label, Path(args.out_metrics))
    write_method_table(labels, rows_by_label, Path(args.out_methods))
    if args.plot:
        make_plot(labels, rows_by_label, Path(args.plot))

    for label in labels:
        row = rows_by_label[label]
        print(
            {
                "method": label,
                "average_latency": as_float(row, "average_latency"),
                "cvar95": as_float(row, "cvar95"),
                "cvar99": as_float(row, "cvar99"),
                "mean_portfolio_concentration": as_float(row, "mean_portfolio_concentration"),
                "global_demand_concentration": as_float(row, "global_demand_concentration"),
            }
        )


if __name__ == "__main__":
    main()
