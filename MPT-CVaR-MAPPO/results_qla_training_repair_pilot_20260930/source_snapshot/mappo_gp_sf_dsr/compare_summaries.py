from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "system_risk_latency",
    "risk_tail_latency",
    "risk_gap_latency",
    "per_user_latency_jain",
    "mean_portfolio_entropy",
    "mean_portfolio_concentration",
    "global_demand_concentration",
    "server_selection_gini",
    "inference_time_ms",
]


def read_one(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new MAPPO-DSR summary CSV files.")
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    old = read_one(Path(args.old))
    new = read_one(Path(args.new))
    rows = []
    for key in KEYS:
        old_value = as_float(old, key)
        new_value = as_float(new, key)
        if old_value is None and new_value is None:
            continue
        delta = None if old_value is None or new_value is None else new_value - old_value
        relative = None if old_value in (None, 0.0) or new_value is None else delta / abs(old_value)
        rows.append(
            {
                "metric": key,
                "old_mappo_dsr": old_value if old_value is not None else "",
                "risk_mappo_dsr": new_value if new_value is not None else "",
                "delta_new_minus_old": delta if delta is not None else "",
                "relative_delta": relative if relative is not None else "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
