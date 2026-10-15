from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else None


def learning_file(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("learning_*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one learning CSV under {run_dir}, found {candidates}")
    return candidates[0]


def untrained_evaluation(
    untrained_root: Path | None,
    variant: str,
    seed: int,
    delay: int,
) -> dict[str, str] | None:
    if untrained_root is None:
        return None
    reference_variant = "M2PPO-original-lr1e-3" if variant.startswith("M2PPO-") else variant
    candidates = sorted(
        (
            untrained_root
            / reference_variant
            / f"seed_{seed}"
            / f"delay_{delay}"
            / "KAIST"
            / "30u"
            / "poisson_bursty"
        ).glob("summary_*.csv")
    )
    return read_rows(candidates[0])[0] if len(candidates) == 1 else None


def summarize(root: Path, untrained_root: Path | None = None) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/seed_*/delay_*/KAIST/30u/poisson_bursty/summary_*.csv")):
        run_dir = summary_path.parent
        rows = read_rows(learning_file(run_dir))
        evaluation = read_rows(summary_path)[0]
        latencies = [float(row["average_latency"]) for row in rows]
        tail_count = max(1, len(latencies) // 5)
        critic_losses = [
            value
            for row in rows
            if (value := as_float(row, "critic_loss")) is not None and value > 0.0
        ]
        dqn_losses = [
            value
            for row in rows
            if (value := as_float(row, "loss")) is not None and value > 0.0
        ]
        variant = summary_path.relative_to(root).parts[0]
        seed = int(summary_path.relative_to(root).parts[1].split("_")[1])
        delay = int(summary_path.relative_to(root).parts[2].split("_")[1])
        untrained = untrained_evaluation(untrained_root, variant, seed, delay)
        eval_latency = float(evaluation["average_latency"])
        untrained_latency = float(untrained["average_latency"]) if untrained else None
        output.append(
            {
                "variant": variant,
                "seed": seed,
                "delay": delay,
                "logged_points": len(rows),
                "train_initial_latency": latencies[0],
                "train_min_latency": min(latencies),
                "train_final_latency": latencies[-1],
                "train_tail_mean_latency": fmean(latencies[-tail_count:]),
                "train_final_over_initial": latencies[-1] / latencies[0],
                "critic_loss_first": critic_losses[0] if critic_losses else "",
                "critic_loss_final": critic_losses[-1] if critic_losses else "",
                "critic_loss_max": max(critic_losses) if critic_losses else "",
                "dqn_loss_first": dqn_losses[0] if dqn_losses else "",
                "dqn_loss_final": dqn_losses[-1] if dqn_losses else "",
                "dqn_loss_max": max(dqn_losses) if dqn_losses else "",
                "eval_average_latency": eval_latency,
                "untrained_eval_average_latency": untrained_latency if untrained else "",
                "eval_minus_untrained_latency": (
                    eval_latency - untrained_latency if untrained_latency is not None else ""
                ),
                "eval_cvar95": float(evaluation["cvar95"]),
                "eval_cvar99": float(evaluation["cvar99"]),
                "eval_global_demand_concentration": float(
                    evaluation["global_demand_concentration"]
                ),
                "eval_queue_oscillation": float(evaluation["queue_oscillation"]),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize isolated stability ablations.")
    parser.add_argument("--root", type=Path, default="results_convergence_stability_ablation")
    parser.add_argument("--untrained-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    untrained_root = args.untrained_root.resolve() if args.untrained_root else None
    rows = summarize(root, untrained_root)
    if not rows:
        raise FileNotFoundError(f"No completed ablations found under {root}")
    output = root / "stability_ablation_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
