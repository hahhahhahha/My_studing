from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TunedV4:
    alpha: float = 0.30
    q_penalty_weight: float = 0.08
    kappa_i: float = 0.02
    kappa_g: float = 0.02
    target_portfolio_entropy: float = 0.99
    feedback_gate_eta: float = 0.10


TUNED = TunedV4()

METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
    "server_selection_gini",
    "portfolio_migration_rate",
    "p90_queue_lifetime",
    "inference_time_ms",
]

TABLE_METRICS = [
    "average_latency",
    "cvar95",
    "cvar99",
    "queue_oscillation",
    "global_demand_concentration",
]

PLOT_METRICS = [
    ("average_latency", "Average latency (s)"),
    ("cvar95", "CVaR95 (s)"),
    ("cvar99", "CVaR99 (s)"),
    ("queue_oscillation", "Queue oscillation"),
    ("global_demand_concentration", "Global concentration"),
    ("server_selection_gini", "Server selection Gini"),
]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def _v4_out_base(args: argparse.Namespace, users: int, seed: int, delay: int) -> Path:
    return Path(args.output_dir) / "v4_agegate_tuned" / f"{users}u" / f"seed_{seed}" / f"delay_{delay}"


def _v4_summary_path(base: Path, users: int, distribution: str) -> Path:
    return base / "KAIST" / f"{users}u" / distribution / "summary_DSR-MTG-Qpen-v4-agegate.csv"


def _existing_v4_summary(args: argparse.Namespace, users: int, seed: int, delay: int) -> Path | None:
    candidates = [
        _v4_summary_path(_v4_out_base(args, users, seed, delay), users, args.distribution),
    ]
    if users == 30:
        candidates.append(
            Path("results_v4_agegate_tuned_multiseed_compare")
            / "coordinate_tuned"
            / f"seed_{seed}"
            / f"delay_{delay}"
            / "KAIST"
            / f"{users}u"
            / args.distribution
            / "summary_DSR-MTG-Qpen-v4-agegate.csv"
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def run_v4(args: argparse.Namespace) -> None:
    for users in args.users:
        for seed in args.seeds:
            for delay in args.delays:
                if _existing_v4_summary(args, users, seed, delay) is not None and not args.force:
                    print(f"[skip] V4-AgeGate-Tuned users={users} seed={seed} delay={delay}", flush=True)
                    continue
                out_base = _v4_out_base(args, users, seed, delay)
                log_dir = out_base / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-B",
                    "-m",
                    "mappo_gp_sf_dsr.train",
                    "--method",
                    "DSR-MTG-Qpen-v4-agegate",
                    "--dataset",
                    "KAIST",
                    "--users",
                    str(users),
                    "--distribution",
                    args.distribution,
                    "--train-steps",
                    str(args.train_steps),
                    "--test-steps",
                    str(args.test_steps),
                    "--feedback-delay",
                    str(delay),
                    "--broadcast-period",
                    str(args.broadcast_period),
                    "--slot-s",
                    str(args.slot_s),
                    "--task-compute-mb",
                    str(args.task_compute_mb),
                    "--poisson-lambda",
                    str(args.poisson_lambda),
                    "--burst-prob",
                    str(args.burst_prob),
                    "--alpha",
                    str(TUNED.alpha),
                    "--q-penalty-weight",
                    str(TUNED.q_penalty_weight),
                    "--kappa-i",
                    str(TUNED.kappa_i),
                    "--kappa-g",
                    str(TUNED.kappa_g),
                    "--target-portfolio-entropy",
                    str(TUNED.target_portfolio_entropy),
                    "--feedback-gate-eta",
                    str(TUNED.feedback_gate_eta),
                    "--max-supported-users",
                    str(max(args.max_supported_users, max(args.users))),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(out_base),
                    "--log-interval",
                    str(args.log_interval),
                    "--device",
                    args.device,
                ]
                print(f"[run] V4-AgeGate-Tuned users={users} seed={seed} delay={delay}", flush=True)
                with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
                    subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
                print(f"[done] V4-AgeGate-Tuned users={users} seed={seed} delay={delay}", flush=True)


def run_calibration_policy(args: argparse.Namespace, policy: str) -> None:
    out_dir = Path(args.output_dir) / policy
    summary = out_dir / "KAIST" / args.distribution / "homogeneous" / "bg0" / "exec_simultaneous" / "calibration_summary.csv"
    if summary.exists() and not args.force:
        print(f"[skip] {policy} calibration", flush=True)
        return
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "mappo_gp_sf_dsr.calibrate_environment",
        "--dataset",
        "KAIST",
        "--output-dir",
        str(out_dir),
        "--users",
        ",".join(str(v) for v in args.users),
        "--feedback-delays",
        ",".join(str(v) for v in args.delays),
        "--broadcast-periods",
        str(args.broadcast_period),
        "--seeds",
        ",".join(str(v) for v in args.seeds),
        "--steps",
        str(args.test_steps),
        "--slot-s",
        str(args.slot_s),
        "--task-compute-mb",
        str(args.task_compute_mb),
        "--workload",
        args.distribution,
        "--poisson-lambda",
        str(args.poisson_lambda),
        "--burst-prob",
        str(args.burst_prob),
        "--policy",
        policy,
        "--score-temperature",
        str(args.score_temperature),
        "--execution-mode",
        "simultaneous",
    ]
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {policy} calibration users={args.users} delays={args.delays}", flush=True)
    with (log_dir / "run.log").open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[done] {policy} calibration", flush=True)


def _read_one_summary(path: Path, method: str, users: int, seed: int, delay: int) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    out = {"method": method, "users": users, "seed": seed, "feedback_delay": delay}
    for metric in METRICS:
        out[metric] = float(row.get(metric, "nan"))
    return out


def collect_v4(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    for users in args.users:
        for seed in args.seeds:
            for delay in args.delays:
                path = _existing_v4_summary(args, users, seed, delay)
                if path is None:
                    missing.append(f"users={users} seed={seed} delay={delay}")
                    continue
                rows.append(_read_one_summary(path, "V4-AgeGate-Tuned", users, seed, delay))
    if missing:
        raise FileNotFoundError("Missing V4 summaries:\n" + "\n".join(missing))
    return rows


def collect_calibration(args: argparse.Namespace, policy: str, label: str) -> list[dict]:
    runs_path = (
        Path(args.output_dir)
        / policy
        / "KAIST"
        / args.distribution
        / "homogeneous"
        / "bg0"
        / "exec_simultaneous"
        / "calibration_runs.csv"
    )
    if not runs_path.exists():
        raise FileNotFoundError(runs_path)
    df = pd.read_csv(runs_path)
    rows: list[dict] = []
    for _, item in df.iterrows():
        if int(item["broadcast_period"]) != args.broadcast_period:
            continue
        rows.append(
            {
                "method": label,
                "users": int(item["users"]),
                "seed": int(item["seed"]),
                "feedback_delay": int(item["feedback_delay"]),
                **{metric: float(item.get(metric, float("nan"))) for metric in METRICS},
            }
        )
    return rows


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_delay = raw.groupby(["users", "method", "feedback_delay"], as_index=False).agg(
        {metric: ["mean", "std"] for metric in METRICS}
    )
    by_delay.columns = ["_".join([x for x in col if x]) for col in by_delay.columns]
    by_users = raw.groupby(["users", "method"], as_index=False).agg({metric: ["mean", "std"] for metric in METRICS})
    by_users.columns = ["_".join([x for x in col if x]) for col in by_users.columns]
    return by_delay, by_users


def plot_scale(by_users: pd.DataFrame, out_dir: Path) -> None:
    methods = ["score-softmax", "delayed-JSQ", "V4-AgeGate-Tuned"]
    colors = {"score-softmax": "#d95f02", "delayed-JSQ": "#7570b3", "V4-AgeGate-Tuned": "#1b9e77"}
    markers = {"score-softmax": "o", "delayed-JSQ": "s", "V4-AgeGate-Tuned": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.8))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, PLOT_METRICS):
        for method in methods:
            sub = by_users[by_users["method"] == method].sort_values("users")
            if sub.empty:
                continue
            x = sub["users"].to_numpy()
            y = sub[f"{metric}_mean"].to_numpy()
            ax.plot(x, y, marker=markers[method], linewidth=2.4, markersize=7, color=colors[method], label=method)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Number of users N", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(sorted(by_users["users"].unique()))
        ax.grid(alpha=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13, frameon=True, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Learned-policy scale study over users N", fontsize=18, y=0.995)
    fig.subplots_adjust(top=0.84, left=0.065, right=0.985, bottom=0.075, hspace=0.42, wspace=0.25)
    fig.savefig(out_dir / "learned_policy_scale_key_metrics.png", dpi=230)
    plt.close(fig)


def write_report(out_dir: Path, by_users: pd.DataFrame) -> None:
    table_cols = ["users", "method"] + [f"{metric}_mean" for metric in TABLE_METRICS]
    table = by_users[table_cols].copy()
    lines = [
        "# Learned-Policy Scale Study: N={20,30,40}",
        "",
        "Setting: KAIST, poisson_bursty workload, task=15MB, background=0, homogeneous servers, broadcast K=2, delay={0,1,2,4,8}, seeds={0,1,2}.",
        "",
        "The table averages each method over the full delay grid. Lower values are better for all listed metrics.",
        "",
        "## Main Scale Table",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- `score-softmax` is the score-chasing mechanism baseline. If its CVaR and queue oscillation grow with N, the stale-feedback crowding problem strengthens with user density.",
        "- `delayed-JSQ` is a classic shortest-queue heuristic with stale load information. High concentration/Gini indicates herd behavior under stale queues.",
        "- `V4-AgeGate-Tuned` should increase more slowly with N and keep global concentration low, showing that the learned policy mitigates systemic risk rather than only optimizing a single scale.",
    ]
    (out_dir / "LEARNED_POLICY_SCALE_STUDY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results_learned_policy_scale_n20_30_40")
    parser.add_argument("--users", default="20,30,40")
    parser.add_argument("--delays", default="0,1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--score-temperature", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    args.users = _parse_ints(args.users)
    args.delays = _parse_ints(args.delays)
    args.seeds = _parse_ints(args.seeds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        run_v4(args)
        run_calibration_policy(args, "score-softmax")
        run_calibration_policy(args, "delayed-jsq")

    rows = []
    rows.extend(collect_v4(args))
    rows.extend(collect_calibration(args, "score-softmax", "score-softmax"))
    rows.extend(collect_calibration(args, "delayed-jsq", "delayed-JSQ"))
    raw = pd.DataFrame(rows)
    by_delay, by_users = aggregate(raw)
    raw.to_csv(out_dir / "learned_policy_scale_raw.csv", index=False)
    by_delay.to_csv(out_dir / "learned_policy_scale_by_delay.csv", index=False)
    by_users.to_csv(out_dir / "learned_policy_scale_by_users.csv", index=False)
    plot_scale(by_users, out_dir)
    write_report(out_dir, by_users)
    print(by_users[["users", "method", "average_latency_mean", "cvar99_mean", "queue_oscillation_mean", "global_demand_concentration_mean"]].to_string(index=False))
    print(f"Wrote {out_dir / 'LEARNED_POLICY_SCALE_STUDY.md'}")


if __name__ == "__main__":
    main()
