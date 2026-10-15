from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .calibrate_environment import _policy_from_obs, _policy_portfolios
from .config import TRACE_ROOT
from .metrics import phase_diagnostics, summarize_results
from .run_priceedge_qt_fixed_checkpoint_robustness import (
    DEFAULT_CKPT_ROOT,
    build_env,
    evaluate_priceedge,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]


def parse_list(value: str, cast=int) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    out: list[dict[str, object]] = []
    for group_key, group in df.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        row["runs"] = len(group)
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(row)
    return out


def analytic_rollout(env, policy: str, steps: int, mode: str = "simultaneous", rng=None):
    env.reset()
    results = []
    rng = np.random.default_rng(20260725) if rng is None else rng
    for _ in range(steps):
        if mode == "simultaneous":
            portfolios = _policy_portfolios(env, policy, 5.0, 5)
            results.append(env.joint_step(portfolios, stats=env.portfolio_stats(portfolios)))
        elif mode == "sequential":
            def policy_fn(_user: int, obs: np.ndarray) -> np.ndarray:
                return _policy_from_obs(env, obs, policy, 5.0, 5)

            results.append(env.sequential_step(policy_fn))
        elif mode == "mixed_age":
            portfolios = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float32)
            ages = rng.choice(np.asarray([0, 1, 2, 4, 8], dtype=int), size=env.cfg.n_users)
            for user, age in enumerate(ages):
                feedback = env.server_feedback() if int(age) == 0 else env.feedback_buffer.get(int(age))
                obs = env.actor_observation_for_user(user, feedback=feedback)
                portfolios[user] = _policy_from_obs(env, obs, policy, 5.0, 5)
            results.append(env.joint_step(portfolios, stats=env.portfolio_stats(portfolios)))
        else:
            raise ValueError(mode)
    return results


def summarize_rollout(results, sys_cfg, meta: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = summarize_results(results)
    row.update(phase_diagnostics(results, sys_cfg))
    row.update(meta)
    return row


def run_queue_correlation(args, out: Path) -> None:
    rows: list[dict[str, object]] = []
    for seed in parse_list(args.mechanism_seeds):
        args.seed = seed
        env, sys_cfg, _ = build_env("KAIST", "poisson_bursty", 30, 8, args)
        results = analytic_rollout(env, "score-softmax", args.test_steps)
        queues = np.asarray([result.queue_load_after_mb for result in results], dtype=np.float64)
        for delay in [1, 2, 4, 8]:
            x = queues[:-delay].reshape(-1)
            y = queues[delay:].reshape(-1)
            rho = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
            rows.append({"seed": seed, "delta": delay, "rho_Q": rho, "samples": x.size})
    write_rows(out / "queue_state_correlation_raw.csv", rows)
    summary = []
    for delay in [1, 2, 4, 8]:
        vals = np.asarray([float(r["rho_Q"]) for r in rows if int(r["delta"]) == delay], dtype=float)
        summary.append(
            {
                "delta": delay,
                "rho_Q_mean": float(np.nanmean(vals)),
                "rho_Q_std": float(np.nanstd(vals, ddof=1)),
                "seeds": len(vals),
            }
        )
    write_rows(out / "queue_state_correlation_summary.csv", summary)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    x = np.asarray([r["delta"] for r in summary], dtype=float)
    y = np.asarray([r["rho_Q_mean"] for r in summary], dtype=float)
    e = np.asarray([r["rho_Q_std"] for r in summary], dtype=float)
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, lw=2)
    for xx, yy in zip(x, y):
        ax.text(xx, yy + 0.035, f"{yy:.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel(r"Feedback age $\delta$")
    ax.set_ylabel(r"$\rho_Q(\delta)=corr(Q(t-\delta),Q(t))$")
    ax.set_title("Temporal information retained by stale queue summaries")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_queue_state_correlation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_asynchrony(args, out: Path) -> None:
    conditions = [
        ("synchronous_fresh", 0, "simultaneous"),
        ("synchronous_shared_stale", 8, "simultaneous"),
        ("heterogeneous_user_age", 8, "mixed_age"),
        ("sequential_fresh_updates", 0, "sequential"),
    ]
    rows: list[dict[str, object]] = []
    for seed in parse_list(args.mechanism_seeds):
        for condition, delay, mode in conditions:
            args.seed = seed
            env, sys_cfg, _ = build_env("KAIST", "poisson_bursty", 30, delay, args)
            results = analytic_rollout(
                env,
                "score-softmax",
                args.test_steps,
                mode=mode,
                rng=np.random.default_rng(seed + 7100),
            )
            rows.append(
                summarize_rollout(
                    results,
                    sys_cfg,
                    {
                        "condition": condition,
                        "feedback_delay": "mixed" if mode == "mixed_age" else delay,
                        "seed": seed,
                        "policy": "score-softmax",
                    },
                )
            )
    write_rows(out / "shared_stale_asynchrony_raw.csv", rows)
    summary = aggregate(rows, ["condition"])
    write_rows(out / "shared_stale_asynchrony_summary.csv", summary)
    order = [c[0] for c in conditions]
    lookup = {str(row["condition"]): row for row in summary}
    labels = ["Sync fresh", "Sync shared stale", "Heterogeneous age", "Sequential fresh"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for ax, metric, title in zip(
        axes,
        ["average_latency", "cvar99", "queue_oscillation"],
        ["Avg latency (s)", "CVaR99 (s)", "Queue oscillation"],
    ):
        vals = [float(lookup[key][f"{metric}_mean"]) for key in order]
        bars = ax.bar(np.arange(4), vals, color=["#4daf4a", "#e41a1c", "#377eb8", "#984ea3"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Shared-stale synchronization mechanism (score-softmax)", y=1.05)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_shared_stale_asynchrony.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_load_scan(args, out: Path) -> None:
    methods = ["score-softmax", "delayed-jsq", "QLA-MAPPO"]
    rows: list[dict[str, object]] = []
    base_checkpoint_seed = int(args.checkpoint_seed)
    for seed in parse_list(args.load_scan_seeds):
        # Pair seed_i -> checkpoint_seed_i so QLA-MAPPO uses matched-checkpoint inference.
        args.checkpoint_seed = seed if seed in {1, 2, 3, 4, 5} else base_checkpoint_seed
        for users in [20, 30, 40]:
            for delay in [0, 1, 2, 4, 8]:
                for method in methods:
                    print(f"[load-scan] method={method} N={users} delay={delay} seed={seed} ckpt_seed={args.checkpoint_seed}", flush=True)
                    args.seed = seed
                    env, sys_cfg, _ = build_env("KAIST", "uniform", users, delay, args)
                    if method == "QLA-MAPPO":
                        results = evaluate_priceedge(env, sys_cfg, args)
                    else:
                        results = analytic_rollout(env, method, args.test_steps)
                    rows.append(
                        summarize_rollout(
                            results,
                            sys_cfg,
                            {
                                "method": method,
                                "dataset": "KAIST",
                                "distribution": "uniform",
                                "users": users,
                                "feedback_delay": delay,
                                "seed": seed,
                                "checkpoint_seed": args.checkpoint_seed if method == "QLA-MAPPO" else "",
                                "protocol": "fixed checkpoint by training delay; descriptive load-regime scan",
                            },
                        )
                    )
                    write_rows(out / "load_regime_scan_raw.csv", rows)
    summary = aggregate(rows, ["method", "users", "feedback_delay"])
    write_rows(out / "load_regime_scan_summary.csv", summary)
    df = pd.DataFrame(summary)
    colors = {"score-softmax": "#d95f02", "delayed-jsq": "#7570b3", "QLA-MAPPO": "#1b9e77"}
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 9.0), sharex=True)
    for row_index, users in enumerate([20, 30, 40]):
        for col_index, (metric, title) in enumerate(
            [("average_latency", "Avg latency (s)"), ("cvar99", "CVaR99 (s)"), ("queue_oscillation", "Queue oscillation")]
        ):
            ax = axes[row_index, col_index]
            for method in methods:
                sub = df[(df["method"].eq(method)) & (df["users"].astype(int).eq(users))].sort_values("feedback_delay")
                ax.plot(
                    sub["feedback_delay"],
                    sub[f"{metric}_mean"],
                    marker="o",
                    lw=1.8,
                    label=method,
                    color=colors[method],
                )
            ax.set_xticks([0, 1, 2, 4, 8])
            ax.grid(alpha=0.25)
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(f"N={users}")
            if row_index == 2:
                ax.set_xlabel(r"Feedback delay $\delta$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Load-regime evidence under Uniform requests", y=1.045)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_load_regime_3x3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Problem-existence and stale-mechanism evidence for PriceEdge V4.")
    parser.add_argument("--output-dir", default="results_priceedge_v4_problem_mechanism")
    parser.add_argument("--tasks", default="correlation,asynchrony,load_scan")
    parser.add_argument("--mechanism-seeds", default="1,2,3,4,5")
    parser.add_argument("--load-scan-seeds", default="1")
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--max-supported-users", type=int, default=80)
    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--queue-constraint", default="resample")
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous")
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--target-portfolio-entropy", type=float, default=0.99)
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--lcc-v", type=float, default=0.5)
    parser.add_argument("--lcc-migration-weight", type=float, default=0.05)
    parser.add_argument("--lcc-cvar-alpha", type=float, default=0.99)
    parser.add_argument("--lcc-cvar-budget", type=float, default=1.30)
    parser.add_argument("--lcc-var-lr", type=float, default=0.05)
    parser.add_argument("--lcc-concentration-factor", type=float, default=2.0)
    parser.add_argument("--lcc-stale-margin", type=float, default=1.0)
    parser.add_argument("--lcc-price-components", default="QT")
    parser.add_argument("--lcc-price-logit-bias", type=float, default=0.75)
    parser.add_argument("--lcc-price-intervention", default="aligned")
    parser.add_argument("--lcc-price-shuffle", action="store_true")
    parser.add_argument("--lcc-no-stale-margin", action="store_true")
    parser.add_argument("--lcc-no-z-queue", action="store_true")
    parser.add_argument("--lcc-no-h-queue", action="store_true")
    args = parser.parse_args()
    args.lcc_price_scale_observation = True
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tasks = set(parse_list(args.tasks, str))
    if "correlation" in tasks:
        run_queue_correlation(args, out)
    if "asynchrony" in tasks:
        run_asynchrony(args, out)
    if "load_scan" in tasks:
        run_load_scan(args, out)
    print(f"Wrote problem/mechanism evidence to {out}")


if __name__ == "__main__":
    main()
