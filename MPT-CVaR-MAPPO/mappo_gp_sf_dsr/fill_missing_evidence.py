from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .calibrate_environment import (
    _edge_capacities,
    _policy_from_obs,
    _policy_portfolios,
    _write_csv,
)
from .config import MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


MAIN_METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "portfolio_migration_rate",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_table(path: Path, rows: list[dict]) -> None:
    _write_csv(path, rows)


def _group_rows(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    return grouped


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_permutation_p(diff: np.ndarray, rng: np.random.Generator, n_perm: int = 20000) -> float:
    """One-sided paired sign-flip p-value for mean(diff) > 0."""

    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    observed = float(np.mean(diff))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size))
    permuted = signs @ diff / diff.size
    return float((np.sum(permuted >= observed) + 1.0) / (n_perm + 1.0))


def main_baseline_statistics(out_dir: Path, n_boot: int) -> None:
    rng = np.random.default_rng(20260719)
    rows = _read_csv(ROOT / "results_selected_baseline_comparison" / "selected_baseline_common_grid_raw.csv")
    grouped = _group_rows(rows, ["method"])

    ci_rows: list[dict] = []
    for method, group in sorted(grouped.items()):
        method_name = method[0]
        for metric in MAIN_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            lo, hi = _bootstrap_ci(values, rng, n_boot=n_boot)
            ci_rows.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "runs": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                }
            )
    _write_table(out_dir / "main_baseline_bootstrap_ci.csv", ci_rows)

    by_unit = _group_rows(rows, ["seed", "feedback_delay"])
    test_rows: list[dict] = []
    for metric in MAIN_METRICS:
        for baseline in sorted(grouped):
            baseline_name = baseline[0]
            if baseline_name == "FACR-MAPPO":
                continue
            paired_diff = []
            for _unit, group in by_unit.items():
                values = {row["method"]: float(row[metric]) for row in group}
                if "FACR-MAPPO" in values and baseline_name in values:
                    paired_diff.append(values[baseline_name] - values["FACR-MAPPO"])
            diff = np.asarray(paired_diff, dtype=np.float64)
            lo, hi = _bootstrap_ci(diff, rng, n_boot=n_boot)
            p_value = _paired_permutation_p(diff, rng, n_perm=max(2000, n_boot))
            test_rows.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "paired_units": int(diff.size),
                    "mean_improvement_baseline_minus_facr": float(np.mean(diff)),
                    "relative_improvement_vs_baseline": float(np.mean(diff) / np.mean([float(r[metric]) for r in grouped[(baseline_name,)]])),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "one_sided_p_baseline_gt_facr": p_value,
                }
            )
    _write_table(out_dir / "main_baseline_paired_tests.csv", test_rows)


def _latency_values(path: Path) -> np.ndarray:
    rows = _read_csv(path)
    vals = []
    for row in rows:
        if "phase" in row and row["phase"] and row["phase"] != "test":
            continue
        if "latency" in row:
            vals.append(float(row["latency"]))
    return np.asarray(vals, dtype=np.float64)


def tail_distribution(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = {
        "FACR-MAPPO": ROOT / "results_v4_agegate_lyapprior_formal" / "lyapprior_w20_smooth80",
        "PPO-MV-A": ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-MV",
        "PPO-DSR-A": ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-DSR",
        "LyFOTO-LC": ROOT / "results_recent_mec_local_lyapunov_multiseed" / "Lyapunov-Q-local-stale",
        "TODA-MAB-A": ROOT / "results_recent_mec_baselines_strong_multiseed" / "TODAuction-MAB-tail",
        "QECO-A": ROOT / "results_recent_mec_baselines_strong_qeco_multiseed" / "QECO-load-topk",
    }
    patterns = {
        "FACR-MAPPO": "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_DSR-MTG-Qpen-v4-agegate-lyapprior.csv",
        "PPO-MV-A": "seed_*/delay_8/Towards-PPO-MV_delay_8/KAIST/30u/poisson_bursty/latency_samples_Towards-PPO-MV.csv",
        "PPO-DSR-A": "seed_*/delay_8/Towards-PPO-DSR_delay_8/KAIST/30u/poisson_bursty/latency_samples_Towards-PPO-DSR.csv",
        "LyFOTO-LC": "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_Lyapunov-Q-local-stale.csv",
        "TODA-MAB-A": "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_TODAuction-MAB-tail.csv",
        "QECO-A": "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_QECO-load-topk.csv",
    }

    quantile_rows: list[dict] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, base in specs.items():
        arrays = [_latency_values(path) for path in sorted(base.glob(patterns[method]))]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            continue
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        values = np.clip(values, 0.0, 10.0)
        xs = np.sort(values)
        ccdf = 1.0 - (np.arange(xs.size) + 1.0) / xs.size
        ax.step(xs, np.maximum(ccdf, 1e-5), where="post", label=method, linewidth=1.8)
        quantile_rows.append(
            {
                "method": method,
                "delay": 8,
                "samples": int(values.size),
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
        )
    ax.set_yscale("log")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("CCDF: Pr{latency > x}")
    ax.set_title("Tail latency distribution at feedback delay δ=8")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig_latency_ccdf_delta8.{ext}", dpi=240)
    plt.close(fig)
    _write_table(out_dir / "latency_tail_quantiles_delta8.csv", quantile_rows)


def _default_env(seed: int, feedback_delay: int, execution_mode: str) -> JointMAPPOEdgeEnv:
    sys_cfg = MAPPOSystemConfig(
        n_users=30,
        n_edges=35,
        slot_s=1.0,
        edge_compute_mb_s=63.0,
        queue_upper_mb=630.0,
        max_requests_per_user=7,
        task_compute_mb=15.0,
        task_package_mb=270.0 / 1024.0,
        max_supported_users=80,
    )
    score_cfg = ServerScoreConfig()
    traces = load_trace_set("KAIST", 30, 35, sys_cfg.slot_s, seed, trace_root=TRACE_ROOT)
    capacities = _edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, "homogeneous", seed)
    return JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution="poisson_bursty",
        seed=seed,
        queue_constraint="resample",
        score_config=score_cfg,
        feedback_delay=feedback_delay,
        broadcast_period=2,
        poisson_lambda=3.5,
        burst_prob=0.1,
        burst_multiplier=2.0,
        edge_capacities_mb_s=capacities,
        background_load=0.0,
    )


def _run_score_softmax(seed: int, feedback_delay: int, execution_mode: str, steps: int) -> tuple[list[SlotResult], np.ndarray]:
    env = _default_env(seed, feedback_delay, execution_mode)
    env.reset()
    results: list[SlotResult] = []
    rng = np.random.default_rng(seed + 2026)
    for _ in range(steps):
        if execution_mode == "simultaneous":
            portfolios = _policy_portfolios(env, "score-softmax", 5.0, 5)
            results.append(env.joint_step(portfolios, stats=env.portfolio_stats(portfolios)))
        elif execution_mode == "sequential":
            def policy_fn(_user: int, obs: np.ndarray) -> np.ndarray:
                return _policy_from_obs(env, obs, "score-softmax", 5.0, 5)

            results.append(env.sequential_step(policy_fn))
        elif execution_mode == "mixed_age":
            # A lightweight heterogeneous-age sensitivity check: users do not
            # share the same feedback age, but the physical slot is still
            # updated jointly. This isolates stale-summary synchronization from
            # the stronger all-users-same-age assumption.
            portfolios = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float32)
            user_ages = rng.choice(np.array([0, 1, 2, 4, 8], dtype=np.int64), size=env.cfg.n_users)
            for user, age in enumerate(user_ages):
                feedback = env.server_feedback() if int(age) == 0 else env.feedback_buffer.get(int(age))
                obs = env.actor_observation_for_user(user, feedback=feedback)
                portfolios[user] = _policy_from_obs(env, obs, "score-softmax", 5.0, 5)
            results.append(env.joint_step(portfolios, stats=env.portfolio_stats(portfolios)))
        else:
            raise ValueError(execution_mode)
    capacities = np.asarray(env.edge_capacities_mb_s, dtype=np.float64)
    return results, capacities


def state_correlation_and_asynchrony(out_dir: Path, steps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr_rows: list[dict] = []
    for seed in [1, 2, 3, 4, 5]:
        results, _capacities = _run_score_softmax(seed, feedback_delay=8, execution_mode="simultaneous", steps=steps)
        q = np.asarray([r.queue_load_after_mb for r in results], dtype=np.float64)
        for delay in [1, 2, 4, 8]:
            if q.shape[0] <= delay:
                continue
            x = q[:-delay].reshape(-1)
            y = q[delay:].reshape(-1)
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                rho = float("nan")
            else:
                rho = float(np.corrcoef(x, y)[0, 1])
            corr_rows.append({"seed": seed, "delta": delay, "rho_Q": rho})
    _write_table(out_dir / "queue_state_correlation.csv", corr_rows)

    corr_summary = []
    for delay in [1, 2, 4, 8]:
        vals = np.asarray([float(r["rho_Q"]) for r in corr_rows if int(r["delta"]) == delay], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        corr_summary.append(
            {
                "delta": delay,
                "rho_Q_mean": float(np.mean(vals)),
                "rho_Q_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            }
        )
    _write_table(out_dir / "queue_state_correlation_summary.csv", corr_summary)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    delays = [int(r["delta"]) for r in corr_summary]
    means = [float(r["rho_Q_mean"]) for r in corr_summary]
    stds = [float(r["rho_Q_std"]) for r in corr_summary]
    ax.errorbar(delays, means, yerr=stds, marker="o", capsize=3, linewidth=2.0)
    ax.set_xlabel("Feedback age δ (slots)")
    ax.set_ylabel(r"$\rho_Q(\delta)=corr(Q(t-\delta),Q(t))$")
    ax.set_title("Temporal correlation of stale queue states")
    ax.set_xticks(delays)
    ax.set_ylim(-0.35, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig_queue_state_correlation.{ext}", dpi=240)
    plt.close(fig)

    async_rows: list[dict] = []
    for mode in ["simultaneous", "sequential"]:
        for delay in [0, 4, 8]:
            for seed in [1, 2, 3, 4, 5]:
                results, capacities = _run_score_softmax(seed, feedback_delay=delay, execution_mode=mode, steps=steps)
                summary = summarize_results(results)
                summary.update(phase_diagnostics(results, MAPPOSystemConfig(n_users=30, n_edges=35, max_supported_users=80)))
                summary.update(
                    {
                        "execution_mode": mode,
                        "seed": seed,
                        "feedback_delay": delay,
                        "policy": "score-softmax",
                        "steps": steps,
                    }
                )
                async_rows.append(summary)
    for seed in [1, 2, 3, 4, 5]:
        results, capacities = _run_score_softmax(seed, feedback_delay=8, execution_mode="mixed_age", steps=steps)
        summary = summarize_results(results)
        summary.update(phase_diagnostics(results, MAPPOSystemConfig(n_users=30, n_edges=35, max_supported_users=80)))
        summary.update(
            {
                "execution_mode": "mixed_age",
                "seed": seed,
                "feedback_delay": "mixed_0_1_2_4_8",
                "policy": "score-softmax",
                "steps": steps,
            }
        )
        async_rows.append(summary)
    _write_table(out_dir / "asynchrony_score_softmax_raw.csv", async_rows)

    async_summary: list[dict] = []
    for mode in ["simultaneous", "sequential"]:
        for delay in [0, 4, 8]:
            group = [r for r in async_rows if r["execution_mode"] == mode and int(r["feedback_delay"]) == delay]
            item = {"execution_mode": mode, "feedback_delay": delay, "runs": len(group)}
            for metric in MAIN_METRICS:
                vals = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
                item[f"{metric}_mean"] = float(np.mean(vals))
                item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            async_summary.append(item)
    group = [r for r in async_rows if r["execution_mode"] == "mixed_age"]
    item = {"execution_mode": "mixed_age", "feedback_delay": "mixed_0_1_2_4_8", "runs": len(group)}
    for metric in MAIN_METRICS:
        vals = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
        item[f"{metric}_mean"] = float(np.mean(vals))
        item[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    async_summary.append(item)
    _write_table(out_dir / "asynchrony_score_softmax_summary.csv", async_summary)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharex=False)
    for ax, metric, ylabel in [
        (axes[0], "cvar99", "CVaR99 (s)"),
        (axes[1], "queue_oscillation", "Queue oscillation"),
    ]:
        for mode, label in [("simultaneous", "Simultaneous"), ("sequential", "Sequential / partial async")]:
            xs = []
            ys = []
            for row in async_summary:
                if row["execution_mode"] == mode:
                    xs.append(int(row["feedback_delay"]))
                    ys.append(float(row[f"{metric}_mean"]))
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", linewidth=2.0, label=label)
        ax.set_xlabel("Feedback delay δ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Synchronous response stress test using score-softmax")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig_asynchrony_scoresoftmax.{ext}", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    labels = ["sync δ=8", "mixed age", "sequential δ=0"]
    lookup = {(str(r["execution_mode"]), str(r["feedback_delay"])): r for r in async_summary}
    vals = [
        float(lookup[("simultaneous", "8")]["cvar99_mean"]),
        float(lookup[("mixed_age", "mixed_0_1_2_4_8")]["cvar99_mean"]),
        float(lookup[("sequential", "0")]["cvar99_mean"]),
    ]
    bars = ax.bar(labels, vals, color=["#d95f02", "#7570b3", "#1b9e77"], alpha=0.86)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CVaR99 (s)")
    ax.set_title("Effect of heterogeneous feedback ages")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig_mixed_age_sensitivity.{ext}", dpi=240)
    plt.close(fig)


def numeric_consistency_audit(out_dir: Path) -> None:
    summary_rows = _read_csv(ROOT / "results_facr_ablation_analysis" / "facr_ablation_summary_mean_std.csv")
    summary = {row["method"]: row for row in summary_rows}
    audit_rows = []
    checks = [
        ("Actor-only MAPPO", "cvar99_mean", 4.263, 4.454),
        ("Prior-only", "cvar99_mean", 3.538, 3.546),
    ]
    for method, metric, old_a, old_b in checks:
        authoritative = float(summary[method][metric]) if method in summary and metric in summary[method] else float("nan")
        audit_rows.append(
            {
                "item": f"{method} {metric}",
                "authoritative_source": "results_facr_ablation_analysis/facr_ablation_summary_mean_std.csv",
                "authoritative_value": authoritative,
                "conflicting_value_A": old_a,
                "conflicting_value_B": old_b,
                "recommended_paper_value_3dp": round(authoritative, 3) if math.isfinite(authoritative) else "",
                "likely_reason": "mixed draft values from earlier partial/old aggregation; use the clean 3 seeds x 3 delays ablation table as the single source of truth",
            }
        )
    _write_table(out_dir / "numeric_consistency_audit.csv", audit_rows)


def baseline_adaptation_transparency(out_dir: Path) -> None:
    rows = [
        {
            "paper_family": "Towards Risk-Averse Edge DRL",
            "baseline_name": "PPO-DSR-A",
            "kept_from_original": "PPO policy optimization and differential Sharpe-ratio style risk-aware reward.",
            "adapted_for_this_problem": "Action mapped to multi-edge portfolio weights; observations use the unified mobile MEC stale-summary interface; training/evaluation grid aligned with FACR.",
            "not_kept_or_limitation": "Not a full reproduction of the original system model; local-execution assumptions are removed to match the portfolio offloading environment.",
            "main_hyperparameters": "lr=1e-3, clip=0.2, gamma=0.99, GAE lambda=0.95, DSR mu=1.5, DSR eta=0.01.",
        },
        {
            "paper_family": "Towards Risk-Averse Edge DRL",
            "baseline_name": "PPO-MV-A",
            "kept_from_original": "PPO optimizer with mean-variance latency-risk objective.",
            "adapted_for_this_problem": "Continuous action adapter produces portfolio weights over 35 MEC servers under stale feedback.",
            "not_kept_or_limitation": "The original local/offload execution model is replaced by probabilistic multi-edge assignment.",
            "main_hyperparameters": "lr=1e-3, clip=0.2, gamma=0.99, entropy coef=0.005, latency norm=3.0.",
        },
        {
            "paper_family": "LyFOTO / Lyapunov queue control",
            "baseline_name": "LyFOTO-LC",
            "kept_from_original": "Drift-plus-penalty queue-pressure decision principle.",
            "adapted_for_this_problem": "Uses only stale broadcast feedback B(t-delta); restricts decisions to local candidate MEC servers; no fresh queue oracle.",
            "not_kept_or_limitation": "Does not reproduce the complete federated actor-critic, battery/energy virtual queues, or original local-execution candidate system.",
            "main_hyperparameters": "local candidate set, fixed V/temperature, broadcast period=2, delays={0,1,2,4,8}.",
        },
        {
            "paper_family": "TODAuction / incomplete-information MAB",
            "baseline_name": "TODA-MAB-A",
            "kept_from_original": "Auction/MAB-style incomplete-information exploration and empirical-delay bidding.",
            "adapted_for_this_problem": "Bids include stale-aware/tail-aware delay estimates and are mapped to portfolio-compatible server selection.",
            "not_kept_or_limitation": "Auction market details are simplified; the implementation is an adapted decision mechanism rather than the full original protocol stack.",
            "main_hyperparameters": "discounted empirical delay, UCB/tail-aware bidding variant, unified 10k/1k train-test protocol.",
        },
        {
            "paper_family": "QECO / QoE-oriented distributed DRL",
            "baseline_name": "QECO-A",
            "kept_from_original": "D3QN/LSTM/dueling network, target network, replay buffer, and QoE reward idea.",
            "adapted_for_this_problem": "Local-or-edge action is adapted to load-aware top-k edge selection under stale MEC summary.",
            "not_kept_or_limitation": "The original binary local/offloading and device-side QoE details are not fully preserved because this paper has no local execution branch.",
            "main_hyperparameters": "D3QN with replay/target update, load-aware top-k adapter, same seed-delay evaluation grid.",
        },
    ]
    _write_table(out_dir / "baseline_adaptation_transparency.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing statistical and diagnostic evidence for FACR paper.")
    parser.add_argument("--output-dir", default="results_missing_evidence")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    main_baseline_statistics(out_dir, args.bootstrap)
    tail_distribution(out_dir)
    state_correlation_and_asynchrony(out_dir, args.steps)
    numeric_consistency_audit(out_dir)
    baseline_adaptation_transparency(out_dir)
    print(f"Wrote missing-evidence outputs to {out_dir}")


if __name__ == "__main__":
    main()
