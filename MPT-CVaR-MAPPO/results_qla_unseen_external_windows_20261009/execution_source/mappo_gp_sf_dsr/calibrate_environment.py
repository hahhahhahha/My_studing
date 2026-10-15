from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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


def _edge_capacities(n_edges: int, base: float, mode: str, seed: int) -> np.ndarray:
    if mode == "homogeneous":
        return np.full(n_edges, base, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if mode == "heterogeneous":
        multipliers = np.ones(n_edges, dtype=np.float32)
        order = rng.permutation(n_edges)
        n_slow = max(1, int(round(0.2 * n_edges)))
        n_fast = max(1, int(round(0.2 * n_edges)))
        multipliers[order[:n_slow]] = 0.5
        multipliers[order[n_slow : n_slow + n_fast]] = 1.5
        return (base * multipliers).astype(np.float32)
    if mode == "uniform_heterogeneous":
        return rng.uniform(0.5 * base, 1.5 * base, size=n_edges).astype(np.float32)
    raise ValueError(f"Unknown capacity mode: {mode}")


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = values * temperature
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _scores_from_obs(obs: np.ndarray, n_edges: int, include_feedback_age: bool) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if include_feedback_age:
        return obs[..., -(n_edges + 1) : -1]
    return obs[..., -n_edges:]


def _policy_from_scores(scores: np.ndarray, policy: str, temperature: float, topk: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    single = scores.ndim == 1
    if scores.ndim == 1:
        scores = scores[None, :]
    if policy == "score-softmax":
        portfolios = _softmax(scores, temperature).astype(np.float32)
    elif policy == "greedy-score":
        portfolios = np.zeros_like(scores, dtype=np.float32)
        best = np.argmax(scores, axis=1)
        portfolios[np.arange(scores.shape[0]), best] = 1.0
    elif policy == "topk-score":
        k = int(np.clip(topk, 1, scores.shape[1]))
        portfolios = np.zeros_like(scores, dtype=np.float32)
        idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        for user in range(scores.shape[0]):
            portfolios[user, idx[user]] = 1.0 / k
    elif policy == "uniform":
        portfolios = np.full_like(scores, 1.0 / scores.shape[1], dtype=np.float32)
    elif policy == "delayed-jsq":
        portfolios = np.zeros_like(scores, dtype=np.float32)
        best = np.argmax(scores, axis=1)
        portfolios[np.arange(scores.shape[0]), best] = 1.0
    elif policy != "score-softmax":
        raise ValueError(f"Unknown policy: {policy}")
    return portfolios[0] if single else portfolios


def _policy_portfolios(env: JointMAPPOEdgeEnv, policy: str, temperature: float, topk: int) -> np.ndarray:
    if policy == "delayed-jsq":
        feedback = env.feedback_buffer.get(env.feedback_delay)
        # JSQ follows the least delayed waiting/queue pressure. Ties are kept
        # deterministic to expose stale-information herding.
        best = int(np.argmin(feedback.wait_norm))
        portfolios = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float32)
        portfolios[:, best] = 1.0
        return portfolios
    obs = env.actor_observations()
    scores = _scores_from_obs(obs, env.cfg.n_edges, env.cfg.include_feedback_age)
    return _policy_from_scores(scores, policy, temperature, topk).astype(np.float32)


def _policy_from_obs(env: JointMAPPOEdgeEnv, obs: np.ndarray, policy: str, temperature: float, topk: int) -> np.ndarray:
    if policy == "delayed-jsq":
        feedback = env.feedback_buffer.get(env.feedback_delay)
        weights = np.zeros(env.cfg.n_edges, dtype=np.float32)
        weights[int(np.argmin(feedback.wait_norm))] = 1.0
        return weights
    scores = _scores_from_obs(obs, env.cfg.n_edges, env.cfg.include_feedback_age)
    return _policy_from_scores(scores, policy, temperature, topk).astype(np.float32)


def _episode_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        arr = arr[:, None]
    for col in range(arr.shape[1]):
        run = 0
        for value in arr[:, col]:
            if value:
                run += 1
            elif run > 0:
                lengths.append(run)
                run = 0
        if run > 0:
            lengths.append(run)
    return lengths


def _score_delay_metrics(scores: np.ndarray, delays: list[int], topk: int) -> dict[str, float]:
    out: dict[str, float] = {}
    eps = 1e-8
    for delay in delays:
        if delay <= 0 or scores.shape[0] <= delay:
            continue
        errs = []
        flips = []
        k = int(np.clip(topk, 1, scores.shape[1]))
        for t in range(delay, scores.shape[0]):
            current = scores[t]
            stale = scores[t - delay]
            errs.append(float(np.linalg.norm(current - stale) / (np.linalg.norm(current) + eps)))
            curr_top = set(np.argpartition(current, -k)[-k:].tolist())
            stale_top = set(np.argpartition(stale, -k)[-k:].tolist())
            flips.append(1.0 - len(curr_top & stale_top) / k)
        out[f"score_err_d{delay}"] = float(np.mean(errs)) if errs else 0.0
        out[f"topk_flip_d{delay}"] = float(np.mean(flips)) if flips else 0.0
    return out


def _calibration_metrics(
    results: list[SlotResult],
    capacities: np.ndarray,
    slot_s: float,
    score_delay_values: list[int],
    topk: int,
) -> dict[str, float]:
    queue_history = np.asarray([result.queue_load_after_mb for result in results], dtype=np.float64)
    non_empty_lengths = _episode_lengths(queue_history > 1e-8)
    if non_empty_lengths:
        lifetimes = np.asarray(non_empty_lengths, dtype=np.float64)
        mean_life = float(np.mean(lifetimes))
        median_life = float(np.median(lifetimes))
        p90_life = float(np.percentile(lifetimes, 90))
    else:
        mean_life = median_life = p90_life = 0.0

    pressure = np.asarray(
        [
            (result.queue_load_before_mb + result.server_arrival_load_mb)
            / (capacities * slot_s + 1e-8)
            for result in results
        ],
        dtype=np.float64,
    )
    overload_lengths = _episode_lengths(pressure > 1.0)
    if overload_lengths:
        overload = np.asarray(overload_lengths, dtype=np.float64)
        mean_over = float(np.mean(overload))
        median_over = float(np.median(overload))
        p90_over = float(np.percentile(overload, 90))
    else:
        mean_over = median_over = p90_over = 0.0

    scores = np.asarray([result.server_feedback.score for result in results], dtype=np.float64)
    utilization = np.asarray(
        [np.sum(result.server_arrival_load_mb) / (np.sum(capacities) * slot_s + 1e-8) for result in results],
        dtype=np.float64,
    )
    out = {
        "mean_queue_lifetime": mean_life,
        "median_queue_lifetime": median_life,
        "p90_queue_lifetime": p90_life,
        "mean_overload_duration": mean_over,
        "median_overload_duration": median_over,
        "p90_overload_duration": p90_over,
        "mean_max_server_pressure": float(np.mean(np.max(pressure, axis=1))) if pressure.size else 0.0,
        "p90_max_server_pressure": float(np.percentile(np.max(pressure, axis=1), 90)) if pressure.size else 0.0,
        "mean_utilization": float(np.mean(utilization)),
        "p90_utilization": float(np.percentile(utilization, 90)),
    }
    out.update(_score_delay_metrics(scores, score_delay_values, topk))
    return out


def _plot_heatmap(rows: list[dict], out_dir: Path, metric: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    users = sorted({int(row["users"]) for row in rows})
    delays = sorted({int(row["feedback_delay"]) for row in rows})
    matrix = np.full((len(users), len(delays)), np.nan)
    for row in rows:
        i = users.index(int(row["users"]))
        j = delays.index(int(row["feedback_delay"]))
        matrix[i, j] = float(row[metric])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(delays)), labels=delays)
    ax.set_yticks(np.arange(len(users)), labels=users)
    ax.set_xlabel("feedback delay delta (slots)")
    ax.set_ylabel("users N")
    ax.set_title(title)
    for i in range(len(users)):
        for j in range(len(delays)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=180)
    plt.close(fig)


def run_one(args: argparse.Namespace, users: int, delay: int, broadcast_period: int, seed: int) -> dict:
    sys_cfg = MAPPOSystemConfig(
        n_users=users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_compute_mb=args.task_compute_mb,
        task_package_mb=args.task_package_mb,
        max_supported_users=max(users, 80),
    )
    capacities = _edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, seed)
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    traces = load_trace_set(
        args.dataset,
        users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        seed,
        trace_root=Path(args.trace_root),
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.workload,
        seed=seed,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=delay,
        broadcast_period=broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    results = []
    env.reset()
    for _ in range(args.steps):
        start = perf_counter()
        if args.execution_mode == "simultaneous":
            portfolios = _policy_portfolios(env, args.policy, args.score_temperature, args.policy_topk)
            decision_latency = perf_counter() - start
            stats = env.portfolio_stats(portfolios)
            results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency))
        elif args.execution_mode == "sequential":
            def policy_fn(_user: int, obs: np.ndarray) -> np.ndarray:
                return _policy_from_obs(env, obs, args.policy, args.score_temperature, args.policy_topk)

            results.append(env.sequential_step(policy_fn, decision_latency_s=perf_counter() - start))
        else:
            raise ValueError(f"Unknown execution mode: {args.execution_mode}")

    summary = summarize_results(results)
    summary.update(phase_diagnostics(results, sys_cfg))
    summary.update(_calibration_metrics(results, capacities, sys_cfg.slot_s, _parse_ints(args.score_delays), args.score_topk))
    summary.update(
        {
            "dataset": args.dataset,
            "users": users,
            "feedback_delay": delay,
            "broadcast_period": broadcast_period,
            "seed": seed,
            "policy": args.policy,
            "execution_mode": args.execution_mode,
            "slot_s": args.slot_s,
            "n_edges": args.n_edges,
            "edge_compute_mb_s": args.edge_compute_mb_s,
            "capacity_mode": args.capacity_mode,
            "queue_upper_mb": args.queue_upper_mb,
            "task_compute_mb": args.task_compute_mb,
            "task_package_mb": args.task_package_mb,
            "workload": args.workload,
            "poisson_lambda": args.poisson_lambda,
            "burst_prob": args.burst_prob,
            "burst_multiplier": args.burst_multiplier,
            "background_load": args.background_load,
            "steps": args.steps,
        }
    )
    return summary


def _aggregate(rows: list[dict]) -> list[dict]:
    keys = ["execution_mode", "users", "feedback_delay", "broadcast_period"]
    metrics = [
        "average_latency",
        "cvar95",
        "cvar99",
        "queue_oscillation",
        "portfolio_overlap",
        "portfolio_migration_rate",
        "global_demand_concentration",
        "server_selection_gini",
        "queue_overflow_rate",
        "mean_queue_lifetime",
        "median_queue_lifetime",
        "p90_queue_lifetime",
        "mean_overload_duration",
        "median_overload_duration",
        "p90_overload_duration",
        "mean_max_server_pressure",
        "p90_max_server_pressure",
        "mean_utilization",
        "p90_utilization",
    ]
    for delay in _parse_ints(rows[0].get("score_delays", "1,2,4,8")) if rows else []:
        metrics += [f"score_err_d{delay}", f"topk_flip_d{delay}"]
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for group_key, group in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["runs"] = len(group)
        for metric in metrics:
            if metric not in group[0]:
                continue
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values))
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate high-utilization urban RSU-MEC environment.")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default="results_environment_calibration")
    parser.add_argument("--users", default="30,50,80")
    parser.add_argument("--feedback-delays", default="0,1,2,4,8")
    parser.add_argument("--broadcast-periods", default="1,2,4,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--workload", default="poisson_bursty", choices=["uniform", "gaussian", "poisson", "poisson_bursty", "bursty"])
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--policy", default="score-softmax", choices=["score-softmax", "greedy-score", "topk-score", "uniform", "delayed-jsq"])
    parser.add_argument("--execution-mode", default="simultaneous", choices=["simultaneous", "sequential"])
    parser.add_argument("--score-temperature", type=float, default=5.0)
    parser.add_argument("--policy-topk", type=int, default=5)
    parser.add_argument("--score-delays", default="1,2,4,8")
    parser.add_argument("--score-topk", type=int, default=5)
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        / args.dataset
        / args.workload
        / args.capacity_mode
        / f"bg{args.background_load:g}"
        / f"exec_{args.execution_mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    users_list = _parse_ints(args.users)
    delays = _parse_ints(args.feedback_delays)
    periods = _parse_ints(args.broadcast_periods)
    seeds = _parse_ints(args.seeds)
    for users in users_list:
        for delay in delays:
            for period in periods:
                for seed in seeds:
                    print(
                        {
                            "execution_mode": args.execution_mode,
                            "users": users,
                            "delay": delay,
                            "broadcast_period": period,
                            "seed": seed,
                        }
                    )
                    row = run_one(args, users, delay, period, seed)
                    row["score_delays"] = args.score_delays
                    rows.append(row)
                    _write_csv(out_dir / "calibration_runs.csv", rows)

    summary = _aggregate(rows)
    _write_csv(out_dir / "calibration_summary.csv", summary)
    with (out_dir / "calibration_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    for metric, title in [
        ("cvar99_mean", "CVaR99"),
        ("queue_oscillation_mean", "Queue oscillation"),
        ("median_queue_lifetime_mean", "Median queue lifetime"),
        ("median_overload_duration_mean", "Median overload duration"),
        ("score_err_d4_mean", "Score stale error d=4"),
        ("topk_flip_d4_mean", "Top-K flip d=4"),
    ]:
        if summary and metric in summary[0]:
            _plot_heatmap(summary, out_dir, metric, title)


if __name__ == "__main__":
    main()
