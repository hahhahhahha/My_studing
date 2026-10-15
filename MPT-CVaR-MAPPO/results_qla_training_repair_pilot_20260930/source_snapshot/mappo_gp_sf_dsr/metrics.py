from __future__ import annotations

import numpy as np

from repro_risk_averse_edge.metrics import empirical_cvar

from .env import DrainResult, JointMAPPOEdgeEnv, SlotResult
from .config import MAPPOSystemConfig


def jain_fairness(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    denom = x.size * np.sum(x * x)
    if denom <= 1e-12:
        return 1.0
    return float((np.sum(x) ** 2) / denom)


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    if np.min(x) < 0:
        x = x - np.min(x)
    total = np.sum(x)
    if total <= 1e-12:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * total)) - (n + 1.0) / n)


def slot_metrics(result: SlotResult) -> dict[str, float]:
    active_lat = result.per_user_mean_latency[result.per_user_tasks > 0]
    return {
        "average_latency": result.system_average_latency,
        "slot_tasks": int(np.sum(result.per_user_tasks)),
        "mean_portfolio_entropy": result.portfolio_stats.mean_entropy,
        "mean_portfolio_concentration": result.portfolio_stats.mean_concentration,
        "global_demand_concentration": result.portfolio_stats.global_demand_concentration,
        "max_server_expected_demand_share": result.portfolio_stats.max_server_expected_demand_share,
        "server_selection_gini": gini(result.server_assignment_counts),
        "jain_fairness_latency": jain_fairness(1.0 / np.maximum(active_lat, 1e-8)),
        "server_load_variance": float(np.var(result.queue_load_after_mb)),
        "queue_task_count_variance": float(np.var(result.queue_task_count_after)),
        "available_resource_variance": float(np.var(result.server_feedback.available_resource_mb)),
        "average_queue_length_mb": float(np.mean(result.queue_load_after_mb)),
        "max_queue_length_mb": float(np.max(result.queue_load_after_mb)),
        "queue_overflow_rate": float(result.overflow_count / max(1, np.sum(result.per_user_tasks))),
        "decision_latency_ms": float(result.decision_latency_s * 1000.0),
    }


def queue_oscillation(queue_history: np.ndarray, capacities_mb_s: np.ndarray, slot_s: float) -> float:
    history = np.asarray(queue_history, dtype=np.float64)
    capacities = np.asarray(capacities_mb_s, dtype=np.float64)
    if history.size == 0:
        return float("nan")
    pressure = history / (capacities[None, :] * slot_s + 1e-8)
    return float(np.std(pressure, axis=0).mean())


def portfolio_overlap(
    portfolios: np.ndarray,
    user_loads_mb: np.ndarray,
    capacities_mb_s: np.ndarray,
    slot_s: float,
) -> float:
    w = np.asarray(portfolios, dtype=np.float64)
    loads = np.asarray(user_loads_mb, dtype=np.float64)
    capacities = np.asarray(capacities_mb_s, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] < 2:
        return 0.0
    active = loads > 0
    if np.sum(active) < 2:
        return 0.0
    x = w[active] * loads[active, None] / (capacities[None, :] * slot_s + 1e-8)
    norms = np.linalg.norm(x, axis=1) + 1e-8
    x_norm = x / norms[:, None]
    sims = x_norm @ x_norm.T
    upper = np.triu_indices(x_norm.shape[0], k=1)
    return float(np.mean(sims[upper])) if upper[0].size else 0.0


def portfolio_migration_rate(results: list[SlotResult]) -> float:
    """Mean L1 distance between consecutive per-user portfolios, averaged over
    users and time. Captures how fast portfolios are swinging — a direct
    signal of herding oscillation under stale feedback.
    """
    if len(results) < 2:
        return 0.0
    diffs = []
    for prev, curr in zip(results[:-1], results[1:]):
        w_prev = np.asarray(prev.portfolios, dtype=np.float64)
        w_curr = np.asarray(curr.portfolios, dtype=np.float64)
        if w_prev.shape != w_curr.shape:
            continue
        # Per-user L1, then mean over users; do not normalize — raw mass moved.
        per_user = np.abs(w_curr - w_prev).sum(axis=1)
        diffs.append(float(np.mean(per_user)) if per_user.size else 0.0)
    return float(np.mean(diffs)) if diffs else 0.0


def episode_lengths(mask: np.ndarray) -> list[int]:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        arr = arr[:, None]
    lengths: list[int] = []
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


def phase_diagnostics(results: list[SlotResult], sys_cfg: MAPPOSystemConfig) -> dict[str, float]:
    capacities = np.full(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, dtype=np.float64)
    queue_history = np.asarray([r.queue_load_after_mb for r in results], dtype=np.float64)
    overlaps = [
        portfolio_overlap(
            r.portfolios,
            r.per_user_tasks.astype(np.float64) * sys_cfg.task_compute_mb,
            capacities,
            sys_cfg.slot_s,
        )
        for r in results
    ]
    non_empty = np.asarray(episode_lengths(queue_history > 1e-8), dtype=np.float64)
    p90_queue_lifetime = float(np.percentile(non_empty, 90)) if non_empty.size else 0.0
    return {
        "queue_oscillation": queue_oscillation(queue_history, capacities, sys_cfg.slot_s),
        "portfolio_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "portfolio_migration_rate": portfolio_migration_rate(results),
        "p90_queue_lifetime": p90_queue_lifetime,
    }


def summarize_results(
    results: list[SlotResult],
    drain: DrainResult | None = None,
    sys_cfg: MAPPOSystemConfig | None = None,
) -> dict[str, float]:
    latencies = np.asarray([lat for result in results for lat in result.latencies], dtype=np.float64)
    per_user: dict[int, list[float]] = {}
    for result in results:
        for record in result.records:
            if record.dropped:
                continue
            per_user.setdefault(record.user, []).append(record.latency_s)

    n_generated = int(np.sum([len(r.records) for r in results]))
    n_accepted = int(np.sum([len(r.accepted_records) for r in results]))
    n_dropped = int(np.sum([r.n_dropped_tasks for r in results]))
    n_unfinished = int(drain.n_unfinished_tasks) if drain is not None else 0

    deadline_s = float(sys_cfg.latency_deadline_s) if sys_cfg is not None else 10.0
    n_deadline_violations = int(np.sum(latencies > deadline_s)) if latencies.size else 0
    p_deadline_given_accepted = (
        float(n_deadline_violations / latencies.size) if latencies.size else float("nan")
    )
    cvar95 = empirical_cvar(latencies, 0.95)
    cvar99 = empirical_cvar(latencies, 0.99)
    capped_latencies = np.minimum(latencies, deadline_s)
    capped_cvar95 = empirical_cvar(capped_latencies, 0.95)
    capped_cvar99 = empirical_cvar(capped_latencies, 0.99)
    tail95_count = int(np.ceil(0.05 * latencies.size)) if latencies.size else 0
    tail99_count = int(np.ceil(0.01 * latencies.size)) if latencies.size else 0

    offered_workload = float(
        np.sum(
            [
                np.sum(r.server_arrival_load_mb)
                for r in results
            ]
        )
    )
    served_during_slots = float(
        np.sum(
            [
                np.sum(r.served_load_mb) if r.served_load_mb is not None else 0.0
                for r in results
            ]
        )
    )
    dropped_workload = float(
        np.sum(
            [
                np.sum(r.dropped_load_mb) if r.dropped_load_mb is not None else 0.0
                for r in results
            ]
        )
    )
    served_during_drain = float(drain.workload_served_mb_during_drain) if drain is not None else 0.0
    unfinished_workload = (
        float(np.sum(drain.workload_unfinished_mb_per_server)) if drain is not None else 0.0
    )
    episode_mass_error = abs(
        offered_workload
        - (served_during_slots + served_during_drain + dropped_workload + unfinished_workload)
    )

    summary = {
        "average_latency": float(np.mean(latencies)) if latencies.size else float("nan"),
        "cvar95": cvar95,
        "cvar99": cvar99,
        "average_latency_uncapped": float(np.mean(latencies)) if latencies.size else float("nan"),
        "cvar95_uncapped": cvar95,
        "cvar99_uncapped": cvar99,
        "average_latency_capped_at_deadline": (
            float(np.mean(capped_latencies)) if capped_latencies.size else float("nan")
        ),
        "cvar95_capped_at_deadline": capped_cvar95,
        "cvar99_capped_at_deadline": capped_cvar99,
        "latency_deadline_s": deadline_s,
        "tail95_sample_count": tail95_count,
        "tail99_sample_count": tail99_count,
        "n_tasks": int(latencies.size),
        "n_generated_tasks": n_generated,
        "n_accepted_tasks": n_accepted,
        "n_dropped_tasks": n_dropped,
        "n_unfinished_tasks": n_unfinished,
        "p_drop": float(n_dropped / max(1, n_generated)),
        "p_drop_or_unfinished": float((n_dropped + n_unfinished) / max(1, n_generated)),
        "n_deadline_violations": n_deadline_violations,
        "p_deadline_given_accepted": p_deadline_given_accepted,
        # Backward-compatible alias; canonical paper text must use the
        # conditional name above and report p_drop separately.
        "p_ddl": p_deadline_given_accepted,
        "drain_horizon": int(drain.drain_horizon) if drain is not None else 0,
        "drain_finished": bool(drain.finished) if drain is not None else True,
        "offered_workload_mb_total": offered_workload,
        "served_workload_mb_during_slots": served_during_slots,
        "served_workload_mb_during_drain": served_during_drain,
        "dropped_workload_mb_total": dropped_workload,
        "unfinished_workload_mb_total": unfinished_workload,
        "episode_mass_conservation_error": episode_mass_error,
        "inference_time_ms": float(np.mean([r.decision_latency_s for r in results]) * 1000.0) if results else float("nan"),
        "server_load_variance": float(np.mean([np.var(r.queue_load_after_mb) for r in results])) if results else float("nan"),
        "queue_task_count_variance": float(np.mean([np.var(r.queue_task_count_after) for r in results])) if results else float("nan"),
        "available_resource_variance": float(np.mean([np.var(r.server_feedback.available_resource_mb) for r in results]))
        if results
        else float("nan"),
        "average_queue_length_mb": float(np.mean([np.mean(r.queue_load_after_mb) for r in results])) if results else float("nan"),
        "max_queue_length_mb": float(np.max([np.max(r.queue_load_after_mb) for r in results])) if results else float("nan"),
        "queue_overflow_rate": float(
            np.sum([r.overflow_count for r in results]) / max(1, n_generated)
        )
        if results
        else float("nan"),
        "mean_portfolio_entropy": float(np.mean([r.portfolio_stats.mean_entropy for r in results])) if results else float("nan"),
        "mean_portfolio_concentration": float(np.mean([r.portfolio_stats.mean_concentration for r in results]))
        if results
        else float("nan"),
        "global_demand_concentration": float(np.mean([r.portfolio_stats.global_demand_concentration for r in results]))
        if results
        else float("nan"),
        "max_server_expected_demand_share": float(
            np.mean([r.portfolio_stats.max_server_expected_demand_share for r in results])
        )
        if results
        else float("nan"),
        "server_selection_gini": float(np.mean([gini(r.server_assignment_counts) for r in results]))
        if results
        else float("nan"),
    }
    if drain is not None:
        summary["workload_unfinished_mb_total"] = unfinished_workload
    for user, values in sorted(per_user.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary[f"user{user}_average_latency"] = float(np.mean(arr)) if arr.size else float("nan")
        summary[f"user{user}_cvar95"] = empirical_cvar(arr, 0.95)
        summary[f"user{user}_cvar99"] = empirical_cvar(arr, 0.99)
    if per_user:
        user_means = np.asarray([summary[f"user{user}_average_latency"] for user in sorted(per_user)], dtype=np.float64)
        summary["per_user_latency_jain"] = jain_fairness(1.0 / np.maximum(user_means, 1e-8))
    else:
        summary["per_user_latency_jain"] = float("nan")
    return summary


def finalize_evaluation(
    env: JointMAPPOEdgeEnv,
    results: list[SlotResult],
    sys_cfg: MAPPOSystemConfig,
) -> tuple[dict[str, float], DrainResult | None]:
    """Apply the same terminal drain and reliability summary to every method."""
    drain = env.drain() if bool(sys_cfg.finite_buffer_protocol) else None
    summary = summarize_results(results, drain=drain, sys_cfg=sys_cfg)
    summary["finite_buffer_protocol"] = bool(sys_cfg.finite_buffer_protocol)
    summary["latency_is_uncapped"] = True
    summary["drop_penalty_s_training_only"] = float(sys_cfg.drop_penalty_s)
    if env.canonical_schedule is not None:
        summary[f"{env.canonical_schedule.role}_split_id"] = env.canonical_schedule.schedule_id
        summary[f"{env.canonical_schedule.role}_schedule_sha256"] = env.canonical_schedule.content_sha256
    return summary, drain
