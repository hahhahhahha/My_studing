from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from repro_risk_averse_edge.data import load_trace_set

from .buffer import MAPPOBuffer
from .canonical_protocol import install_schedule_for_env
from .config import MAPPOConfig, MAPPOSystemConfig, ROOT, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv
from .evaluate import evaluate_policy
from .metrics import finalize_evaluation, slot_metrics
from .models import MAPPOPolicy, MAPPOUpdateStats
from .plots import make_training_plots
from .rewards import MAPPORewarder, compute_dsr_input


def build_output_dir(base: Path, dataset: str, users: int, distribution: str) -> Path:
    if base.name == distribution and base.parent.name == f"{users}u":
        return base
    return base / dataset / f"{users}u" / distribution


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def edge_capacities(n_edges: int, base: float, mode: str, seed: int) -> np.ndarray:
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


def eval_diagnostics(results, capacities: np.ndarray, slot_s: float) -> dict[str, float]:
    if not results:
        return {}
    queue_history = np.asarray([result.queue_load_after_mb for result in results], dtype=np.float64)
    pressure_history = queue_history / (capacities[None, :] * slot_s + 1e-8)
    non_empty = np.asarray(episode_lengths(queue_history > 1e-8), dtype=np.float64)
    pressure = np.asarray(
        [
            (result.queue_load_before_mb + result.server_arrival_load_mb)
            / (capacities * slot_s + 1e-8)
            for result in results
        ],
        dtype=np.float64,
    )
    overload = np.asarray(episode_lengths(pressure > 1.0), dtype=np.float64)
    max_pressure = np.max(pressure, axis=1) if pressure.size else np.asarray([], dtype=np.float64)
    utilization = np.asarray(
        [np.sum(result.server_arrival_load_mb) / (np.sum(capacities) * slot_s + 1e-8) for result in results],
        dtype=np.float64,
    )
    selected_pressures: list[float] = []
    portfolio_pressures: list[float] = []
    high_pressure_selection: list[float] = []
    low_pressure_selection: list[float] = []
    for result, slot_pressure in zip(results, pressure):
        counts = np.asarray(result.server_assignment_counts, dtype=np.float64)
        total = float(np.sum(counts))
        if total > 0:
            selected_pressures.append(float(np.sum(counts * slot_pressure) / total))
            high_pressure_selection.append(float(np.sum(counts[slot_pressure > 1.0]) / total))
            low_pressure_selection.append(float(np.sum(counts[slot_pressure < 0.5]) / total))
        active = np.asarray(result.per_user_tasks) > 0
        if np.any(active):
            user_pressure = np.asarray(result.portfolios, dtype=np.float64) @ slot_pressure
            portfolio_pressures.append(float(np.mean(user_pressure[active])))

    def mean_or_zero(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else 0.0

    def median_or_zero(values: np.ndarray) -> float:
        return float(np.median(values)) if values.size else 0.0

    def p90_or_zero(values: np.ndarray) -> float:
        return float(np.percentile(values, 90)) if values.size else 0.0

    migrations = []
    for prev, curr in zip(results[:-1], results[1:]):
        if prev.portfolios.shape == curr.portfolios.shape:
            migrations.append(float(np.mean(np.abs(curr.portfolios - prev.portfolios).sum(axis=1))))

    return {
        "queue_oscillation": float(np.std(pressure_history, axis=0).mean()),
        "mean_queue_lifetime": mean_or_zero(non_empty),
        "median_queue_lifetime": median_or_zero(non_empty),
        "p90_queue_lifetime": p90_or_zero(non_empty),
        "mean_overload_duration": mean_or_zero(overload),
        "median_overload_duration": median_or_zero(overload),
        "p90_overload_duration": p90_or_zero(overload),
        "mean_max_server_pressure": mean_or_zero(max_pressure),
        "p90_max_server_pressure": p90_or_zero(max_pressure),
        "selected_server_pressure": float(np.mean(selected_pressures)) if selected_pressures else 0.0,
        "p90_selected_server_pressure": float(np.percentile(selected_pressures, 90)) if selected_pressures else 0.0,
        "portfolio_weighted_pressure": float(np.mean(portfolio_pressures)) if portfolio_pressures else 0.0,
        "high_pressure_selection_ratio": float(np.mean(high_pressure_selection)) if high_pressure_selection else 0.0,
        "low_pressure_selection_ratio": float(np.mean(low_pressure_selection)) if low_pressure_selection else 0.0,
        "mean_utilization": mean_or_zero(utilization),
        "p90_utilization": p90_or_zero(utilization),
        "portfolio_migration_rate": float(np.mean(migrations)) if migrations else 0.0,
    }


def softmax_negative(cost: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -np.asarray(cost, dtype=np.float64) / max(float(temperature), 1e-8)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def lyapunov_prior_portfolios(
    env: JointMAPPOEdgeEnv,
    tradeoff_v: float,
    temperature: float,
    fresh_state: bool = False,
) -> np.ndarray:
    """Stale/fresh drift-plus-delay prior used only as a soft portfolio guide.

    The prior mirrors the strengthened Lyapunov-Q baseline's cost model, but it
    remains an action adapter: V4 still provides the learned portfolio and this
    prior only nudges it toward low estimated delay / low stale queue pressure.
    """

    feedback = env.server_feedback() if fresh_state else env.feedback_buffer.get(env.feedback_delay)
    locations = env._locations()
    costs = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float64)
    normalized_load = feedback.wait_s / max(env.cfg.slot_s, 1e-8)
    service_time = env.cfg.task_compute_mb / np.maximum(env.edge_capacities_mb_s.astype(np.float64), 1e-8)
    for user in range(env.cfg.n_users):
        communication = np.asarray(
            [
                np.linalg.norm(locations[user] - env.traces.edge_positions[edge]) / 10_000.0
                + env.cfg.task_package_mb / max(env.edge_capacities_mb_s[edge], 1e-8)
                for edge in range(env.cfg.n_edges)
            ],
            dtype=np.float64,
        )
        estimated_delay = communication + service_time + feedback.wait_s
        costs[user] = float(tradeoff_v) * estimated_delay + normalized_load * service_time
    return softmax_negative(costs, temperature)


def normalize_portfolios(portfolios: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(portfolios, dtype=np.float64), 1e-8)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def local_critic_observations(actor_obs: np.ndarray, portfolios: np.ndarray) -> np.ndarray:
    """Critic input for local-critic ablations.

    It intentionally uses only the same stale/local observation visible to the
    actor plus the executing portfolio of that user. It does not read fresh
    queues, global server summaries, or other users' current portfolios.
    """

    return np.concatenate(
        [
            np.asarray(actor_obs, dtype=np.float32),
            normalize_portfolios(portfolios),
        ],
        axis=1,
    ).astype(np.float32)


def apply_lyapunov_prior(
    portfolios: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    previous_portfolios: np.ndarray | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> np.ndarray:
    weight = float(getattr(args, "lyapunov_prior_weight", 0.0))
    weight = float(np.clip(weight, 0.0, 1.0))
    actor_portfolios = normalize_portfolios(portfolios)
    prior: np.ndarray | None = None
    if weight > 0.0:
        prior = lyapunov_prior_portfolios(
            env,
            tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
            temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
            fresh_state=bool(getattr(args, "lyapunov_prior_fresh_state", False)),
        )
        mixed = (1.0 - weight) * actor_portfolios + weight * prior
    else:
        mixed = actor_portfolios

    projection_activation_rate = 0.0
    pre_projection_l1_mean = 0.0
    pre_projection_l1_max = 0.0
    if previous_portfolios is not None and previous_portfolios.shape == mixed.shape:
        smoothing = float(np.clip(getattr(args, "lyapunov_prior_smoothing", 0.0), 0.0, 1.0))
        if smoothing > 0.0:
            mixed = (1.0 - smoothing) * mixed + smoothing * previous_portfolios

        max_step = float(getattr(args, "lyapunov_prior_max_step", 0.0))
        if max_step > 0.0:
            delta = mixed - previous_portfolios
            l1 = np.sum(np.abs(delta), axis=1, keepdims=True)
            pre_projection_l1_mean = float(np.mean(l1))
            pre_projection_l1_max = float(np.max(l1))
            scale = np.minimum(1.0, max_step / np.maximum(l1, 1e-12))
            projection_activation_rate = float(np.mean(scale < 1.0 - 1e-9))
            mixed = previous_portfolios + delta * scale

    shaped = normalize_portfolios(mixed)
    if diagnostics is not None:
        eps = 1e-12
        if prior is not None:
            kl = np.sum(shaped * (np.log(np.maximum(shaped, eps)) - np.log(np.maximum(prior, eps))), axis=1)
            diagnostics.setdefault("lyapunov_prior_kl_w_to_prior", []).append(float(np.mean(kl)))
        else:
            diagnostics.setdefault("lyapunov_prior_kl_w_to_prior", []).append(float("nan"))
        diagnostics.setdefault("projection_activation_rate", []).append(projection_activation_rate)
        diagnostics.setdefault("pre_projection_l1_mean", []).append(pre_projection_l1_mean)
        diagnostics.setdefault("pre_projection_l1_max", []).append(pre_projection_l1_max)
        if previous_portfolios is not None and previous_portfolios.shape == shaped.shape:
            post_l1 = np.sum(np.abs(shaped - previous_portfolios), axis=1)
            diagnostics.setdefault("post_step_l1_mean", []).append(float(np.mean(post_l1)))
            diagnostics.setdefault("post_step_l1_max", []).append(float(np.max(post_l1)))
        else:
            diagnostics.setdefault("post_step_l1_mean", []).append(0.0)
            diagnostics.setdefault("post_step_l1_max", []).append(0.0)
    return shaped


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project row-wise vectors onto the probability simplex."""

    x = np.asarray(v, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for simplex projection, got shape {x.shape}")
    u = np.sort(x, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, x.shape[1] + 1, dtype=np.float64)
    cond = u - cssv / ind > 0
    rho = np.sum(cond, axis=1) - 1
    theta = cssv[np.arange(x.shape[0]), rho] / (rho + 1.0)
    return np.maximum(x - theta[:, None], 0.0).astype(np.float32)


def apply_residual_lyapunov_prior(
    latent_actions: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    previous_portfolios: np.ndarray | None = None,
    diagnostics: dict[str, list[float]] | None = None,
    state: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Confidence-calibrated residual policy around a stale Lyapunov prior.

    The actor no longer proposes a complete portfolio. Its latent vector is
    centered into a zero-sum direction in the tangent space of the simplex, and
    a confidence-controlled radius determines how far the executed portfolio may
    deviate from the stale Lyapunov prior.
    """

    prior = lyapunov_prior_portfolios(
        env,
        tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
        temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
        fresh_state=bool(getattr(args, "lyapunov_prior_fresh_state", False)),
    ).astype(np.float32)
    feedback = env.feedback_buffer.get(env.feedback_delay)
    wait = np.asarray(feedback.wait_s, dtype=np.float64)
    if state is not None and "_last_residual_prior_wait" in state:
        prev_wait = np.asarray(state["_last_residual_prior_wait"], dtype=np.float64)
        volatility = float(np.mean(np.abs(wait - prev_wait) / max(env.cfg.slot_s, 1e-8)))
    else:
        volatility = 0.0
    if state is not None:
        state["_last_residual_prior_wait"] = wait.copy()

    delta_age = float(env.feedback_source_age(feedback))
    eta_delta = float(getattr(args, "residual_confidence_eta_delta", 0.10))
    eta_vol = float(getattr(args, "residual_confidence_eta_volatility", 0.10))
    confidence = float(np.exp(-eta_delta * delta_age - eta_vol * volatility))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    r_min = float(getattr(args, "residual_radius_min", 0.0))
    r_max = float(getattr(args, "residual_radius_max", 0.25))
    radius = r_min + (r_max - r_min) * (1.0 - confidence)

    raw = np.asarray(latent_actions, dtype=np.float64)
    centered = raw - np.mean(raw, axis=1, keepdims=True)
    norm = np.sum(np.abs(centered), axis=1, keepdims=True)
    direction = centered / np.maximum(norm, 1e-8)
    proposed = project_simplex(prior + radius * direction)

    projection_activation_rate = 0.0
    pre_projection_l1_mean = 0.0
    pre_projection_l1_max = 0.0
    mixed = proposed.astype(np.float64)
    if previous_portfolios is not None and previous_portfolios.shape == mixed.shape:
        smoothing = float(np.clip(getattr(args, "lyapunov_prior_smoothing", 0.0), 0.0, 1.0))
        if smoothing > 0.0:
            mixed = (1.0 - smoothing) * mixed + smoothing * previous_portfolios

        max_step = float(getattr(args, "lyapunov_prior_max_step", 0.0))
        if max_step > 0.0:
            delta = mixed - previous_portfolios
            l1 = np.sum(np.abs(delta), axis=1, keepdims=True)
            pre_projection_l1_mean = float(np.mean(l1))
            pre_projection_l1_max = float(np.max(l1))
            scale = np.minimum(1.0, max_step / np.maximum(l1, 1e-12))
            projection_activation_rate = float(np.mean(scale < 1.0 - 1e-9))
            mixed = previous_portfolios + delta * scale

    shaped = normalize_portfolios(mixed)
    if diagnostics is not None:
        eps = 1e-12
        kl = np.sum(shaped * (np.log(np.maximum(shaped, eps)) - np.log(np.maximum(prior, eps))), axis=1)
        diagnostics.setdefault("lyapunov_prior_kl_w_to_prior", []).append(float(np.mean(kl)))
        diagnostics.setdefault("residual_prior_confidence", []).append(confidence)
        diagnostics.setdefault("residual_radius", []).append(radius)
        diagnostics.setdefault("residual_wait_volatility", []).append(volatility)
        diagnostics.setdefault("residual_direction_l1", []).append(float(np.mean(np.sum(np.abs(direction), axis=1))))
        diagnostics.setdefault("projection_activation_rate", []).append(projection_activation_rate)
        diagnostics.setdefault("pre_projection_l1_mean", []).append(pre_projection_l1_mean)
        diagnostics.setdefault("pre_projection_l1_max", []).append(pre_projection_l1_max)
        if previous_portfolios is not None and previous_portfolios.shape == shaped.shape:
            post_l1 = np.sum(np.abs(shaped - previous_portfolios), axis=1)
            diagnostics.setdefault("post_step_l1_mean", []).append(float(np.mean(post_l1)))
            diagnostics.setdefault("post_step_l1_max", []).append(float(np.max(post_l1)))
        else:
            diagnostics.setdefault("post_step_l1_mean", []).append(0.0)
            diagnostics.setdefault("post_step_l1_max", []).append(0.0)
    return shaped


def apply_actor_amplitude_residual_prior(
    latent_actions: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    previous_portfolios: np.ndarray | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Actor-amplitude residual around stale Lyapunov prior.

    Unlike ``apply_residual_lyapunov_prior``, this v2 adapter does not force the
    residual direction to unit L1 norm and does not enlarge the radius as a fixed
    function of feedback age. The actor directly observes age and score features
    and learns both residual direction and amplitude.
    """

    prior = lyapunov_prior_portfolios(
        env,
        tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
        temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
        fresh_state=bool(getattr(args, "lyapunov_prior_fresh_state", False)),
    ).astype(np.float32)
    latent = np.asarray(latent_actions, dtype=np.float64)
    if latent.shape[1] < env.cfg.n_edges + 1:
        raise ValueError(
            f"Residual-v2 expects latent shape (N, M+1), got {latent.shape} for M={env.cfg.n_edges}"
        )
    raw_direction = latent[:, : env.cfg.n_edges]
    centered = raw_direction - np.mean(raw_direction, axis=1, keepdims=True)
    direction = np.tanh(centered)
    raw_l1 = np.sum(np.abs(direction), axis=1, keepdims=True)
    direction_l1_cap = float(getattr(args, "residual_direction_l1_cap", 1.0))
    scale = np.minimum(1.0, direction_l1_cap / np.maximum(raw_l1, 1e-12))
    clipped_direction = direction * scale
    amplitude = 1.0 / (1.0 + np.exp(-np.clip(latent[:, env.cfg.n_edges : env.cfg.n_edges + 1], -30.0, 30.0)))
    r_max = float(getattr(args, "residual_radius_max", 0.05))
    proposed = project_simplex(prior + r_max * amplitude * clipped_direction)

    projection_activation_rate = 0.0
    pre_projection_l1_mean = 0.0
    pre_projection_l1_max = 0.0
    mixed = proposed.astype(np.float64)
    if previous_portfolios is not None and previous_portfolios.shape == mixed.shape:
        smoothing = float(np.clip(getattr(args, "lyapunov_prior_smoothing", 0.0), 0.0, 1.0))
        if smoothing > 0.0:
            mixed = (1.0 - smoothing) * mixed + smoothing * previous_portfolios

        max_step = float(getattr(args, "lyapunov_prior_max_step", 0.0))
        if max_step > 0.0:
            delta = mixed - previous_portfolios
            l1 = np.sum(np.abs(delta), axis=1, keepdims=True)
            pre_projection_l1_mean = float(np.mean(l1))
            pre_projection_l1_max = float(np.max(l1))
            step_scale = np.minimum(1.0, max_step / np.maximum(l1, 1e-12))
            projection_activation_rate = float(np.mean(step_scale < 1.0 - 1e-9))
            mixed = previous_portfolios + delta * step_scale

    shaped = normalize_portfolios(mixed)
    if diagnostics is not None:
        eps = 1e-12
        kl = np.sum(shaped * (np.log(np.maximum(shaped, eps)) - np.log(np.maximum(prior, eps))), axis=1)
        diagnostics.setdefault("lyapunov_prior_kl_w_to_prior", []).append(float(np.mean(kl)))
        diagnostics.setdefault("residual_actor_amplitude", []).append(float(np.mean(amplitude)))
        diagnostics.setdefault("residual_radius", []).append(float(r_max * np.mean(amplitude)))
        diagnostics.setdefault("residual_direction_l1_raw", []).append(float(np.mean(raw_l1)))
        diagnostics.setdefault("residual_direction_l1", []).append(float(np.mean(np.sum(np.abs(clipped_direction), axis=1))))
        diagnostics.setdefault("projection_activation_rate", []).append(projection_activation_rate)
        diagnostics.setdefault("pre_projection_l1_mean", []).append(pre_projection_l1_mean)
        diagnostics.setdefault("pre_projection_l1_max", []).append(pre_projection_l1_max)
        if previous_portfolios is not None and previous_portfolios.shape == shaped.shape:
            post_l1 = np.sum(np.abs(shaped - previous_portfolios), axis=1)
            diagnostics.setdefault("post_step_l1_mean", []).append(float(np.mean(post_l1)))
            diagnostics.setdefault("post_step_l1_max", []).append(float(np.max(post_l1)))
        else:
            diagnostics.setdefault("post_step_l1_mean", []).append(0.0)
            diagnostics.setdefault("post_step_l1_max", []).append(0.0)
    return shaped


def parse_mixed_feedback_delays(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    delays: list[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            delays.append(max(0, int(item)))
    return sorted(set(delays))


def residual_v3_prior_and_volatility(
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    state: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    prior = lyapunov_prior_portfolios(
        env,
        tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
        temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
        fresh_state=False,
    ).astype(np.float32)
    feedback = env.feedback_buffer.get(env.feedback_delay)
    wait = np.asarray(feedback.wait_s, dtype=np.float64)
    volatility = 0.0
    state_key = f"_last_residual_v3_wait_delay_{int(env.feedback_delay)}"
    if state is not None and state_key in state:
        prev_wait = np.asarray(state[state_key], dtype=np.float64)
        if prev_wait.shape == wait.shape:
            volatility = float(np.mean(np.abs(wait - prev_wait)) / max(float(env.cfg.slot_s), 1e-8))
    if state is not None:
        state[state_key] = wait.copy()
    return prior, float(max(volatility, 0.0))


def augment_residual_v3_actor_observations(
    base_actor_obs: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    state: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    prior, volatility = residual_v3_prior_and_volatility(env, args, state=state)
    vol_col = np.full((env.cfg.n_users, 1), volatility, dtype=np.float32)
    augmented = np.concatenate([base_actor_obs.astype(np.float32), prior, vol_col], axis=1).astype(np.float32)
    return augmented, prior, volatility


def augment_lcc_actor_observations(
    base_actor_obs: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    rewarder: MAPPORewarder,
    previous_portfolios: np.ndarray | None = None,
    state: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Append stale Lyapunov-coordinator state without exposing fresh queues.

    The actor still outputs the complete portfolio.  The extra state contains
    only stale broadcast wait summaries, virtual risk queues, and the previous
    executed portfolio so MAPPO can learn coordinated, low-migration actions.
    """

    feedback = env.feedback_buffer.get(env.feedback_delay)
    wait_norm = np.asarray(feedback.wait_norm, dtype=np.float32)
    age_norm = float(env.feedback_source_age(feedback)) / max(float(env.cfg.feedback_age_norm), 1e-8)
    use_stale_margin = bool(getattr(args, "lcc_use_stale_margin", True)) and not bool(
        getattr(args, "lcc_no_stale_margin", False)
    )
    if use_stale_margin:
        state_key = f"_last_lcc_wait_norm_delay_{int(env.feedback_delay)}"
        if state is not None and state_key in state and np.asarray(state[state_key]).shape == wait_norm.shape:
            server_volatility = np.abs(wait_norm - np.asarray(state[state_key], dtype=np.float32))
        else:
            server_volatility = np.zeros_like(wait_norm, dtype=np.float32)
        if state is not None:
            state[state_key] = wait_norm.copy()
        stale_margin = float(getattr(args, "lcc_stale_margin", 1.0)) * age_norm * server_volatility
        wait_norm = np.clip(wait_norm + stale_margin, 0.0, 2.0).astype(np.float32)
    stale_risk = np.tile(wait_norm[None, :], (env.cfg.n_users, 1))
    if previous_portfolios is None or np.asarray(previous_portfolios).shape != (env.cfg.n_users, env.cfg.n_edges):
        previous = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
    else:
        previous = np.asarray(previous_portfolios, dtype=np.float32)
    z_norm = np.full(
        (env.cfg.n_users, 1),
        np.tanh(float(getattr(rewarder, "z_queue", 0.0)) / 10.0),
        dtype=np.float32,
    )
    h_norm = np.full(
        (env.cfg.n_users, 1),
        np.tanh(float(getattr(rewarder, "h_queue", 0.0)) / 10.0),
        dtype=np.float32,
    )
    return np.concatenate([base_actor_obs.astype(np.float32), stale_risk, previous, z_norm, h_norm], axis=1).astype(np.float32)


def lcc_marginal_risk_prices(
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    rewarder: MAPPORewarder,
    previous_portfolios: np.ndarray | None = None,
    state: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-user, per-edge marginal risk prices for LCC-price MAPPO.

    This is intentionally not a portfolio prior.  It produces an M-dimensional
    risk-price feature for each user and lets the MAPPO actor retain full
    control over the final portfolio.  To preserve the stale-feedback setting,
    queue and wait terms are computed from the delayed broadcast summary rather
    than ``env.server_feedback()``.
    """

    feedback = env.feedback_buffer.get(env.feedback_delay)
    n_users, n_edges = env.cfg.n_users, env.cfg.n_edges
    eps = 1e-8
    latency_scale = max(float(args.latency_norm), eps)
    capacities = np.asarray(env.edge_capacities_mb_s, dtype=np.float64)
    service_time = float(env.cfg.task_compute_mb) / np.maximum(capacities, eps)
    wait_s = np.asarray(feedback.wait_s, dtype=np.float64)
    wait_norm = np.asarray(feedback.wait_norm, dtype=np.float64)
    age_norm = float(env.feedback_source_age(feedback)) / max(float(env.cfg.feedback_age_norm), eps)
    use_stale_margin_price = bool(getattr(args, "lcc_use_stale_margin", True)) and not bool(
        getattr(args, "lcc_no_stale_margin", False)
    )
    if use_stale_margin_price:
        state_key = f"_last_lcc_price_wait_norm_delay_{int(env.feedback_delay)}"
        if state is not None and state_key in state and np.asarray(state[state_key]).shape == wait_norm.shape:
            server_volatility = np.abs(wait_norm - np.asarray(state[state_key], dtype=np.float64))
        else:
            server_volatility = np.zeros_like(wait_norm, dtype=np.float64)
        if state is not None:
            state[state_key] = wait_norm.copy()
        stale_margin_norm = float(getattr(args, "lcc_stale_margin", 1.0)) * age_norm * server_volatility
        wait_norm = np.clip(wait_norm + stale_margin_norm, 0.0, 2.0)
        wait_s = np.clip(wait_s + stale_margin_norm * max(float(env.cfg.slot_s), eps), 0.0, None)
    locations = env._locations()
    requests = np.asarray(env.requests, dtype=np.float64)
    load_norm = (requests / max(float(env.cfg.max_requests_per_user), 1.0))[:, None]

    if previous_portfolios is None or np.asarray(previous_portfolios).shape != (n_users, n_edges):
        prev = np.full((n_users, n_edges), 1.0 / max(float(n_edges), 1.0), dtype=np.float64)
    else:
        prev = normalize_portfolios(np.asarray(previous_portfolios, dtype=np.float64)).astype(np.float64)
    demand = np.sum((requests * float(env.cfg.task_compute_mb))[:, None] * prev, axis=0)
    demand_share = demand / max(float(np.sum(demand)), eps)

    z_pressure = 0.0 if getattr(args, "lcc_no_z_queue", False) else float(np.tanh(float(getattr(rewarder, "z_queue", 0.0)) / 10.0))
    h_pressure = 0.0 if getattr(args, "lcc_no_h_queue", False) else float(np.tanh(float(getattr(rewarder, "h_queue", 0.0)) / 5.0))
    alpha = float(np.clip(getattr(args, "lcc_cvar_alpha", 0.99), 0.0, 0.9999))
    nu = float(max(getattr(rewarder, "nu_var", 0.0), 0.0))
    v_weight = float(getattr(args, "lcc_v", 1.0))

    prices = np.zeros((n_users, n_edges), dtype=np.float64)
    components = str(getattr(args, "lcc_price_components", "Full")).upper()
    if components in {"NOPRICE", "NONE", "ZERO"}:
        normalized = np.zeros((n_users, n_edges), dtype=np.float32)
        scale = np.zeros((n_users, 1), dtype=np.float32)
        return normalized, scale
    for user in range(n_users):
        distances = np.linalg.norm(env.traces.edge_positions - locations[user][None, :], axis=1)
        communication = distances.astype(np.float64) / 10_000.0
        estimated_delay = communication + service_time + wait_s
        estimated_norm = estimated_delay / latency_scale

        # Queue marginal price: increasing user i's weight on edge j adds more
        # pressure when the stale queue/wait summary for j is high.
        pi_q = load_norm[user, 0] * wait_norm

        # Concentration marginal price: under fixed slot load, assigning more
        # demand to an already popular server has larger marginal concentration
        # risk.  The normalized HHI derivative is represented up to a positive
        # constant, then scaled by H queue pressure.
        pi_h = h_pressure * load_norm[user, 0] * demand_share

        # Tail marginal price: RU-tail subgradient proxy.  A server whose
        # estimated task delay lies above the current VaR proxy receives higher
        # price; the 1/(1-alpha) factor is clipped by the final normalization.
        tail_mask = (estimated_norm > nu).astype(np.float64)
        pi_z = z_pressure * tail_mask * estimated_norm / max(1.0 - alpha, 1e-4)

        # Latency penalty price.
        pi_t = v_weight * estimated_norm
        # The formal QLA signal is the sum of a stale queue-pressure term and
        # a predicted-latency term.  Q/T are exposed as composition ablations;
        # the existing QT/QLTZ/QTH/Full variants retain their prior semantics.
        if components == "Q":
            base_price = pi_q
        elif components in {"T", "L", "LATENCY"}:
            base_price = pi_t
        else:
            base_price = pi_q + pi_t
        total = base_price.copy()
        if components == "FULLSAFE":
            # Conservative alignment gate for noisy tail/concentration price
            # proxies.  In a local softmax-logit model, price bias improves a
            # one-step proxy cost when Cov(cost, price) is positive.  We do not
            # observe the true marginal cost online, so use QT as a stable
            # short-horizon cost surrogate and only retain Z/H residual price
            # when it is positively aligned with QT's server ranking.
            residual_price = pi_z + pi_h
            centered_base = base_price - float(np.mean(base_price))
            centered_residual = residual_price - float(np.mean(residual_price))
            denom = float(np.dot(centered_residual, centered_residual)) + eps
            gamma = float(np.dot(centered_base, centered_residual)) / denom
            gamma = float(np.clip(max(0.0, gamma), 0.0, 1.0))
            total = base_price + gamma * residual_price
        else:
            if "Z" in components or components == "FULL":
                total = total + pi_z
            if "H" in components or components == "FULL":
                total = total + pi_h
        prices[user] = total

    # Stabilize the directional prices before feeding them to the MLP.  The
    # ranking and relative gaps are preserved per user; the separate log-scale
    # feature carries the absolute risk magnitude.
    scale = np.log1p(np.mean(np.maximum(prices, 0.0), axis=1, keepdims=True))
    scale = np.clip(scale, 0.0, 10.0).astype(np.float32)
    price_min = np.min(prices, axis=1, keepdims=True)
    price_span = np.maximum(np.max(prices, axis=1, keepdims=True) - price_min, eps)
    normalized = (prices - price_min) / price_span
    intervention = str(getattr(args, "lcc_price_intervention", "aligned")).lower()
    if intervention in {"zero", "zeros"}:
        normalized = np.zeros_like(normalized, dtype=np.float64)
        scale = np.zeros_like(scale, dtype=np.float32)
    elif intervention in {"constant", "const"}:
        normalized = np.full_like(normalized, 0.5, dtype=np.float64)
    shuffle_prices = bool(getattr(args, "lcc_price_shuffle", False)) or intervention == "shuffle"
    if shuffle_prices:
        rng = np.random.default_rng(int(env.slot_index) + 1009 * int(env.feedback_delay) + 9173)
        for user in range(n_users):
            rng.shuffle(normalized[user])
    return normalized.astype(np.float32), scale


def price_only_portfolios(actor_obs: np.ndarray, n_edges: int, price_bias: float) -> np.ndarray:
    """Analytic QT-price-only policy used as a non-learning mechanism baseline."""
    base_dim = 2 + n_edges + 3 + n_edges + 1
    prices = np.asarray(actor_obs[:, base_dim : base_dim + n_edges], dtype=np.float64)
    return price_only_portfolios_from_prices(prices, price_bias)


def price_only_portfolios_from_prices(prices: np.ndarray, price_bias: float) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float64)
    logits = -float(price_bias) * prices
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
    return weights.astype(np.float32)


def augment_lcc_price_actor_observations(
    base_actor_obs: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    rewarder: MAPPORewarder,
    previous_portfolios: np.ndarray | None = None,
    state: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    prices, scale = lcc_marginal_risk_prices(env, args, rewarder, previous_portfolios=previous_portfolios, state=state)
    if bool(getattr(args, "lcc_price_scale_observation", False)):
        return np.concatenate([base_actor_obs.astype(np.float32), prices, scale], axis=1).astype(np.float32)
    return np.concatenate([base_actor_obs.astype(np.float32), prices], axis=1).astype(np.float32)


def augment_lcc_price_clean_actor_observations(
    base_actor_obs: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    rewarder: MAPPORewarder,
    previous_portfolios: np.ndarray | None = None,
    state: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Append clean price coordinator features with the same dimension as LCC-H.

    Added dimensions are ``[price_i, w_i(t-1), z_bar, h_bar]``.  Compared with
    the ordinary LCC observation ``[stale_wait, w_i(t-1), z_bar, h_bar]``, this
    replaces the stale wait vector by a lagged edge-wise risk-price proxy while
    preserving previous portfolio and virtual-queue urgency scalars.
    """

    prices, _ = lcc_marginal_risk_prices(env, args, rewarder, previous_portfolios=previous_portfolios, state=state)
    if previous_portfolios is None or np.asarray(previous_portfolios).shape != (env.cfg.n_users, env.cfg.n_edges):
        previous = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
    else:
        previous = np.asarray(previous_portfolios, dtype=np.float32)
    z_norm = np.full(
        (env.cfg.n_users, 1),
        np.tanh(float(getattr(rewarder, "z_queue", 0.0)) / 10.0),
        dtype=np.float32,
    )
    h_norm = np.full(
        (env.cfg.n_users, 1),
        np.tanh(float(getattr(rewarder, "h_queue", 0.0)) / 5.0),
        dtype=np.float32,
    )
    return np.concatenate([base_actor_obs.astype(np.float32), prices, previous, z_norm, h_norm], axis=1).astype(np.float32)


def mean_price_action_spearman(prices: np.ndarray, portfolios: np.ndarray) -> float:
    """Mean Spearman correlation between ``-price`` and portfolio weights."""

    p = np.asarray(prices, dtype=np.float64)
    w = np.asarray(portfolios, dtype=np.float64)
    if p.shape != w.shape or p.ndim != 2:
        return float("nan")

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(x.size, dtype=np.float64)
        return r

    vals: list[float] = []
    for row_p, row_w in zip(p, w):
        a = ranks(-row_p)
        b = ranks(row_w)
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            continue
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def apply_proposal_residual_prior(
    latent_actions: np.ndarray,
    env: JointMAPPOEdgeEnv,
    args: argparse.Namespace,
    previous_portfolios: np.ndarray | None = None,
    actor_obs: np.ndarray | None = None,
    diagnostics: dict[str, list[float]] | None = None,
) -> np.ndarray:
    """Prior-conditioned, age-capped proposal residual (Residual-v3).

    The actor outputs a correction proposal q and an amplitude logit. The final
    action is a convex interpolation between stale Lyapunov prior p^L and q:
    w_hat = (1 - beta) p^L + beta q.  This avoids signed-direction saturation
    and keeps execution on the probability simplex without projection.
    """

    latent = np.asarray(latent_actions, dtype=np.float64)
    if latent.shape[1] < env.cfg.n_edges + 1:
        raise ValueError(f"Residual-v3 expects latent shape (N, M+1), got {latent.shape}")
    if actor_obs is not None and actor_obs.shape[1] >= env.cfg.n_edges + 1:
        prior = np.asarray(actor_obs[:, -(env.cfg.n_edges + 1) : -1], dtype=np.float64)
        prior = normalize_portfolios(prior).astype(np.float64)
        volatility = np.asarray(actor_obs[:, -1:], dtype=np.float64)
        if env.cfg.include_feedback_age:
            base_dim = actor_obs.shape[1] - (env.cfg.n_edges + 1)
            age_norm = np.asarray(actor_obs[:, base_dim - 1 : base_dim], dtype=np.float64)
        else:
            selected = env.feedback_buffer.get(env.feedback_delay)
            age_norm = np.full(
                (env.cfg.n_users, 1),
                env.feedback_source_age(selected) / max(float(env.cfg.feedback_age_norm), 1e-8),
            )
    else:
        prior = lyapunov_prior_portfolios(
            env,
            tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
            temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
            fresh_state=False,
        ).astype(np.float64)
        volatility = np.zeros((env.cfg.n_users, 1), dtype=np.float64)
        selected = env.feedback_buffer.get(env.feedback_delay)
        age_norm = np.full(
            (env.cfg.n_users, 1),
            env.feedback_source_age(selected) / max(float(env.cfg.feedback_age_norm), 1e-8),
        )

    tau_r = max(float(getattr(args, "residual_proposal_temperature", 1.0)), 1e-8)
    logits = latent[:, : env.cfg.n_edges] / tau_r
    logits = logits - np.max(logits, axis=1, keepdims=True)
    proposal = np.exp(np.clip(logits, -60.0, 0.0))
    proposal = proposal / np.maximum(proposal.sum(axis=1, keepdims=True), 1e-12)

    beta_max = float(np.clip(getattr(args, "residual_beta_max", 0.05), 0.0, 1.0))
    amplitude = 1.0 / (1.0 + np.exp(-np.clip(latent[:, env.cfg.n_edges : env.cfg.n_edges + 1], -30.0, 30.0)))
    envelope = np.exp(
        -float(getattr(args, "residual_beta_gamma_delta", 1.0)) * np.maximum(age_norm, 0.0)
        -float(getattr(args, "residual_beta_gamma_volatility", 1.0)) * np.maximum(volatility, 0.0)
    )
    beta = np.clip(beta_max * amplitude * envelope, 0.0, 1.0)
    mixed = (1.0 - beta) * prior + beta * proposal

    projection_activation_rate = 0.0
    pre_projection_l1_mean = 0.0
    pre_projection_l1_max = 0.0
    if previous_portfolios is not None and previous_portfolios.shape == mixed.shape:
        smoothing = float(np.clip(getattr(args, "lyapunov_prior_smoothing", 0.0), 0.0, 1.0))
        if smoothing > 0.0:
            mixed = (1.0 - smoothing) * mixed + smoothing * previous_portfolios
        max_step = float(getattr(args, "lyapunov_prior_max_step", 0.0))
        if max_step > 0.0:
            delta = mixed - previous_portfolios
            l1 = np.sum(np.abs(delta), axis=1, keepdims=True)
            pre_projection_l1_mean = float(np.mean(l1))
            pre_projection_l1_max = float(np.max(l1))
            step_scale = np.minimum(1.0, max_step / np.maximum(l1, 1e-12))
            projection_activation_rate = float(np.mean(step_scale < 1.0 - 1e-9))
            mixed = previous_portfolios + delta * step_scale

    shaped = normalize_portfolios(mixed)
    if diagnostics is not None:
        eps = 1e-12
        kl_to_prior = np.sum(shaped * (np.log(np.maximum(shaped, eps)) - np.log(np.maximum(prior, eps))), axis=1)
        proposal_kl = np.sum(proposal * (np.log(np.maximum(proposal, eps)) - np.log(np.maximum(prior, eps))), axis=1)
        correction_l1 = np.sum(np.abs(shaped - prior), axis=1)
        fresh = lyapunov_prior_portfolios(
            env,
            tradeoff_v=float(getattr(args, "lyapunov_prior_v", 1.0)),
            temperature=float(getattr(args, "lyapunov_prior_temperature", 2.5)),
            fresh_state=True,
        )
        stale_gap = np.sum(np.abs(fresh - prior), axis=1)
        proposal_gap = np.sum(np.abs(fresh - proposal), axis=1)
        pre_smooth_gap = np.sum(np.abs(fresh - ((1.0 - beta) * prior + beta * proposal)), axis=1)
        final_gap = np.sum(np.abs(fresh - shaped), axis=1)
        kl_fresh_to_prior = np.sum(fresh * (np.log(np.maximum(fresh, eps)) - np.log(np.maximum(prior, eps))), axis=1)
        kl_fresh_to_q = np.sum(fresh * (np.log(np.maximum(fresh, eps)) - np.log(np.maximum(proposal, eps))), axis=1)
        diagnostics.setdefault("lyapunov_prior_kl_w_to_prior", []).append(float(np.mean(kl_to_prior)))
        diagnostics.setdefault("residual_proposal_kl_q_to_prior", []).append(float(np.mean(proposal_kl)))
        diagnostics.setdefault("residual_kl_fresh_to_prior", []).append(float(np.mean(kl_fresh_to_prior)))
        diagnostics.setdefault("residual_kl_fresh_to_q", []).append(float(np.mean(kl_fresh_to_q)))
        diagnostics.setdefault("residual_proposal_improvement_ratio", []).append(float(np.mean(kl_fresh_to_q < kl_fresh_to_prior)))
        diagnostics.setdefault("residual_actor_amplitude", []).append(float(np.mean(amplitude)))
        diagnostics.setdefault("residual_beta", []).append(float(np.mean(beta)))
        diagnostics.setdefault("residual_beta_p10", []).append(float(np.percentile(beta, 10)))
        diagnostics.setdefault("residual_beta_p50", []).append(float(np.percentile(beta, 50)))
        diagnostics.setdefault("residual_beta_p90", []).append(float(np.percentile(beta, 90)))
        diagnostics.setdefault("residual_authority_envelope", []).append(float(np.mean(envelope)))
        diagnostics.setdefault("residual_correction_l1", []).append(float(np.mean(correction_l1)))
        diagnostics.setdefault("residual_fresh_stale_prior_l1", []).append(float(np.mean(stale_gap)))
        diagnostics.setdefault("residual_fresh_proposal_l1", []).append(float(np.mean(proposal_gap)))
        diagnostics.setdefault("residual_fresh_presmooth_l1", []).append(float(np.mean(pre_smooth_gap)))
        diagnostics.setdefault("residual_fresh_final_l1", []).append(float(np.mean(final_gap)))
        diagnostics.setdefault("residual_oracle_gap_improvement_ratio", []).append(float(np.mean(final_gap < stale_gap)))
        if np.std(beta.ravel()) > 1e-12 and np.std(stale_gap.ravel()) > 1e-12:
            diagnostics.setdefault("residual_beta_oracle_gap_corr", []).append(float(np.corrcoef(beta.ravel(), stale_gap.ravel())[0, 1]))
        else:
            diagnostics.setdefault("residual_beta_oracle_gap_corr", []).append(0.0)
        diagnostics.setdefault("projection_activation_rate", []).append(projection_activation_rate)
        diagnostics.setdefault("pre_projection_l1_mean", []).append(pre_projection_l1_mean)
        diagnostics.setdefault("pre_projection_l1_max", []).append(pre_projection_l1_max)
        if previous_portfolios is not None and previous_portfolios.shape == shaped.shape:
            post_l1 = np.sum(np.abs(shaped - previous_portfolios), axis=1)
            diagnostics.setdefault("post_step_l1_mean", []).append(float(np.mean(post_l1)))
            diagnostics.setdefault("post_step_l1_max", []).append(float(np.max(post_l1)))
        else:
            diagnostics.setdefault("post_step_l1_mean", []).append(0.0)
            diagnostics.setdefault("post_step_l1_max", []).append(0.0)
    return shaped


def train_once(args: argparse.Namespace) -> dict[str, float | int | str]:
    tail_fraction = args.tail_fraction
    if args.cvar_alpha is not None:
        tail_fraction = float(np.clip(1.0 - args.cvar_alpha, 0.0, 1.0))
    v4_agegate_methods = {
        "DSR-MTG-Qpen-v4-agegate",
        "DSR-MTG-Qpen-v4-agegate-lyapprior",
        "DSR-MTG-Qpen-v4-agegate-lyapresidual",
        "DSR-MTG-Qpen-v4-agegate-noQpen",
        "DSR-MTG-Qpen-v4-agegate-noConc",
    }
    residual_v2_method = "DSR-MTG-Qpen-v4-lyapresidual-v2"
    residual_v3_method = "DSR-MTG-Qpen-v4-lyapresidual-v3"
    residual_v3p1_method = "DSR-MTG-Qpen-v4-lyapresidual-v3p1"
    lcc_method = "LCC-MAPPO"
    is_m2ppo = args.method == "M2PPO"
    is_lcc = args.method == lcc_method
    use_lcc_price_observation = is_lcc and bool(args.lcc_price_observation)
    use_lcc_price_clean_observation = is_lcc and bool(args.lcc_price_clean_observation)
    use_lcc_price_scale_observation = is_lcc and bool(args.lcc_price_scale_observation)
    use_lcc_edgewise_price_actor = is_lcc and bool(args.lcc_edgewise_price_actor)
    use_lcc_flat_price_bias_actor = is_lcc and bool(args.lcc_flat_price_bias_actor)
    use_lcc_price_only_policy = is_lcc and bool(args.lcc_price_only_policy)
    use_lcc_local_critic = is_lcc and bool(args.lcc_local_critic)
    if use_lcc_edgewise_price_actor:
        use_lcc_price_observation = True
        use_lcc_price_scale_observation = True
    if use_lcc_flat_price_bias_actor or use_lcc_price_only_policy:
        use_lcc_price_observation = True
        use_lcc_price_scale_observation = True
    residual_v3_methods = {residual_v3_method, residual_v3p1_method}
    is_v4 = args.method in {"DSR-MTG-Qpen-v4", residual_v2_method, residual_v3p1_method, *residual_v3_methods, *v4_agegate_methods}
    use_residual_actor = args.method in {residual_v2_method, *residual_v3_methods}
    use_residual_v3 = args.method in residual_v3_methods
    use_feedback_age_gate = (args.feedback_age_aware_score_gate or args.method in v4_agegate_methods) and not use_residual_actor
    include_feedback_age = use_feedback_age_gate or use_residual_actor or is_lcc
    twin_critic = args.twin_critic or is_v4
    critic_ensemble_count = max(1, int(args.critic_ensemble_count))
    if is_m2ppo and critic_ensemble_count == 1:
        critic_ensemble_count = 5
    if critic_ensemble_count > 1:
        twin_critic = False
    auto_portfolio_entropy = args.auto_portfolio_entropy or (is_v4 and not use_residual_actor) or is_lcc
    target_portfolio_entropy = args.target_portfolio_entropy
    kappa_i = args.kappa_i
    kappa_g = args.kappa_g
    feedback_gate_eta = args.feedback_gate_eta
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    extra_actor_obs_dim = (
        (2 * args.n_edges + 2)
        if use_lcc_price_clean_observation
        else (
            (args.n_edges + (1 if use_lcc_price_scale_observation else 0))
            if (use_lcc_price_observation or use_lcc_price_scale_observation)
            else ((2 * args.n_edges + 2) if is_lcc else ((args.n_edges + 1) if use_residual_v3 else 0))
        )
    )
    base_actor_obs_dim = 2 + args.n_edges + 3 + args.n_edges + int(include_feedback_age) + extra_actor_obs_dim
    sys_cfg = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_package_mb=args.task_package_mb,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        max_supported_users=max(args.max_supported_users, args.users),
        include_feedback_age=include_feedback_age,
        feedback_age_norm=args.feedback_age_norm,
        extra_actor_obs_dim=extra_actor_obs_dim,
        critic_obs_override_dim=(base_actor_obs_dim + args.n_edges if use_lcc_local_critic else None),
    )
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, args.seed)
    mappo_cfg = MAPPOConfig(
        rollout_size=args.rollout_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        twin_critic=twin_critic,
        critic_ensemble_count=critic_ensemble_count,
        auto_portfolio_entropy=auto_portfolio_entropy,
        target_portfolio_entropy=target_portfolio_entropy,
        auto_entropy_lr=args.auto_entropy_lr,
        auto_entropy_init=args.auto_entropy_init,
        log_std_min=args.log_std_min,
        residual_actor=use_residual_actor,
        residual_actor_prior_conditioned=use_residual_v3,
        residual_amplitude_init_bias=args.residual_amplitude_init_bias,
        residual_amplitude_penalty=args.residual_amplitude_penalty,
        residual_proposal_temperature=args.residual_proposal_temperature,
        residual_beta_max=args.residual_beta_max,
        residual_beta_gamma_delta=args.residual_beta_gamma_delta,
        residual_beta_gamma_volatility=args.residual_beta_gamma_volatility,
        residual_aux_kl_weight=args.residual_aux_kl_weight,
        residual_aux_proposal_kl_weight=args.residual_aux_proposal_kl_weight,
        residual_aux_amplitude_weight=args.residual_aux_amplitude_weight,
        residual_beta_penalty=args.residual_beta_penalty,
        edgewise_price_actor=use_lcc_edgewise_price_actor,
        flat_price_bias_actor=use_lcc_flat_price_bias_actor,
        feedback_age_gate=use_feedback_age_gate,
        feedback_gate_eta=feedback_gate_eta,
        lcc_enabled=is_lcc,
        lcc_v=args.lcc_v,
        lcc_migration_weight=args.lcc_migration_weight,
        lcc_cvar_alpha=args.lcc_cvar_alpha,
        lcc_cvar_budget=args.lcc_cvar_budget,
        lcc_var_lr=args.lcc_var_lr,
        lcc_concentration_factor=args.lcc_concentration_factor,
        lcc_stale_margin=args.lcc_stale_margin,
        lcc_use_z_queue=not args.lcc_no_z_queue,
        lcc_use_h_queue=not args.lcc_no_h_queue,
        lcc_use_stale_margin=not args.lcc_no_stale_margin,
        lcc_static_reward=args.lcc_static_reward,
        lcc_price_observation=use_lcc_price_observation,
        lcc_price_clean_observation=use_lcc_price_clean_observation,
        lcc_price_scale_observation=use_lcc_price_scale_observation,
        lcc_edgewise_price_actor=use_lcc_edgewise_price_actor,
        lcc_flat_price_bias_actor=use_lcc_flat_price_bias_actor,
        lcc_price_components=args.lcc_price_components,
        lcc_price_shuffle=args.lcc_price_shuffle,
        lcc_price_logit_bias=args.lcc_price_logit_bias,
        lcc_local_critic=use_lcc_local_critic,
        dsr_alpha=args.alpha,
        dsr_temperature=args.dsr_temperature,
        dsr_beta_mean=args.beta_mean,
        dsr_beta_tail=args.beta_tail,
        dsr_beta_gap=args.beta_gap,
        dsr_beta_qpressure=args.beta_qpressure,
        dsr_tail_topk=args.tail_topk,
        dsr_tail_fraction=tail_fraction,
        kappa_i=kappa_i,
        kappa_g=kappa_g,
        latency_norm=args.latency_norm,
        q_pressure_max=args.q_pressure_max,
        q_penalty_weight=args.q_penalty_weight,
        dsr_mu=args.dsr_mu,
        dsr_eta=args.dsr_eta,
        score=score_cfg,
    )
    traces = load_trace_set(
        args.dataset,
        args.users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    mixed_feedback_delays = parse_mixed_feedback_delays(args.mixed_feedback_delays)
    train_feedback_delay = max(mixed_feedback_delays) if mixed_feedback_delays else args.feedback_delay
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=args.seed,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=train_feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    train_schedule = install_schedule_for_env(
        env,
        root=Path(args.canonical_schedule_root),
        distribution=args.distribution,
        role="train",
        seed=args.seed,
        n_steps=args.train_steps,
        max_feedback_delay=args.canonical_max_feedback_delay,
    )
    policy = MAPPOPolicy(sys_cfg, mappo_cfg, seed=args.seed, device=args.device)
    if args.load_checkpoint:
        policy.load_checkpoint(args.load_checkpoint)
    rewarder = MAPPORewarder(mappo_cfg, method=args.method)
    buffer = MAPPOBuffer(max_size=mappo_cfg.buffer_size)
    out_dir = build_output_dir(Path(args.output_dir), args.dataset, args.users, args.distribution)
    out_dir.mkdir(parents=True, exist_ok=True)

    learning_rows: list[dict] = []
    latency_rows: list[dict] = []
    last_update = MAPPOUpdateStats(0.0, 0.0, 0.0, 0.0)
    start_all = perf_counter()
    env.reset()
    user_ids = np.arange(sys_cfg.n_users, dtype=np.int64)
    previous_train_portfolios: np.ndarray | None = None
    train_residual_state: dict[str, np.ndarray] = {}
    train_v3_obs_state: dict[str, np.ndarray] = {}
    train_lcc_obs_state: dict[str, np.ndarray] = {}
    mixed_delay_index = 0
    current_train_delay = args.feedback_delay
    for slot in range(1, args.train_steps + 1):
        if mixed_feedback_delays and len(buffer) == 0:
            current_train_delay = int(mixed_feedback_delays[mixed_delay_index % len(mixed_feedback_delays)])
            mixed_delay_index += 1
        env.feedback_delay = current_train_delay if mixed_feedback_delays else args.feedback_delay
        base_actor_obs = env.actor_observations()
        if is_lcc:
            if use_lcc_price_clean_observation:
                actor_obs = augment_lcc_price_clean_actor_observations(
                    base_actor_obs,
                    env,
                    args,
                    rewarder,
                    previous_portfolios=previous_train_portfolios,
                    state=train_lcc_obs_state,
                )
            elif use_lcc_price_observation:
                actor_obs = augment_lcc_price_actor_observations(
                    base_actor_obs,
                    env,
                    args,
                    rewarder,
                    previous_portfolios=previous_train_portfolios,
                    state=train_lcc_obs_state,
                )
            else:
                actor_obs = augment_lcc_actor_observations(
                    base_actor_obs,
                    env,
                    args,
                    rewarder,
                    previous_portfolios=previous_train_portfolios,
                    state=train_lcc_obs_state,
                )
            stale_prior_for_aux = None
        elif use_residual_v3:
            actor_obs, stale_prior_for_aux, _ = augment_residual_v3_actor_observations(
                base_actor_obs,
                env,
                args,
                state=train_v3_obs_state,
            )
        else:
            actor_obs = base_actor_obs
            stale_prior_for_aux = None
        aux_targets = None
        aux_weights = None
        needs_v3_aux = (
            use_residual_v3
            and (
                args.residual_aux_kl_weight > 0.0
                or args.residual_aux_proposal_kl_weight > 0.0
                or args.residual_aux_amplitude_weight > 0.0
                or args.residual_beta_penalty > 0.0
            )
        )
        if needs_v3_aux:
            aux_targets = lyapunov_prior_portfolios(
                env,
                tradeoff_v=float(args.lyapunov_prior_v),
                temperature=float(args.lyapunov_prior_temperature),
                fresh_state=True,
            )
            aux_weights = np.ones(sys_cfg.n_users, dtype=np.float32)
        t0 = perf_counter()
        portfolios, latents, log_probs = policy.act(actor_obs, deterministic=False)
        if is_lcc and use_lcc_price_only_policy:
            portfolios = price_only_portfolios(actor_obs, sys_cfg.n_edges, args.lcc_price_logit_bias)
        elif is_lcc:
            portfolios = normalize_portfolios(portfolios)
        elif args.method == "DSR-MTG-Qpen-v4-agegate-lyapresidual":
            portfolios = apply_residual_lyapunov_prior(
                latents,
                env,
                args,
                previous_train_portfolios,
                state=train_residual_state,
            )
        elif args.method == residual_v2_method:
            portfolios = apply_actor_amplitude_residual_prior(
                latents,
                env,
                args,
                previous_train_portfolios,
            )
        elif use_residual_v3:
            portfolios = apply_proposal_residual_prior(
                latents,
                env,
                args,
                previous_train_portfolios,
                actor_obs=actor_obs,
            )
        else:
            portfolios = apply_lyapunov_prior(portfolios, env, args, previous_train_portfolios)
        decision_latency = perf_counter() - t0
        pstats = env.portfolio_stats(portfolios)
        critic_obs = (
            local_critic_observations(actor_obs, portfolios)
            if use_lcc_local_critic
            else env.critic_observations(portfolios, stats=pstats)
        )
        values = policy.value(critic_obs, user_ids)
        previous_for_reward = None if previous_train_portfolios is None else np.asarray(previous_train_portfolios, dtype=np.float32).copy()
        result = env.joint_step(portfolios, stats=pstats, decision_latency_s=decision_latency)
        previous_train_portfolios = np.asarray(result.portfolios, dtype=np.float32)
        rewards = rewarder.compute(
            result.per_user_mean_latency,
            result.per_user_tasks,
            result.system_average_latency,
            pstats,
            portfolios=result.portfolios,
            queues_before_mb=result.queue_load_before_mb,
            sys_cfg=sys_cfg,
            previous_portfolios=previous_for_reward,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )
        buffer.add_batch(
            slot=slot,
            actor_obs=actor_obs,
            critic_obs=critic_obs,
            latent_actions=latents,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            aux_targets=aux_targets,
            aux_weights=aux_weights,
        )
        if len(buffer) >= args.rollout_size:
            last_update = policy.update(buffer)

        for record in result.records:
            latency_rows.append(
                {
                    "phase": "train",
                    "slot": slot,
                    "step": slot * sys_cfg.n_users,
                    "user": record.user,
                    "task": record.task,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                }
            )

        if slot == 1 or slot % args.log_interval == 0 or slot == args.train_steps:
            row = {
                "slot": slot,
                "step": slot * sys_cfg.n_users,
                "mean_reward": float(np.mean(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "elapsed_s": perf_counter() - start_all,
                "actor_loss": last_update.actor_loss,
                "critic_loss": last_update.critic_loss,
                "entropy_loss_term": last_update.entropy,
                "portfolio_entropy_update": last_update.portfolio_entropy,
                "auto_entropy_coef": last_update.auto_entropy_coef,
                "auto_entropy_loss": last_update.auto_entropy_loss,
                "residual_amplitude_mean_update": last_update.residual_amplitude_mean,
                "residual_amplitude_penalty_update": last_update.residual_amplitude_penalty,
                "residual_aux_loss_update": last_update.residual_aux_loss,
                "residual_aux_proposal_loss_update": last_update.residual_aux_proposal_loss,
                "residual_aux_amplitude_loss_update": last_update.residual_aux_amplitude_loss,
                "residual_beta_penalty_update": last_update.residual_beta_penalty,
                "approx_kl": last_update.approx_kl,
            }
            if rewarder.is_dsr_method or rewarder.is_lcc_method:
                row.update(rewarder.last_components)
            row.update(slot_metrics(result))
            learning_rows.append(row)

    if len(buffer) > 0:
        last_update = policy.update(buffer)
        if learning_rows:
            learning_rows[-1]["actor_loss"] = last_update.actor_loss
            learning_rows[-1]["critic_loss"] = last_update.critic_loss
            learning_rows[-1]["entropy_loss_term"] = last_update.entropy
            learning_rows[-1]["portfolio_entropy_update"] = last_update.portfolio_entropy
            learning_rows[-1]["auto_entropy_coef"] = last_update.auto_entropy_coef
            learning_rows[-1]["auto_entropy_loss"] = last_update.auto_entropy_loss
            learning_rows[-1]["residual_amplitude_mean_update"] = last_update.residual_amplitude_mean
            learning_rows[-1]["residual_amplitude_penalty_update"] = last_update.residual_amplitude_penalty
            learning_rows[-1]["residual_aux_loss_update"] = last_update.residual_aux_loss
            learning_rows[-1]["residual_aux_proposal_loss_update"] = last_update.residual_aux_proposal_loss
            learning_rows[-1]["residual_aux_amplitude_loss_update"] = last_update.residual_aux_amplitude_loss
            learning_rows[-1]["residual_beta_penalty_update"] = last_update.residual_beta_penalty
            learning_rows[-1]["approx_kl"] = last_update.approx_kl

    eval_env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=args.distribution,
        seed=args.seed + 10_000,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    evaluation_split_role = str(args.evaluation_split_role)
    test_schedule = install_schedule_for_env(
        eval_env,
        root=Path(args.canonical_schedule_root),
        distribution=args.distribution,
        role=evaluation_split_role,
        seed=args.seed + 10_000,
        n_steps=args.test_steps,
        max_feedback_delay=args.canonical_max_feedback_delay,
    )
    previous_eval_portfolios: np.ndarray | None = None
    eval_prior_diagnostics: dict[str, list[float]] = {}
    eval_residual_state: dict[str, np.ndarray] = {}
    eval_v3_obs_state: dict[str, np.ndarray] = {}
    eval_lcc_obs_state: dict[str, np.ndarray] = {}
    eval_lcc_rewarder = MAPPORewarder(mappo_cfg, method=args.method) if is_lcc else None
    eval_lcc_components: list[dict[str, float]] = []
    last_eval_v3_actor_obs: np.ndarray | None = None
    last_eval_lcc_price_obs: np.ndarray | None = None

    def transform_eval_observations(
        base_actor_obs: np.ndarray,
        current_env: JointMAPPOEdgeEnv,
    ) -> np.ndarray:
        nonlocal last_eval_v3_actor_obs, last_eval_lcc_price_obs
        if is_lcc and eval_lcc_rewarder is not None:
            if use_lcc_price_clean_observation:
                augmented = augment_lcc_price_clean_actor_observations(
                    base_actor_obs,
                    current_env,
                    args,
                    eval_lcc_rewarder,
                    previous_portfolios=previous_eval_portfolios,
                    state=eval_lcc_obs_state,
                )
                base_dim = 2 + current_env.cfg.n_edges + 3 + current_env.cfg.n_edges + int(current_env.cfg.include_feedback_age)
                last_eval_lcc_price_obs = augmented[:, base_dim : base_dim + current_env.cfg.n_edges].copy()
                return augmented
            if use_lcc_price_observation:
                augmented = augment_lcc_price_actor_observations(
                    base_actor_obs,
                    current_env,
                    args,
                    eval_lcc_rewarder,
                    previous_portfolios=previous_eval_portfolios,
                    state=eval_lcc_obs_state,
                )
                base_dim = 2 + current_env.cfg.n_edges + 3 + current_env.cfg.n_edges + int(current_env.cfg.include_feedback_age)
                last_eval_lcc_price_obs = augmented[:, base_dim : base_dim + current_env.cfg.n_edges].copy()
                return augmented
            return augment_lcc_actor_observations(
                    base_actor_obs,
                    current_env,
                    args,
                    eval_lcc_rewarder,
                    previous_portfolios=previous_eval_portfolios,
                    state=eval_lcc_obs_state,
                )
        if not use_residual_v3:
            return base_actor_obs
        augmented, _, _ = augment_residual_v3_actor_observations(
            base_actor_obs,
            current_env,
            args,
            state=eval_v3_obs_state,
        )
        last_eval_v3_actor_obs = augmented
        return augmented

    def transform_eval_portfolios(
        portfolios: np.ndarray,
        current_env: JointMAPPOEdgeEnv,
        latents: np.ndarray | None = None,
    ) -> np.ndarray:
        nonlocal previous_eval_portfolios, last_eval_lcc_price_obs
        if is_lcc:
            if use_lcc_price_only_policy and last_eval_lcc_price_obs is not None:
                shaped = price_only_portfolios_from_prices(last_eval_lcc_price_obs, args.lcc_price_logit_bias)
            else:
                shaped = normalize_portfolios(portfolios)
        elif args.method == "DSR-MTG-Qpen-v4-agegate-lyapresidual":
            shaped = apply_residual_lyapunov_prior(
                latents if latents is not None else portfolios,
                current_env,
                args,
                previous_eval_portfolios,
                diagnostics=eval_prior_diagnostics,
                state=eval_residual_state,
            )
        elif args.method == residual_v2_method:
            shaped = apply_actor_amplitude_residual_prior(
                latents if latents is not None else portfolios,
                current_env,
                args,
                previous_eval_portfolios,
                diagnostics=eval_prior_diagnostics,
            )
        elif use_residual_v3:
            shaped = apply_proposal_residual_prior(
                latents if latents is not None else portfolios,
                current_env,
                args,
                previous_eval_portfolios,
                actor_obs=last_eval_v3_actor_obs,
                diagnostics=eval_prior_diagnostics,
            )
        else:
            shaped = apply_lyapunov_prior(
                portfolios,
                current_env,
                args,
                previous_eval_portfolios,
                diagnostics=eval_prior_diagnostics,
            )
        previous_eval_portfolios = np.asarray(shaped, dtype=np.float32)
        if is_lcc and last_eval_lcc_price_obs is not None:
            eval_prior_diagnostics.setdefault("lcc_price_action_spearman", []).append(
                mean_price_action_spearman(last_eval_lcc_price_obs, shaped)
            )
        return shaped

    def post_eval_step(result, current_env: JointMAPPOEdgeEnv) -> None:
        if not is_lcc or eval_lcc_rewarder is None:
            return
        prev = getattr(post_eval_step, "_previous_reward_portfolios", None)
        eval_lcc_rewarder.compute(
            result.per_user_mean_latency,
            result.per_user_tasks,
            result.system_average_latency,
            result.portfolio_stats,
            portfolios=result.portfolios,
            queues_before_mb=result.queue_load_before_mb,
            sys_cfg=sys_cfg,
            previous_portfolios=prev,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )
        component_row = {"slot": len(eval_lcc_components) + 1}
        component_row.update(eval_lcc_rewarder.last_components)
        eval_lcc_components.append(component_row)
        setattr(post_eval_step, "_previous_reward_portfolios", np.asarray(result.portfolios, dtype=np.float32))

    eval_results = evaluate_policy(
        policy,
        eval_env,
        args.test_steps,
        deterministic=True,
        portfolio_transform=transform_eval_portfolios,
        observation_transform=transform_eval_observations if (use_residual_v3 or is_lcc) else None,
        post_step_callback=post_eval_step if is_lcc else None,
    )
    for slot, result in enumerate(eval_results, start=1):
        for record in result.records:
            latency_rows.append(
                {
                    "phase": "test",
                    "slot": slot,
                    "step": slot * sys_cfg.n_users,
                    "user": record.user,
                    "task": record.task,
                    "edge": record.edge,
                    "latency": record.latency_s,
                    "tran_s": record.tran_s,
                    "queue_s": record.queue_s,
                    "comp_s": record.comp_s,
                    "dropped": int(record.dropped),
                }
            )

    summary, eval_drain = finalize_evaluation(eval_env, eval_results, sys_cfg)
    summary.update(eval_diagnostics(eval_results, capacities.astype(np.float64), sys_cfg.slot_s))
    for key, values in eval_prior_diagnostics.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary[key] = float(np.mean(finite)) if finite.size else float("nan")
    if eval_lcc_components:
        for key in eval_lcc_components[0].keys():
            if key == "slot":
                continue
            vals = np.asarray([item.get(key, np.nan) for item in eval_lcc_components], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                summary[key] = float(np.mean(vals))
    if rewarder.is_dsr_method:
        eval_components = [
            compute_dsr_input(
                per_user_latency=result.per_user_mean_latency,
                per_user_tasks=result.per_user_tasks,
                system_average_latency=result.system_average_latency,
                portfolios=result.portfolios,
                queues_before_mb=result.queue_load_before_mb,
                cfg=mappo_cfg,
                sys_cfg=sys_cfg,
                use_q_pressure=rewarder.uses_q_pressure,
            )
            for result in eval_results
        ]
        for key in (
            "system_risk_latency",
            "dsr_input_x",
            "t_mean_norm",
            "t_topk_norm",
            "gap_norm",
            "q_pressure",
            "q_pressure_for_dsr",
            "q_port_pressure_mean",
            "risk_mean_latency",
            "risk_tail_latency",
            "risk_gap_latency",
            "risk_tail_topk",
        ):
            summary[key] = float(np.mean([item[key] for item in eval_components])) if eval_components else float("nan")
        summary["q_penalty_mean"] = (
            args.q_penalty_weight * summary["q_port_pressure_mean"]
            if rewarder.uses_direct_q_penalty
            else 0.0
        )
    summary.update(
        {
            "method": args.method,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "train_split_id": train_schedule.schedule_id,
            "train_schedule_sha256": train_schedule.content_sha256,
            "evaluation_split_role": evaluation_split_role,
            "evaluation_split_id": test_schedule.schedule_id,
            "evaluation_schedule_sha256": test_schedule.content_sha256,
            "alpha": args.alpha,
            "beta_mean": args.beta_mean,
            "beta_tail": args.beta_tail,
            "beta_gap": args.beta_gap,
            "beta_qpressure": args.beta_qpressure,
            "tail_topk": args.tail_topk,
            "tail_fraction": tail_fraction,
            "cvar_alpha": args.cvar_alpha if args.cvar_alpha is not None else 1.0 - tail_fraction,
            "kappa_i": kappa_i,
            "kappa_g": kappa_g,
            "latency_norm": args.latency_norm,
            "q_pressure_max": args.q_pressure_max,
            "q_penalty_weight": args.q_penalty_weight,
            "twin_critic": twin_critic,
            "critic_ensemble_count": critic_ensemble_count,
            "auto_portfolio_entropy": auto_portfolio_entropy,
            "target_portfolio_entropy": target_portfolio_entropy,
            "auto_entropy_lr": args.auto_entropy_lr,
            "auto_entropy_init": args.auto_entropy_init,
            "feedback_age_aware_score_gate": use_feedback_age_gate,
            "feedback_gate_eta": feedback_gate_eta,
            "feedback_age_norm": args.feedback_age_norm,
            "lyapunov_prior_weight": args.lyapunov_prior_weight,
            "lyapunov_prior_v": args.lyapunov_prior_v,
            "lyapunov_prior_temperature": args.lyapunov_prior_temperature,
            "lyapunov_prior_smoothing": args.lyapunov_prior_smoothing,
            "lyapunov_prior_max_step": args.lyapunov_prior_max_step,
            "lyapunov_prior_fresh_state": args.lyapunov_prior_fresh_state,
            "residual_radius_min": args.residual_radius_min,
            "residual_radius_max": args.residual_radius_max,
            "residual_direction_l1_cap": args.residual_direction_l1_cap,
            "residual_amplitude_init_bias": args.residual_amplitude_init_bias,
            "residual_amplitude_penalty": args.residual_amplitude_penalty,
            "residual_proposal_temperature": args.residual_proposal_temperature,
            "residual_beta_max": args.residual_beta_max,
            "residual_beta_gamma_delta": args.residual_beta_gamma_delta,
            "residual_beta_gamma_volatility": args.residual_beta_gamma_volatility,
            "residual_aux_kl_weight": args.residual_aux_kl_weight,
            "residual_aux_proposal_kl_weight": args.residual_aux_proposal_kl_weight,
            "residual_aux_amplitude_weight": args.residual_aux_amplitude_weight,
            "residual_beta_penalty": args.residual_beta_penalty,
            "residual_confidence_eta_delta": args.residual_confidence_eta_delta,
            "residual_confidence_eta_volatility": args.residual_confidence_eta_volatility,
            "lcc_v": args.lcc_v,
            "lcc_migration_weight": args.lcc_migration_weight,
            "lcc_cvar_alpha": args.lcc_cvar_alpha,
            "lcc_cvar_budget": args.lcc_cvar_budget,
            "lcc_var_lr": args.lcc_var_lr,
            "lcc_concentration_factor": args.lcc_concentration_factor,
            "lcc_stale_margin": args.lcc_stale_margin,
            "lcc_use_z_queue": not args.lcc_no_z_queue,
            "lcc_use_h_queue": not args.lcc_no_h_queue,
            "lcc_use_stale_margin": not args.lcc_no_stale_margin,
            "lcc_static_reward": args.lcc_static_reward,
            "lcc_price_observation": use_lcc_price_observation,
            "lcc_price_clean_observation": use_lcc_price_clean_observation,
            "lcc_price_scale_observation": use_lcc_price_scale_observation,
            "lcc_edgewise_price_actor": use_lcc_edgewise_price_actor,
            "lcc_flat_price_bias_actor": use_lcc_flat_price_bias_actor,
            "lcc_price_only_policy": use_lcc_price_only_policy,
            "lcc_price_components": args.lcc_price_components,
            "lcc_price_shuffle": args.lcc_price_shuffle,
            "lcc_price_intervention": args.lcc_price_intervention,
            "lcc_price_logit_bias": args.lcc_price_logit_bias,
            "lcc_local_critic": use_lcc_local_critic,
            "mixed_feedback_delays": args.mixed_feedback_delays,
            "train_feedback_delay_buffer_max": train_feedback_delay,
            "portfolio_entropy_update": last_update.portfolio_entropy,
            "auto_entropy_coef": last_update.auto_entropy_coef,
            "auto_entropy_loss": last_update.auto_entropy_loss,
            "residual_amplitude_mean_update": last_update.residual_amplitude_mean,
            "residual_amplitude_penalty_update": last_update.residual_amplitude_penalty,
            "residual_aux_loss_update": last_update.residual_aux_loss,
            "residual_aux_proposal_loss_update": last_update.residual_aux_proposal_loss,
            "residual_aux_amplitude_loss_update": last_update.residual_aux_amplitude_loss,
            "residual_beta_penalty_update": last_update.residual_beta_penalty,
            "dsr_mu": args.dsr_mu,
            "dsr_eta": args.dsr_eta,
            "dsr_temperature": args.dsr_temperature,
            "log_std_min": args.log_std_min,
            "score_resource_weight": args.score_resource_weight,
            "score_capacity_weight": args.score_capacity_weight,
            "score_task_count_weight": args.score_task_count_weight,
            "score_wait_weight": args.score_wait_weight,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
            "slot_s": args.slot_s,
            "n_edges": args.n_edges,
            "edge_compute_mb_s": sys_cfg.edge_compute_mb_s,
            "task_compute_mb": sys_cfg.task_compute_mb,
            "task_package_mb": sys_cfg.task_package_mb,
            "queue_upper_mb": sys_cfg.queue_upper_mb,
            "max_requests_per_user": sys_cfg.max_requests_per_user,
            "poisson_lambda": args.poisson_lambda,
            "burst_prob": args.burst_prob,
            "burst_multiplier": args.burst_multiplier,
            "background_load": args.background_load,
            "capacity_mode": args.capacity_mode,
            "edge_compute_scale": args.edge_compute_scale,
            "task_compute_scale": args.task_compute_scale,
        }
    )
    if evaluation_split_role == "validation":
        summary["validation_split_id"] = test_schedule.schedule_id
        summary["validation_schedule_sha256"] = test_schedule.content_sha256
    else:
        summary["test_split_id"] = test_schedule.schedule_id
        summary["test_schedule_sha256"] = test_schedule.content_sha256

    method_file = args.method.replace("/", "_")
    write_csv(out_dir / f"learning_{method_file}.csv", learning_rows)
    write_csv(out_dir / f"latency_samples_{method_file}.csv", latency_rows)
    if eval_lcc_components:
        write_csv(out_dir / f"evaluation_lcc_components_{method_file}.csv", eval_lcc_components)
    write_summary(out_dir / f"summary_{method_file}.csv", summary)
    checkpoint_path = out_dir / f"checkpoint_{method_file}.pt"
    config_path = out_dir / f"config_{method_file}.json"
    manifest_path = out_dir / "manifest.json"
    policy.save_checkpoint(checkpoint_path)
    config_payload = {
        **vars(args),
        "finite_buffer_protocol": bool(sys_cfg.finite_buffer_protocol),
        "drop_penalty_s": float(sys_cfg.drop_penalty_s),
        "checkpoint_rule": "final_10000_slots_or_requested_budget_no_test_early_stopping",
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, ensure_ascii=False)
    manifest = {
        "method": args.method,
        "seed": int(args.seed),
        "device": str(args.device),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "source_sha256": source_tree_sha256(),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_split_id": train_schedule.schedule_id,
        "train_schedule_sha256": train_schedule.content_sha256,
        "test_split_id": test_schedule.schedule_id,
        "test_schedule_sha256": test_schedule.content_sha256,
        "finite_buffer_protocol": bool(sys_cfg.finite_buffer_protocol),
        "latency_is_uncapped": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_training_plots(out_dir, args.method)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO-GP-SF-DSR extension.")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--distribution", default="uniform", choices=["uniform", "gaussian", "poisson", "poisson_bursty", "bursty"])
    parser.add_argument(
        "--method",
        default="DSR-MTG-Qpen-v3",
        choices=[
            "MAPPO-L",
            "MAPPO-TeamL",
            "MAPPO-DSR",
            "MAPPO-DSR-risk",
            "M2PPO",
            "DSR-MTG",
            "DSR-MTG-Q",
            "DSR-MTG-Q-noConc",
            "DSR-MTG-Qpen-v3",
            "DSR-MTG-noQpen-v3",
            "DSR-MTG-Qpen-v4",
            "DSR-MTG-Qpen-v4-agegate",
            "DSR-MTG-Qpen-v4-agegate-lyapprior",
            "DSR-MTG-Qpen-v4-agegate-lyapresidual",
            "DSR-MTG-Qpen-v4-lyapresidual-v2",
            "DSR-MTG-Qpen-v4-lyapresidual-v3",
            "DSR-MTG-Qpen-v4-lyapresidual-v3p1",
            "DSR-MTG-Qpen-v4-agegate-noQpen",
            "DSR-MTG-Qpen-v4-agegate-noConc",
            "LCC-MAPPO",
        ],
    )
    parser.add_argument("--train-steps", type=int, default=1_000, help="Number of joint time slots.")
    parser.add_argument("--test-steps", type=int, default=200, help="Number of evaluation time slots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_mappo_gp_sf_dsr"))
    parser.add_argument(
        "--canonical-schedule-root",
        default=str(ROOT / "results_qla_canonical_v1" / "shared_schedules"),
    )
    parser.add_argument("--canonical-max-feedback-delay", type=int, default=8)
    parser.add_argument(
        "--evaluation-split-role",
        choices=["validation", "test"],
        default="test",
        help="Canonical split used for post-training evaluation. Parameter selection must use validation; final reporting uses test.",
    )
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--feedback-delay", type=int, default=0)
    parser.add_argument("--mixed-feedback-delays", default="",
                        help="Comma-separated feedback delays sampled per rollout during training, e.g. 0,4,8.")
    parser.add_argument("--broadcast-period", type=int, default=1)
    parser.add_argument("--slot-s", type=float, default=10.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=10.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--poisson-lambda", type=float, default=None)
    parser.add_argument("--burst-prob", type=float, default=0.0)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--edge-compute-scale", type=float, default=1.0,
                        help="Multiplier on edge_compute_mb_s (default 63 MB/s). 0.1 = stressed capacity.")
    parser.add_argument("--task-compute-scale", type=float, default=1.0,
                        help="Multiplier on task_compute_mb (default 10 MB).")
    parser.add_argument("--rollout-size", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--twin-critic", action="store_true")
    parser.add_argument("--critic-ensemble-count", type=int, default=1)
    parser.add_argument("--auto-portfolio-entropy", action="store_true")
    parser.add_argument("--target-portfolio-entropy", type=float, default=0.95)
    parser.add_argument("--auto-entropy-lr", type=float, default=1e-3)
    parser.add_argument("--auto-entropy-init", type=float, default=0.01)
    parser.add_argument("--log-std-min", type=float, default=-1.5)
    parser.add_argument("--feedback-age-aware-score-gate", action="store_true",
                        help="Use local branch + exp(-eta*age) gated score branch in the actor.")
    parser.add_argument("--feedback-gate-eta", type=float, default=0.25)
    parser.add_argument("--feedback-age-norm", type=float, default=8.0,
                        help="Normalizes feedback delay before appending it to actor observations.")
    parser.add_argument("--lyapunov-prior-weight", type=float, default=0.0,
                        help="Blend weight for the stale Lyapunov portfolio prior; 0 disables the adapter.")
    parser.add_argument("--lyapunov-prior-v", type=float, default=1.0,
                        help="Delay trade-off V used by the Lyapunov prior.")
    parser.add_argument("--lyapunov-prior-temperature", type=float, default=2.5,
                        help="Softmax temperature for the Lyapunov prior.")
    parser.add_argument("--lyapunov-prior-smoothing", type=float, default=0.0,
                        help="Exponential smoothing against the previous portfolio after prior blending.")
    parser.add_argument("--lyapunov-prior-max-step", type=float, default=0.0,
                        help="Optional per-user L1 trust-region on portfolio changes; 0 disables.")
    parser.add_argument("--lyapunov-prior-fresh-state", action="store_true",
                        help="Use true current queue state for the prior. For oracle diagnostics only; not paper-fair.")
    parser.add_argument("--residual-radius-min", type=float, default=0.0,
                        help="Minimum L1-scaled residual radius around the stale Lyapunov prior.")
    parser.add_argument("--residual-radius-max", type=float, default=0.25,
                        help="Maximum L1-scaled residual radius around the stale Lyapunov prior.")
    parser.add_argument("--residual-direction-l1-cap", type=float, default=1.0,
                        help="L1 cap for residual-v2 direction before multiplying by actor amplitude and radius.")
    parser.add_argument("--residual-amplitude-init-bias", type=float, default=0.0,
                        help="Initial bias for residual-v2 amplitude logit; negative values start more conservative.")
    parser.add_argument("--residual-amplitude-penalty", type=float, default=0.0,
                        help="Light actor-side penalty on squared residual-v2 amplitude mean.")
    parser.add_argument("--residual-proposal-temperature", type=float, default=1.0,
                        help="Softmax temperature for residual-v3 correction proposal q.")
    parser.add_argument("--residual-beta-max", type=float, default=0.05,
                        help="Maximum residual-v3 interpolation authority beta.")
    parser.add_argument("--residual-beta-gamma-delta", type=float, default=1.0,
                        help="Monotone feedback-age envelope coefficient for residual-v3 beta.")
    parser.add_argument("--residual-beta-gamma-volatility", type=float, default=1.0,
                        help="Stale-summary volatility envelope coefficient for residual-v3 beta.")
    parser.add_argument("--residual-aux-kl-weight", type=float, default=0.0,
                        help="Training-only CTDE KL weight from fresh Lyapunov prior to residual-v3 executed portfolio.")
    parser.add_argument("--residual-aux-proposal-kl-weight", type=float, default=0.0,
                        help="Training-only direct KL weight from fresh Lyapunov prior to residual-v3 proposal q.")
    parser.add_argument("--residual-aux-amplitude-weight", type=float, default=0.0,
                        help="Training-only weight for residual-v3 amplitude calibration against oracle correction gain.")
    parser.add_argument("--residual-beta-penalty", type=float, default=0.0,
                        help="Training-only penalty on residual-v3 beta authority.")
    parser.add_argument("--residual-confidence-eta-delta", type=float, default=0.10,
                        help="Feedback-age coefficient for residual-prior confidence.")
    parser.add_argument("--residual-confidence-eta-volatility", type=float, default=0.10,
                        help="Stale-summary volatility coefficient for residual-prior confidence.")
    parser.add_argument("--lcc-v", type=float, default=1.0,
                        help="Lyapunov-coordinated MAPPO latency-vs-drift weight V.")
    parser.add_argument("--lcc-migration-weight", type=float, default=0.02,
                        help="Single portfolio migration penalty lambda_m for LCC-MAPPO.")
    parser.add_argument("--lcc-cvar-alpha", type=float, default=0.99,
                        help="Tail level for the LCC virtual CVaR queue.")
    parser.add_argument("--lcc-cvar-budget", type=float, default=1.30,
                        help="Normalized CVaR service budget zeta for the LCC virtual risk queue.")
    parser.add_argument("--lcc-var-lr", type=float, default=0.05,
                        help="EWMA update rate for the LCC Rockafellar-Uryasev VaR auxiliary variable.")
    parser.add_argument("--lcc-concentration-factor", type=float, default=2.0,
                        help="Concentration budget factor; budget=max(0.02, factor/M).")
    parser.add_argument("--lcc-stale-margin", type=float, default=1.0,
                        help="Fixed stale-margin multiplier added to the delayed wait summary in actor state.")
    parser.add_argument("--lcc-no-z-queue", action="store_true",
                        help="Ablation: remove the CVaR virtual queue Z.")
    parser.add_argument("--lcc-no-h-queue", action="store_true",
                        help="Ablation: remove the concentration virtual queue H.")
    parser.add_argument("--lcc-no-stale-margin", action="store_true",
                        help="Ablation: remove the fixed stale uncertainty margin from actor state.")
    parser.add_argument("--lcc-static-reward", action="store_true",
                        help="Ablation: use static latency/migration team reward without Lyapunov drift queues.")
    parser.add_argument("--lcc-price-observation", action="store_true",
                        help="Append per-user edge marginal risk prices to LCC actor observations without action prior/fusion.")
    parser.add_argument("--lcc-price-clean-observation", action="store_true",
                        help="Replace stale wait vector with edge risk prices while keeping previous portfolio and Z/H scalars.")
    parser.add_argument("--lcc-price-scale-observation", action="store_true",
                        help="Append one log-scale scalar carrying absolute LCC price magnitude.")
    parser.add_argument("--lcc-edgewise-price-actor", action="store_true",
                        help="Use a per-edge shared scoring actor for price-augmented LCC observations.")
    parser.add_argument("--lcc-flat-price-bias-actor", action="store_true",
                        help="Use a flattened MLP actor with the same explicit price-to-logit bias as the edge-wise actor.")
    parser.add_argument("--lcc-price-only-policy", action="store_true",
                        help="Use analytic softmax(-lambda*price) portfolios as a non-learning price-only baseline.")
    parser.add_argument("--lcc-price-components", default="Full", choices=["NoPrice", "Q", "T", "QT", "QTZ", "QTH", "Full", "FullSafe"],
                        help="Which marginal-risk price components to expose: NoPrice, QT, QTZ, QTH, Full, or FullSafe.")
    parser.add_argument("--lcc-price-shuffle", action="store_true",
                        help="Diagnostic: shuffle per-user price dimensions before feeding the actor.")
    parser.add_argument("--lcc-price-intervention", default="aligned", choices=["aligned", "shuffle", "zero", "constant"],
                        help="Test-time/diagnostic price intervention applied inside the price observation path.")
    parser.add_argument("--lcc-price-logit-bias", type=float, default=0.0,
                        help="For edge-wise price actor, subtract lambda*price from each action logit mean.")
    parser.add_argument("--lcc-local-critic", action="store_true",
                        help="Ablation: train critic on local actor-visible observation plus own portfolio, not centralized fresh state.")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta-mean", type=float, default=0.4)
    parser.add_argument("--beta-tail", type=float, default=0.4)
    parser.add_argument("--beta-gap", type=float, default=0.2)
    parser.add_argument("--beta-qpressure", type=float, default=0.0)
    parser.add_argument("--tail-topk", type=int, default=0, help="Top-K users for tail latency; <=0 uses tail fraction.")
    parser.add_argument("--tail-fraction", type=float, default=0.6)
    parser.add_argument("--cvar-alpha", type=float, default=None, help="User-tail CVaR alpha; sets tail_fraction=1-alpha.")
    parser.add_argument("--kappa-i", type=float, default=0.01)
    parser.add_argument("--kappa-g", type=float, default=0.01)
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--q-pressure-max", type=float, default=3.0)
    parser.add_argument("--q-penalty-weight", type=float, default=0.05)
    parser.add_argument("--dsr-mu", type=float, default=1.5)
    parser.add_argument("--dsr-eta", type=float, default=0.01)
    parser.add_argument("--dsr-temperature", type=float, default=5.0)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-checkpoint", default="",
                        help="Optional MAPPO checkpoint to load before evaluation; use with --train-steps 0 for test-time interventions.")
    return parser.parse_args()


def main() -> None:
    summary = train_once(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
