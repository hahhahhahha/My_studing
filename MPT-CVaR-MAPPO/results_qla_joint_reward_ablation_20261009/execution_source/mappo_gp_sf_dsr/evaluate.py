from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from .env import JointMAPPOEdgeEnv, SlotResult
from .models import MAPPOPolicy


def evaluate_policy(
    policy: MAPPOPolicy,
    env: JointMAPPOEdgeEnv,
    steps: int,
    deterministic: bool = True,
    portfolio_transform: Callable[..., np.ndarray] | None = None,
    observation_transform: Callable[..., np.ndarray] | None = None,
    post_step_callback: Callable[..., None] | None = None,
) -> list[SlotResult]:
    results: list[SlotResult] = []
    env.reset()
    for _ in range(steps):
        actor_obs = env.actor_observations()
        if observation_transform is not None:
            actor_obs = observation_transform(actor_obs, env)
        start = perf_counter()
        portfolios, latents, _ = policy.act(actor_obs, deterministic=deterministic)
        if portfolio_transform is not None:
            portfolios = portfolio_transform(portfolios, env, latents)
        decision_latency = perf_counter() - start
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency)
        results.append(result)
        if post_step_callback is not None:
            post_step_callback(result, env)
    return results


def random_policy_results(env: JointMAPPOEdgeEnv, steps: int) -> list[SlotResult]:
    results: list[SlotResult] = []
    env.reset()
    for _ in range(steps):
        portfolios = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=0.0))
    return results
