"""Numerically testable GAE helpers for adapted policy-gradient baselines."""
from __future__ import annotations

import numpy as np


def gae_with_bootstrap(
    rewards: np.ndarray,
    values: np.ndarray,
    user_ids: np.ndarray,
    slots: np.ndarray,
    bootstrap_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    dones: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE per trajectory, including a continuing rollout's last V.

    Transitions may be stored in interleaved user-major batches.  ``slots``
    restores temporal order within each user's trajectory.  A true terminal
    flag masks both the bootstrap and future advantage; an administrative
    rollout boundary is represented by ``dones=0`` and uses ``bootstrap_values``.
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    user_ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32).reshape(-1)
    if dones is None:
        dones = np.zeros_like(rewards, dtype=np.float32)
    else:
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    n = len(rewards)
    if any(len(x) != n for x in (values, user_ids, slots, dones)):
        raise ValueError("GAE inputs must have the same transition count")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("Rewards and values must be finite")
    if not np.all(np.isfinite(bootstrap_values)):
        raise ValueError("Bootstrap values must be finite")
    if n and (user_ids.min() < 0 or user_ids.max() >= len(bootstrap_values)):
        raise ValueError("Every user id must have a corresponding bootstrap value")

    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    for user in np.unique(user_ids):
        indices = np.flatnonzero(user_ids == user)
        indices = indices[np.argsort(slots[indices], kind="stable")]
        gae = 0.0
        for pos in reversed(range(len(indices))):
            idx = indices[pos]
            if dones[idx] >= 1.0:
                next_value = 0.0
            elif pos + 1 < len(indices):
                next_value = values[indices[pos + 1]]
            else:
                next_value = bootstrap_values[int(user)]
            nonterminal = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * next_value * nonterminal - values[idx]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
    return advantages, returns
