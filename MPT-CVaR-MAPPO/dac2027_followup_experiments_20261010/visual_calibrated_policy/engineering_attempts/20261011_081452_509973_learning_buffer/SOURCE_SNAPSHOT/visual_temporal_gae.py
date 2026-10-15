"""Time-by-user physical-time GAE; no model, optimizer or metric normalization."""
from __future__ import annotations

import numpy as np


def per_user_temporal_gae(rewards, values, next_values, terminal, duration_seconds,
                          gamma_per_second=.99, lambda_per_second=.95):
    rewards, values, next_values = (np.asarray(x, dtype=np.float64) for x in (rewards, values, next_values))
    terminal = np.asarray(terminal)
    duration = np.asarray(duration_seconds, dtype=np.float64)
    if (rewards.ndim != 2 or min(rewards.shape) < 1 or values.shape != rewards.shape
            or next_values.shape != rewards.shape or terminal.shape != rewards.shape
            or terminal.dtype != np.bool_ or duration.shape != (rewards.shape[0],)
            or not all(np.isfinite(x).all() for x in (rewards, values, next_values, duration))
            or np.any(duration <= 0)):
        raise ValueError('Finite [time,user] arrays, explicit boolean terminal and positive physical durations required')
    if not (np.isfinite(gamma_per_second) and np.isfinite(lambda_per_second)
            and 0 < gamma_per_second <= 1 and 0 < lambda_per_second <= 1):
        raise ValueError('Fixed per-second discount and trace constants required')
    gamma = np.power(gamma_per_second, duration)[:, None]
    trace = np.power(lambda_per_second, duration)[:, None]
    alive = ~terminal
    delta = rewards + gamma * alive * next_values - values
    advantages = np.empty_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for time in range(rewards.shape[0] - 1, -1, -1):
        running = delta[time] + gamma[time] * trace[time] * alive[time] * running
        advantages[time] = running
    returns = advantages + values
    if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
        raise RuntimeError('GAE overflow; do not silently clip training targets')
    advantages.setflags(write=False)
    returns.setflags(write=False)
    return advantages, returns


def flatten_after_gae(features, advantages, returns):
    """Flatten for an eventual optimizer ONLY AFTER independent user GAE."""
    advantages, returns = np.asarray(advantages), np.asarray(returns)
    if advantages.ndim != 2 or returns.shape != advantages.shape:
        raise ValueError('Prepared [time,user] advantages/returns required')
    flat = {}
    for name, value in features.items():
        value = np.asarray(value)
        if value.ndim < 2 or value.shape[:2] != advantages.shape or not np.isfinite(value).all():
            raise ValueError('All optimizer fields must share the time/user layout')
        flat[name] = value.reshape((-1, *value.shape[2:]))
    flat['advantages'] = advantages.reshape(-1)
    flat['returns'] = returns.reshape(-1)
    return flat
