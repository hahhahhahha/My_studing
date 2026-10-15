from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DSRReward:
    mu: float = 11.0
    eta: float = 1e-3
    eps: float = 1e-8
    a: float = 0.0
    b: float = 0.0

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0

    def __call__(self, latency: float) -> float:
        offload_return = self.mu - latency
        prev_a = self.a
        prev_b = self.b
        delta_a = offload_return - prev_a
        delta_b = offload_return * offload_return - prev_b
        denom = max(prev_b - prev_a * prev_a, self.eps) ** 1.5
        reward = (prev_b * delta_a - 0.5 * prev_a * delta_b) / denom
        self.a = prev_a + self.eta * delta_a
        self.b = prev_b + self.eta * delta_b
        return float(np.clip(reward, -1.0, 1.0))


@dataclass
class MVReward:
    mu: float = 11.0
    eps: float = 1e-8
    count: int = 0
    sum_return: float = 0.0
    sumsq_return: float = 0.0
    prev_sr: float = -np.inf

    def reset(self) -> None:
        self.count = 0
        self.sum_return = 0.0
        self.sumsq_return = 0.0
        self.prev_sr = -np.inf

    def __call__(self, latency: float) -> float:
        value = self.mu - latency
        self.count += 1
        self.sum_return += value
        self.sumsq_return += value * value
        mean = self.sum_return / self.count
        var = max(self.sumsq_return / self.count - mean * mean, 0.0)
        sr = mean / np.sqrt(var + self.eps)
        if self.prev_sr == -np.inf:
            reward = 0.0
        else:
            reward = 1.0 if sr >= self.prev_sr else 0.0
        self.prev_sr = sr
        return reward


def latency_reward(latency: float) -> float:
    return -float(latency)
