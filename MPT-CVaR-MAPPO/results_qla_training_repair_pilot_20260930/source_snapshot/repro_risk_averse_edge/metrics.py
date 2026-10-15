from __future__ import annotations

import numpy as np


def empirical_cvar(latencies: list[float] | np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(latencies, dtype=np.float64))
    if values.size == 0:
        return float("nan")
    idx = int(np.ceil(alpha * values.size))
    idx = min(max(idx, 0), values.size - 1)
    return float(np.mean(values[idx:]))


def latency_summary(latencies: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "average_latency": float(np.mean(values)) if values.size else float("nan"),
        "cvar95": empirical_cvar(values, 0.95),
        "cvar99": empirical_cvar(values, 0.99),
        "n_tasks": int(values.size),
    }

