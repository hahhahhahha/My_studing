from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TRACE_ROOT


@dataclass
class TraceSet:
    dataset: str
    trajectories: np.ndarray  # shape: users, slots, 2
    edge_positions: np.ndarray  # shape: edges, 2


def _read_trace_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Trace {path} must have time/x/y columns")
    return data[:, :3]


def load_trace_set(
    dataset: str,
    n_users: int,
    n_edges: int,
    slot_s: float,
    seed: int,
    trace_root: Path = TRACE_ROOT,
) -> TraceSet:
    """Load pre-interpolated 10-second GPS traces.

    The trace files already contain 10-second sampled positions, matching the
    paper's τ = 10 s slot duration. Edge server positions are sampled uniformly
    inside the dataset bounding box with a fixed seed for reproducibility.
    """
    folder = trace_root / dataset
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    files = sorted(folder.glob("*.txt"))
    if len(files) < n_users:
        raise ValueError(f"{dataset} has {len(files)} traces, need {n_users}")

    selected = files[:n_users]
    traces = [_read_trace_file(path)[:, 1:] for path in selected]
    min_len = min(len(t) for t in traces)
    trajectories = np.stack([t[:min_len] for t in traces], axis=0).astype(np.float32)

    coords = trajectories.reshape(-1, 2)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    rng = np.random.default_rng(seed)
    edge_positions = rng.uniform(lo, lo + span, size=(n_edges, 2)).astype(np.float32)
    return TraceSet(dataset=dataset, trajectories=trajectories, edge_positions=edge_positions)
