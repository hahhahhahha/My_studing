"""Canonical train/validation/test schedules and artifact provenance.

The schedule fixes exogenous randomness independently of the policy: trace
indices, request counts, and the uniforms used to map a portfolio CDF to a
server.  Different methods therefore see the same tasks and random numbers
while still producing method-dependent assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TraceSplit:
    role: str
    start: int
    length: int
    guard: int


@dataclass
class CanonicalSchedule:
    schedule_id: str
    role: str
    trace_indices: np.ndarray
    requests: np.ndarray
    assignment_uniforms: np.ndarray
    capacity_scale: np.ndarray
    background_scale: np.ndarray
    summary_loss: np.ndarray
    content_sha256: str
    path: Path

    @property
    def n_steps(self) -> int:
        return int(self.requests.shape[0])


def trace_splits(n_trace_slots: int, guard: int) -> dict[str, TraceSplit]:
    guard = max(0, int(guard))
    usable = int(n_trace_slots) - 2 * guard
    if usable < 15:
        raise ValueError(
            f"Trace has {n_trace_slots} slots, insufficient for 60/20/20 splits "
            f"with two guards of {guard} slots"
        )
    n_train = max(1, int(np.floor(0.60 * usable)))
    n_val = max(1, int(np.floor(0.20 * usable)))
    n_test = usable - n_train - n_val
    train = TraceSplit("train", 0, n_train, guard)
    val = TraceSplit("validation", n_train + guard, n_val, guard)
    test = TraceSplit("test", n_train + guard + n_val + guard, n_test, guard)
    return {"train": train, "validation": val, "test": test}


def _request_schedule(
    rng: np.random.Generator,
    distribution: str,
    n_steps: int,
    n_users: int,
    max_requests: int,
    poisson_lambda: float | None,
    burst_prob: float,
    burst_multiplier: float,
) -> np.ndarray:
    high = int(max_requests)
    if distribution == "uniform":
        return rng.integers(0, high + 1, size=(n_steps, n_users), dtype=np.int64)
    if distribution == "gaussian":
        values = np.rint(rng.normal(high / 2.0, high / 6.0, size=(n_steps, n_users)))
        return np.clip(values, 0, high).astype(np.int64)
    if distribution in {"poisson", "poisson_bursty", "bursty"}:
        base = float(poisson_lambda if poisson_lambda is not None else high / 2.0)
        out = np.zeros((n_steps, n_users), dtype=np.int64)
        for slot in range(n_steps):
            lam = base
            if distribution in {"poisson_bursty", "bursty"} and rng.random() < float(burst_prob):
                lam *= float(burst_multiplier)
            out[slot] = np.clip(rng.poisson(lam, size=n_users), 0, high)
        return out
    raise ValueError(f"Unknown request distribution: {distribution}")


def _content_hash(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_or_load_schedule(
    *,
    root: Path,
    dataset: str,
    distribution: str,
    role: str,
    seed: int,
    n_steps: int,
    n_trace_slots: int,
    n_users: int,
    n_edges: int,
    max_requests: int,
    max_feedback_delay: int,
    broadcast_period: int,
    poisson_lambda: float | None,
    burst_prob: float,
    burst_multiplier: float,
) -> CanonicalSchedule:
    if role not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split role: {role}")
    guard = max(0, int(max_feedback_delay)) + max(1, int(broadcast_period))
    split = trace_splits(n_trace_slots, guard)[role]
    metadata: dict[str, object] = {
        "dataset": dataset,
        "distribution": distribution,
        "role": role,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "n_trace_slots": int(n_trace_slots),
        "split_start": split.start,
        "split_length": split.length,
        "guard": split.guard,
        "n_users": int(n_users),
        "n_edges": int(n_edges),
        "max_requests": int(max_requests),
        "poisson_lambda": None if poisson_lambda is None else float(poisson_lambda),
        "burst_prob": float(burst_prob),
        "burst_multiplier": float(burst_multiplier),
    }
    identity = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    schedule_id = f"{dataset}_{distribution}_{role}_s{seed}_{identity}"
    directory = Path(root) / dataset / distribution / role
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{schedule_id}.npz"
    manifest_path = directory / f"{schedule_id}.json"

    if npz_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        content_hash = _content_hash(arrays, metadata)
        if content_hash != manifest.get("content_sha256"):
            raise RuntimeError(f"Schedule hash mismatch: {npz_path}")
    else:
        rng = np.random.default_rng(int(seed))
        trace_indices = split.start + (np.arange(n_steps, dtype=np.int64) % split.length)
        requests = _request_schedule(
            rng,
            distribution,
            n_steps,
            n_users,
            max_requests,
            poisson_lambda,
            burst_prob,
            burst_multiplier,
        )
        assignment_uniforms = rng.random((n_steps, n_users, max_requests), dtype=np.float64)
        arrays = {
            "trace_indices": trace_indices,
            "requests": requests,
            "assignment_uniforms": assignment_uniforms,
            "capacity_scale": np.ones((n_steps, n_edges), dtype=np.float32),
            "background_scale": np.ones((n_steps, n_edges), dtype=np.float32),
            "summary_loss": np.zeros((n_steps, n_edges), dtype=np.bool_),
        }
        content_hash = _content_hash(arrays, metadata)
        np.savez_compressed(npz_path, **arrays)
        manifest_path.write_text(
            json.dumps(
                {**metadata, "schedule_id": schedule_id, "content_sha256": content_hash},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return CanonicalSchedule(
        schedule_id=schedule_id,
        role=role,
        trace_indices=np.asarray(arrays["trace_indices"], dtype=np.int64),
        requests=np.asarray(arrays["requests"], dtype=np.int64),
        assignment_uniforms=np.asarray(arrays["assignment_uniforms"], dtype=np.float64),
        capacity_scale=np.asarray(arrays["capacity_scale"], dtype=np.float32),
        background_scale=np.asarray(arrays["background_scale"], dtype=np.float32),
        summary_loss=np.asarray(arrays["summary_loss"], dtype=np.bool_),
        content_sha256=content_hash,
        path=npz_path,
    )


def install_schedule_for_env(
    env,
    *,
    root: Path,
    distribution: str,
    role: str,
    seed: int,
    n_steps: int,
    max_feedback_delay: int = 8,
) -> CanonicalSchedule:
    schedule = build_or_load_schedule(
        root=Path(root),
        dataset=str(env.traces.dataset),
        distribution=distribution,
        role=role,
        seed=int(seed),
        n_steps=int(n_steps),
        n_trace_slots=int(env.traces.trajectories.shape[1]),
        n_users=int(env.cfg.n_users),
        n_edges=int(env.cfg.n_edges),
        max_requests=int(env.cfg.max_requests_per_user),
        max_feedback_delay=int(max_feedback_delay),
        broadcast_period=int(env.broadcast_period),
        poisson_lambda=env.poisson_lambda,
        burst_prob=float(env.burst_prob),
        burst_multiplier=float(env.burst_multiplier),
    )
    env.install_canonical_schedule(schedule)
    return schedule


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_source_sha256() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_run_manifest(
    *,
    output_dir: Path,
    method: str,
    args,
    summary: dict[str, object],
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "method": method,
        "seed": int(getattr(args, "seed", -1)),
        "python": platform.python_version(),
        "source_sha256": package_source_sha256(),
        "finite_buffer_protocol": bool(summary.get("finite_buffer_protocol", False)),
        "latency_is_uncapped": bool(summary.get("latency_is_uncapped", False)),
        "checkpoint_rule": "final_requested_training_budget_no_test_early_stopping",
    }
    for key in (
        "train_split_id",
        "train_schedule_sha256",
        "validation_split_id",
        "validation_schedule_sha256",
        "test_split_id",
        "test_schedule_sha256",
        "evaluation_split_role",
        "evaluation_split_id",
        "evaluation_schedule_sha256",
    ):
        if key in summary:
            payload[key] = summary[key]
    if config_path is not None and Path(config_path).exists():
        payload["config_sha256"] = file_sha256(Path(config_path))
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        payload["checkpoint_sha256"] = file_sha256(Path(checkpoint_path))
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
