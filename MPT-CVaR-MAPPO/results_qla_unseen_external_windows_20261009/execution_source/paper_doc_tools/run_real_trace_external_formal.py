from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mappo_gp_sf_dsr.canonical_protocol import CanonicalSchedule, trace_splits
from mappo_gp_sf_dsr.env import JointMAPPOEdgeEnv, ServerFeedback


METHOD_Q = "QLA-MAPPO"
METHOD_PPO = "PPO-MV-A"
METHOD_DSR = "PPO-DSR-A"
METHOD_QECO = "QECO-A"
METHOD_LYFOTO = "LyFOTO-LC"
METHOD_JSEQ = "JSEQ-A"
METHOD_MFC = "MFC-DI-A"
METHOD_SCORE = "Score-Softmax"
METHOD_JSQ = "JSQ"
METHODS = (
    METHOD_Q,
    METHOD_PPO,
    METHOD_DSR,
    METHOD_QECO,
    METHOD_LYFOTO,
    METHOD_JSEQ,
    METHOD_MFC,
    METHOD_SCORE,
    METHOD_JSQ,
)

FILE_METHOD = {
    METHOD_Q: "LCC-MAPPO",
    METHOD_PPO: "Towards-PPO-MV",
    METHOD_DSR: "Towards-PPO-DSR",
    METHOD_QECO: "QECO-load-topk",
    METHOD_LYFOTO: "Lyapunov-Q-local-stale",
}


@dataclass
class ExternalTraceSchedule(CanonicalSchedule):
    task_compute_scale: np.ndarray
    source_age_extra: np.ndarray
    protocol_metadata: dict[str, object]


def _stable_hash(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _numeric_series_from_csv(path: Path, max_rows: int = 500_000) -> np.ndarray | None:
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                nums = []
                for cell in row:
                    try:
                        value = float(cell)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        nums.append(value)
                if nums:
                    values.append(float(np.mean(nums)))
    except OSError:
        return None
    if len(values) < 16:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size >= 16 else None


def _find_first_numeric_trace(root: Path, patterns: Iterable[str]) -> tuple[np.ndarray | None, str]:
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.stat().st_size <= 0:
                continue
            series = _numeric_series_from_csv(path)
            if series is not None:
                return series, str(path)
    return None, "deterministic_proxy"


def _normalize_series(series: np.ndarray, n: int, low: float, high: float) -> np.ndarray:
    if series.size == 0:
        raise ValueError("empty series")
    x = np.resize(np.asarray(series, dtype=np.float64), n)
    lo, hi = np.percentile(x, [5.0, 95.0])
    if hi <= lo:
        y = np.full(n, 0.5, dtype=np.float64)
    else:
        y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (low + (high - low) * y).astype(np.float32)


def _proxy_azure_series(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    diurnal = 1.0 + 0.35 * np.sin(2.0 * np.pi * t / max(24, n // 3))
    burst = rng.gamma(shape=1.4, scale=0.45, size=n)
    shock = (rng.random(n) < 0.04) * rng.uniform(1.5, 3.0, size=n)
    return np.maximum(0.05, diurnal + burst + shock)


def _proxy_alibaba_series(n: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    base = 0.78 + 0.14 * np.sin(2.0 * np.pi * t[:, None] / 96.0 + rng.uniform(0, 2 * np.pi, size=(1, n_edges)))
    noise = rng.normal(0.0, 0.035, size=(n, n_edges))
    shocks = np.ones((n, n_edges), dtype=np.float64)
    for _ in range(max(1, n // 160)):
        start = int(rng.integers(0, max(1, n)))
        width = int(rng.integers(8, 40))
        edges = rng.choice(n_edges, size=max(1, n_edges // 6), replace=False)
        shocks[start : min(n, start + width), edges] *= rng.uniform(0.45, 0.75)
    return np.clip(base + noise, 0.35, 1.25) * shocks


def build_or_load_external_schedule(
    *,
    root: Path,
    data_root: Path,
    dataset: str,
    role: str,
    seed: int,
    n_steps: int,
    n_trace_slots: int,
    n_users: int,
    n_edges: int,
    max_requests: int,
    max_feedback_delay: int,
    broadcast_period: int,
    require_real_external_data: bool = False,
) -> ExternalTraceSchedule:
    guard = max(0, int(max_feedback_delay)) + max(1, int(broadcast_period))
    split = trace_splits(n_trace_slots, guard)[role]
    metadata = {
        "protocol": "real_trace_external_v1",
        "dataset": dataset,
        "role": role,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "split_start": int(split.start),
        "split_length": int(split.length),
        "n_users": int(n_users),
        "n_edges": int(n_edges),
        "max_requests": int(max_requests),
        "max_feedback_delay": int(max_feedback_delay),
        "broadcast_period": int(broadcast_period),
    }
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    out_dir = root / dataset / role
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{dataset}_{role}_s{seed}_{identity}.npz"
    json_path = out_dir / f"{dataset}_{role}_s{seed}_{identity}.json"
    if npz_path.exists() and json_path.exists():
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        digest = _stable_hash(arrays, metadata)
        if digest != manifest.get("content_sha256"):
            raise RuntimeError(f"External schedule hash mismatch: {npz_path}")
        protocol_metadata = dict(manifest.get("protocol_metadata", {}))
        if require_real_external_data and (
            protocol_metadata.get("azure_invocation_kind") != "local_csv"
            or protocol_metadata.get("azure_duration_kind") != "local_csv"
            or protocol_metadata.get("alibaba_capacity_kind") != "local_csv"
        ):
            raise FileNotFoundError(
                "Existing external schedule was built from deterministic proxy data. "
                f"Delete/regenerate it after extracting real Azure and Alibaba CSV files under {data_root}."
            )
    else:
        rng = np.random.default_rng(int(seed))
        total_needed = max(int(n_steps), int(split.length), 64)
        azure_inv_raw, azure_inv_source = _find_first_numeric_trace(
            data_root,
            (
                "*invocations_per_function*.csv",
                "*invocation*.csv",
                "*Azure*/*invocation*.csv",
                "*azure*/*invocation*.csv",
            ),
        )
        azure_dur_raw, azure_dur_source = _find_first_numeric_trace(
            data_root,
            (
                "*function_durations_percentiles*.csv",
                "*duration*.csv",
                "*Azure*/*duration*.csv",
                "*azure*/*duration*.csv",
            ),
        )
        if azure_inv_raw is None:
            azure_inv_raw = _proxy_azure_series(total_needed, rng)
            azure_inv_kind = "deterministic_proxy"
        else:
            azure_inv_kind = "local_csv"
        if azure_dur_raw is None:
            azure_dur_raw = np.sqrt(np.maximum(azure_inv_raw, 1e-6))
            azure_dur_kind = "derived_from_invocation_proxy"
            azure_dur_source = azure_inv_source
        else:
            azure_dur_kind = "local_csv"
        alibaba_raw, alibaba_source = _find_first_numeric_trace(
            data_root,
            ("*Alibaba*/*.csv", "*alibaba*/*.csv", "*machine*/*.csv", "*cluster*/*.csv", "*.csv"),
        )
        if require_real_external_data and (
            azure_inv_kind != "local_csv"
            or azure_dur_kind != "local_csv"
            or alibaba_raw is None
        ):
            missing = []
            if azure_inv_kind != "local_csv":
                missing.append("Azure Functions invocation CSV")
            if azure_dur_kind != "local_csv":
                missing.append("Azure Functions duration CSV")
            if alibaba_raw is None:
                missing.append("Alibaba machine utilization CSV")
            raise FileNotFoundError(
                "Required real external trace file(s) are missing: "
                + ", ".join(missing)
                + f". Put extracted CSV files under {data_root}."
            )
        if alibaba_raw is None:
            capacity_scale = _proxy_alibaba_series(total_needed, n_edges, rng)
            alibaba_kind = "deterministic_proxy"
        else:
            base = _normalize_series(alibaba_raw, total_needed, 0.45, 1.20)
            phases = rng.uniform(0.85, 1.15, size=(1, n_edges))
            capacity_scale = np.clip(base[:, None] * phases, 0.35, 1.25).astype(np.float32)
            alibaba_kind = "local_csv"

        inv_scale = _normalize_series(azure_inv_raw, total_needed, 0.20, 1.00)
        compute_scale_all = _normalize_series(azure_dur_raw, total_needed, 0.70, 1.40)
        trace_indices = split.start + (np.arange(n_steps, dtype=np.int64) % split.length)
        requests = np.zeros((n_steps, n_users), dtype=np.int64)
        user_weight = rng.lognormal(mean=0.0, sigma=0.25, size=n_users)
        user_weight /= np.mean(user_weight)
        for slot in range(n_steps):
            base_lam = 0.65 + 4.2 * float(inv_scale[slot % total_needed])
            lam = np.clip(base_lam * user_weight, 0.05, max_requests)
            requests[slot] = np.clip(rng.poisson(lam), 0, max_requests)
        arrays = {
            "trace_indices": trace_indices,
            "requests": requests,
            "assignment_uniforms": rng.random((n_steps, n_users, max_requests), dtype=np.float64),
            "capacity_scale": np.resize(capacity_scale, (n_steps, n_edges)).astype(np.float32),
            "background_scale": np.ones((n_steps, n_edges), dtype=np.float32),
            "summary_loss": (rng.random((n_steps, n_edges)) < 0.03).astype(np.bool_),
            "task_compute_scale": np.resize(compute_scale_all, n_steps).astype(np.float32),
            "source_age_extra": rng.integers(0, max(1, max_feedback_delay + 1), size=(n_steps, n_edges), dtype=np.int64),
        }
        protocol_metadata = {
            "azure_invocation_source": azure_inv_source,
            "azure_invocation_kind": azure_inv_kind,
            "azure_duration_source": azure_dur_source,
            "azure_duration_kind": azure_dur_kind,
            "alibaba_machine_source": alibaba_source,
            "alibaba_capacity_kind": alibaba_kind,
            "request_model": "Azure invocation intensity mapped to per-user Poisson arrivals",
            "compute_model": "Azure intensity mapped to slot-level task compute scaling",
            "capacity_model": "Alibaba utilization mapped to per-edge residual service-rate scaling",
            "summary_loss_probability": 0.03,
            "capacity_shock_model": "server-specific shock embedded in capacity_scale",
        }
        digest = _stable_hash(arrays, metadata)
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(
            json.dumps(
                {**metadata, "schedule_id": npz_path.stem, "content_sha256": digest, "protocol_metadata": protocol_metadata},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return ExternalTraceSchedule(
        schedule_id=npz_path.stem,
        role=role,
        trace_indices=np.asarray(arrays["trace_indices"], dtype=np.int64),
        requests=np.asarray(arrays["requests"], dtype=np.int64),
        assignment_uniforms=np.asarray(arrays["assignment_uniforms"], dtype=np.float64),
        capacity_scale=np.asarray(arrays["capacity_scale"], dtype=np.float32),
        background_scale=np.asarray(arrays["background_scale"], dtype=np.float32),
        summary_loss=np.asarray(arrays["summary_loss"], dtype=np.bool_),
        content_sha256=digest,
        path=npz_path,
        task_compute_scale=np.asarray(arrays["task_compute_scale"], dtype=np.float32),
        source_age_extra=np.asarray(arrays["source_age_extra"], dtype=np.int64),
        protocol_metadata=protocol_metadata,
    )


class ExternalTraceEnv(JointMAPPOEdgeEnv):
    def __init__(self, *args, **kwargs) -> None:
        self._external_schedule: ExternalTraceSchedule | None = None
        self._base_capacities_mb_s: np.ndarray | None = None
        self._base_task_compute_mb: float | None = None
        self._true_feedback_history: dict[int, ServerFeedback] = {}
        self._last_composite_feedback: ServerFeedback | None = None
        super().__init__(*args, **kwargs)
        self._base_capacities_mb_s = self.edge_capacities_mb_s.astype(np.float32).copy()
        self._base_task_compute_mb = float(self.cfg.task_compute_mb)

    def install_canonical_schedule(self, schedule: CanonicalSchedule) -> None:
        self._external_schedule = schedule if isinstance(schedule, ExternalTraceSchedule) else None
        super().install_canonical_schedule(schedule)

    def reset(self) -> np.ndarray:
        obs = super().reset()
        self._true_feedback_history = {}
        self._last_composite_feedback = None
        self._apply_external_slot()
        true_feedback = super().server_feedback()
        self._true_feedback_history[int(self.slot_index)] = true_feedback
        self.last_broadcast_feedback = true_feedback
        self.feedback_buffer.clear()
        self.feedback_buffer.push(true_feedback)
        return self.actor_observations()

    def _apply_external_slot(self) -> None:
        schedule = self._external_schedule
        if schedule is None or schedule.n_steps <= 0:
            return
        idx = min(max(0, int(self.schedule_cursor)), schedule.n_steps - 1)
        if self._base_capacities_mb_s is not None:
            self.edge_capacities_mb_s = (
                self._base_capacities_mb_s * np.asarray(schedule.capacity_scale[idx], dtype=np.float32)
            ).astype(np.float32)
        if self._base_task_compute_mb is not None:
            object.__setattr__(
                self.cfg,
                "task_compute_mb",
                float(self._base_task_compute_mb * float(schedule.task_compute_scale[idx])),
            )

    def _composite_feedback(self, current_slot: int, true_feedback: ServerFeedback) -> ServerFeedback:
        schedule = self._external_schedule
        if schedule is None:
            return true_feedback
        idx = min(max(0, int(self.schedule_cursor)), schedule.n_steps - 1)
        n_edges = self.cfg.n_edges
        out = ServerFeedback(
            score=true_feedback.score.copy(),
            queue_task_count_norm=true_feedback.queue_task_count_norm.copy(),
            available_resource_norm=true_feedback.available_resource_norm.copy(),
            capacity_norm=true_feedback.capacity_norm.copy(),
            wait_norm=true_feedback.wait_norm.copy(),
            available_resource_mb=true_feedback.available_resource_mb.copy(),
            wait_s=true_feedback.wait_s.copy(),
            source_slot=int(current_slot),
        )
        source_slots: list[int] = []
        for edge in range(n_edges):
            if bool(schedule.summary_loss[idx, edge]) and self._last_composite_feedback is not None:
                src = self._last_composite_feedback
                source_slots.append(int(src.source_slot))
            else:
                extra = int(schedule.source_age_extra[idx, edge])
                src_slot = max(0, int(current_slot) - extra)
                available = [s for s in self._true_feedback_history if s <= src_slot]
                src = self._true_feedback_history[max(available)] if available else true_feedback
                source_slots.append(int(src.source_slot))
            out.score[edge] = src.score[edge]
            out.queue_task_count_norm[edge] = src.queue_task_count_norm[edge]
            out.available_resource_norm[edge] = src.available_resource_norm[edge]
            out.capacity_norm[edge] = src.capacity_norm[edge]
            out.wait_norm[edge] = src.wait_norm[edge]
            out.available_resource_mb[edge] = src.available_resource_mb[edge]
            out.wait_s[edge] = src.wait_s[edge]
        out.source_slot = int(min(source_slots)) if source_slots else int(current_slot)
        self._last_composite_feedback = out
        return out

    def joint_step(self, *args, **kwargs):
        self._apply_external_slot()
        result = super().joint_step(*args, **kwargs)
        self._apply_external_slot()
        true_feedback = super().server_feedback()
        self._true_feedback_history[int(self.slot_index)] = true_feedback
        if self.last_broadcast_feedback is None or self.slot_index % self.broadcast_period == 0:
            self.last_broadcast_feedback = self._composite_feedback(int(self.slot_index), true_feedback)
        if self.feedback_buffer.buffer:
            self.feedback_buffer.buffer[-1] = self.last_broadcast_feedback
        return result


def install_external_schedule_for_env(
    env,
    *,
    root: Path,
    distribution: str,
    role: str,
    seed: int,
    n_steps: int,
    max_feedback_delay: int = 8,
) -> ExternalTraceSchedule:
    return_schedule = build_or_load_external_schedule(
        root=Path(root),
        data_root=ROOT / "datasets",
        dataset=str(env.traces.dataset),
        role=role,
        seed=int(seed),
        n_steps=int(n_steps),
        n_trace_slots=int(env.traces.trajectories.shape[1]),
        n_users=int(env.cfg.n_users),
        n_edges=int(env.cfg.n_edges),
        max_requests=int(env.cfg.max_requests_per_user),
        max_feedback_delay=int(max_feedback_delay),
        broadcast_period=int(env.broadcast_period),
        require_real_external_data=bool(getattr(env, "require_real_external_data", False)),
    )
    env.install_canonical_schedule(return_schedule)
    return return_schedule


@contextmanager
def patched_external_protocol():
    import mappo_gp_sf_dsr.train as train_mod
    import mappo_gp_sf_dsr.paper_policy_baselines as ppo_mod
    import mappo_gp_sf_dsr.delayed_information_baselines as delayed_mod
    import mappo_gp_sf_dsr.m2ppo_paper as m2ppo_mod
    import mappo_gp_sf_dsr.recent_mec_baselines as recent_mod

    originals = []
    for mod in (train_mod, ppo_mod, m2ppo_mod):
        originals.append((mod, mod.JointMAPPOEdgeEnv, mod.install_schedule_for_env))
        mod.JointMAPPOEdgeEnv = ExternalTraceEnv
        mod.install_schedule_for_env = install_external_schedule_for_env
    try:
        yield train_mod, ppo_mod, delayed_mod, recent_mod
    finally:
        for mod, cls, installer in originals:
            mod.JointMAPPOEdgeEnv = cls
            mod.install_schedule_for_env = installer


def _parse_with(module, args: list[str]):
    old = sys.argv[:]
    try:
        sys.argv = [module.__file__ or module.__name__, *args]
        return module.parse_args()
    finally:
        sys.argv = old


def _qla_args(common: list[str], device: str) -> list[str]:
    return [
        "--method", "LCC-MAPPO",
        *common,
        "--device", device,
        "--learning-rate", "0.001",
        "--rollout-size", "1024",
        "--batch-size", "128",
        "--target-portfolio-entropy", "0.99",
        "--lcc-v", "0.5",
        "--lcc-migration-weight", "0.05",
        "--lcc-price-observation",
        "--lcc-price-scale-observation",
        "--lcc-edgewise-price-actor",
        "--lcc-price-components", "QT",
        "--lcc-price-logit-bias", "0.5",
        "--lcc-no-stale-margin",
        "--lcc-no-z-queue",
    ]


def _ppo_args(method: str, common: list[str], device: str) -> list[str]:
    file_method = FILE_METHOD[method]
    return [
        "--method", file_method,
        *common,
        "--device", device,
        "--learning-rate", "0.001",
        "--rollout-size", "1024",
        "--batch-size", "128",
        "--entropy-coef", "0.005",
        "--latency-norm", "3.0",
    ]


def _recent_args(method: str, common: list[str], device: str) -> list[str]:
    args = ["--method", FILE_METHOD[method], *common, "--device", device]
    if method == METHOD_QECO:
        args += [
            "--learning-rate", "0.001",
            "--qeco-load-mask",
            "--qeco-topk", "4",
            "--qeco-topk-temperature", "0.7",
            "--qeco-mask-quantile", "0.70",
        ]
    return args


def _delayed_args(method: str, common: list[str], device: str) -> list[str]:
    args = ["--method", method, *common, "--device", device]
    if method == METHOD_MFC:
        args += ["--mfc-learning-rate", "0.0003", "--mfc-rollout", "256"]
    return args


def _summary_exists(method: str, out_base: Path, dataset: str, users: int, distribution: str) -> bool:
    file_method = FILE_METHOD.get(method, method)
    return any(out_base.rglob(f"summary_{file_method}.csv"))


def _summary_parent(method: str, out_base: Path) -> Path:
    file_method = FILE_METHOD.get(method, method)
    matches = sorted(out_base.rglob(f"summary_{file_method}.csv"))
    if not matches:
        raise FileNotFoundError(f"No summary_{file_method}.csv under {out_base}")
    return matches[0].parent


def _mechanism_env(train_mod, common: list[str], method: str):
    parser_args = _qla_args(common, "cpu")
    parsed = _parse_with(train_mod, parser_args)
    parsed.method = "LCC-MAPPO"
    parsed.train_steps = 0
    import mappo_gp_sf_dsr.train as train_ref
    from mappo_gp_sf_dsr.config import MAPPOSystemConfig, ServerScoreConfig
    from repro_risk_averse_edge.data import load_trace_set

    score_cfg = ServerScoreConfig(
        resource_weight=parsed.score_resource_weight,
        capacity_weight=parsed.score_capacity_weight,
        task_count_weight=parsed.score_task_count_weight,
        wait_weight=parsed.score_wait_weight,
    )
    sys_cfg = MAPPOSystemConfig(
        n_users=parsed.users,
        n_edges=parsed.n_edges,
        slot_s=parsed.slot_s,
        edge_compute_mb_s=parsed.edge_compute_mb_s * parsed.edge_compute_scale,
        queue_upper_mb=parsed.queue_upper_mb,
        max_requests_per_user=parsed.max_requests_per_user,
        task_package_mb=parsed.task_package_mb,
        task_compute_mb=parsed.task_compute_mb * parsed.task_compute_scale,
        max_supported_users=max(parsed.max_supported_users, parsed.users),
        include_feedback_age=True,
        feedback_age_norm=parsed.feedback_age_norm,
    )
    traces = load_trace_set(
        parsed.dataset,
        parsed.users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        parsed.seed,
        trace_root=Path(parsed.trace_root),
    )
    capacities = train_ref.edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, parsed.capacity_mode, parsed.seed)
    env = ExternalTraceEnv(
        sys_cfg,
        traces,
        request_distribution=parsed.distribution,
        seed=parsed.seed + 10_000,
        queue_constraint=parsed.queue_constraint,
        score_config=score_cfg,
        feedback_delay=parsed.feedback_delay,
        broadcast_period=parsed.broadcast_period,
        poisson_lambda=parsed.poisson_lambda,
        burst_prob=parsed.burst_prob,
        burst_multiplier=parsed.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=parsed.background_load,
    )
    ExternalTraceEnv.require_real_external_data = True
    test_schedule = install_external_schedule_for_env(
        env,
        root=Path(parsed.canonical_schedule_root),
        distribution=parsed.distribution,
        role="test",
        seed=parsed.seed + 10_000,
        n_steps=parsed.test_steps,
        max_feedback_delay=parsed.canonical_max_feedback_delay,
    )
    return parsed, env, sys_cfg, capacities, test_schedule


def _mechanism_portfolios(method: str, env: ExternalTraceEnv, temperature: float = 0.25) -> np.ndarray:
    feedback = env.feedback_buffer.get(env.feedback_delay)
    if method == METHOD_SCORE:
        logits = np.asarray(feedback.score, dtype=np.float64) / max(float(temperature), 1e-8)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs) if np.sum(probs) > 0 else 1.0
        return np.tile(probs.astype(np.float32), (env.cfg.n_users, 1))
    if method == METHOD_JSQ:
        locs = env._locations()
        capacity = np.maximum(env.edge_capacities_mb_s.astype(np.float64), 1e-8)
        wait = env.queues_mb.astype(np.float64) / capacity
        out = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float32)
        for user in range(env.cfg.n_users):
            distances = np.linalg.norm(env.traces.edge_positions - locs[user][None, :], axis=1)
            # Fresh JSQ with communication/service tie-breaking. It does not
            # know other users' simultaneous current-slot choices.
            tran = np.asarray([env._transmission_delay_s(locs[user], edge) for edge in range(env.cfg.n_edges)])
            comp = float(env.cfg.task_compute_mb) / capacity
            edge = int(np.argmin(wait + tran + comp + 1e-6 * distances))
            out[user, edge] = 1.0
        return out
    raise ValueError(method)


def _run_mechanism(train_mod, method: str, common: list[str], cell_base: Path, dataset: str, users: int, distribution: str) -> dict[str, object]:
    from mappo_gp_sf_dsr.metrics import finalize_evaluation, slot_metrics
    from mappo_gp_sf_dsr.train import eval_diagnostics, write_csv, write_summary
    from mappo_gp_sf_dsr.canonical_protocol import write_run_manifest

    parsed, env, sys_cfg, capacities, test_schedule = _mechanism_env(train_mod, common, method)
    out_dir = cell_base / dataset / f"{users}u" / distribution
    out_dir.mkdir(parents=True, exist_ok=True)
    learning_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    results = []
    env.reset()
    for slot in range(1, parsed.test_steps + 1):
        portfolios = _mechanism_portfolios(method, env)
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios, stats=stats, decision_latency_s=0.0)
        results.append(result)
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
        if slot == 1 or slot % parsed.log_interval == 0 or slot == parsed.test_steps:
            row: dict[str, object] = {"slot": slot, "step": slot * sys_cfg.n_users}
            row.update(slot_metrics(result))
            learning_rows.append(row)
    summary, _ = finalize_evaluation(env, results, sys_cfg)
    summary.update(eval_diagnostics(results, capacities.astype(np.float64), sys_cfg.slot_s))
    summary.update(
        {
            "method": method,
            "dataset": dataset,
            "users": users,
            "distribution": distribution,
            "train_steps": 0,
            "test_steps": parsed.test_steps,
            "feedback_delay": parsed.feedback_delay,
            "broadcast_period": parsed.broadcast_period,
            "uses_fresh_queue": bool(method == METHOD_JSQ),
            "mechanism_baseline": True,
            "test_split_id": test_schedule.schedule_id,
            "test_schedule_sha256": test_schedule.content_sha256,
        }
    )
    write_csv(out_dir / f"learning_{method}.csv", learning_rows)
    write_csv(out_dir / f"latency_samples_{method}.csv", latency_rows)
    write_summary(out_dir / f"summary_{method}.csv", summary)
    config_path = out_dir / f"config_{method}.json"
    config_path.write_text(json.dumps({**vars(parsed), "method": method}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_manifest(output_dir=out_dir, method=method, args=parsed, summary=summary, config_path=config_path)
    return summary


def run_grid(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    schedule_root = Path(args.external_schedule_root) if args.external_schedule_root else output_root / "external_schedules"
    datasets = args.datasets
    methods = args.methods
    jobs: list[tuple[str, str, int, int]] = [
        (method, dataset, seed, delay)
        for dataset in datasets
        for method in methods
        for seed in args.seeds
        for delay in args.delays
    ]
    with patched_external_protocol() as (train_mod, ppo_mod, delayed_mod, recent_mod):
        for index, (method, dataset, seed, delay) in enumerate(jobs, start=1):
            cell_base = output_root / "cells" / method / dataset / f"seed_{seed}" / f"delay_{delay}"
            if _summary_exists(method, cell_base, dataset, args.users, "poisson_bursty") and not args.force:
                print(f"[skip {index}/{len(jobs)}] {method} {dataset} seed={seed} delay={delay}", flush=True)
                continue
            common = [
                "--dataset", dataset,
                "--users", str(args.users),
                "--distribution", "poisson_bursty",
                "--seed", str(seed),
                "--feedback-delay", str(delay),
                "--broadcast-period", str(args.broadcast_period),
                "--train-steps", str(args.train_steps),
                "--test-steps", str(args.test_steps),
                "--evaluation-split-role", args.evaluation_split_role,
                "--output-dir", str(cell_base),
                "--canonical-schedule-root", str(schedule_root),
                "--canonical-max-feedback-delay", str(args.canonical_max_feedback_delay),
                "--trace-root", str(ROOT / "Traces_TimeXY_10sec_txt"),
                "--slot-s", str(args.slot_s),
                "--n-edges", str(args.n_edges),
                "--max-requests-per-user", str(args.max_requests_per_user),
                "--edge-compute-mb-s", "63.0",
                "--queue-upper-mb", "630.0",
                "--task-compute-mb", "15.0",
                "--task-package-mb", str(270.0 / 1024.0),
                "--poisson-lambda", "3.5",
                "--burst-prob", "0.1",
                "--burst-multiplier", "2.0",
            ]
            if args.load_checkpoint:
                common += ["--load-checkpoint", args.load_checkpoint]
            ExternalTraceEnv.require_real_external_data = bool(args.require_real_external_data)
            print(f"[run {index}/{len(jobs)}] {method} {dataset} seed={seed} delay={delay}", flush=True)
            if method == METHOD_Q:
                raise RuntimeError("Formal QLA trace cells must use paired_training_repair_worker_trace.py")
            elif method in {METHOD_PPO, METHOD_DSR}:
                parsed = _parse_with(ppo_mod, _ppo_args(method, common, args.device))
                summary = (
                    ppo_mod.evaluate_towards_ppo_checkpoint(parsed)
                    if parsed.load_checkpoint else ppo_mod.train_towards_ppo(parsed)
                )
            elif method in {METHOD_QECO, METHOD_LYFOTO}:
                parsed = _parse_with(recent_mod, _recent_args(method, common, args.device))
                summary = recent_mod.train_once(parsed)
            elif method in {METHOD_SCORE, METHOD_JSQ}:
                summary = _run_mechanism(train_mod, method, common, cell_base, dataset, args.users, "poisson_bursty")
            else:
                parsed = _parse_with(delayed_mod, _delayed_args(method, common, args.device))
                summary = delayed_mod.train_once(parsed)
            summary_path = _summary_parent(method, cell_base) / "external_protocol_manifest.json"
            schedule_manifest = {
                "external_trace_protocol": "real_trace_external_v1",
                "method_alias": method,
                "dataset": dataset,
                "seed": seed,
                "feedback_delay": delay,
                "train_steps": args.train_steps,
                "test_steps": args.test_steps,
                "summary": {k: summary.get(k) for k in ("average_latency", "cvar99", "queue_oscillation", "portfolio_migration")},
                "notes": "Uses local Azure/Alibaba CSVs when present; otherwise deterministic trace-shaped proxy schedules are recorded in external_schedules manifests.",
            }
            summary_path.write_text(json.dumps(schedule_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in (root / "cells").rglob("summary_*.csv"):
        parts = path.parts
        try:
            method = parts[parts.index("cells") + 1]
            dataset = parts[parts.index("cells") + 2]
            seed = int(parts[parts.index("cells") + 3].split("_")[-1])
            delay = int(parts[parts.index("cells") + 4].split("_")[-1])
        except Exception:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row.update({"method_alias": method, "mobility_dataset": dataset, "seed": seed, "feedback_delay": delay})
                rows.append(row)
    return rows


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    raw = collect_rows(root)
    write_table(root / "analysis" / "real_trace_external_raw.csv", raw)
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw:
        groups.setdefault((str(row["method_alias"]), int(row["feedback_delay"])), []).append(row)
    summary_rows = []
    for (method, delay), vals in sorted(groups.items()):
        out = {"method": method, "feedback_delay": delay, "n": len(vals)}
        for metric in ("average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate", "p_drop", "p_ddl"):
            arr = np.asarray([float(v.get(metric, "nan")) for v in vals], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else float("nan")
            out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        summary_rows.append(out)
    write_table(root / "analysis" / "real_trace_external_summary_by_delay.csv", summary_rows)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(summary_rows)
        fig_dir = root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel in [("average_latency", "Avg latency (s)"), ("cvar99", "CVaR99 (s)")]:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for method in METHODS:
                sub = df[df["method"] == method].sort_values("feedback_delay")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["feedback_delay"],
                    sub[f"{metric}_mean"],
                    yerr=sub[f"{metric}_std"],
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=method,
                )
            ax.set_xlabel("Feedback delay")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / f"real_trace_external_{metric}_by_delay.png", dpi=220)
            fig.savefig(fig_dir / f"real_trace_external_{metric}_by_delay.pdf")
            plt.close(fig)
    except Exception as exc:
        print(f"[warn] plotting skipped: {exc}", flush=True)


def status(args: argparse.Namespace) -> None:
    expected = len(args.datasets) * len(args.methods) * len(args.seeds) * len(args.delays)
    rows = collect_rows(Path(args.output_dir))
    print(json.dumps({"completed": len(rows), "expected": expected}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace-driven Azure/Alibaba interface experiment for QLA and close baselines.")
    parser.add_argument("--output-dir", default=str(ROOT / "results_real_trace_external_qla_baselines"))
    parser.add_argument("--datasets", nargs="+", default=["KAIST", "NCSU", "NewYork"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 1, 2, 4])
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--canonical-max-feedback-delay", type=int, default=8)
    parser.add_argument("--external-schedule-root", default="")
    parser.add_argument("--evaluation-split-role", choices=["validation", "test"], default="test")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-real-external-data", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status:
        status(args)
        return
    if not args.summarize_only:
        run_grid(args)
    if not args.no_summarize:
        summarize(args)


if __name__ == "__main__":
    main()
