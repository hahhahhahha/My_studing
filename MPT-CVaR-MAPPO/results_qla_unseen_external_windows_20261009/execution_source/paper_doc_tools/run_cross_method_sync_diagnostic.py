from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro_risk_averse_edge.data import load_trace_set

from mappo_gp_sf_dsr.canonical_protocol import install_schedule_for_env
from mappo_gp_sf_dsr.config import MAPPOConfig, MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from mappo_gp_sf_dsr.delayed_information_baselines import JSEQPolicy, MFCDIPolicy
from mappo_gp_sf_dsr.env import JointMAPPOEdgeEnv
from mappo_gp_sf_dsr.metrics import finalize_evaluation, phase_diagnostics
from mappo_gp_sf_dsr.models import MAPPOPolicy
from mappo_gp_sf_dsr.paper_policy_baselines import SharedGaussianPPO
from mappo_gp_sf_dsr.recent_mec_baselines import (
    QECONetwork,
    _lyapunov_local_stale_portfolios,
    _qeco_action_portfolios,
    _qeco_agents,
    _qeco_state,
)
from mappo_gp_sf_dsr.rewards import MAPPORewarder
from mappo_gp_sf_dsr.train import (
    augment_lcc_price_actor_observations,
    edge_capacities,
    lcc_marginal_risk_prices,
    normalize_portfolios,
)


FORMAL = ROOT / "results_qla_canonical_v1"
SCHEDULES = FORMAL / "shared_schedules"
METHODS = ("QLA-MAPPO", "JSEQ-A", "MFC-DI-A", "PPO-MV-A", "QL-Bias-Shuffle")
REPAIRED_CHECKPOINTS: dict[tuple[str, int, int], tuple[Path, str]] = {}


def load_repaired_checkpoints() -> None:
    sources = (
        (ROOT / "results_qla_formal_bothfix_20260930" / "FROZEN_BEFORE_TEST.json",
         {"Full": "QLA-MAPPO", "Shuffle": "QL-Bias-Shuffle"}, "variant"),
        (ROOT / "results_qla_formal_baselines_bothfix_20261001" / "FROZEN_BEFORE_TEST.json",
         {"Towards-PPO-MV": "PPO-MV-A", "MFC-DI-A": "MFC-DI-A"}, "method"),
    )
    for registry_path, labels, field in sources:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        for entry in registry["checkpoints"]:
            label = labels.get(entry[field])
            if label:
                key = (label, int(entry["seed"]), int(entry["age"]))
                REPAIRED_CHECKPOINTS[key] = (Path(entry["checkpoint"]), entry["checkpoint_sha256"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cli_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cli", data)


def locate(method: str, seed: int, delay: int) -> tuple[Path | None, Path]:
    key = (method, seed, delay)
    if REPAIRED_CHECKPOINTS and method != "JSEQ-A":
        checkpoint, expected_hash = REPAIRED_CHECKPOINTS[key]
        if not checkpoint.is_file() or sha256(checkpoint) != expected_hash:
            raise AssertionError(f"Frozen repaired checkpoint lock failed: {key}")
        config_files = list(checkpoint.parent.glob("config_*.json"))
        if len(config_files) != 1:
            raise AssertionError(f"Expected one config beside {checkpoint}, found {len(config_files)}")
        return checkpoint, config_files[0]
    if method == "Uniform" or method == "QL-Only" or method.startswith("Distance@"):
        root = FORMAL / "qla" / "main" / "QLA-MAPPO-NoMargin" / f"seed_{seed}" / f"delay_{delay}"
        return None, next(root.rglob("config_LCC-MAPPO.json"))
    if method == "QLA-MAPPO":
        root = FORMAL / "qla" / "main" / "QLA-MAPPO-NoMargin" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_LCC-MAPPO.pt"))
        cfg = next(root.rglob("config_LCC-MAPPO.json"))
        return ckpt, cfg
    if method == "QL-NoQL":
        root = ROOT / "results_qla_signal_components_v1" / "cells" / "QLA-No-QL" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_LCC-MAPPO.pt"))
        cfg = next(root.rglob("config_LCC-MAPPO.json"))
        return ckpt, cfg
    if method == "QL-Bias-Shuffle":
        root = FORMAL / "qla" / "ablation" / "QL-Bias-Shuffle-NoMargin" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_LCC-MAPPO.pt"))
        cfg = next(root.rglob("config_LCC-MAPPO.json"))
        return ckpt, cfg
    if method == "PPO-MV-A":
        root = FORMAL / "baselines" / "Towards-PPO-MV" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_Towards-PPO-MV.pt"))
        cfg = next(root.rglob("config_Towards-PPO-MV.json"))
        return ckpt, cfg
    if method == "PPO-DSR-A":
        root = FORMAL / "baselines" / "Towards-PPO-DSR" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_Towards-PPO-DSR.pt"))
        cfg = next(root.rglob("config_Towards-PPO-DSR.json"))
        return ckpt, cfg
    if method == "QECO-A":
        root = FORMAL / "baselines" / "QECO-load-topk" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_QECO-load-topk.pt"))
        cfg = next(root.rglob("config_QECO-load-topk.json"))
        return ckpt, cfg
    if method == "LyFOTO-LC":
        root = FORMAL / "baselines" / "Lyapunov-Q-local-stale" / f"seed_{seed}" / f"delay_{delay}"
        return None, next(root.rglob("config_Lyapunov-Q-local-stale.json"))
    if method == "MFC-DI-A":
        root = FORMAL / "baselines" / "MFC-DI-A" / f"seed_{seed}" / f"delay_{delay}"
        ckpt = next(root.rglob("checkpoint_MFC-DI-A.pt"))
        cfg = next(root.rglob("config_MFC-DI-A.json"))
        return ckpt, cfg
    if method == "JSEQ-A":
        root = FORMAL / "baselines" / "JSEQ-A" / f"seed_{seed}" / f"delay_{delay}"
        return None, next(root.rglob("config_JSEQ-A.json"))
    raise KeyError(method)


def build_env(seed: int, delay: int, test_steps: int, schedule_role: str = "test") -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig]:
    n_edges = 35
    sys_cfg = MAPPOSystemConfig(
        n_users=30, n_edges=n_edges, slot_s=1.0, edge_compute_mb_s=63.0,
        queue_upper_mb=630.0, max_requests_per_user=7,
        task_package_mb=270.0 / 1024.0, task_compute_mb=15.0,
        max_supported_users=50, include_feedback_age=True,
        feedback_age_norm=8.0, extra_actor_obs_dim=n_edges + 1,
    )
    capacities = edge_capacities(n_edges, sys_cfg.edge_compute_mb_s, "homogeneous", seed)
    traces = load_trace_set("KAIST", 30, n_edges, 1.0, seed, trace_root=TRACE_ROOT)
    env = JointMAPPOEdgeEnv(
        sys_cfg, traces, request_distribution="poisson_bursty", seed=seed + 10_000,
        queue_constraint="resample", score_config=ServerScoreConfig(),
        feedback_delay=delay, broadcast_period=2, poisson_lambda=3.5,
        burst_prob=0.1, burst_multiplier=2.0,
        edge_capacities_mb_s=capacities, background_load=0.0,
    )
    install_schedule_for_env(
        env, root=SCHEDULES, distribution="poisson_bursty", role=schedule_role,
        seed=seed + 10_000, n_steps=test_steps, max_feedback_delay=8,
    )
    return env, sys_cfg


def qla_config(args: argparse.Namespace) -> MAPPOConfig:
    return MAPPOConfig(
        rollout_size=args.rollout_size, batch_size=args.batch_size,
        learning_rate=args.learning_rate, auto_portfolio_entropy=args.auto_portfolio_entropy,
        target_portfolio_entropy=args.target_portfolio_entropy,
        log_std_min=args.log_std_min, edgewise_price_actor=True,
        lcc_enabled=True, lcc_v=args.lcc_v,
        lcc_migration_weight=args.lcc_migration_weight,
        lcc_cvar_alpha=args.lcc_cvar_alpha, lcc_cvar_budget=args.lcc_cvar_budget,
        lcc_var_lr=args.lcc_var_lr, lcc_concentration_factor=args.lcc_concentration_factor,
        lcc_use_z_queue=not args.lcc_no_z_queue,
        lcc_use_h_queue=not args.lcc_no_h_queue,
        lcc_use_stale_margin=not args.lcc_no_stale_margin,
        lcc_price_observation=True, lcc_price_scale_observation=True,
        lcc_edgewise_price_actor=True, lcc_price_components=args.lcc_price_components,
        lcc_price_shuffle=args.lcc_price_shuffle,
        lcc_price_logit_bias=args.lcc_price_logit_bias,
        latency_norm=args.latency_norm,
    )


class Adapter:
    def act(self, env: JointMAPPOEdgeEnv) -> np.ndarray:
        raise NotImplementedError

    def observe(self, result) -> None:
        pass


class UniformAdapter(Adapter):
    def act(self, env):
        return np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)


class DistanceRandomAdapter(Adapter):
    """Queue-free randomized routing from current local distance and static service time."""

    def __init__(self, beta: float):
        self.beta = float(beta)

    def act(self, env):
        locations = env._locations()
        service = env.cfg.task_compute_mb / np.maximum(env.edge_capacities_mb_s, 1e-8)
        costs = np.empty((env.cfg.n_users, env.cfg.n_edges), dtype=np.float64)
        for user in range(env.cfg.n_users):
            costs[user] = np.asarray(
                [env._transmission_delay_s(locations[user], edge) for edge in range(env.cfg.n_edges)]
            ) + service
        logits = -self.beta * costs / 3.0
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        return normalize_portfolios(weights)


class QLOnlyAdapter(Adapter):
    """Training-free softmax policy using the frozen joint QL signal only."""

    def __init__(self, cfg_path, sys_cfg):
        cfg = cli_config(cfg_path)
        cfg["device"] = "cpu"
        self.args = argparse.Namespace(**cfg)
        self.cfg = qla_config(self.args)
        self.rewarder = MAPPORewarder(self.cfg, method="LCC-MAPPO")
        self.sys_cfg = sys_cfg
        self.previous = None
        self.state: dict[str, np.ndarray] = {}

    def act(self, env):
        prices, _ = lcc_marginal_risk_prices(
            env, self.args, self.rewarder,
            previous_portfolios=self.previous, state=self.state,
        )
        logits = -0.5 * np.asarray(prices, dtype=np.float64)
        logits -= np.max(logits, axis=1, keepdims=True)
        return normalize_portfolios(np.exp(logits))

    def observe(self, result):
        self.rewarder.compute(
            result.per_user_mean_latency, result.per_user_tasks,
            result.system_average_latency, result.portfolio_stats,
            portfolios=result.portfolios, queues_before_mb=result.queue_load_before_mb,
            sys_cfg=self.sys_cfg,
            previous_portfolios=self.previous,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )
        self.previous = np.asarray(result.portfolios, dtype=np.float32)


class QLAAdapter(Adapter):
    def __init__(self, env, sys_cfg, seed, cfg_path, checkpoint):
        cfg_dict = cli_config(cfg_path)
        cfg_dict["device"] = "cpu"
        self.args = argparse.Namespace(**cfg_dict)
        self.cfg = qla_config(self.args)
        self.policy = MAPPOPolicy(sys_cfg, self.cfg, seed=seed, device="cpu")
        self.policy.load_checkpoint(checkpoint)
        self.sys_cfg = sys_cfg
        self.rewarder = MAPPORewarder(self.cfg, method="LCC-MAPPO")
        self.previous = None
        self.state: dict[str, np.ndarray] = {}

    def act(self, env):
        obs = augment_lcc_price_actor_observations(
            env.actor_observations(), env, self.args, self.rewarder,
            previous_portfolios=self.previous, state=self.state,
        )
        portfolios, _, _ = self.policy.act(obs, deterministic=True)
        return normalize_portfolios(portfolios)

    def observe(self, result):
        self.rewarder.compute(
            result.per_user_mean_latency, result.per_user_tasks,
            result.system_average_latency, result.portfolio_stats,
            portfolios=result.portfolios, queues_before_mb=result.queue_load_before_mb,
            sys_cfg=self.sys_cfg,
            previous_portfolios=self.previous,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )
        self.previous = np.asarray(result.portfolios, dtype=np.float32)


class PPOAdapter(Adapter):
    def __init__(self, env, cfg_path, checkpoint):
        cfg = cli_config(cfg_path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        obs_dim = int(state["actor.0.weight"].shape[1])
        self.obs_dim = obs_dim
        self.policy = SharedGaussianPPO(obs_dim, env.cfg.n_edges, log_std_min=cfg.get("log_std_min", -1.5))
        self.policy.load_state_dict(state)
        self.policy.eval()

    def act(self, env):
        obs = env.actor_observations()
        return self.policy.act(obs[:, : self.obs_dim], deterministic=True)[3]


class MFCAdapter(Adapter):
    def __init__(self, env, cfg_path, checkpoint):
        cfg = cli_config(cfg_path)
        cfg["device"] = "cpu"
        cfg["seed"] = int(cfg["seed"]) + 10_000
        self.policy = MFCDIPolicy(env, argparse.Namespace(**cfg))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.policy.actor.load_state_dict(state["actor"])
        self.policy.critic.load_state_dict(state["critic"])
        self.policy.actor.eval()
        self.policy.critic.eval()

    def act(self, env):
        return self.policy.act(deterministic=True)[0]


class JSEQAdapter(Adapter):
    def __init__(self, env, cfg_path):
        cfg = cli_config(cfg_path)
        self.policy = JSEQPolicy(env, cfg.get("jseq_temperature", 0.25))

    def act(self, env):
        return self.policy.portfolios()

    def observe(self, result):
        self.policy.observe(result)


class QECOAdapter(Adapter):
    """Frozen QECO-load-topk policy under the diagnostic feedback wrapper."""

    def __init__(self, env, cfg_path, checkpoint):
        cfg = cli_config(cfg_path)
        cfg["device"] = "cpu"
        self.args = argparse.Namespace(**cfg)
        self.env = env
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(int(cfg["seed"]) + 31_337)
        checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        models = checkpoint_state["online_models"]
        first_fc = models[0]["fc1.weight"]
        self.state_dim = int(first_fc.shape[1]) - int(self.args.qeco_lstm_hidden)
        states, workload = _qeco_state(env)
        self.states = states[:, : self.state_dim]
        self.agents = _qeco_agents(
            self.args, self.state_dim, workload.size, self.device
        )
        if len(models) != len(self.agents):
            raise ValueError(
                f"QECO checkpoint has {len(models)} agents; expected {len(self.agents)}"
            )
        for agent, model_state in zip(self.agents, models):
            agent.online.load_state_dict(model_state)
            agent.online.eval()
        self.histories = np.repeat(
            workload[None, None, :],
            self.args.users * self.args.history_steps,
            axis=0,
        ).reshape(self.args.users, self.args.history_steps, workload.size)

    def act(self, env):
        _, portfolios = _qeco_action_portfolios(
            self.agents,
            self.states,
            self.histories,
            env,
            1.0,
            self.rng,
            self.device,
            deterministic=True,
            topk=self.args.qeco_topk,
            temperature=self.args.qeco_topk_temperature,
            load_mask=self.args.qeco_load_mask,
            mask_quantile=self.args.qeco_mask_quantile,
        )
        return portfolios

    def observe(self, result):
        next_states, next_workload = _qeco_state(self.env)
        self.histories = np.concatenate(
            [
                self.histories[:, 1:],
                np.repeat(next_workload[None, None, :], self.args.users, axis=0),
            ],
            axis=1,
        )
        self.states = next_states[:, : self.state_dim]


class LyFOTOAdapter(Adapter):
    """Training-free local-candidate stale Lyapunov adapter."""

    def __init__(self, cfg_path):
        self.cfg = cli_config(cfg_path)

    def act(self, env):
        return _lyapunov_local_stale_portfolios(
            env,
            tradeoff_v=float(self.cfg["lyapunov_local_v"]),
            temperature=float(self.cfg["lyapunov_local_temperature"]),
            candidate_edges=int(self.cfg["lyapunov_local_candidates"]),
        )


def make_adapter(method, env, sys_cfg, seed, cfg_path, checkpoint):
    if method == "Uniform":
        return UniformAdapter()
    if method.startswith("Distance@"):
        return DistanceRandomAdapter(float(method.split("@", 1)[1]))
    if method == "QL-Only":
        return QLOnlyAdapter(cfg_path, sys_cfg)
    if method in {"QLA-MAPPO", "QL-NoQL", "QL-Bias-Shuffle"}:
        return QLAAdapter(env, sys_cfg, seed, cfg_path, checkpoint)
    if method in {"PPO-MV-A", "PPO-DSR-A"}:
        return PPOAdapter(env, cfg_path, checkpoint)
    if method == "QECO-A":
        return QECOAdapter(env, cfg_path, checkpoint)
    if method == "LyFOTO-LC":
        return LyFOTOAdapter(cfg_path)
    if method == "MFC-DI-A":
        return MFCAdapter(env, cfg_path, checkpoint)
    if method == "JSEQ-A":
        return JSEQAdapter(env, cfg_path)
    raise KeyError(method)


def pairwise_cosine(change: np.ndarray, mask: np.ndarray) -> float:
    x = np.asarray(change, dtype=np.float64)[mask]
    if x.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(x, axis=1)
    x = x[norms > 1e-12]
    norms = norms[norms > 1e-12]
    if x.shape[0] < 2:
        return float("nan")
    x = x / norms[:, None]
    ix = np.triu_indices(x.shape[0], 1)
    return float(np.mean((x @ x.T)[ix]))


def run_cell(method: str, seed: int, delay: int, test_steps: int, schedule_role: str = "test"):
    checkpoint, cfg_path = locate(method, seed, delay)
    env, sys_cfg = build_env(seed, delay, test_steps, schedule_role=schedule_role)
    env.reset()
    adapter = make_adapter(method, env, sys_cfg, seed, cfg_path, checkpoint)
    execution_audit: list[dict[str, float]] = []
    original_sample_edge = env._sample_edge

    def audited_sample_edge(weights, slot_load, *, user=None, task_idx=None):
        raw = np.asarray(weights, dtype=np.float64)
        raw = np.where(np.isfinite(raw), np.maximum(raw, 0.0), 0.0)
        raw = raw / raw.sum() if raw.sum() > 0 else np.full(env.cfg.n_edges, 1.0 / env.cfg.n_edges)
        if env.cfg.finite_buffer_protocol:
            service = env.edge_capacities_mb_s * env.cfg.slot_s
            residual = np.maximum(0.0, env.queues_mb - service)
            feasible = residual + slot_load + env.cfg.task_compute_mb <= env.cfg.queue_upper_mb + 1e-8
        else:
            feasible = env.queues_mb + slot_load + env.cfg.task_compute_mb <= env.cfg.queue_upper_mb
        effective = np.where(feasible, raw, 0.0)
        if effective.sum() <= 0:
            effective = raw if env.queue_constraint == "penalty" else feasible.astype(np.float64)
        if effective.sum() <= 0:
            effective = np.ones(env.cfg.n_edges, dtype=np.float64)
        effective /= effective.sum()
        result = original_sample_edge(weights, slot_load, user=user, task_idx=task_idx)
        changed = float("nan")
        if (
            env.canonical_schedule is not None and user is not None and task_idx is not None
            and env.schedule_cursor < env.canonical_schedule.n_steps
        ):
            u = float(env.canonical_schedule.assignment_uniforms[env.schedule_cursor, user, task_idx])
            raw_edge = int(np.searchsorted(np.cumsum(raw), min(u, np.nextafter(1.0, 0.0)), side="right"))
            effective_edge = int(np.searchsorted(np.cumsum(effective), min(u, np.nextafter(1.0, 0.0)), side="right"))
            if result[0] != effective_edge:
                raise AssertionError("Execution audit changed or mis-reconstructed the canonical assignment")
            changed = float(raw_edge != effective_edge)
        execution_audit.append({
            "masked_probability_mass": float(np.sum(raw[~feasible])),
            "any_infeasible_edge": float(np.any(~feasible)),
            "raw_assignment_changed": changed,
            "all_edges_infeasible": float(not np.any(feasible)),
        })
        return result

    env._sample_edge = audited_sample_edge
    previous = None
    previous_source = None
    results, slots, server_rows, receipt_slots = [], [], [], []
    capacities = np.asarray(env.edge_capacities_mb_s, dtype=np.float64)
    schedule_hash = env.canonical_schedule.content_sha256
    for slot in range(test_steps):
        task_audit_start = len(execution_audit)
        feedback = env.feedback_buffer.get(env.feedback_delay)
        source_slot = int(feedback.source_slot)
        action_source_age = int(env.slot_index) - source_slot
        if previous_source is None or source_slot != previous_source:
            receipt_slots.append(slot)
        previous_source = source_slot
        active = np.asarray(env.requests) > 0
        portfolios = normalize_portfolios(adapter.act(env))
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios, stats=stats, decision_latency_s=0.0)
        adapter.observe(result)
        task_audit = execution_audit[task_audit_start:]
        masked_mass = [x["masked_probability_mass"] for x in task_audit]
        changed_assignment = [x["raw_assignment_changed"] for x in task_audit if np.isfinite(x["raw_assignment_changed"])]
        results.append(result)
        realized_total = int(np.sum(result.server_assignment_counts))
        realized_share = result.server_assignment_counts / max(realized_total, 1)
        expected_share = np.asarray(stats.demand_share, dtype=np.float64)
        offered_pressure = (
            np.asarray(result.queue_load_before_mb) + np.asarray(result.task_offered_load_mb)
        ) / np.maximum(capacities * sys_cfg.slot_s, 1e-8)
        queue_jump = np.maximum(
            np.asarray(result.queue_load_after_mb) - np.asarray(result.queue_load_before_mb), 0.0
        ) / np.maximum(capacities * sys_cfg.slot_s, 1e-8)
        if previous is None:
            sync_active = cos_active = float("nan")
        else:
            delta = portfolios - previous
            if int(np.sum(active)):
                targets = np.argmax(delta[active], axis=1)
                sync_active = float(np.max(np.bincount(targets, minlength=sys_cfg.n_edges)) / len(targets))
            else:
                sync_active = float("nan")
            cos_active = pairwise_cosine(delta, active)
        slots.append({
            "method": method, "seed": seed, "feedback_age": delay, "slot": slot + 1,
            "source_slot": source_slot, "source_age": action_source_age,
            "active_users": int(np.sum(active)), "synchronization_rate": sync_active,
            "portfolio_change_cosine": cos_active,
            "max_expected_server_share": float(np.max(expected_share)),
            "max_realized_server_share": float(np.max(realized_share)),
            "overloaded_server_fraction": float(np.mean(offered_pressure > 1.0)),
            "any_overloaded_server": float(np.any(offered_pressure > 1.0)),
            "queue_spike": float(np.max(queue_jump) > 1.0),
            "max_queue_jump_service_slots": float(np.max(queue_jump)),
            "mean_masked_probability_mass": float(np.mean(masked_mass)) if masked_mass else 0.0,
            "mask_active_task_fraction": float(np.mean([x["any_infeasible_edge"] for x in task_audit])) if task_audit else 0.0,
            "raw_assignment_changed_fraction": float(np.mean(changed_assignment)) if changed_assignment else 0.0,
            "all_edges_infeasible_task_fraction": float(np.mean([x["all_edges_infeasible"] for x in task_audit])) if task_audit else 0.0,
            "audited_tasks": len(task_audit),
            "lcc_h_queue": float(adapter.rewarder.h_queue) if hasattr(adapter, "rewarder") else float("nan"),
        })
        for edge in range(sys_cfg.n_edges):
            server_rows.append({
                "method": method, "seed": seed, "feedback_age": delay, "slot": slot + 1,
                "edge": edge + 1, "expected_share": float(expected_share[edge]),
                "realized_share": float(realized_share[edge]),
                "offered_pressure": float(offered_pressure[edge]),
                "queue_before_mb": float(result.queue_load_before_mb[edge]),
                "queue_after_mb": float(result.queue_load_after_mb[edge]),
            })
        previous = portfolios.copy()
    summary, drain = finalize_evaluation(env, results, sys_cfg)
    summary.update(phase_diagnostics(results, sys_cfg))
    slot_df = pd.DataFrame(slots)
    window_events = []
    for event in receipt_slots:
        window = slot_df.iloc[event : min(event + 3, len(slot_df))]
        if len(window):
            window_events.append(float(window["queue_spike"].max()))
    cell = {
        "method": method, "seed": seed, "feedback_age": delay,
        "schedule_role": schedule_role,
        "average_latency": summary["average_latency"], "cvar99": summary["cvar99"],
        "queue_oscillation": summary["queue_oscillation"],
        "portfolio_migration_rate": summary["portfolio_migration_rate"],
        "per_user_latency_jain": summary["per_user_latency_jain"],
        "p90_queue_lifetime": summary["p90_queue_lifetime"],
        "synchronization_rate_mean": slot_df["synchronization_rate"].mean(),
        "synchronization_rate_p95": slot_df["synchronization_rate"].quantile(0.95),
        "portfolio_change_cosine_mean": slot_df["portfolio_change_cosine"].mean(),
        "max_expected_share_p95": slot_df["max_expected_server_share"].quantile(0.95),
        "max_expected_share_p99": slot_df["max_expected_server_share"].quantile(0.99),
        "max_realized_share_p95": slot_df["max_realized_server_share"].quantile(0.95),
        "max_realized_share_p99": slot_df["max_realized_server_share"].quantile(0.99),
        "broadcast_window_queue_spike_rate": float(np.mean(window_events)) if window_events else float("nan"),
        "overloaded_server_probability": slot_df["overloaded_server_fraction"].mean(),
        "any_overloaded_slot_probability": slot_df["any_overloaded_server"].mean(),
        "mean_masked_probability_mass": float(np.mean([x["masked_probability_mass"] for x in execution_audit])) if execution_audit else 0.0,
        "mask_active_task_fraction": float(np.mean([x["any_infeasible_edge"] for x in execution_audit])) if execution_audit else 0.0,
        "raw_assignment_changed_fraction": float(np.mean([x["raw_assignment_changed"] for x in execution_audit if np.isfinite(x["raw_assignment_changed"])])) if execution_audit else 0.0,
        "all_edges_infeasible_task_fraction": float(np.mean([x["all_edges_infeasible"] for x in execution_audit])) if execution_audit else 0.0,
        "lcc_h_queue_max": slot_df["lcc_h_queue"].max(),
        "lcc_h_queue_active_slot_fraction": float((slot_df["lcc_h_queue"] > 0).mean()) if slot_df["lcc_h_queue"].notna().any() else float("nan"),
        "mass_conservation_error": max(float(r.mass_conservation_error) for r in results),
        "drain_finished": bool(drain.finished), "test_schedule_sha256": schedule_hash,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else "analytic-no-checkpoint",
        "checkpoint": str(checkpoint) if checkpoint else "analytic-no-checkpoint",
    }
    return cell, slot_df, pd.DataFrame(server_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "results_qla_cross_method_sync_v1")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--delays", default="4,8")
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--schedule-role", default="test", choices=["validation", "test"])
    p.add_argument("--use-repaired", action="store_true", help="use frozen R1/R2 checkpoints")
    args = p.parse_args()
    if args.use_repaired:
        load_repaired_checkpoints()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    delays = [int(x) for x in args.delays.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    design = {"methods": methods, "seeds": seeds, "feedback_ages": delays,
              "test_steps": args.test_steps, "schedule_role": args.schedule_role,
              "use_repaired": args.use_repaired,
              "random_replicate": "training seed",
              "experimental_condition": "feedback age",
              "queue_spike_definition": "max positive queue jump > one service-slot capacity",
              "broadcast_window": "first three decision slots after receipt of a new selected summary"}
    (out / "DESIGN.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    cells, slot_parts, server_parts = [], [], []
    total = len(methods) * len(seeds) * len(delays)
    index = 0
    for method in methods:
        for seed in seeds:
            for delay in delays:
                index += 1
                print(f"[run] {index}/{total} {method} seed={seed} age={delay}", flush=True)
                cell, slot_df, server_df = run_cell(
                    method, seed, delay, args.test_steps, schedule_role=args.schedule_role
                )
                cells.append(cell); slot_parts.append(slot_df); server_parts.append(server_df)
                pd.DataFrame(cells).to_csv(out / "cell_summary.csv", index=False)
    pd.concat(slot_parts, ignore_index=True).to_csv(out / "slot_sync_metrics.csv", index=False)
    pd.concat(server_parts, ignore_index=True).to_csv(out / "slot_server_metrics.csv", index=False)
    print(f"Completed {len(cells)}/{total} fixed-checkpoint evaluations -> {out}")


if __name__ == "__main__":
    main()
