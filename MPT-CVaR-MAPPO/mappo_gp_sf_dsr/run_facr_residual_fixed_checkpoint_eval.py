from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results
from .models import MAPPOPolicy
from .train import (
    apply_actor_amplitude_residual_prior,
    apply_lyapunov_prior,
    apply_proposal_residual_prior,
    augment_residual_v3_actor_observations,
    edge_capacities,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_facr_residual_fixed_checkpoint_eval"


@dataclass(frozen=True)
class FixedVariant:
    name: str
    method: str
    summary_name: str
    checkpoint: Path | None = None
    train_args: dict[str, object] = field(default_factory=dict)


COMMON_TRAIN_ARGS: dict[str, object] = {
    "dataset": "KAIST",
    "users": 30,
    "distribution": "poisson_bursty",
    "train-steps": 10_000,
    "test-steps": 1_000,
    "broadcast-period": 2,
    "slot-s": 1.0,
    "n-edges": 35,
    "edge-compute-mb-s": 63.0,
    "queue-upper-mb": 630.0,
    "task-compute-mb": 15.0,
    "task-package-mb": 270.0 / 1024.0,
    "max-requests-per-user": 7,
    "poisson-lambda": 3.5,
    "burst-prob": 0.1,
    "burst-multiplier": 2.0,
    "background-load": 0.0,
    "capacity-mode": "homogeneous",
    "edge-compute-scale": 1.0,
    "task-compute-scale": 1.0,
    "score-resource-weight": 1.0,
    "score-capacity-weight": 0.5,
    "score-task-count-weight": 0.5,
    "score-wait-weight": 1.0,
    "max-supported-users": 50,
    "alpha": 0.20,
    "q-penalty-weight": 0.08,
    "kappa-i": 0.02,
    "kappa-g": 0.02,
    "target-portfolio-entropy": 0.99,
    "feedback-gate-eta": 0.25,
    "lyapunov-prior-v": 1.0,
    "lyapunov-prior-temperature": 2.5,
    "lyapunov-prior-max-step": 0.03,
    "seed": 1,
    "feedback-delay": 0,
    "mixed-feedback-delays": "0,4,8",
}


VARIANTS: dict[str, FixedVariant] = {
    "FACR": FixedVariant(
        name="FACR",
        method="DSR-MTG-Qpen-v4-agegate-lyapprior",
        summary_name="DSR-MTG-Qpen-v4-agegate-lyapprior",
        train_args={
            "lyapunov-prior-weight": 0.20,
            "lyapunov-prior-smoothing": 0.80,
        },
    ),
    "v2-s95": FixedVariant(
        name="v2-s95",
        method="DSR-MTG-Qpen-v4-lyapresidual-v2",
        summary_name="DSR-MTG-Qpen-v4-lyapresidual-v2",
        train_args={
            "lyapunov-prior-weight": 0.0,
            "lyapunov-prior-smoothing": 0.95,
            "residual-radius-max": 0.03,
            "residual-direction-l1-cap": 0.5,
            "residual-amplitude-init-bias": -2.0,
            "residual-amplitude-penalty": 0.001,
        },
    ),
    "v3.1-B": FixedVariant(
        name="v3.1-B",
        method="DSR-MTG-Qpen-v4-lyapresidual-v3p1",
        summary_name="DSR-MTG-Qpen-v4-lyapresidual-v3p1",
        checkpoint=ROOT
        / "results_facr_residual_v3p1_B_prescreen"
        / "FACR-Residual-v3.1"
        / "seed_1"
        / "delay_0"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "checkpoint_DSR-MTG-Qpen-v4-lyapresidual-v3p1.pt",
        train_args={
            "lyapunov-prior-weight": 0.0,
            "lyapunov-prior-smoothing": 0.95,
            "residual-amplitude-init-bias": -2.0,
            "residual-proposal-temperature": 1.0,
            "residual-beta-max": 0.05,
            "residual-beta-gamma-delta": 1.0,
            "residual-beta-gamma-volatility": 1.0,
            "residual-aux-proposal-kl-weight": 0.10,
            "residual-aux-amplitude-weight": 0.02,
            "residual-beta-penalty": 0.001,
        },
    ),
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_file(method: str) -> str:
    return method.replace("/", "_")


def checkpoint_dir(output_dir: Path, variant: FixedVariant) -> Path:
    return output_dir / "checkpoints" / variant.name / "seed_1" / "mixed_0_4_8"


def trained_checkpoint(output_dir: Path, variant: FixedVariant) -> Path:
    if variant.checkpoint is not None and variant.checkpoint.exists():
        return variant.checkpoint
    return (
        checkpoint_dir(output_dir, variant)
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"checkpoint_{method_file(variant.method)}.pt"
    )


def checkpoint_config_path(checkpoint: Path, variant: FixedVariant) -> Path:
    return checkpoint.with_name(f"config_{method_file(variant.method)}.json")


def build_train_command(output_dir: Path, variant: FixedVariant, device: str | None) -> list[str]:
    values = {
        **COMMON_TRAIN_ARGS,
        **variant.train_args,
        "output-dir": str(checkpoint_dir(output_dir, variant)),
    }
    if device:
        values["device"] = device
    command = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train", "--method", variant.method]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def ensure_checkpoint(output_dir: Path, variant: FixedVariant, device: str | None, force_train: bool) -> Path:
    checkpoint = trained_checkpoint(output_dir, variant)
    if checkpoint.exists() and not force_train:
        print(f"[checkpoint] reuse {variant.name}: {checkpoint}", flush=True)
        return checkpoint
    command = build_train_command(output_dir, variant, device)
    print(f"[train] {variant.name} single mixed-delay checkpoint", flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint after training: {checkpoint}")
    return checkpoint


def cfg_get(cfg: dict[str, object], key: str, default: object) -> object:
    value = cfg.get(key, default)
    return default if value is None else value


def build_policy_and_args(checkpoint: Path, variant: FixedVariant, eval_seed: int, eval_delay: int, device: str) -> tuple[MAPPOPolicy, SimpleNamespace, MAPPOSystemConfig, ServerScoreConfig, np.ndarray]:
    config_path = checkpoint_config_path(checkpoint, variant)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    method = str(cfg_get(cfg, "method", variant.method))
    residual_v2_method = "DSR-MTG-Qpen-v4-lyapresidual-v2"
    residual_v3_methods = {"DSR-MTG-Qpen-v4-lyapresidual-v3", "DSR-MTG-Qpen-v4-lyapresidual-v3p1"}
    v4_agegate_methods = {
        "DSR-MTG-Qpen-v4-agegate",
        "DSR-MTG-Qpen-v4-agegate-lyapprior",
        "DSR-MTG-Qpen-v4-agegate-lyapresidual",
        "DSR-MTG-Qpen-v4-agegate-noQpen",
        "DSR-MTG-Qpen-v4-agegate-noConc",
    }
    use_residual_actor = method in {residual_v2_method, *residual_v3_methods}
    use_residual_v3 = method in residual_v3_methods
    use_feedback_age_gate = method in v4_agegate_methods and not use_residual_actor
    include_feedback_age = use_feedback_age_gate or use_residual_actor

    score_cfg = ServerScoreConfig(
        resource_weight=float(cfg_get(cfg, "score_resource_weight", 1.0)),
        capacity_weight=float(cfg_get(cfg, "score_capacity_weight", 0.5)),
        task_count_weight=float(cfg_get(cfg, "score_task_count_weight", 0.5)),
        wait_weight=float(cfg_get(cfg, "score_wait_weight", 1.0)),
    )
    sys_cfg = MAPPOSystemConfig(
        n_users=int(cfg_get(cfg, "users", 30)),
        n_edges=int(cfg_get(cfg, "n_edges", 35)),
        slot_s=float(cfg_get(cfg, "slot_s", 1.0)),
        edge_compute_mb_s=float(cfg_get(cfg, "edge_compute_mb_s", 63.0)) * float(cfg_get(cfg, "edge_compute_scale", 1.0)),
        queue_upper_mb=float(cfg_get(cfg, "queue_upper_mb", 630.0)),
        max_requests_per_user=int(cfg_get(cfg, "max_requests_per_user", 7)),
        task_package_mb=float(cfg_get(cfg, "task_package_mb", 270.0 / 1024.0)),
        task_compute_mb=float(cfg_get(cfg, "task_compute_mb", 15.0)) * float(cfg_get(cfg, "task_compute_scale", 1.0)),
        max_supported_users=max(int(cfg_get(cfg, "max_supported_users", 50)), int(cfg_get(cfg, "users", 30))),
        include_feedback_age=include_feedback_age,
        feedback_age_norm=float(cfg_get(cfg, "feedback_age_norm", 8.0)),
        extra_actor_obs_dim=(int(cfg_get(cfg, "n_edges", 35)) + 1) if use_residual_v3 else 0,
    )
    mappo_cfg = MAPPOConfig(
        rollout_size=int(cfg_get(cfg, "rollout_size", 1024)),
        batch_size=int(cfg_get(cfg, "batch_size", 128)),
        learning_rate=float(cfg_get(cfg, "learning_rate", 1e-3)),
        twin_critic=bool(cfg_get(cfg, "twin_critic", True)),
        critic_ensemble_count=int(cfg_get(cfg, "critic_ensemble_count", 1)),
        auto_portfolio_entropy=bool(cfg_get(cfg, "auto_portfolio_entropy", method in v4_agegate_methods)),
        target_portfolio_entropy=float(cfg_get(cfg, "target_portfolio_entropy", 0.99)),
        auto_entropy_lr=float(cfg_get(cfg, "auto_entropy_lr", 1e-3)),
        auto_entropy_init=float(cfg_get(cfg, "auto_entropy_init", 0.01)),
        log_std_min=float(cfg_get(cfg, "log_std_min", -1.5)),
        residual_actor=use_residual_actor,
        residual_actor_prior_conditioned=use_residual_v3,
        residual_amplitude_init_bias=float(cfg_get(cfg, "residual_amplitude_init_bias", 0.0)),
        residual_amplitude_penalty=float(cfg_get(cfg, "residual_amplitude_penalty", 0.0)),
        residual_proposal_temperature=float(cfg_get(cfg, "residual_proposal_temperature", 1.0)),
        residual_beta_max=float(cfg_get(cfg, "residual_beta_max", 0.05)),
        residual_beta_gamma_delta=float(cfg_get(cfg, "residual_beta_gamma_delta", 1.0)),
        residual_beta_gamma_volatility=float(cfg_get(cfg, "residual_beta_gamma_volatility", 1.0)),
        residual_aux_kl_weight=float(cfg_get(cfg, "residual_aux_kl_weight", 0.0)),
        residual_aux_proposal_kl_weight=float(cfg_get(cfg, "residual_aux_proposal_kl_weight", 0.0)),
        residual_aux_amplitude_weight=float(cfg_get(cfg, "residual_aux_amplitude_weight", 0.0)),
        residual_beta_penalty=float(cfg_get(cfg, "residual_beta_penalty", 0.0)),
        feedback_age_gate=use_feedback_age_gate,
        feedback_gate_eta=float(cfg_get(cfg, "feedback_gate_eta", 0.25)),
        dsr_alpha=float(cfg_get(cfg, "alpha", 0.2)),
        dsr_temperature=float(cfg_get(cfg, "dsr_temperature", 5.0)),
        dsr_beta_mean=float(cfg_get(cfg, "beta_mean", 0.4)),
        dsr_beta_tail=float(cfg_get(cfg, "beta_tail", 0.4)),
        dsr_beta_gap=float(cfg_get(cfg, "beta_gap", 0.2)),
        dsr_beta_qpressure=float(cfg_get(cfg, "beta_qpressure", 0.0)),
        dsr_tail_topk=int(cfg_get(cfg, "tail_topk", 0)),
        dsr_tail_fraction=float(cfg_get(cfg, "tail_fraction", 0.6)),
        kappa_i=float(cfg_get(cfg, "kappa_i", 0.02)),
        kappa_g=float(cfg_get(cfg, "kappa_g", 0.02)),
        latency_norm=float(cfg_get(cfg, "latency_norm", 3.0)),
        q_pressure_max=float(cfg_get(cfg, "q_pressure_max", 3.0)),
        q_penalty_weight=float(cfg_get(cfg, "q_penalty_weight", 0.08)),
        dsr_mu=float(cfg_get(cfg, "dsr_mu", 1.5)),
        dsr_eta=float(cfg_get(cfg, "dsr_eta", 0.01)),
        score=score_cfg,
    )
    policy = MAPPOPolicy(sys_cfg, mappo_cfg, seed=int(cfg_get(cfg, "seed", 1)), device=device)
    policy.load_checkpoint(checkpoint, actor_only=True)

    args = SimpleNamespace(**cfg)
    args.method = method
    args.seed = int(eval_seed)
    args.feedback_delay = int(eval_delay)
    args.device = device
    args.trace_root = str(cfg_get(cfg, "trace_root", TRACE_ROOT))
    args.queue_constraint = str(cfg_get(cfg, "queue_constraint", "resample"))
    args.capacity_mode = str(cfg_get(cfg, "capacity_mode", "homogeneous"))
    args.lyapunov_prior_weight = float(cfg_get(cfg, "lyapunov_prior_weight", 0.0))
    args.lyapunov_prior_v = float(cfg_get(cfg, "lyapunov_prior_v", 1.0))
    args.lyapunov_prior_temperature = float(cfg_get(cfg, "lyapunov_prior_temperature", 2.5))
    args.lyapunov_prior_smoothing = float(cfg_get(cfg, "lyapunov_prior_smoothing", 0.0))
    args.lyapunov_prior_max_step = float(cfg_get(cfg, "lyapunov_prior_max_step", 0.0))
    args.lyapunov_prior_fresh_state = False
    capacities = edge_capacities(sys_cfg.n_edges, sys_cfg.edge_compute_mb_s, args.capacity_mode, int(cfg_get(cfg, "seed", 1)))
    return policy, args, sys_cfg, score_cfg, capacities


def build_eval_env(args: SimpleNamespace, sys_cfg: MAPPOSystemConfig, score_cfg: ServerScoreConfig, capacities: np.ndarray) -> JointMAPPOEdgeEnv:
    traces = load_trace_set(
        str(cfg_get(vars(args), "dataset", "KAIST")),
        sys_cfg.n_users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        int(args.seed),
        trace_root=Path(str(args.trace_root)),
    )
    return JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=str(cfg_get(vars(args), "distribution", "poisson_bursty")),
        seed=int(args.seed),
        queue_constraint=str(args.queue_constraint),
        score_config=score_cfg,
        feedback_delay=int(args.feedback_delay),
        broadcast_period=int(cfg_get(vars(args), "broadcast_period", 2)),
        poisson_lambda=float(cfg_get(vars(args), "poisson_lambda", 3.5)),
        burst_prob=float(cfg_get(vars(args), "burst_prob", 0.1)),
        burst_multiplier=float(cfg_get(vars(args), "burst_multiplier", 2.0)),
        edge_capacities_mb_s=capacities,
        background_load=float(cfg_get(vars(args), "background_load", 0.0)),
    )


def evaluate_fixed_checkpoint(checkpoint: Path, variant: FixedVariant, eval_seed: int, eval_delay: int, test_steps: int, device: str) -> dict[str, object]:
    policy, args, sys_cfg, score_cfg, capacities = build_policy_and_args(checkpoint, variant, eval_seed, eval_delay, device)
    env = build_eval_env(args, sys_cfg, score_cfg, capacities)
    residual_v2_method = "DSR-MTG-Qpen-v4-lyapresidual-v2"
    residual_v3_methods = {"DSR-MTG-Qpen-v4-lyapresidual-v3", "DSR-MTG-Qpen-v4-lyapresidual-v3p1"}
    use_residual_v3 = args.method in residual_v3_methods
    previous: np.ndarray | None = None
    diagnostics: dict[str, list[float]] = {}
    v3_obs_state: dict[str, np.ndarray] = {}
    last_v3_actor_obs: np.ndarray | None = None
    results: list[SlotResult] = []
    env.reset()
    for _ in range(test_steps):
        base_obs = env.actor_observations()
        actor_obs = base_obs
        if use_residual_v3:
            actor_obs, _, _ = augment_residual_v3_actor_observations(base_obs, env, args, state=v3_obs_state)
            last_v3_actor_obs = actor_obs
        started = perf_counter()
        portfolios, latents, _ = policy.act(actor_obs, deterministic=True)
        if args.method == residual_v2_method:
            shaped = apply_actor_amplitude_residual_prior(latents, env, args, previous, diagnostics=diagnostics)
        elif use_residual_v3:
            shaped = apply_proposal_residual_prior(latents, env, args, previous, actor_obs=last_v3_actor_obs, diagnostics=diagnostics)
        else:
            shaped = apply_lyapunov_prior(portfolios, env, args, previous, diagnostics=diagnostics)
        decision_latency = perf_counter() - started
        previous = np.asarray(shaped, dtype=np.float32)
        stats = env.portfolio_stats(shaped)
        results.append(env.joint_step(shaped, stats=stats, decision_latency_s=decision_latency))

    row: dict[str, object] = summarize_results(results)
    row.update(phase_diagnostics(results, sys_cfg))
    for key, values in diagnostics.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        row[key] = float(np.mean(finite)) if finite.size else float("nan")
    row.update(
        {
            "variant": variant.name,
            "method": args.method,
            "checkpoint": str(checkpoint),
            "checkpoint_config": str(checkpoint_config_path(checkpoint, variant)),
            "checkpoint_protocol": "one fixed checkpoint evaluated across delays and eval seeds; no retraining during evaluation",
            "eval_seed": eval_seed,
            "feedback_delay": eval_delay,
            "test_steps": test_steps,
            "dataset": str(cfg_get(vars(args), "dataset", "KAIST")),
            "distribution": str(cfg_get(vars(args), "distribution", "poisson_bursty")),
            "users": sys_cfg.n_users,
        }
    )
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "average_latency",
        "cvar99",
        "queue_oscillation",
        "portfolio_migration_rate",
        "lyapunov_prior_kl_w_to_prior",
        "residual_proposal_improvement_ratio",
        "residual_kl_fresh_to_prior",
        "residual_kl_fresh_to_q",
        "residual_oracle_gap_improvement_ratio",
        "residual_actor_amplitude",
        "residual_beta",
    ]
    out: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["feedback_delay"])), []).append(row)
        groups.setdefault((str(row["variant"]), "aggregate"), []).append(row)
    for (variant, delay), group in sorted(groups.items()):
        item: dict[str, object] = {"variant": variant, "feedback_delay": delay, "runs": len(group)}
        for metric in metrics:
            values = np.asarray([float(row.get(metric, np.nan)) for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            item[f"{metric}_std"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One-checkpoint residual/FACR evaluation protocol.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", default="FACR,v2-s95,v3.1-B")
    parser.add_argument("--eval-seeds", default="1,2,3")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [VARIANTS[name.strip()] for name in args.variants.split(",") if name.strip()]
    eval_seeds = parse_csv_ints(args.eval_seeds)
    delays = parse_csv_ints(args.delays)

    checkpoints: dict[str, Path] = {}
    for variant in selected:
        checkpoint = trained_checkpoint(output_dir, variant)
        if args.skip_train:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        else:
            checkpoint = ensure_checkpoint(output_dir, variant, device, args.force_train)
        checkpoints[variant.name] = checkpoint

    raw_path = output_dir / "fixed_checkpoint_residual_raw.csv"
    summary_path = output_dir / "fixed_checkpoint_residual_summary.csv"
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, int, int]] = set()
    if raw_path.exists() and not args.force_eval:
        import pandas as pd

        old = pd.read_csv(raw_path)
        rows = old.to_dict("records")
        for row in rows:
            completed.add((str(row["variant"]), int(row["feedback_delay"]), int(row["eval_seed"])))
        print(f"[resume] {len(completed)} evaluation rows", flush=True)

    total = len(selected) * len(delays) * len(eval_seeds)
    done = 0
    for variant in selected:
        checkpoint = checkpoints[variant.name]
        for delay in delays:
            for eval_seed in eval_seeds:
                done += 1
                key = (variant.name, delay, eval_seed)
                if key in completed:
                    print(f"[skip] {done}/{total} {variant.name} delay={delay} eval_seed={eval_seed}", flush=True)
                    continue
                print(f"[eval] {done}/{total} {variant.name} delay={delay} eval_seed={eval_seed}", flush=True)
                row = evaluate_fixed_checkpoint(checkpoint, variant, eval_seed, delay, args.test_steps, device)
                rows.append(row)
                write_csv(raw_path, rows)
                write_csv(summary_path, aggregate(rows))
    write_csv(raw_path, rows)
    write_csv(summary_path, aggregate(rows))
    print(f"Raw: {raw_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
