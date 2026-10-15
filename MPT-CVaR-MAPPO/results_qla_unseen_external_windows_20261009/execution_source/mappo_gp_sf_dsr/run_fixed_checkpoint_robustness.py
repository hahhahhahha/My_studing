from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, ServerScoreConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results
from .models import MAPPOPolicy
from .paper_policy_baselines import SharedGaussianPPO
from .recent_mec_baselines import QECONetwork, QECOAgent, QECOReplay, _qeco_action_portfolios, _qeco_state
from .train import apply_lyapunov_prior, edge_capacities


ROOT = Path(__file__).resolve().parents[1]
FACR_ROOT = ROOT / "results_v4_lyapprior_factorial_formal" / "AL_BH_CH_DH_EH"
PPO_ROOT = ROOT / "results_paper_algorithms_v4scenario_multiseed"
QECO_ROOT = ROOT / "results_recent_mec_baselines_strong_qeco_multiseed" / "QECO-load-topk"


def parse_csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
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


def build_system(users: int, args: argparse.Namespace, include_feedback_age: bool) -> tuple[MAPPOSystemConfig, ServerScoreConfig, np.ndarray]:
    sys_cfg = MAPPOSystemConfig(
        n_users=users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_package_mb=args.task_package_mb,
        task_compute_mb=args.task_compute_mb,
        max_supported_users=max(users, args.max_supported_users),
        include_feedback_age=include_feedback_age,
        feedback_age_norm=args.feedback_age_norm,
    )
    score_cfg = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    capacities = edge_capacities(args.n_edges, args.edge_compute_mb_s, args.capacity_mode, args.seed)
    return sys_cfg, score_cfg, capacities


def build_env(dataset: str, distribution: str, users: int, delay: int, args: argparse.Namespace, include_feedback_age: bool) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig]:
    sys_cfg, score_cfg, capacities = build_system(users, args, include_feedback_age)
    traces = load_trace_set(
        dataset,
        users,
        sys_cfg.n_edges,
        sys_cfg.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    env = JointMAPPOEdgeEnv(
        sys_cfg,
        traces,
        request_distribution=distribution,
        seed=args.seed,
        queue_constraint=args.queue_constraint,
        score_config=score_cfg,
        feedback_delay=delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    return env, sys_cfg


def facr_checkpoint(delay: int, seed: int) -> Path:
    path = (
        FACR_ROOT
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "checkpoint_DSR-MTG-Qpen-v4-agegate-lyapprior.pt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ppo_checkpoint(method: str, delay: int, seed: int) -> Path:
    original = "Towards-PPO-MV" if method == "PPO-MV-A" else "Towards-PPO-DSR"
    path = (
        PPO_ROOT
        / original
        / f"seed_{seed}"
        / f"delay_{delay}"
        / f"{original}_delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / f"checkpoint_{original}.pt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def qeco_checkpoint(delay: int, seed: int) -> Path:
    path = (
        QECO_ROOT
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "checkpoint_QECO-load-topk.pt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def summarize_run(results: list[SlotResult], sys_cfg: MAPPOSystemConfig) -> dict[str, float]:
    summary = summarize_results(results)
    summary.update(phase_diagnostics(results, sys_cfg))
    latencies = np.asarray([lat for result in results for lat in result.latencies], dtype=np.float64)
    summary["p95_latency"] = float(np.percentile(latencies, 95)) if latencies.size else float("nan")
    summary["p99_latency"] = float(np.percentile(latencies, 99)) if latencies.size else float("nan")
    return summary


def latency_rows(results: list[SlotResult], meta: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot, result in enumerate(results):
        for record in result.records:
            rows.append({**meta, "slot": slot, "user": record.user, "edge": record.edge, "latency_s": record.latency_s})
    return rows


def run_facr(env: JointMAPPOEdgeEnv, delay: int, args: argparse.Namespace) -> list[SlotResult]:
    _, score_cfg, _ = build_system(env.cfg.n_users, args, include_feedback_age=True)
    alg_cfg = MAPPOConfig(
        twin_critic=True,
        auto_portfolio_entropy=True,
        feedback_age_gate=True,
        feedback_gate_eta=args.feedback_gate_eta,
        target_portfolio_entropy=args.target_portfolio_entropy,
        log_std_min=args.log_std_min,
        q_penalty_weight=args.q_penalty_weight,
        kappa_i=args.kappa_i,
        kappa_g=args.kappa_g,
        score=score_cfg,
    )
    policy = MAPPOPolicy(env.cfg, alg_cfg, seed=args.seed, device=args.device)
    policy.load_checkpoint(facr_checkpoint(delay, args.checkpoint_seed), actor_only=True)
    lyap_args = argparse.Namespace(
        lyapunov_prior_weight=args.lyapunov_prior_weight,
        lyapunov_prior_v=args.lyapunov_prior_v,
        lyapunov_prior_temperature=args.lyapunov_prior_temperature,
        lyapunov_prior_smoothing=args.lyapunov_prior_smoothing,
        lyapunov_prior_max_step=args.lyapunov_prior_max_step,
        lyapunov_prior_fresh_state=False,
    )
    results: list[SlotResult] = []
    previous: np.ndarray | None = None
    env.reset()
    for _ in range(args.test_steps):
        obs = env.actor_observations()
        started = perf_counter()
        portfolios, _, _ = policy.act(obs, deterministic=True)
        portfolios = apply_lyapunov_prior(portfolios, env, lyap_args, previous)
        decision_latency_s = perf_counter() - started
        previous = portfolios.copy()
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s))
    return results


def run_ppo(env: JointMAPPOEdgeEnv, method: str, delay: int, args: argparse.Namespace) -> list[SlotResult]:
    obs = env.reset()
    policy = SharedGaussianPPO(obs.shape[1], env.cfg.n_edges, log_std_min=-1.5).to(args.device)
    state = torch.load(ppo_checkpoint(method, delay, args.checkpoint_seed), map_location=args.device)
    policy.load_state_dict(state)
    policy.eval()
    results: list[SlotResult] = []
    for _ in range(args.test_steps):
        obs = env.actor_observations()
        started = perf_counter()
        portfolios = policy.act(obs, deterministic=True)[3]
        decision_latency_s = perf_counter() - started
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s))
    return results


def make_qeco_agents(env: JointMAPPOEdgeEnv, users: int, delay: int, args: argparse.Namespace) -> tuple[list[QECOAgent], np.ndarray]:
    states, workload = _qeco_state(env)
    checkpoint = torch.load(qeco_checkpoint(delay, args.checkpoint_seed), map_location=args.device)
    saved_models = checkpoint["online_models"]
    agents: list[QECOAgent] = []
    for user in range(users):
        model = QECONetwork(states.shape[1], workload.size, env.cfg.n_edges, args.qeco_hidden, args.qeco_lstm_hidden).to(args.device)
        source = saved_models[user % len(saved_models)]
        model.load_state_dict(source)
        model.eval()
        target = QECONetwork(states.shape[1], workload.size, env.cfg.n_edges, args.qeco_hidden, args.qeco_lstm_hidden).to(args.device)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        agents.append(QECOAgent(model, target, optimizer, QECOReplay(1)))
    return agents, workload


def run_qeco(env: JointMAPPOEdgeEnv, delay: int, args: argparse.Namespace) -> list[SlotResult]:
    rng = np.random.default_rng(args.seed)
    env.reset()
    agents, workload = make_qeco_agents(env, env.cfg.n_users, delay, args)
    states, workload = _qeco_state(env)
    histories = np.repeat(workload[None, None, :], env.cfg.n_users * args.history_steps, axis=0).reshape(
        env.cfg.n_users, args.history_steps, workload.size
    )
    results: list[SlotResult] = []
    for _ in range(args.test_steps):
        states, workload = _qeco_state(env)
        histories = np.concatenate(
            [histories[:, 1:], np.repeat(workload[None, None, :], env.cfg.n_users, axis=0)],
            axis=1,
        )
        started = perf_counter()
        _, portfolios = _qeco_action_portfolios(
            agents,
            states,
            histories,
            env,
            greedy_probability=1.0,
            rng=rng,
            device=torch.device(args.device),
            deterministic=True,
            topk=args.qeco_topk,
            temperature=args.qeco_topk_temperature,
            load_mask=args.qeco_load_mask,
            mask_quantile=args.qeco_mask_quantile,
        )
        decision_latency_s = perf_counter() - started
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s))
    return results


def run_one(method: str, dataset: str, distribution: str, users: int, delay: int, args: argparse.Namespace) -> tuple[dict[str, object], list[SlotResult], MAPPOSystemConfig]:
    include_age = method == "FACR-MAPPO"
    env, sys_cfg = build_env(dataset, distribution, users, delay, args, include_feedback_age=include_age)
    if method == "FACR-MAPPO":
        results = run_facr(env, delay, args)
    elif method in {"PPO-MV-A", "PPO-DSR-A"}:
        results = run_ppo(env, method, delay, args)
    elif method == "QECO-A":
        results = run_qeco(env, delay, args)
    else:
        raise ValueError(method)
    row = summarize_run(results, sys_cfg)
    row.update(
        {
            "method": method,
            "dataset": dataset,
            "distribution": distribution,
            "users": users,
            "feedback_delay": delay,
            "seed": args.seed,
            "checkpoint_seed": args.checkpoint_seed,
            "status": "ok",
            "checkpoint_protocol": "fixed pretrained checkpoint selected by method and feedback delay; no retraining in target environment",
        }
    )
    if method == "QECO-A" and users > 30:
        row["checkpoint_protocol"] += "; per-user QECO agents are reused cyclically beyond the 30 trained agents"
    return row, results, sys_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-checkpoint cross-environment robustness for FACR and selected baselines.")
    parser.add_argument("--output-dir", default="results_fixed_checkpoint_robustness")
    parser.add_argument("--datasets", default="KAIST,NCSU,NewYork")
    parser.add_argument("--distributions", default="uniform,gaussian")
    parser.add_argument("--users", default="15,20,25,30")
    parser.add_argument("--delays", default="0,8")
    parser.add_argument("--methods", default="FACR-MAPPO,PPO-MV-A,PPO-DSR-A,QECO-A")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-latency-filter", default="feedback_delay=8,users=30")

    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=80)
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)

    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--feedback-gate-eta", type=float, default=0.25)
    parser.add_argument("--target-portfolio-entropy", type=float, default=0.99)
    parser.add_argument("--log-std-min", type=float, default=-1.5)
    parser.add_argument("--q-penalty-weight", type=float, default=0.08)
    parser.add_argument("--kappa-i", type=float, default=0.02)
    parser.add_argument("--kappa-g", type=float, default=0.02)
    parser.add_argument("--lyapunov-prior-weight", type=float, default=0.20)
    parser.add_argument("--lyapunov-prior-v", type=float, default=1.0)
    parser.add_argument("--lyapunov-prior-temperature", type=float, default=2.5)
    parser.add_argument("--lyapunov-prior-smoothing", type=float, default=0.80)
    parser.add_argument("--lyapunov-prior-max-step", type=float, default=0.03)

    parser.add_argument("--qeco-hidden", type=int, default=20)
    parser.add_argument("--qeco-lstm-hidden", type=int, default=20)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--qeco-topk", type=int, default=4)
    parser.add_argument("--qeco-topk-temperature", type=float, default=0.7)
    parser.add_argument("--qeco-load-mask", action="store_true", default=True)
    parser.add_argument("--qeco-mask-quantile", type=float, default=0.7)

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fixed_checkpoint_robustness_raw.csv"
    latency_path = output_dir / "fixed_checkpoint_robustness_latency_samples.csv"

    datasets = parse_csv_list(args.datasets, str)
    distributions = parse_csv_list(args.distributions, str)
    users_list = parse_csv_list(args.users, int)
    delays = parse_csv_list(args.delays, int)
    methods = parse_csv_list(args.methods, str)

    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, int, int, str]] = set()
    if summary_path.exists() and not args.force:
        import pandas as pd

        old = pd.read_csv(summary_path)
        rows = old.to_dict("records")
        for row in rows:
            if str(row.get("status", "ok")) == "ok":
                completed.add((str(row["dataset"]), str(row["distribution"]), int(row["users"]), int(row["feedback_delay"]), str(row["method"])))
        print(f"[resume] {len(completed)} completed rows")

    lat_rows: list[dict[str, object]] = []
    total = len(datasets) * len(distributions) * len(users_list) * len(delays) * len(methods)
    done = 0
    for dataset in datasets:
        for distribution in distributions:
            for users in users_list:
                for delay in delays:
                    for method in methods:
                        done += 1
                        key = (dataset, distribution, users, delay, method)
                        print(f"[run] {done}/{total} {method} {dataset} {distribution} users={users} delay={delay}", flush=True)
                        if key in completed:
                            print("[skip] completed", flush=True)
                            continue
                        meta = {"method": method, "dataset": dataset, "distribution": distribution, "users": users, "feedback_delay": delay, "seed": args.seed}
                        try:
                            row, results, _ = run_one(method, dataset, distribution, users, delay, args)
                            if delay == 8 and users in {30, 40}:
                                lat_rows.extend(latency_rows(results, meta))
                        except Exception as exc:
                            row = {**meta, "status": "error", "error": str(exc)}
                            print(f"[error] {row}", flush=True)
                        rows.append(row)
                        write_csv(summary_path, rows)
    write_csv(summary_path, rows)
    write_csv(latency_path, lat_rows)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
