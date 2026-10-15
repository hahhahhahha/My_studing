from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, TRACE_ROOT, ServerScoreConfig
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import phase_diagnostics, summarize_results
from .models import MAPPOPolicy
from .train import apply_lyapunov_prior, edge_capacities


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = ROOT / "results_v4_lyapprior_factorial_formal" / "AL_BH_CH_DH_EH"


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


def checkpoint_for_delay(root: Path, delay: int, seed: int) -> Path:
    path = (
        root
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


def build_system(users: int, args: argparse.Namespace) -> tuple[MAPPOSystemConfig, ServerScoreConfig, np.ndarray]:
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
        include_feedback_age=True,
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


def build_env(dataset: str, distribution: str, users: int, delay: int, args: argparse.Namespace) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig]:
    sys_cfg, score_cfg, capacities = build_system(users, args)
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


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64) * float(temperature)
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    return (exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def score_softmax_portfolios(env: JointMAPPOEdgeEnv, temperature: float) -> np.ndarray:
    obs = env.actor_observations()
    scores = obs[:, -(env.cfg.n_edges + 1) : -1] if env.cfg.include_feedback_age else obs[:, -env.cfg.n_edges :]
    return softmax(scores, temperature)


def delayed_jsq_portfolios(env: JointMAPPOEdgeEnv) -> np.ndarray:
    feedback = env.feedback_buffer.get(env.feedback_delay)
    best = int(np.argmin(feedback.wait_norm))
    portfolios = np.zeros((env.cfg.n_users, env.cfg.n_edges), dtype=np.float32)
    portfolios[:, best] = 1.0
    return portfolios


def run_facr(env: JointMAPPOEdgeEnv, delay: int, args: argparse.Namespace) -> list[SlotResult]:
    _, score_cfg, _ = build_system(env.cfg.n_users, args)
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
    policy.load_checkpoint(checkpoint_for_delay(Path(args.checkpoint_root), delay, args.checkpoint_seed), actor_only=True)

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


def run_mechanism(env: JointMAPPOEdgeEnv, method: str, args: argparse.Namespace) -> list[SlotResult]:
    results: list[SlotResult] = []
    env.reset()
    for _ in range(args.test_steps):
        started = perf_counter()
        if method == "score-softmax":
            portfolios = score_softmax_portfolios(env, args.score_policy_temperature)
        elif method == "delayed-JSQ":
            portfolios = delayed_jsq_portfolios(env)
        else:
            raise ValueError(f"Unknown mechanism: {method}")
        decision_latency_s = perf_counter() - started
        stats = env.portfolio_stats(portfolios)
        results.append(env.joint_step(portfolios, stats=stats, decision_latency_s=decision_latency_s))
    return results


def summarize_run(results: list[SlotResult], sys_cfg: MAPPOSystemConfig) -> dict[str, float]:
    summary = summarize_results(results)
    summary.update(phase_diagnostics(results, sys_cfg))
    return summary


def latency_rows(results: list[SlotResult], meta: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot, result in enumerate(results):
        for record in result.records:
            rows.append(
                {
                    **meta,
                    "slot": slot,
                    "user": record.user,
                    "edge": record.edge,
                    "latency_s": record.latency_s,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="FACR-MAPPO real-trace robustness evaluation.")
    parser.add_argument("--output-dir", default="results_facr_real_trace_robustness")
    parser.add_argument("--datasets", default="KAIST,NCSU,NewYork")
    parser.add_argument("--distributions", default="uniform,gaussian")
    parser.add_argument("--users", default="20,30,40")
    parser.add_argument("--delays", default="0,1,2,4,8")
    parser.add_argument("--methods", default="FACR-MAPPO,score-softmax,delayed-JSQ")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--test-steps", type=int, default=1000)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=50)
    parser.add_argument("--feedback-age-norm", type=float, default=8.0)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--capacity-mode", default="homogeneous", choices=["homogeneous", "heterogeneous", "uniform_heterogeneous"])
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.0)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)

    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--score-policy-temperature", type=float, default=5.0)

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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv_list(args.datasets, str)
    distributions = parse_csv_list(args.distributions, str)
    users_list = parse_csv_list(args.users, int)
    delays = parse_csv_list(args.delays, int)
    methods = parse_csv_list(args.methods, str)

    summary_path = output_dir / "facr_real_trace_robustness_raw.csv"
    latency_path = output_dir / "facr_real_trace_latency_samples.csv"
    rows: list[dict[str, object]] = []
    lat_rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, int, int, str]] = set()
    if summary_path.exists() and not args.force:
        import pandas as pd

        old = pd.read_csv(summary_path)
        rows = old.to_dict("records")
        for row in rows:
            if str(row.get("status", "ok")) == "error":
                continue
            completed.add(
                (
                    str(row["dataset"]),
                    str(row["distribution"]),
                    int(row["users"]),
                    int(row["feedback_delay"]),
                    str(row["method"]),
                )
            )
        print(f"[resume] loaded {len(rows)} summary rows; completed={len(completed)}")

    total = len(datasets) * len(distributions) * len(users_list) * len(delays) * len(methods)
    done = 0
    for dataset in datasets:
        for distribution in distributions:
            for users in users_list:
                for delay in delays:
                    for method in methods:
                        done += 1
                        print(f"[run] {done}/{total} {method} {dataset} {distribution} users={users} delay={delay}", flush=True)
                        meta = {
                            "method": method,
                            "dataset": dataset,
                            "distribution": distribution,
                            "users": users,
                            "feedback_delay": delay,
                            "seed": args.seed,
                        }
                        key = (dataset, distribution, users, delay, method)
                        if key in completed:
                            print(f"[skip] completed {method} {dataset} {distribution} users={users} delay={delay}", flush=True)
                            continue
                        try:
                            env, sys_cfg = build_env(dataset, distribution, users, delay, args)
                            if method == "FACR-MAPPO":
                                results = run_facr(env, delay, args)
                            else:
                                results = run_mechanism(env, method, args)
                            row = {**meta, "status": "ok", **summarize_run(results, sys_cfg)}
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
