"""Estimate dependence between stale wait summaries and QLA portfolio actions.

For each evaluation delay, the script loads the matching no-margin checkpoint,
collects (B_j(t-delta).wait_norm, w_ij(t)) pairs, and reports mutual information
and rank correlation.  Samples are flattened over slots, users, and edges; a
fixed RNG subsample bounds the kNN MI estimator cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from mappo_gp_sf_dsr.rewards import MAPPORewarder
from mappo_gp_sf_dsr.run_priceedge_qt_fixed_checkpoint_robustness import (
    build_env,
    checkpoint_path,
    make_policy,
)
from mappo_gp_sf_dsr.train import augment_lcc_price_actor_observations, normalize_portfolios


ROOT = Path(__file__).resolve().parents[1]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def evaluate_delay(delay: int, args: argparse.Namespace) -> dict[str, object]:
    env, sys_cfg, _ = build_env(args.dataset, args.distribution, args.users, delay, args)
    policy = make_policy(sys_cfg, args)
    ckpt = checkpoint_path(Path(args.checkpoint_root), args.checkpoint_seed, delay)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    policy.load_checkpoint(ckpt, actor_only=True)
    policy.actor.eval()
    rewarder = MAPPORewarder(policy.cfg, method="LCC-MAPPO")
    state: dict[str, np.ndarray] = {}
    previous: np.ndarray | None = None
    waits: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    env.reset()
    for _ in range(args.test_steps):
        feedback = env.feedback_buffer.get(env.feedback_delay)
        wait_norm = np.asarray(feedback.wait_norm, dtype=np.float32)
        base_obs = env.actor_observations()
        actor_obs = augment_lcc_price_actor_observations(
            base_obs, env, args, rewarder,
            previous_portfolios=previous, state=state,
        )
        portfolios, _, _ = policy.act(actor_obs, deterministic=True)
        portfolios = normalize_portfolios(portfolios)
        # Each user observes the same per-edge broadcast summary in this regime.
        waits.append(np.broadcast_to(wait_norm[None, :], portfolios.shape).reshape(-1).copy())
        weights.append(np.asarray(portfolios, dtype=np.float32).reshape(-1).copy())
        stats = env.portfolio_stats(portfolios)
        result = env.joint_step(portfolios, stats=stats, decision_latency_s=0.0)
        prev_for_reward = None if previous is None else previous.copy()
        previous = np.asarray(result.portfolios, dtype=np.float32)
        rewarder.compute(
            result.per_user_mean_latency,
            result.per_user_tasks,
            result.system_average_latency,
            result.portfolio_stats,
            portfolios=result.portfolios,
            queues_before_mb=result.queue_load_before_mb,
            sys_cfg=sys_cfg,
            previous_portfolios=prev_for_reward,
            slot_latencies=np.asarray(result.latencies, dtype=np.float32),
        )

    x_all = np.concatenate(waits).astype(np.float64)
    y_all = np.concatenate(weights).astype(np.float64)
    rng = np.random.default_rng(args.seed * 1009 + delay)
    if x_all.size > args.max_mi_samples:
        idx = rng.choice(x_all.size, args.max_mi_samples, replace=False)
        x, y = x_all[idx], y_all[idx]
    else:
        x, y = x_all, y_all
    mi = float(mutual_info_regression(
        x.reshape(-1, 1), y,
        discrete_features=False,
        n_neighbors=args.mi_neighbors,
        random_state=args.seed,
    )[0])
    rho = float(spearmanr(-x, y, nan_policy="omit").statistic)
    return {
        "method": "QLA-MAPPO-NoMargin",
        "feedback_delay": delay,
        "checkpoint_seed": args.checkpoint_seed,
        "evaluation_seed": args.seed,
        "test_steps": args.test_steps,
        "all_pairs": int(x_all.size),
        "mi_samples": int(x.size),
        "mutual_information_nats_wait_to_weight": mi,
        "spearman_neg_wait_vs_weight": rho,
        "mean_wait_norm": float(x_all.mean()),
        "mean_portfolio_weight": float(y_all.mean()),
        "lcc_use_stale_margin": False,
        "lcc_use_z_queue": True,
        "checkpoint": str(ckpt),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results_qla_simplified_rerun/mi_audit"))
    p.add_argument("--checkpoint-root", type=Path,
                   default=Path("results_qla_simplified_rerun/main/QLA-MAPPO-NoMargin"))
    p.add_argument("--checkpoint-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delays", default="0,1,2,4,8")
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--max-mi-samples", type=int, default=200_000)
    p.add_argument("--mi-neighbors", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset", default="KAIST")
    p.add_argument("--distribution", default="poisson_bursty")
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--trace-root", default=str(ROOT / "Traces_TimeXY_10sec_txt"))
    p.add_argument("--slot-s", type=float, default=1.0)
    p.add_argument("--n-edges", type=int, default=35)
    p.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    p.add_argument("--edge-compute-scale", type=float, default=1.0)
    p.add_argument("--queue-upper-mb", type=float, default=630.0)
    p.add_argument("--max-requests-per-user", type=int, default=7)
    p.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    p.add_argument("--task-compute-mb", type=float, default=15.0)
    p.add_argument("--task-compute-scale", type=float, default=1.0)
    p.add_argument("--max-supported-users", type=int, default=80)
    p.add_argument("--feedback-age-norm", type=float, default=8.0)
    p.add_argument("--queue-constraint", default="resample")
    p.add_argument("--broadcast-period", type=int, default=2)
    p.add_argument("--poisson-lambda", type=float, default=3.5)
    p.add_argument("--burst-prob", type=float, default=0.1)
    p.add_argument("--burst-multiplier", type=float, default=2.0)
    p.add_argument("--background-load", type=float, default=0.0)
    p.add_argument("--capacity-mode", default="homogeneous")
    p.add_argument("--score-resource-weight", type=float, default=1.0)
    p.add_argument("--score-capacity-weight", type=float, default=0.5)
    p.add_argument("--score-task-count-weight", type=float, default=0.5)
    p.add_argument("--score-wait-weight", type=float, default=1.0)
    p.add_argument("--target-portfolio-entropy", type=float, default=0.99)
    p.add_argument("--latency-norm", type=float, default=3.0)
    p.add_argument("--lcc-v", type=float, default=0.5)
    p.add_argument("--lcc-migration-weight", type=float, default=0.05)
    p.add_argument("--lcc-cvar-alpha", type=float, default=0.99)
    p.add_argument("--lcc-cvar-budget", type=float, default=1.30)
    p.add_argument("--lcc-var-lr", type=float, default=0.05)
    p.add_argument("--lcc-concentration-factor", type=float, default=2.0)
    p.add_argument("--lcc-stale-margin", type=float, default=1.0)
    p.add_argument("--lcc-price-components", default="QT")
    p.add_argument("--lcc-price-logit-bias", type=float, default=0.75)
    p.add_argument("--lcc-price-intervention", default="aligned")
    p.add_argument("--lcc-price-shuffle", action="store_true")
    p.add_argument("--lcc-no-stale-margin", action="store_true", default=True)
    p.add_argument("--lcc-no-z-queue", action="store_true", default=False)
    p.add_argument("--lcc-no-h-queue", action="store_true", default=False)
    args = p.parse_args()
    args.lcc_price_scale_observation = True
    rows = [evaluate_delay(delay, args) for delay in parse_ints(args.delays)]
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "qla_wait_action_mutual_information.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
