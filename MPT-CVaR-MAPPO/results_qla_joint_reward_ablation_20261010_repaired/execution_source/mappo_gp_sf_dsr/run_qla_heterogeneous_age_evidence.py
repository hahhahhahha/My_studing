"""Fixed-checkpoint QLA test under shared versus heterogeneous feedback age.

The same delay-8 trained checkpoint is used in both regimes.  This isolates the
test-time information structure: all users reading B(t-8), versus each user
reading B(t-delta_i) with delta_i sampled from {0,1,2,4,8}.  No retraining is
performed and stale-margin correction remains disabled.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from mappo_gp_sf_dsr.metrics import phase_diagnostics, summarize_results
from mappo_gp_sf_dsr.rewards import MAPPORewarder
from mappo_gp_sf_dsr.run_priceedge_qt_fixed_checkpoint_robustness import (
    build_env,
    checkpoint_path,
    make_policy,
)
from mappo_gp_sf_dsr.train import augment_lcc_price_actor_observations, normalize_portfolios


ROOT = Path(__file__).resolve().parents[1]
AGES = np.asarray([0, 1, 2, 4, 8], dtype=int)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def actor_obs_for_ages(env, args, rewarder, previous, state, ages: np.ndarray) -> np.ndarray:
    original_delay = int(env.feedback_delay)
    rows: list[np.ndarray] = []
    try:
        for user, age in enumerate(ages):
            env.feedback_delay = int(age)
            base = env.actor_observations()
            augmented = augment_lcc_price_actor_observations(
                base, env, args, rewarder,
                previous_portfolios=previous, state=state,
            )
            rows.append(np.asarray(augmented[user], dtype=np.float32))
    finally:
        env.feedback_delay = original_delay
    return np.stack(rows, axis=0)


def actor_obs_for_user_age(env, args, rewarder, previous, state, user_idx: int, age: int) -> np.ndarray:
    """Build augmented actor obs for a single user at a given feedback age.

    Used by ``sequential_fresh`` where each user's obs must be built independently
    inside ``env.sequential_step``'s callback.
    """
    original_delay = int(env.feedback_delay)
    try:
        env.feedback_delay = int(age)
        base = env.actor_observations()
        augmented = augment_lcc_price_actor_observations(
            base, env, args, rewarder,
            previous_portfolios=previous, state=state,
        )
        return np.asarray(augmented[user_idx], dtype=np.float32)
    finally:
        env.feedback_delay = original_delay


def run_one(condition: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = int(seed)
    env, sys_cfg, _ = build_env(
        run_args.dataset, run_args.distribution, run_args.users,
        run_args.shared_delay, run_args,
    )
    policy = make_policy(sys_cfg, run_args)
    ckpt = checkpoint_path(
        Path(run_args.checkpoint_root),
        run_args.checkpoint_seed,
        run_args.checkpoint_delay,
    )
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    policy.load_checkpoint(ckpt, actor_only=True)
    policy.actor.eval()
    rewarder = MAPPORewarder(policy.cfg, method="LCC-MAPPO")
    rng = np.random.default_rng(seed + 9173)
    previous: np.ndarray | None = None
    state: dict[str, np.ndarray] = {}
    age_counts = {int(age): 0 for age in AGES}
    results = []
    env.reset()
    for _ in range(run_args.test_steps):
        if condition == "shared_stale":
            ages = np.full(run_args.users, run_args.shared_delay, dtype=int)
        elif condition == "sync_fresh":
            ages = np.zeros(run_args.users, dtype=int)
        elif condition == "heterogeneous_age":
            ages = rng.choice(AGES, size=run_args.users, replace=True)
        elif condition == "sequential_fresh":
            ages = None  # handled via env.sequential_step below
        else:
            raise ValueError(condition)
        if ages is not None:
            for age in ages:
                age_counts[int(age)] += 1
            actor_obs = actor_obs_for_ages(env, run_args, rewarder, previous, state, ages)
            portfolios, _, _ = policy.act(actor_obs, deterministic=True)
            portfolios = normalize_portfolios(portfolios)
            stats = env.portfolio_stats(portfolios)
            result = env.joint_step(portfolios, stats=stats, decision_latency_s=0.0)
        else:
            # Sequential fresh: each user reads age=0 state, earlier users' load
            # contributions are visible to later users within the same slot.
            for age in np.zeros(run_args.users, dtype=int):
                age_counts[int(age)] += 1

            # Build augmented obs once per slot (env.feedback_delay=0 → fresh).
            # prices/scale are system-wide; the per-user variation comes from
            # `_obs` which carries sequential_step's accumulated slot_load.
            original_delay_seq = int(env.feedback_delay)
            try:
                env.feedback_delay = 0
                full_base = env.actor_observations()
                full_aug = augment_lcc_price_actor_observations(
                    full_base, env, run_args, rewarder,
                    previous_portfolios=previous, state=state,
                )
            finally:
                env.feedback_delay = original_delay_seq
            base_dim = full_base.shape[1]

            def policy_fn(user: int, _obs: np.ndarray) -> np.ndarray:
                aug_extra = full_aug[user, base_dim:]
                single_obs = np.concatenate(
                    [_obs.astype(np.float32), aug_extra.astype(np.float32)], axis=0,
                )[None, :]
                weights, _, _ = policy.act(single_obs, deterministic=True)
                return normalize_portfolios(weights)[0]

            result = env.sequential_step(policy_fn)
            portfolios = np.asarray(result.portfolios, dtype=np.float32)
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
        results.append(result)
    row: dict[str, object] = summarize_results(results)
    row.update(phase_diagnostics(results, sys_cfg))
    row.update({
        "method": "QLA-MAPPO-NoMargin",
        "condition": condition,
        "evaluation_seed": seed,
        "checkpoint_seed": args.checkpoint_seed,
        "checkpoint_delay": args.checkpoint_delay,
        "test_steps": args.test_steps,
        "lcc_use_stale_margin": False,
        "lcc_use_z_queue": True,
        "age_histogram": ";".join(f"{k}:{v}" for k, v in age_counts.items()),
        "checkpoint": str(ckpt),
    })
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results_qla_simplified_rerun/heterogeneous_age"))
    p.add_argument("--checkpoint-root", type=Path,
                   default=Path("results_qla_simplified_rerun/main/QLA-MAPPO-NoMargin"))
    p.add_argument("--checkpoint-seed", type=int, default=1)
    p.add_argument("--checkpoint-delay", type=int, default=8)
    p.add_argument("--evaluation-seeds", default="1,2,3")
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset", default="KAIST")
    p.add_argument("--distribution", default="poisson_bursty")
    p.add_argument("--users", type=int, default=30)
    p.add_argument("--shared-delay", type=int, default=8)
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
    rows = [
        run_one(condition, seed, args)
        for seed in parse_ints(args.evaluation_seeds)
        for condition in ("sync_fresh", "shared_stale", "heterogeneous_age", "sequential_fresh")
    ]
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "qla_shared_vs_heterogeneous_age_raw.csv", rows)
    df = pd.DataFrame(rows)
    metrics = ["average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"]
    summary_rows: list[dict[str, object]] = []
    for condition, group in df.groupby("condition"):
        row: dict[str, object] = {"condition": condition, "n_eval_seeds": len(group)}
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1))
        summary_rows.append(row)
    write_rows(out / "qla_shared_vs_heterogeneous_age_summary.csv", summary_rows)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
