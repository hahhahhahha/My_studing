"""Run one isolated repair condition without editing the canonical package.

Bootstrap-only still retains the problematic action-conditioned critic. Its
next-state value is an antithetic 16-sample policy expectation; independent
local generators leave the training RNG streams untouched. This diagnostic
arm does NOT make the original action-conditioned advantage estimator valid.
"""
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_training_repair_core import (
    gae_with_bootstrap, instrument_train_source, mask_action_features, sha256,
    source_digest,
)


def write_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", choices=["original", "critic", "bootstrap", "both"], required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    opts, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if source_digest(ROOT) != opts.expected_source:
        raise SystemExit("Canonical package changed after pilot preparation")
    import numpy as np
    import torch
    from mappo_gp_sf_dsr import train
    from mappo_gp_sf_dsr.env import JointMAPPOEdgeEnv
    from mappo_gp_sf_dsr.models import MAPPOPolicy

    torch.set_num_threads(1)
    sys.argv = [str(__file__), *remaining]
    args = train.parse_args()
    if args.method != "LCC-MAPPO" or args.lcc_local_critic or args.lcc_price_clean_observation:
        raise SystemExit("Pilot is restricted to the canonical shared edge-wise LCC actor")
    opts.diagnostic_dir.mkdir(parents=True, exist_ok=True)
    fix_critic = opts.repair in {"critic", "both"}
    fix_bootstrap = opts.repair in {"bootstrap", "both"}
    updates, boundaries, queue_rows = [], [], []
    phase = "train"
    original_critic_obs = JointMAPPOEdgeEnv.critic_observations
    original_gae = MAPPOPolicy._compute_gae
    original_clip = torch.nn.utils.clip_grad_norm_
    original_evaluate = train.evaluate_policy
    original_step = JointMAPPOEdgeEnv.joint_step
    grad_norms = []

    def critic_observations(env, portfolios, stats=None, feedback=None):
        obs = original_critic_obs(env, portfolios, stats=stats, feedback=feedback)
        return mask_action_features(obs, env.cfg.n_edges, len(env._system_summary())) if fix_critic else obs

    def clip_grad_norm(*a, **kw):
        norm = original_clip(*a, **kw)
        grad_norms.append(float(norm.detach().cpu()))
        return norm

    def evaluate(*a, **kw):
        nonlocal phase
        phase = args.evaluation_split_role
        return original_evaluate(*a, **kw)

    def joint_step(env, *a, **kw):
        result = original_step(env, *a, **kw)
        jump = np.maximum(result.queue_load_after_mb - result.queue_load_before_mb, 0.0)
        jump = jump / np.maximum(env.edge_capacities_mb_s * env.cfg.slot_s, 1e-8)
        queue_rows.append({"phase": phase, "slot": int(env.slot_index),
                           "max_queue_jump_service_slots": float(np.max(jump)),
                           "spike_gt_one": int(np.max(jump) > 1.0)})
        return result

    def next_values(policy, env, train_args, rewarder, previous, obs_state, update_id):
        uniform = np.full((env.cfg.n_users, env.cfg.n_edges), 1.0 / env.cfg.n_edges, dtype=np.float32)
        user_ids = np.arange(env.cfg.n_users, dtype=np.int64)
        if fix_critic:
            return policy.value(env.critic_observations(uniform), user_ids)
        # Bootstrap-only: E_a[V(s_next,a)] rather than an arbitrary new action
        # or the last action. No global RNG consumption or observation mutation.
        obs = train.augment_lcc_price_actor_observations(
            env.actor_observations(), env, train_args, copy.deepcopy(rewarder),
            previous_portfolios=previous, state=copy.deepcopy(obs_state),
        )
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
        generator = torch.Generator(device=policy.device)
        generator.manual_seed(910000 + int(args.seed) * 10000 + update_id)
        vals = []
        with torch.no_grad():
            dist = policy.actor.distribution(tensor)
            for _ in range(8):
                noise = torch.randn(dist.mean.shape, generator=generator, device=policy.device)
                for sign in (1.0, -1.0):
                    w = policy.actor.portfolio(dist.mean + sign * dist.stddev * noise, tensor)
                    portfolios = train.normalize_portfolios(w.cpu().numpy())
                    vals.append(policy.value(env.critic_observations(portfolios), user_ids))
        return np.asarray(vals, dtype=np.float32).mean(axis=0)

    def repair_update(policy, buffer, env, train_args, rewarder, previous, obs_state):
        arrays = buffer.arrays()
        update_id = len(updates) + 1
        boot = next_values(policy, env, train_args, rewarder, previous, obs_state, update_id) if fix_bootstrap else np.zeros(env.cfg.n_users, dtype=np.float32)
        policy._repair_bootstrap = boot
        advantages, returns = (gae_with_bootstrap(arrays, policy.cfg.gamma, policy.cfg.gae_lambda, boot)
                               if fix_bootstrap else original_gae(policy, arrays))
        for array in (boot, advantages, returns):
            if not np.isfinite(array).all():
                raise FloatingPointError("Non-finite GAE or bootstrap")
        for user in np.unique(arrays["user_id"]):
            idxs = np.flatnonzero(arrays["user_id"] == user)
            idxs = idxs[np.argsort(arrays["slot"][idxs])]
            for idx in dict.fromkeys([*idxs[:3], *idxs[-3:]]):
                boundaries.append({"update": update_id, "user": int(user), "slot": int(arrays["slot"][idx]),
                                   "offset_from_end": int(arrays["slot"][idx] - arrays["slot"][idxs[-1]]),
                                   "value": float(arrays["value"][idx]), "reward": float(arrays["reward"][idx]),
                                   "advantage_raw": float(advantages[idx]), "return": float(returns[idx]),
                                   "bootstrap_value": float(boot[int(user)]), "done": float(arrays["done"][idx])})
        grad_norms.clear()
        stats = original_update(policy, buffer)
        with torch.no_grad():
            obs = torch.as_tensor(arrays["actor_obs"], dtype=torch.float32, device=policy.device)
            latent = torch.as_tensor(arrays["latent_action"], dtype=torch.float32, device=policy.device)
            old_logp = torch.as_tensor(arrays["log_prob"], dtype=torch.float32, device=policy.device)
            log_ratio = policy.actor.distribution(obs).log_prob(latent).sum(-1) - old_logp
            stable_kl = float(((torch.exp(log_ratio) - 1.0) - log_ratio).mean().cpu())
        post_value = policy.value(arrays["critic_obs"], arrays["user_id"])
        updates.append({"update": update_id, "first_slot": int(arrays["slot"].min()), "last_slot": int(arrays["slot"].max()),
                        "critic_mse_before": float(np.mean((returns - arrays["value"]) ** 2)),
                        "critic_mse_after": float(np.mean((returns - post_value) ** 2)),
                        "advantage_mean": float(advantages.mean()), "advantage_sd": float(advantages.std()),
                        "advantage_p01": float(np.quantile(advantages, .01)), "advantage_p99": float(np.quantile(advantages, .99)),
                        "advantage_min": float(advantages.min()), "advantage_max": float(advantages.max()),
                        "approx_kl": stats.approx_kl, "nonnegative_sample_kl": stable_kl,
                        "grad_norm_preclip_mean": float(np.mean(grad_norms)) if grad_norms else 0.,
                        "grad_norm_preclip_max": float(np.max(grad_norms)) if grad_norms else 0.,
                        "bootstrap_mean": float(boot.mean()), "bootstrap_min": float(boot.min()), "bootstrap_max": float(boot.max())})
        # Flush at every update: diagnostics survive an interrupted run.
        write_rows(opts.diagnostic_dir / "updates.csv", updates)
        if update_id % 10 == 0:
            write_rows(opts.diagnostic_dir / "rollout_boundaries.csv", boundaries)
        return stats

    def compute_gae(policy, arrays):
        if fix_bootstrap:
            return gae_with_bootstrap(arrays, policy.cfg.gamma, policy.cfg.gae_lambda, policy._repair_bootstrap)
        return original_gae(policy, arrays)

    original_update = MAPPOPolicy.update
    JointMAPPOEdgeEnv.critic_observations = critic_observations
    JointMAPPOEdgeEnv.joint_step = joint_step
    MAPPOPolicy._compute_gae = compute_gae
    torch.nn.utils.clip_grad_norm_ = clip_grad_norm
    train.evaluate_policy = evaluate
    train.repair_update = repair_update
    changed_source = instrument_train_source(inspect.getsource(train.train_once))
    (opts.diagnostic_dir / "effective_train_once.py").write_text(changed_source, encoding="utf-8")
    exec(compile(changed_source, str(opts.diagnostic_dir / "effective_train_once.py"), "exec"), train.__dict__)
    started = perf_counter()
    try:
        summary = train.train_once(args)
        write_rows(opts.diagnostic_dir / "queue_jumps.csv", queue_rows)
        eval_jump = np.asarray([r["max_queue_jump_service_slots"] for r in queue_rows if r["phase"] == args.evaluation_split_role])
        metrics = {"repair": opts.repair, "critic_action_features_zeroed": fix_critic,
                   "rollout_bootstrap": fix_bootstrap, "final_training_cut_is_time_limit": True,
                   "bootstrap_only_policy_expectation_samples": 16 if opts.repair == "bootstrap" else 0,
                   "elapsed_s": perf_counter() - started, "source_sha256": opts.expected_source,
                   "worker_sha256": sha256(__file__), "helper_sha256": sha256(Path(__file__).with_name("paired_training_repair_core.py")),
                   "evaluation_split_role": args.evaluation_split_role,
                   "mean_queue_jump_service_slots": float(eval_jump.mean()),
                   "p95_queue_jump_service_slots": float(np.quantile(eval_jump, .95)),
                   "p99_queue_jump_service_slots": float(np.quantile(eval_jump, .99)),
                   "primary_summary": summary}
        (opts.diagnostic_dir / "repair_manifest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        write_rows(opts.diagnostic_dir / "queue_jumps.csv", queue_rows)
        write_rows(opts.diagnostic_dir / "updates.csv", updates)
        write_rows(opts.diagnostic_dir / "rollout_boundaries.csv", boundaries)


if __name__ == "__main__":
    main()
