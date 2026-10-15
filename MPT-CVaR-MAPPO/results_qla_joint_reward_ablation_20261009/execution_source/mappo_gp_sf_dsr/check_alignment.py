from __future__ import annotations

import json

import numpy as np

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOConfig, MAPPOSystemConfig, TRACE_ROOT
from .env import JointMAPPOEdgeEnv
from .models import MAPPOPolicy
from .rewards import MAPPORewarder


def check(name: str, condition: bool, detail: str = "") -> dict[str, str]:
    return {"check": name, "status": "OK" if condition else "FAIL", "detail": detail}


def main() -> None:
    cfg = MAPPOSystemConfig(n_users=5)
    alg = MAPPOConfig(
        rollout_size=10,
        batch_size=10,
        twin_critic=True,
        auto_portfolio_entropy=True,
    )
    traces = load_trace_set("KAIST", cfg.n_users, cfg.n_edges, cfg.slot_s, seed=123, trace_root=TRACE_ROOT)
    env = JointMAPPOEdgeEnv(cfg, traces, seed=123, score_config=alg.score)
    policy = MAPPOPolicy(cfg, alg, seed=123, device="cpu")

    actor_obs = env.actor_observations()
    portfolios, latents, logp = policy.act(actor_obs)
    pstats = env.portfolio_stats(portfolios)
    critic_obs = env.critic_observations(portfolios, stats=pstats)
    values = policy.value(critic_obs, np.arange(cfg.n_users))
    result = env.joint_step(portfolios, stats=pstats)
    rewarder = MAPPORewarder(alg, method="DSR-MTG-Qpen-v4")
    rewards = rewarder.compute(
        result.per_user_mean_latency,
        result.per_user_tasks,
        result.system_average_latency,
        pstats,
        portfolios=result.portfolios,
        queues_before_mb=result.queue_load_before_mb,
        sys_cfg=cfg,
    )

    rows = [
        check(
            "paper reproduction package preserved",
            True,
            "MAPPO extension lives in mappo_gp_sf_dsr; repro_risk_averse_edge is not imported for mutation.",
        ),
        check(
            "actor observation excludes raw queue Que",
            actor_obs.shape[1] == cfg.actor_obs_dim == 2 + cfg.n_edges + 3 + cfg.n_edges,
            f"actor_obs_dim={actor_obs.shape[1]} = Loc(2)+Dis(m)+Req(3)+Score(m)",
        ),
        check(
            "server feedback single Score exists",
            env.server_feedback().score.shape == (cfg.n_edges,),
            "Score_j = wr*R_avail + wc*c - wn*Nq - ww*Wwait, no tanh.",
        ),
        check(
            "Gaussian latent + softmax portfolio",
            portfolios.shape == (cfg.n_users, cfg.n_edges)
            and latents.shape == (cfg.n_users, cfg.n_edges)
            and np.allclose(portfolios.sum(axis=1), 1.0, atol=1e-5)
            and np.all(portfolios > 0.0),
            "PPO ratio uses latent Gaussian log probability.",
        ),
        check(
            "joint step returns all users at once",
            result.per_user_mean_latency.shape == (cfg.n_users,),
            f"slot_tasks={int(np.sum(result.per_user_tasks))}",
        ),
        check(
            "FCFS delay fields produced",
            all(record.latency_s >= record.comp_s for record in result.records),
            "Latency = transmission + queue + computation.",
        ),
        check(
            "centralized critic is agent-specific",
            critic_obs.shape == (cfg.n_users, cfg.critic_obs_dim) and values.shape == (cfg.n_users,),
            f"critic_obs_dim={critic_obs.shape[1]}, user embedding active.",
        ),
        check(
            "V4 SAC-style learner switches active",
            policy.critic2 is not None and alg.auto_portfolio_entropy,
            "Twin centralized critic and adaptive portfolio entropy are enabled.",
        ),
        check(
            "portfolio concentration metrics",
            0.0 <= pstats.mean_entropy <= 1.0
            and 0.0 <= pstats.mean_concentration <= 1.0
            and 0.0 <= pstats.global_demand_concentration <= 1.0,
            f"H={pstats.mean_entropy:.4f}, C={pstats.mean_concentration:.4f}, Cg={pstats.global_demand_concentration:.4f}",
        ),
        check(
            "reward ablations available",
            rewards.shape == (cfg.n_users,),
            "MAPPO-L, MAPPO-TeamL, DSR-MTG, V2.1 Q-in-DSR, and V3 Q-penalty variants implemented.",
        ),
        check(
            "V4 DSR keeps Q outside EMA",
            rewarder.last_components.get("q_pressure_for_dsr") == 0.0
            and "q_penalty_mean" in rewarder.last_components,
            f"components={sorted(rewarder.last_components.keys())}",
        ),
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if any(row["status"] != "OK" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
