"""Paper-structure mixed MAPPO adapted to the repository's MEC interface.

The learning structure follows Yao et al., "Multi-Agent Reinforcement
Learning for Task Offloading in Crowd-Edge Computing":

* one recurrent actor per mobile user;
* five independently initialized recurrent centralized critics;
* equal-weight critic-value averaging;
* different critic learning rates; and
* PPO clipping epsilon 0.1.

Only the action interface is adapted: the actor's continuous output is mapped
to the existing edge-portfolio action through softmax.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from repro_risk_averse_edge.data import load_trace_set

from .config import MAPPOSystemConfig, ROOT, TRACE_ROOT, ServerScoreConfig
from .canonical_protocol import install_schedule_for_env
from .env import JointMAPPOEdgeEnv, SlotResult
from .metrics import slot_metrics, summarize_results
from .plots import make_training_plots
from .train import edge_capacities, eval_diagnostics


METHOD = "M2PPO-paper"


@dataclass(frozen=True)
class PaperM2PPOConfig:
    hidden: int = 64
    critic_count: int = 5
    critic_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    critic_lr_scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    update_epochs: int = 4
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    latency_norm: float = 3.0
    log_std_min: float = -1.5
    log_std_max: float = 1.0

    def __post_init__(self) -> None:
        if self.critic_count != 5:
            raise ValueError("The source mixed-MAPPO structure uses exactly five critics.")
        if len(self.critic_weights) != self.critic_count:
            raise ValueError("critic_weights must contain one weight per critic")
        if len(self.critic_lr_scales) != self.critic_count:
            raise ValueError("critic_lr_scales must contain one scale per critic")
        if not np.isclose(sum(self.critic_weights), 1.0):
            raise ValueError("critic_weights must sum to one")
        if not np.isclose(self.clip_eps, 0.1):
            raise ValueError("The source paper fixes PPO clip epsilon to 0.1")


class RecurrentGaussianActor(nn.Module):
    """Paper actor: linear(64), linear(64), RNN(64), output layer."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        log_std_min: float = -1.5,
        log_std_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.rnn = nn.RNNCell(hidden, hidden, nonlinearity="tanh")
        self.output = nn.Linear(hidden, 2 * action_dim)

    def step(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        next_hidden = self.rnn(x, hidden)
        mean, log_std = self.output(next_hidden).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp()), next_hidden

    def sequence(self, observations: torch.Tensor) -> Normal:
        hidden = torch.zeros(1, self.hidden, dtype=observations.dtype, device=observations.device)
        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        for observation in observations:
            dist, hidden = self.step(observation.unsqueeze(0), hidden)
            means.append(dist.mean.squeeze(0))
            stds.append(dist.stddev.squeeze(0))
        return Normal(torch.stack(means), torch.stack(stds))


class RecurrentCentralizedCritic(nn.Module):
    """Independent recurrent critic with the paper's 64-unit structure."""

    def __init__(self, state_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.rnn = nn.RNNCell(hidden, hidden, nonlinearity="tanh")
        self.output = nn.Linear(hidden, 1)

    def step(self, state: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        next_hidden = self.rnn(x, hidden)
        return self.output(next_hidden).squeeze(-1), next_hidden

    def sequence(self, states: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(1, self.hidden, dtype=states.dtype, device=states.device)
        values: list[torch.Tensor] = []
        for state in states:
            value, hidden = self.step(state.unsqueeze(0), hidden)
            values.append(value.squeeze(0))
        return torch.stack(values)


@dataclass
class RecurrentRollout:
    actor_observations: list[np.ndarray]
    critic_states: list[np.ndarray]
    latent_actions: list[np.ndarray]
    log_probs: list[np.ndarray]
    values: list[float]
    rewards: list[float]

    @classmethod
    def empty(cls) -> "RecurrentRollout":
        return cls([], [], [], [], [], [])

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.actor_observations.clear()
        self.critic_states.clear()
        self.latent_actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()


@dataclass
class M2UpdateStats:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    critic_disagreement: float = 0.0


class PaperM2PPOPolicy:
    def __init__(
        self,
        n_users: int,
        actor_obs_dim: int,
        critic_state_dim: int,
        action_dim: int,
        config: PaperM2PPOConfig,
        seed: int,
        device: str | None,
    ) -> None:
        self.n_users = n_users
        self.action_dim = action_dim
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        actors = []
        for user in range(n_users):
            torch.manual_seed(seed + 101 * user)
            actors.append(
                RecurrentGaussianActor(
                    actor_obs_dim,
                    action_dim,
                    config.hidden,
                    config.log_std_min,
                    config.log_std_max,
                )
            )
        self.actors = nn.ModuleList(actors).to(self.device)

        critics = []
        for index in range(config.critic_count):
            torch.manual_seed(seed + 10_000 + 997 * index)
            critics.append(RecurrentCentralizedCritic(critic_state_dim, config.hidden))
        self.critics = nn.ModuleList(critics).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate) for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(
                critic.parameters(),
                lr=config.critic_learning_rate * config.critic_lr_scales[index],
            )
            for index, critic in enumerate(self.critics)
        ]
        self.actor_hidden: list[torch.Tensor] = []
        self.critic_hidden: list[torch.Tensor] = []
        self.reset_hidden()

    def reset_hidden(self) -> None:
        self.actor_hidden = [
            torch.zeros(1, self.cfg.hidden, device=self.device) for _ in range(self.n_users)
        ]
        self.critic_hidden = [
            torch.zeros(1, self.cfg.hidden, device=self.device) for _ in range(self.cfg.critic_count)
        ]

    def act(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        portfolios: list[np.ndarray] = []
        latents: list[np.ndarray] = []
        log_probs: list[float] = []
        with torch.no_grad():
            for user, actor in enumerate(self.actors):
                obs = torch.as_tensor(observations[user : user + 1], dtype=torch.float32, device=self.device)
                dist, hidden = actor.step(obs, self.actor_hidden[user])
                self.actor_hidden[user] = hidden.detach()
                latent = dist.mean if deterministic else dist.sample()
                portfolio = torch.softmax(latent, dim=-1)
                portfolios.append(portfolio.squeeze(0).cpu().numpy().astype(np.float32))
                latents.append(latent.squeeze(0).cpu().numpy().astype(np.float32))
                log_probs.append(float(dist.log_prob(latent).sum(dim=-1).item()))
        return (
            np.asarray(portfolios, dtype=np.float32),
            np.asarray(latents, dtype=np.float32),
            np.asarray(log_probs, dtype=np.float32),
        )

    def value(self, state: np.ndarray) -> tuple[float, float]:
        state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=self.device)
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for index, critic in enumerate(self.critics):
                value, hidden = critic.step(state_t, self.critic_hidden[index])
                self.critic_hidden[index] = hidden.detach()
                values.append(value.squeeze(0))
            stacked = torch.stack(values)
            weights = torch.as_tensor(self.cfg.critic_weights, dtype=torch.float32, device=self.device)
            averaged = torch.sum(weights * stacked)
            disagreement = torch.std(stacked, unbiased=False)
        return float(averaged.item()), float(disagreement.item())

    def update(self, rollout: RecurrentRollout) -> M2UpdateStats:
        if len(rollout) < 2:
            return M2UpdateStats()
        actor_obs = torch.as_tensor(np.asarray(rollout.actor_observations), dtype=torch.float32, device=self.device)
        critic_states = torch.as_tensor(np.asarray(rollout.critic_states), dtype=torch.float32, device=self.device)
        latent_actions = torch.as_tensor(np.asarray(rollout.latent_actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(rollout.log_probs), dtype=torch.float32, device=self.device)
        advantages_np, returns_np = self._gae(
            np.asarray(rollout.rewards, dtype=np.float32),
            np.asarray(rollout.values, dtype=np.float32),
        )
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        for user, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            for _ in range(self.cfg.update_epochs):
                dist = actor.sequence(actor_obs[:, user, :])
                new_log_prob = dist.log_prob(latent_actions[:, user, :]).sum(dim=-1)
                ratio = torch.exp(new_log_prob - old_log_probs[:, user])
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * advantages
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = -torch.min(unclipped, clipped).mean() - self.cfg.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                actor_losses.append(float(loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                with torch.no_grad():
                    kls.append(float((old_log_probs[:, user] - new_log_prob).mean().item()))

        critic_losses: list[float] = []
        critic_predictions: list[torch.Tensor] = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            for _ in range(self.cfg.update_epochs):
                predicted = critic.sequence(critic_states)
                loss = nn.functional.mse_loss(predicted, returns)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
                critic_losses.append(float(loss.detach().item()))
            with torch.no_grad():
                critic_predictions.append(critic.sequence(critic_states))
        disagreement = torch.stack(critic_predictions).std(dim=0, unbiased=False).mean()

        rollout.clear()
        self.reset_hidden()
        return M2UpdateStats(
            actor_loss=float(np.mean(actor_losses)),
            critic_loss=float(np.mean(critic_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(kls)),
            critic_disagreement=float(disagreement.item()),
        )

    def _gae(self, rewards: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            next_value = values[index + 1] if index + 1 < len(values) else 0.0
            delta = rewards[index] + self.cfg.gamma * next_value - values[index]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[index] = gae
        return advantages, advantages + values

    def save(self, path: Path, metadata: dict[str, object]) -> None:
        torch.save(
            {
                "actors": [actor.state_dict() for actor in self.actors],
                "critics": [critic.state_dict() for critic in self.critics],
                "paper_config": asdict(self.cfg),
                "metadata": metadata,
            },
            path,
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_environment(args: argparse.Namespace, seed: int) -> tuple[JointMAPPOEdgeEnv, MAPPOSystemConfig, np.ndarray]:
    score = ServerScoreConfig(
        resource_weight=args.score_resource_weight,
        capacity_weight=args.score_capacity_weight,
        task_count_weight=args.score_task_count_weight,
        wait_weight=args.score_wait_weight,
    )
    system = MAPPOSystemConfig(
        n_users=args.users,
        n_edges=args.n_edges,
        slot_s=args.slot_s,
        edge_compute_mb_s=args.edge_compute_mb_s * args.edge_compute_scale,
        queue_upper_mb=args.queue_upper_mb,
        max_requests_per_user=args.max_requests_per_user,
        task_compute_mb=args.task_compute_mb * args.task_compute_scale,
        task_package_mb=args.task_package_mb,
        max_supported_users=max(args.users, args.max_supported_users),
    )
    capacities = edge_capacities(system.n_edges, system.edge_compute_mb_s, args.capacity_mode, args.seed)
    traces = load_trace_set(
        args.dataset,
        args.users,
        system.n_edges,
        system.slot_s,
        args.seed,
        trace_root=Path(args.trace_root),
    )
    environment = JointMAPPOEdgeEnv(
        system,
        traces,
        request_distribution=args.distribution,
        seed=seed,
        queue_constraint=args.queue_constraint,
        score_config=score,
        feedback_delay=args.feedback_delay,
        broadcast_period=args.broadcast_period,
        poisson_lambda=args.poisson_lambda,
        burst_prob=args.burst_prob,
        burst_multiplier=args.burst_multiplier,
        edge_capacities_mb_s=capacities,
        background_load=args.background_load,
    )
    is_training = int(seed) == int(args.seed)
    evaluation_role = str(getattr(args, "evaluation_split_role", "test"))
    install_schedule_for_env(
        environment,
        root=Path(
            getattr(
                args,
                "canonical_schedule_root",
                ROOT / "results_qla_canonical_v1" / "shared_schedules",
            )
        ),
        distribution=args.distribution,
        role="train" if is_training else evaluation_role,
        seed=int(seed),
        n_steps=int(args.train_steps if is_training else args.test_steps),
        max_feedback_delay=int(getattr(args, "canonical_max_feedback_delay", 8)),
    )
    return environment, system, capacities


def _critic_state(environment: JointMAPPOEdgeEnv) -> np.ndarray:
    """Centralized state before the current action.

    The shared environment's critic-observation helper also expects portfolio
    statistics.  A fixed uniform portfolio keeps those six fields constant so
    the critic cannot leak the current action into the paper's state-value
    estimate.
    """
    neutral = np.full(
        (environment.cfg.n_users, environment.cfg.n_edges),
        1.0 / environment.cfg.n_edges,
        dtype=np.float32,
    )
    return environment.critic_observations(neutral).reshape(-1).astype(np.float32)


def _record_latencies(rows: list[dict[str, object]], result: SlotResult, phase: str, slot: int) -> None:
    for record in result.records:
        rows.append(
            {
                "phase": phase,
                "slot": slot,
                "step": slot * result.portfolios.shape[0],
                "user": record.user,
                "task": record.task,
                "edge": record.edge,
                "latency": record.latency_s,
                "tran_s": record.tran_s,
                "queue_s": record.queue_s,
                "comp_s": record.comp_s,
            }
        )


def _evaluate(
    policy: PaperM2PPOPolicy,
    environment: JointMAPPOEdgeEnv,
    steps: int,
) -> list[SlotResult]:
    results: list[SlotResult] = []
    environment.reset()
    policy.reset_hidden()
    for _ in range(steps):
        observations = environment.actor_observations()
        started = perf_counter()
        portfolios, _, _ = policy.act(observations, deterministic=True)
        decision_latency = perf_counter() - started
        results.append(environment.joint_step(portfolios, decision_latency_s=decision_latency))
    return results


def train_once(args: argparse.Namespace) -> dict[str, object]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    environment, system, capacities = _build_environment(args, args.seed)
    paper_config = PaperM2PPOConfig(
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        critic_lr_scales=tuple(args.critic_lr_scales),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        entropy_coef=args.entropy_coef,
        rollout_steps=args.rollout_steps,
        latency_norm=args.latency_norm,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    )
    critic_state_dim = _critic_state(environment).size
    policy = PaperM2PPOPolicy(
        system.n_users,
        system.actor_obs_dim,
        critic_state_dim,
        system.n_edges,
        paper_config,
        args.seed,
        args.device,
    )
    output = Path(args.output_dir) / args.dataset / f"{args.users}u" / args.distribution
    output.mkdir(parents=True, exist_ok=True)
    learning_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    rollout = RecurrentRollout.empty()
    last_update = M2UpdateStats()
    last_disagreement = 0.0
    started_all = perf_counter()
    environment.reset()
    policy.reset_hidden()

    for slot in range(1, args.train_steps + 1):
        observations = environment.actor_observations()
        critic_state = _critic_state(environment)
        value, disagreement = policy.value(critic_state)
        action_started = perf_counter()
        portfolios, latents, log_probs = policy.act(observations, deterministic=False)
        result = environment.joint_step(
            portfolios,
            decision_latency_s=perf_counter() - action_started,
        )
        reward = -result.system_average_latency / max(paper_config.latency_norm, 1e-8)
        rollout.actor_observations.append(observations.copy())
        rollout.critic_states.append(critic_state)
        rollout.latent_actions.append(latents.copy())
        rollout.log_probs.append(log_probs.copy())
        rollout.values.append(value)
        rollout.rewards.append(float(reward))
        last_disagreement = disagreement
        _record_latencies(latency_rows, result, "train", slot)

        if len(rollout) >= paper_config.rollout_steps:
            last_update = policy.update(rollout)
            environment.reset()

        if slot == 1 or slot % args.log_interval == 0 or slot == args.train_steps:
            row: dict[str, object] = {
                "slot": slot,
                "step": slot * system.n_users,
                "mean_reward": reward,
                "actor_loss": last_update.actor_loss,
                "critic_loss": last_update.critic_loss,
                "entropy_loss_term": last_update.entropy,
                "approx_kl": last_update.approx_kl,
                "critic_disagreement": last_update.critic_disagreement or last_disagreement,
                "elapsed_s": perf_counter() - started_all,
            }
            row.update(slot_metrics(result))
            learning_rows.append(row)

    if len(rollout) >= 2:
        last_update = policy.update(rollout)
        if learning_rows:
            learning_rows[-1].update(
                {
                    "actor_loss": last_update.actor_loss,
                    "critic_loss": last_update.critic_loss,
                    "entropy_loss_term": last_update.entropy,
                    "approx_kl": last_update.approx_kl,
                    "critic_disagreement": last_update.critic_disagreement,
                }
            )

    evaluation_environment, _, _ = _build_environment(args, args.seed + 10_000)
    evaluation_results = _evaluate(policy, evaluation_environment, args.test_steps)
    for slot, result in enumerate(evaluation_results, start=1):
        _record_latencies(latency_rows, result, "test", slot)
    summary: dict[str, object] = summarize_results(evaluation_results)
    summary.update(eval_diagnostics(evaluation_results, capacities.astype(np.float64), system.slot_s))
    summary.update(
        {
            "method": METHOD,
            "source_paper_doi": "10.1109/TMC.2025.3531793",
            "architecture": "per-agent Linear64-Linear64-RNN64 actor; 5 independent RNN64 critics",
            "action_adapter": "Gaussian latent to MEC edge portfolio by softmax",
            "reward_adapter": "shared negative system average latency",
            "critic_weights": json.dumps(paper_config.critic_weights),
            "critic_lr_scales": json.dumps(paper_config.critic_lr_scales),
            "clip_eps": paper_config.clip_eps,
            "dataset": args.dataset,
            "users": args.users,
            "distribution": args.distribution,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
        }
    )
    _write_csv(output / f"learning_{METHOD}.csv", learning_rows)
    _write_csv(output / f"latency_samples_{METHOD}.csv", latency_rows)
    _write_csv(output / f"summary_{METHOD}.csv", [summary])
    with (output / f"config_{METHOD}.json").open("w", encoding="utf-8") as handle:
        json.dump({"cli": vars(args), "paper_structure": asdict(paper_config)}, handle, indent=2, ensure_ascii=False)
    policy.save(output / f"checkpoint_{METHOD}.pt", {"cli": vars(args)})
    try:
        make_training_plots(output, METHOD)
    except Exception:
        pass
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-structure mixed MAPPO on the current MEC environment.")
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_m2ppo_paper"))
    parser.add_argument("--queue-constraint", default="resample", choices=["resample", "penalty"])
    parser.add_argument("--feedback-delay", type=int, default=0)
    parser.add_argument("--broadcast-period", type=int, default=2)
    parser.add_argument("--slot-s", type=float, default=1.0)
    parser.add_argument("--n-edges", type=int, default=35)
    parser.add_argument("--edge-compute-mb-s", type=float, default=63.0)
    parser.add_argument("--queue-upper-mb", type=float, default=630.0)
    parser.add_argument("--task-compute-mb", type=float, default=15.0)
    parser.add_argument("--task-package-mb", type=float, default=270.0 / 1024.0)
    parser.add_argument("--max-requests-per-user", type=int, default=7)
    parser.add_argument("--max-supported-users", type=int, default=80)
    parser.add_argument("--poisson-lambda", type=float, default=3.5)
    parser.add_argument("--burst-prob", type=float, default=0.1)
    parser.add_argument("--burst-multiplier", type=float, default=2.0)
    parser.add_argument("--background-load", type=float, default=0.0)
    parser.add_argument("--capacity-mode", default="homogeneous")
    parser.add_argument("--edge-compute-scale", type=float, default=1.0)
    parser.add_argument("--task-compute-scale", type=float, default=1.0)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-lr-scales", type=float, nargs=5, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--latency-norm", type=float, default=3.0)
    parser.add_argument("--log-std-min", type=float, default=-1.5)
    parser.add_argument("--log-std-max", type=float, default=1.0)
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train_once(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
