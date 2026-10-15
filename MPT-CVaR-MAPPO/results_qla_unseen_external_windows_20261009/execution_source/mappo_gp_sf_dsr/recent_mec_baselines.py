"""Recent MEC baselines adapted to the shared stale-feedback environment.

The implementations deliberately separate paper-faithful algorithm structure
from environment adapters:

* ``QECO-adapted`` keeps per-user D3QN agents, a workload-history LSTM,
  replay buffers, Double-Q targets, dueling value/advantage heads and the
  paper's QoE-style completion/delay/energy reward.
* ``Lyapunov-Q-stale`` is explicitly a LyFOTO-inspired drift-plus-penalty
  baseline, not a reproduction of LyFOTO's energy-harvesting/FL system.
* ``TODAuction-MAB-adapted`` keeps cyclic basis-arm exploration, discounted
  empirical delay learning and distributed auction matching over server
  resource groups.

None of these baselines uses AgeGate, CVaR, portfolio concentration, or the
proposed method's Q-pressure reward term.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .canonical_protocol import write_run_manifest
from .config import ROOT, TRACE_ROOT
from .env import JointMAPPOEdgeEnv, SlotResult
from .m2ppo_paper import _build_environment
from .metrics import finalize_evaluation, phase_diagnostics, slot_metrics
from .plots import make_training_plots


QECO = "QECO-adapted"
QECO_LOAD_TOPK = "QECO-load-topk"
LYAPUNOV = "Lyapunov-Q-stale"
LYAPUNOV_LOCAL_STALE = "Lyapunov-Q-local-stale"
LYAPUNOV_LOCAL_STALE_TUNED = "Lyapunov-Q-local-stale-tuned"
LYAPUNOV_FRESH = "Lyapunov-Q-fresh"
LYAPUNOV_STALE_TUNED = "Lyapunov-Q-stale-tuned"
LYAPUNOV_TAIL = "Lyapunov-Q-tail"
MAB = "TODAuction-MAB-adapted"
MAB_TAIL = "TODAuction-MAB-tail"
SAFETAIL = "SafeTail-redundancy-adapted"
METHODS = (
    QECO,
    QECO_LOAD_TOPK,
    LYAPUNOV,
    LYAPUNOV_LOCAL_STALE,
    LYAPUNOV_LOCAL_STALE_TUNED,
    LYAPUNOV_FRESH,
    LYAPUNOV_STALE_TUNED,
    LYAPUNOV_TAIL,
    MAB,
    MAB_TAIL,
    SAFETAIL,
)
# Methods that route through the local-candidate stale drift-plus-penalty
# controller. The "tuned" variant shares the same algorithmic core as the
# paper-facing local-stale baseline; only (candidates, V, temperature) differ
# and are selected on a separate validation grid (see run_lyapunov_local_stale_tuning).
LYAPUNOV_LOCAL_STALE_FAMILY = {LYAPUNOV_LOCAL_STALE, LYAPUNOV_LOCAL_STALE_TUNED}


def _device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _record_latencies(
    rows: list[dict[str, object]], result: SlotResult, phase: str, slot: int
) -> None:
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
                "dropped": int(record.dropped),
            }
        )


def _output_dir(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir) / args.dataset / f"{args.users}u" / args.distribution
    output.mkdir(parents=True, exist_ok=True)
    return output


def _one_hot(actions: np.ndarray, n_edges: int) -> np.ndarray:
    portfolios = np.zeros((actions.size, n_edges), dtype=np.float32)
    portfolios[np.arange(actions.size), actions.astype(np.int64)] = 1.0
    return portfolios


def _save_outputs(
    args: argparse.Namespace,
    method: str,
    learning_rows: list[dict[str, object]],
    latency_rows: list[dict[str, object]],
    summary: dict[str, object],
    extra_config: dict[str, object],
) -> None:
    output = _output_dir(args)
    _write_csv(output / f"learning_{method}.csv", learning_rows)
    _write_csv(output / f"latency_samples_{method}.csv", latency_rows)
    _write_csv(output / f"summary_{method}.csv", [summary])
    config_path = output / f"config_{method}.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"cli": vars(args), "implementation": extra_config},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    write_run_manifest(
        output_dir=output,
        method=method,
        args=args,
        summary=summary,
        config_path=config_path,
    )
    try:
        make_training_plots(output, method)
    except Exception:
        pass


class QECONetwork(nn.Module):
    """PyTorch equivalent of the authors' LSTM + dueling Double-DQN."""

    def __init__(
        self,
        state_dim: int,
        workload_dim: int,
        action_dim: int,
        hidden: int,
        lstm_hidden: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(workload_dim, lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(state_dim + lstm_hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor, workload_history: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(workload_history)
        features = torch.cat([state, recurrent[:, -1]], dim=-1)
        features = torch.relu(self.fc1(features))
        features = torch.relu(self.fc2(features))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class QECOReplay:
    def __init__(self, capacity: int) -> None:
        self.data: deque[tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def add(
        self,
        state: np.ndarray,
        history: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_history: np.ndarray,
        done: bool,
    ) -> None:
        self.data.append(
            (
                state.copy(),
                history.copy(),
                int(action),
                float(reward),
                next_state.copy(),
                next_history.copy(),
                float(done),
            )
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        indices = rng.integers(0, len(self.data), size=batch_size)
        batch = [self.data[int(index)] for index in indices]
        states, histories, actions, rewards, next_states, next_histories, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(histories, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(next_histories, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class QECOAgent:
    online: QECONetwork
    target: QECONetwork
    optimizer: torch.optim.Optimizer
    replay: QECOReplay
    updates: int = 0


def _qeco_state(environment: JointMAPPOEdgeEnv) -> tuple[np.ndarray, np.ndarray]:
    """Return paper-mapped local state and delayed/broadcast workload vector."""

    states = environment.actor_observations().astype(np.float32)
    feedback = environment.feedback_buffer.get(environment.feedback_delay)
    workload = feedback.queue_task_count_norm.astype(np.float32)
    return states, workload


def _qeco_agents(
    args: argparse.Namespace,
    state_dim: int,
    workload_dim: int,
    device: torch.device,
) -> list[QECOAgent]:
    agents: list[QECOAgent] = []
    for user in range(args.users):
        torch.manual_seed(args.seed * 10_000 + user)
        online = QECONetwork(
            state_dim,
            workload_dim,
            args.n_edges,
            args.qeco_hidden,
            args.qeco_lstm_hidden,
        ).to(device)
        target = copy.deepcopy(online).to(device)
        optimizer = torch.optim.RMSprop(online.parameters(), lr=args.learning_rate)
        agents.append(QECOAgent(online, target, optimizer, QECOReplay(args.replay_size)))
    return agents


def _qeco_actions(
    agents: list[QECOAgent],
    states: np.ndarray,
    histories: np.ndarray,
    greedy_probability: float,
    rng: np.random.Generator,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    actions = np.zeros(len(agents), dtype=np.int64)
    with torch.no_grad():
        for user, agent in enumerate(agents):
            if not deterministic and rng.random() >= greedy_probability:
                actions[user] = int(rng.integers(0, histories.shape[-1]))
                continue
            q_values = agent.online(
                torch.as_tensor(states[user : user + 1], device=device),
                torch.as_tensor(histories[user : user + 1], device=device),
            )
            actions[user] = int(torch.argmax(q_values, dim=-1).item())
    return actions


def _qeco_action_portfolios(
    agents: list[QECOAgent],
    states: np.ndarray,
    histories: np.ndarray,
    environment: JointMAPPOEdgeEnv,
    greedy_probability: float,
    rng: np.random.Generator,
    device: torch.device,
    deterministic: bool,
    topk: int,
    temperature: float,
    load_mask: bool,
    mask_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return QECO primary discrete actions and an optional soft top-k adapter.

    The learned QECO object remains a per-user D3QN over edge actions.  The
    stronger adapter only changes how the selected Q-values are exposed to the
    shared portfolio simulator: it can mask obviously congested delayed-state
    edges and spread probability mass over the top-k remaining edges.  It does
    not use AgeGate, CVaR reward, Q-pressure reward, or the proposed global
    concentration penalty.
    """

    n_users = len(agents)
    n_edges = histories.shape[-1]
    portfolios = np.zeros((n_users, n_edges), dtype=np.float32)
    actions = np.zeros(n_users, dtype=np.int64)
    feedback = environment.feedback_buffer.get(environment.feedback_delay)
    mask = np.ones(n_edges, dtype=bool)
    if load_mask:
        load_score = feedback.wait_norm + feedback.queue_task_count_norm
        threshold = float(np.quantile(load_score, np.clip(mask_quantile, 0.05, 1.0)))
        mask = load_score <= threshold
        if not np.any(mask):
            mask[:] = True

    with torch.no_grad():
        for user, agent in enumerate(agents):
            exploratory = (not deterministic) and rng.random() >= greedy_probability
            if exploratory:
                allowed = np.flatnonzero(mask)
                chosen = int(rng.choice(allowed if allowed.size else np.arange(n_edges)))
                actions[user] = chosen
                portfolios[user, chosen] = 1.0
                continue
            q_values = (
                agent.online(
                    torch.as_tensor(states[user : user + 1], device=device),
                    torch.as_tensor(histories[user : user + 1], device=device),
                )
                .detach()
                .cpu()
                .numpy()[0]
                .astype(np.float64)
            )
            masked = q_values.copy()
            masked[~mask] = -1e12
            if not np.any(np.isfinite(masked)) or float(np.max(masked)) < -1e11:
                masked = q_values
            actions[user] = int(np.argmax(masked))
            k = max(1, min(int(topk), n_edges))
            top = np.argsort(masked)[-k:]
            top = top[np.isfinite(masked[top]) & (masked[top] > -1e11)]
            if top.size == 0:
                top = np.asarray([actions[user]], dtype=np.int64)
            logits = masked[top] / max(float(temperature), 1e-8)
            logits -= float(np.max(logits))
            weights = np.exp(np.clip(logits, -60.0, 0.0))
            weights = weights / max(float(weights.sum()), 1e-12)
            portfolios[user, top] = weights.astype(np.float32)
    return actions, portfolios


def _qeco_rewards(
    result: SlotResult,
    battery_modes: np.ndarray,
    deadline_s: float,
) -> np.ndarray:
    rewards = np.zeros(battery_modes.size, dtype=np.float32)
    for user in range(battery_modes.size):
        records = [record for record in result.records if record.user == user]
        if not records:
            continue
        latency = float(np.mean([record.training_latency_s for record in records]))
        transmission = float(np.mean([record.tran_s for record in records]))
        completion = float(
            np.mean([(not record.dropped) and record.latency_s <= deadline_s for record in records])
        )
        delay_cost = min(latency / max(deadline_s, 1e-8), 1.0)
        energy_cost = min(transmission / max(deadline_s, 1e-8), 1.0)
        delay_weight = float(battery_modes[user])
        rewards[user] = completion - (
            delay_weight * delay_cost + (1.0 - delay_weight) * energy_cost
        )
    return rewards


def _qeco_update(
    agent: QECOAgent,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if len(agent.replay) < args.batch_size:
        return 0.0
    states, histories, actions, rewards, next_states, next_histories, dones = agent.replay.sample(
        args.batch_size, rng
    )
    state_t = torch.as_tensor(states, device=device)
    history_t = torch.as_tensor(histories, device=device)
    action_t = torch.as_tensor(actions, device=device)
    reward_t = torch.as_tensor(rewards, device=device)
    next_state_t = torch.as_tensor(next_states, device=device)
    next_history_t = torch.as_tensor(next_histories, device=device)
    done_t = torch.as_tensor(dones, device=device)

    predicted = agent.online(state_t, history_t).gather(1, action_t[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = torch.argmax(agent.online(next_state_t, next_history_t), dim=-1)
        next_values = agent.target(next_state_t, next_history_t).gather(
            1, next_actions[:, None]
        ).squeeze(1)
        targets = reward_t + args.gamma * (1.0 - done_t) * next_values
    loss = torch.mean((predicted - targets) ** 2)
    agent.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.optimizer.step()
    agent.updates += 1
    if agent.updates % args.target_update_interval == 0:
        agent.target.load_state_dict(agent.online.state_dict())
    return float(loss.detach().cpu())


def _run_qeco(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cpu" or (args.device is None and not torch.cuda.is_available()):
        torch.set_num_threads(max(1, args.torch_threads))
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    environment, system, _ = _build_environment(args, args.seed)
    states, workload = _qeco_state(environment)
    agents = _qeco_agents(args, states.shape[1], workload.size, device)
    histories = np.repeat(workload[None, None, :], args.users * args.history_steps, axis=0).reshape(
        args.users, args.history_steps, workload.size
    )
    battery_modes = rng.choice(np.asarray([0.25, 0.50, 0.75], dtype=np.float32), size=args.users)
    learning_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    greedy_probability = 0.0
    started_all = perf_counter()
    method = QECO_LOAD_TOPK if args.method == QECO_LOAD_TOPK else QECO
    use_strong_adapter = method == QECO_LOAD_TOPK

    for slot in range(1, args.train_steps + 1):
        if slot > 1 and (slot - 1) % args.qeco_episode_steps == 0:
            environment.reset()
            states, workload = _qeco_state(environment)
            histories = np.repeat(
                workload[None, None, :], args.users * args.history_steps, axis=0
            ).reshape(args.users, args.history_steps, workload.size)
        action_started = perf_counter()
        if use_strong_adapter:
            actions, portfolios = _qeco_action_portfolios(
                agents,
                states,
                histories,
                environment,
                greedy_probability,
                rng,
                device,
                deterministic=False,
                topk=args.qeco_topk,
                temperature=args.qeco_topk_temperature,
                load_mask=args.qeco_load_mask,
                mask_quantile=args.qeco_mask_quantile,
            )
        else:
            actions = _qeco_actions(
                agents, states, histories, greedy_probability, rng, device, deterministic=False
            )
            portfolios = _one_hot(actions, args.n_edges)
        result = environment.joint_step(
            portfolios,
            decision_latency_s=perf_counter() - action_started,
        )
        rewards = _qeco_rewards(result, battery_modes, args.deadline_s)
        next_states, next_workload = _qeco_state(environment)
        next_histories = np.concatenate(
            [histories[:, 1:], np.repeat(next_workload[None, None, :], args.users, axis=0)],
            axis=1,
        )
        done = slot % args.qeco_episode_steps == 0
        losses: list[float] = []
        for user, agent in enumerate(agents):
            agent.replay.add(
                states[user],
                histories[user],
                int(actions[user]),
                float(rewards[user]),
                next_states[user],
                next_histories[user],
                done,
            )
            if slot % args.train_every == 0:
                losses.append(_qeco_update(agent, args, rng, device))
        states, histories = next_states, next_histories
        greedy_probability = min(
            args.epsilon_max, greedy_probability + args.epsilon_increment
        )
        _record_latencies(latency_rows, result, "train", slot)
        if slot == 1 or slot % args.log_interval == 0 or slot == args.train_steps:
            row: dict[str, object] = {
                "slot": slot,
                "step": slot * system.n_users,
                "mean_reward": float(np.mean(rewards)),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "greedy_probability": greedy_probability,
                "elapsed_s": perf_counter() - started_all,
            }
            row.update(slot_metrics(result))
            learning_rows.append(row)

    evaluation_environment, _, _ = _build_environment(args, args.seed + 10_000)
    eval_states, eval_workload = _qeco_state(evaluation_environment)
    eval_histories = np.repeat(
        eval_workload[None, None, :], args.users * args.history_steps, axis=0
    ).reshape(args.users, args.history_steps, eval_workload.size)
    evaluation_results: list[SlotResult] = []
    for slot in range(1, args.test_steps + 1):
        action_started = perf_counter()
        if use_strong_adapter:
            actions, portfolios = _qeco_action_portfolios(
                agents,
                eval_states,
                eval_histories,
                evaluation_environment,
                1.0,
                rng,
                device,
                deterministic=True,
                topk=args.qeco_topk,
                temperature=args.qeco_topk_temperature,
                load_mask=args.qeco_load_mask,
                mask_quantile=args.qeco_mask_quantile,
            )
        else:
            actions = _qeco_actions(
                agents, eval_states, eval_histories, 1.0, rng, device, deterministic=True
            )
            portfolios = _one_hot(actions, args.n_edges)
        result = evaluation_environment.joint_step(
            portfolios,
            decision_latency_s=perf_counter() - action_started,
        )
        evaluation_results.append(result)
        _record_latencies(latency_rows, result, "test", slot)
        eval_states, eval_workload = _qeco_state(evaluation_environment)
        eval_histories = np.concatenate(
            [
                eval_histories[:, 1:],
                np.repeat(eval_workload[None, None, :], args.users, axis=0),
            ],
            axis=1,
        )

    summary, _ = finalize_evaluation(evaluation_environment, evaluation_results, system)
    summary.update(phase_diagnostics(evaluation_results, system))
    summary.update(
        {
            "method": method,
            "source_paper_doi": "10.1109/TNSE.2025.3556809",
            "reproduction_level": (
                "stronger structure-faithful environment adaptation"
                if use_strong_adapter
                else "structure-faithful environment adaptation"
            ),
            "architecture": "per-user D3QN with 10-step LSTM workload history",
            "action_adapter": (
                "D3QN edge values exposed as load-masked soft top-k portfolio"
                if use_strong_adapter
                else "paper local-or-edge action mapped to one edge per user-slot"
            ),
            "reward_adapter": "QoE completion reward minus battery-adaptive delay/energy cost",
            "uses_agegate": False,
            "uses_cvar_reward": False,
            "uses_crowding_penalty": False,
            "dataset": args.dataset,
            "users": args.users,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
        }
    )
    if environment.canonical_schedule is not None:
        summary["train_split_id"] = environment.canonical_schedule.schedule_id
        summary["train_schedule_sha256"] = environment.canonical_schedule.content_sha256
    _save_outputs(
        args,
        method,
        learning_rows,
        latency_rows,
        summary,
        {
            "paper_network": "LSTM20 + FC20 + FC20 + dueling heads",
            "paper_optimizer": "RMSprop",
            "paper_defaults": {
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "target_update_interval": args.target_update_interval,
                "replay_size": args.replay_size,
                "history_steps": args.history_steps,
            },
            "adapter_limitations": [
                "shared environment has no local-execution action",
                (
                    "multiple arrivals in one user-slot share one soft top-k portfolio"
                    if use_strong_adapter
                    else "multiple arrivals in one user-slot share one categorical action"
                ),
                "task outcome is available at simulated slot completion",
            ],
            "strong_adapter": {
                "enabled": use_strong_adapter,
                "topk": args.qeco_topk,
                "temperature": args.qeco_topk_temperature,
                "load_mask": args.qeco_load_mask,
                "mask_quantile": args.qeco_mask_quantile,
            },
        },
    )
    checkpoint_path = _output_dir(args) / f"checkpoint_{method}.pt"
    torch.save(
        {
            "online_models": [agent.online.state_dict() for agent in agents],
            "config": vars(args),
        },
        checkpoint_path,
    )
    write_run_manifest(
        output_dir=_output_dir(args),
        method=method,
        args=args,
        summary=summary,
        config_path=_output_dir(args) / f"config_{method}.json",
        checkpoint_path=checkpoint_path,
    )
    return summary


def _softmax_negative(cost: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -np.asarray(cost, dtype=np.float64) / max(temperature, 1e-8)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(np.clip(scaled, -60.0, 0.0))
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _lyapunov_portfolios(
    environment: JointMAPPOEdgeEnv,
    tradeoff_v: float,
    temperature: float,
    fresh_state: bool = False,
    tail_penalty: np.ndarray | None = None,
    tail_weight: float = 0.0,
) -> np.ndarray:
    """Drift-plus-delay rule using stale or fresh server feedback."""

    feedback = environment.server_feedback() if fresh_state else environment.feedback_buffer.get(environment.feedback_delay)
    locations = environment._locations()
    costs = np.zeros((environment.cfg.n_users, environment.cfg.n_edges), dtype=np.float64)
    normalized_load = feedback.wait_s / max(environment.cfg.slot_s, 1e-8)
    task_load = environment.cfg.task_compute_mb / np.maximum(
        environment.edge_capacities_mb_s.astype(np.float64), 1e-8
    )
    for user in range(environment.cfg.n_users):
        communication = np.asarray(
            [
                environment._transmission_delay_s(locations[user], edge)
                for edge in range(environment.cfg.n_edges)
            ],
            dtype=np.float64,
        )
        estimated_delay = communication + task_load + feedback.wait_s
        costs[user] = tradeoff_v * estimated_delay + normalized_load * task_load
        if tail_penalty is not None and tail_weight > 0.0:
            costs[user] += tail_weight * np.asarray(tail_penalty, dtype=np.float64)
    return _softmax_negative(costs, temperature)


def _lyapunov_local_stale_portfolios(
    environment: JointMAPPOEdgeEnv,
    tradeoff_v: float,
    temperature: float,
    candidate_edges: int,
) -> np.ndarray:
    """Local-candidate stale drift-plus-penalty rule.

    This is the paper-facing LyFOTO-inspired adapter: it uses only stale
    broadcast feedback, restricts each user to nearby edge candidates, and does
    not use empirical tail memory or fresh queue state.
    """

    feedback = environment.feedback_buffer.get(environment.feedback_delay)
    locations = environment._locations()
    n_users = environment.cfg.n_users
    n_edges = environment.cfg.n_edges
    portfolios = np.zeros((n_users, n_edges), dtype=np.float32)
    normalized_load = feedback.wait_s / max(environment.cfg.slot_s, 1e-8)
    service_time = environment.cfg.task_compute_mb / np.maximum(
        environment.edge_capacities_mb_s.astype(np.float64), 1e-8
    )
    k = max(1, min(int(candidate_edges), n_edges))
    for user in range(n_users):
        distances = np.linalg.norm(environment.traces.edge_positions - locations[user][None, :], axis=1)
        candidates = np.argsort(distances)[:k]
        communication = np.asarray(
            [environment._transmission_delay_s(locations[user], int(edge)) for edge in candidates],
            dtype=np.float64,
        )
        estimated_delay = communication + service_time[candidates] + feedback.wait_s[candidates]
        drift = normalized_load[candidates] * service_time[candidates]
        costs = float(tradeoff_v) * estimated_delay + drift
        local_weights = _softmax_negative(costs[None, :], temperature)[0]
        portfolios[user, candidates] = local_weights
    return portfolios


def _edge_tail_penalty(
    latency_sum: np.ndarray,
    latency_sq_sum: np.ndarray,
    latency_count: np.ndarray,
) -> np.ndarray:
    mean = np.divide(
        latency_sum,
        np.maximum(latency_count, 1e-8),
        out=np.zeros_like(latency_sum),
        where=latency_count > 1e-8,
    )
    second = np.divide(
        latency_sq_sum,
        np.maximum(latency_count, 1e-8),
        out=np.zeros_like(latency_sq_sum),
        where=latency_count > 1e-8,
    )
    std = np.sqrt(np.maximum(second - mean**2, 0.0))
    fallback = float(np.nanmax(mean + 1.65 * std)) if np.any(latency_count > 1e-8) else 1.0
    penalty = mean + 1.65 * std
    penalty[latency_count <= 1e-8] = fallback
    return penalty


def _update_edge_latency_stats(
    result: SlotResult,
    latency_sum: np.ndarray,
    latency_sq_sum: np.ndarray,
    latency_count: np.ndarray,
    gamma: float,
) -> None:
    latency_sum *= gamma
    latency_sq_sum *= gamma
    latency_count *= gamma
    for edge in range(latency_sum.size):
        values = [record.training_latency_s for record in result.records if record.edge == edge]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        latency_sum[edge] += float(np.sum(arr))
        latency_sq_sum[edge] += float(np.sum(arr**2))
        latency_count[edge] += float(arr.size)


def _run_lyapunov(args: argparse.Namespace) -> dict[str, object]:
    environment, system, _ = _build_environment(args, args.seed + 10_000)
    evaluation_results: list[SlotResult] = []
    latency_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    started_all = perf_counter()
    method = args.method
    fresh_state = method == LYAPUNOV_FRESH
    use_tail = method == LYAPUNOV_TAIL
    tradeoff_v = args.lyapunov_v
    temperature = args.lyapunov_temperature
    if method == LYAPUNOV_STALE_TUNED:
        tradeoff_v = args.lyapunov_tuned_v
        temperature = args.lyapunov_tuned_temperature
    latency_sum = np.zeros(args.n_edges, dtype=np.float64)
    latency_sq_sum = np.zeros(args.n_edges, dtype=np.float64)
    latency_count = np.zeros(args.n_edges, dtype=np.float64)
    for slot in range(1, args.test_steps + 1):
        action_started = perf_counter()
        tail_penalty = (
            _edge_tail_penalty(latency_sum, latency_sq_sum, latency_count)
            if use_tail
            else None
        )
        if method in LYAPUNOV_LOCAL_STALE_FAMILY:
            portfolios = _lyapunov_local_stale_portfolios(
                environment,
                tradeoff_v=args.lyapunov_local_v,
                temperature=args.lyapunov_local_temperature,
                candidate_edges=args.lyapunov_local_candidates,
            )
        else:
            portfolios = _lyapunov_portfolios(
                environment,
                tradeoff_v,
                temperature,
                fresh_state=fresh_state,
                tail_penalty=tail_penalty,
                tail_weight=args.lyapunov_tail_weight,
            )
        result = environment.joint_step(
            portfolios, decision_latency_s=perf_counter() - action_started
        )
        if use_tail:
            _update_edge_latency_stats(
                result,
                latency_sum,
                latency_sq_sum,
                latency_count,
                args.lyapunov_tail_discount,
            )
        evaluation_results.append(result)
        _record_latencies(latency_rows, result, "test", slot)
        if slot == 1 or slot % args.log_interval == 0 or slot == args.test_steps:
            row: dict[str, object] = {
                "slot": slot,
                "step": slot * system.n_users,
                "mean_reward": -result.system_average_latency,
                "elapsed_s": perf_counter() - started_all,
            }
            row.update(slot_metrics(result))
            learning_rows.append(row)
    summary, _ = finalize_evaluation(environment, evaluation_results, system)
    summary.update(phase_diagnostics(evaluation_results, system))
    summary.update(
        {
            "method": method,
            "source_paper_doi": "10.1109/TMC.2025.3592450",
            "reproduction_level": (
                "LyFOTO-inspired fresh-state oracle analytic baseline"
                if fresh_state
                else "LyFOTO-inspired local-candidate stale analytic baseline"
                if method == LYAPUNOV_LOCAL_STALE
                else "LyFOTO-inspired local-candidate stale analytic baseline (validation-tuned)"
                if method == LYAPUNOV_LOCAL_STALE_TUNED
                else "LyFOTO-inspired strengthened analytic baseline"
                if method != LYAPUNOV
                else "LyFOTO-inspired analytic baseline"
            ),
            "architecture": (
                "stale local-candidate drift-plus-penalty controller"
                if method in LYAPUNOV_LOCAL_STALE_FAMILY
                else
                "fresh drift-plus-delay softmax controller"
                if fresh_state
                else "stale drift-plus-delay plus empirical tail softmax controller"
                if use_tail
                else "stale drift-plus-delay softmax controller"
            ),
            "training_free": True,
            "uses_agegate": False,
            "uses_cvar_reward": False,
            "uses_crowding_penalty": False,
            "dataset": args.dataset,
            "users": args.users,
            "train_steps": 0,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
        }
    )
    _save_outputs(
        args,
        method,
        learning_rows,
        latency_rows,
        summary,
        {
            "decision_rule": "softmax(-(V*T_hat(t-delta)+Q_hat(t-delta)*task_service_time)/temperature)",
            "fresh_state_oracle": fresh_state,
            "local_candidate_edges": args.lyapunov_local_candidates if method in LYAPUNOV_LOCAL_STALE_FAMILY else 0,
            "tradeoff_v": args.lyapunov_local_v if method in LYAPUNOV_LOCAL_STALE_FAMILY else tradeoff_v,
            "temperature": args.lyapunov_local_temperature if method in LYAPUNOV_LOCAL_STALE_FAMILY else temperature,
            "tail_penalty": {
                "enabled": use_tail,
                "weight": args.lyapunov_tail_weight,
                "discount": args.lyapunov_tail_discount,
                "statistic": "mean + 1.65 * std per edge from completed latencies",
            },
            "not_reproduced": [
                "energy-harvesting battery queue",
                "five-layer federated actor network",
                "NOP quantization and continuous resource-allocation critic",
            ],
        },
    )
    return summary


def _safetail_portfolios(
    environment: JointMAPPOEdgeEnv,
    target_latency_s: float,
    max_redundancy: int,
    temperature: float,
) -> np.ndarray:
    """SafeTail-inspired controlled-redundancy portfolio adapter.

    The SafeTail paper executes duplicate services on a subset of edge servers
    and uses the fastest response.  The shared portfolio environment does not
    expose duplicate execution for the same task, so this adapter maps the same
    idea to a soft portfolio over a selected low-latency subset.  It uses only
    delayed/broadcast server state, service size and communication estimates.
    """

    feedback = environment.feedback_buffer.get(environment.feedback_delay)
    locations = environment._locations()
    n_users = environment.cfg.n_users
    n_edges = environment.cfg.n_edges
    portfolios = np.zeros((n_users, n_edges), dtype=np.float32)
    service_time = environment.cfg.task_compute_mb / np.maximum(
        environment.edge_capacities_mb_s.astype(np.float64), 1e-8
    )
    for user in range(n_users):
        communication = np.asarray(
            [
                environment._transmission_delay_s(locations[user], edge)
                for edge in range(n_edges)
            ],
            dtype=np.float64,
        )
        estimated = communication + service_time + feedback.wait_s
        order = np.argsort(estimated)
        eligible = order[estimated[order] <= target_latency_s]
        if eligible.size == 0:
            chosen = order[: max(1, max_redundancy)]
        else:
            chosen = eligible[: max(1, min(max_redundancy, eligible.size))]
        logits = -estimated[chosen] / max(temperature, 1e-8)
        logits -= np.max(logits)
        weights = np.exp(np.clip(logits, -60.0, 0.0))
        weights = weights / max(float(weights.sum()), 1e-12)
        portfolios[user, chosen] = weights.astype(np.float32)
    return portfolios


def _run_safetail(args: argparse.Namespace) -> dict[str, object]:
    environment, system, _ = _build_environment(args, args.seed + 10_000)
    evaluation_results: list[SlotResult] = []
    latency_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    started_all = perf_counter()
    for slot in range(1, args.test_steps + 1):
        action_started = perf_counter()
        portfolios = _safetail_portfolios(
            environment,
            args.safetail_target_latency_s,
            args.safetail_max_redundancy,
            args.safetail_temperature,
        )
        result = environment.joint_step(
            portfolios, decision_latency_s=perf_counter() - action_started
        )
        evaluation_results.append(result)
        _record_latencies(latency_rows, result, "test", slot)
        if slot == 1 or slot % args.log_interval == 0 or slot == args.test_steps:
            row: dict[str, object] = {
                "slot": slot,
                "step": slot * system.n_users,
                "mean_reward": -result.system_average_latency,
                "elapsed_s": perf_counter() - started_all,
            }
            row.update(slot_metrics(result))
            learning_rows.append(row)
    summary, _ = finalize_evaluation(environment, evaluation_results, system)
    summary.update(phase_diagnostics(evaluation_results, system))
    summary.update(
        {
            "method": SAFETAIL,
            "source_paper": "SafeTail: Efficient Tail Latency Optimization in Edge Service Scheduling via Computational Redundancy Management",
            "source_paper_arxiv": "2408.17171",
            "reproduction_level": "algorithm-idea environment adaptation",
            "architecture": "controlled low-latency edge subset with soft portfolio adapter",
            "action_adapter": "duplicate-execution subset is mapped to a portfolio over selected edges",
            "uses_agegate": False,
            "uses_cvar_reward": False,
            "uses_crowding_penalty": False,
            "training_free": True,
            "dataset": args.dataset,
            "users": args.users,
            "train_steps": 0,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
            "safetail_target_latency_s": args.safetail_target_latency_s,
            "safetail_max_redundancy": args.safetail_max_redundancy,
        }
    )
    _save_outputs(
        args,
        SAFETAIL,
        learning_rows,
        latency_rows,
        summary,
        {
            "paper_core": "choose an edge subset to control tail latency under dynamic network/compute state",
            "adapter_limitations": [
                "the common environment does not support duplicate execution of the same task",
                "latency is generated by portfolio sampling rather than min latency across replicas",
                "uses delayed/broadcast server state to match the stale-feedback setting",
            ],
        },
    )
    return summary


class TODAuctionMAB:
    """Discounted delay MAB with cyclic exploration and auction matching."""

    def __init__(
        self,
        n_users: int,
        capacities: np.ndarray,
        task_compute_mb: float,
        gamma: float,
        exploration: float,
        bid_increment: float,
        tail_weight: float = 0.0,
        stale_weight: float = 0.0,
        capacity_weight: float = 0.0,
    ) -> None:
        self.n_users = int(n_users)
        self.n_edges = int(capacities.size)
        self.gamma = float(gamma)
        self.exploration = float(exploration)
        self.bid_increment = float(bid_increment)
        self.tail_weight = float(tail_weight)
        self.stale_weight = float(stale_weight)
        self.capacity_weight = float(capacity_weight)
        self.discounted_count = np.zeros((n_users, self.n_edges), dtype=np.float64)
        self.discounted_cost = np.zeros((n_users, self.n_edges), dtype=np.float64)
        self.discounted_sq_cost = np.zeros((n_users, self.n_edges), dtype=np.float64)
        units = np.maximum(1, np.floor(capacities / max(task_compute_mb, 1e-8)).astype(int))
        self.capacity_units = units.astype(np.float64)
        self.resource_edges = np.repeat(np.arange(self.n_edges, dtype=np.int64), units)
        self.time = 0

    def clone(self) -> "TODAuctionMAB":
        return copy.deepcopy(self)

    def estimated_cost(self, feedback: object | None = None) -> np.ndarray:
        means = np.divide(
            self.discounted_cost,
            np.maximum(self.discounted_count, 1e-8),
            out=np.zeros_like(self.discounted_cost),
            where=self.discounted_count > 1e-8,
        )
        second = np.divide(
            self.discounted_sq_cost,
            np.maximum(self.discounted_count, 1e-8),
            out=np.zeros_like(self.discounted_sq_cost),
            where=self.discounted_count > 1e-8,
        )
        std = np.sqrt(np.maximum(second - means**2, 0.0))
        unseen = self.discounted_count <= 1e-8
        fallback = float(np.max(means[~unseen])) if np.any(~unseen) else 1.0
        means[unseen] = fallback
        std[unseen] = 0.0
        bonus = self.exploration * np.sqrt(
            np.log(max(self.time + 2, 2)) / np.maximum(self.discounted_count, 1e-8)
        )
        optimistic = means + self.tail_weight * std - bonus
        if feedback is not None and self.stale_weight > 0.0:
            stale_pressure = np.asarray(feedback.wait_norm + feedback.queue_task_count_norm, dtype=np.float64)
            optimistic += self.stale_weight * stale_pressure[None, :]
        if self.capacity_weight > 0.0:
            capacity_pressure = 1.0 / np.maximum(self.capacity_units, 1.0)
            optimistic += self.capacity_weight * capacity_pressure[None, :]
        optimistic[unseen] = -1e6
        return optimistic

    def actions(self, feedback: object | None = None) -> np.ndarray:
        if self.time < self.n_edges:
            return (np.arange(self.n_users, dtype=np.int64) + self.time) % self.n_edges
        return self._auction(self.estimated_cost(feedback))

    def _auction(self, edge_cost: np.ndarray) -> np.ndarray:
        utilities = -edge_cost[:, self.resource_edges]
        prices = np.zeros(self.resource_edges.size, dtype=np.float64)
        owners = np.full(self.resource_edges.size, -1, dtype=np.int64)
        assigned = np.full(self.n_users, -1, dtype=np.int64)
        pending = deque(range(self.n_users))
        iterations = 0
        max_iterations = max(1000, 50 * self.n_users * self.resource_edges.size)
        while pending and iterations < max_iterations:
            user = int(pending.popleft())
            net = utilities[user] - prices
            best = int(np.argmax(net))
            best_value = float(net[best])
            if net.size > 1:
                second = float(np.partition(net, -2)[-2])
            else:
                second = best_value - self.bid_increment
            prices[best] += best_value - second + self.bid_increment
            previous = int(owners[best])
            owners[best] = user
            assigned[user] = best
            if previous >= 0 and previous != user:
                assigned[previous] = -1
                pending.append(previous)
            iterations += 1
        for user in np.flatnonzero(assigned < 0):
            assigned[user] = int(np.argmin(edge_cost[user, self.resource_edges]))
        return self.resource_edges[assigned]

    def update(self, result: SlotResult, actions: np.ndarray) -> None:
        self.discounted_count *= self.gamma
        self.discounted_cost *= self.gamma
        self.discounted_sq_cost *= self.gamma
        for user, edge in enumerate(actions):
            records = [record for record in result.records if record.user == user]
            if not records:
                continue
            observed = float(np.mean([record.training_latency_s for record in records]))
            self.discounted_count[user, edge] += 1.0
            self.discounted_cost[user, edge] += observed
            self.discounted_sq_cost[user, edge] += observed**2
        self.time += 1


def _run_mab(args: argparse.Namespace) -> dict[str, object]:
    train_environment, system, capacities = _build_environment(args, args.seed)
    method = MAB_TAIL if args.method == MAB_TAIL else MAB
    use_tail = method == MAB_TAIL
    policy = TODAuctionMAB(
        args.users,
        capacities,
        system.task_compute_mb,
        args.mab_discount,
        args.mab_exploration,
        args.auction_bid_increment,
        tail_weight=args.mab_tail_weight if use_tail else 0.0,
        stale_weight=args.mab_stale_weight if use_tail else 0.0,
        capacity_weight=args.mab_capacity_weight if use_tail else 0.0,
    )
    learning_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    started_all = perf_counter()
    for slot in range(1, args.train_steps + 1):
        action_started = perf_counter()
        feedback = (
            train_environment.feedback_buffer.get(train_environment.feedback_delay)
            if use_tail
            else None
        )
        actions = policy.actions(feedback)
        result = train_environment.joint_step(
            _one_hot(actions, args.n_edges),
            decision_latency_s=perf_counter() - action_started,
        )
        policy.update(result, actions)
        _record_latencies(latency_rows, result, "train", slot)
        if slot == 1 or slot % args.log_interval == 0 or slot == args.train_steps:
            row: dict[str, object] = {
                "slot": slot,
                "step": slot * system.n_users,
                "mean_reward": -result.system_average_latency,
                "explored_fraction": float(
                    np.mean(policy.discounted_count > args.mab_min_effective_count)
                ),
                "elapsed_s": perf_counter() - started_all,
            }
            row.update(slot_metrics(result))
            learning_rows.append(row)

    evaluation_policy = policy.clone()
    evaluation_environment, _, _ = _build_environment(args, args.seed + 10_000)
    evaluation_results: list[SlotResult] = []
    for slot in range(1, args.test_steps + 1):
        action_started = perf_counter()
        feedback = (
            evaluation_environment.feedback_buffer.get(evaluation_environment.feedback_delay)
            if use_tail
            else None
        )
        actions = evaluation_policy.actions(feedback)
        result = evaluation_environment.joint_step(
            _one_hot(actions, args.n_edges),
            decision_latency_s=perf_counter() - action_started,
        )
        evaluation_policy.update(result, actions)
        evaluation_results.append(result)
        _record_latencies(latency_rows, result, "test", slot)

    summary, _ = finalize_evaluation(evaluation_environment, evaluation_results, system)
    summary.update(phase_diagnostics(evaluation_results, system))
    if train_environment.canonical_schedule is not None:
        summary["train_split_id"] = train_environment.canonical_schedule.schedule_id
        summary["train_schedule_sha256"] = train_environment.canonical_schedule.content_sha256
    summary.update(
        {
            "method": method,
            "source_paper_doi": "10.1007/s10586-025-05560-1",
            "reproduction_level": (
                "stronger algorithm-structure environment adaptation"
                if use_tail
                else "algorithm-structure environment adaptation"
            ),
            "architecture": (
                "tail/stale/capacity-aware discounted MAB plus distributed auction"
                if use_tail
                else "discounted empirical-delay MAB plus distributed auction"
            ),
            "action_adapter": "server capacity represented by repeated resource groups",
            "uses_server_state": use_tail,
            "uses_agegate": False,
            "uses_cvar_reward": False,
            "uses_crowding_penalty": False,
            "dataset": args.dataset,
            "users": args.users,
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "feedback_delay": args.feedback_delay,
            "broadcast_period": args.broadcast_period,
        }
    )
    _save_outputs(
        args,
        method,
        learning_rows,
        latency_rows,
        summary,
        {
            "initial_exploration": "cyclic user-server basis-arm schedule for M slots",
            "nonstationarity": f"discounted empirical delay, gamma={args.mab_discount}",
            "matching": "Bertsekas-style auction over capacity resource groups",
            "feedback": (
                "completed task latency plus delayed/broadcast queue pressure"
                if use_tail
                else "completed task latency only"
            ),
            "tail_stale_capacity_adapter": {
                "enabled": use_tail,
                "tail_weight": args.mab_tail_weight,
                "stale_weight": args.mab_stale_weight,
                "capacity_weight": args.mab_capacity_weight,
            },
        },
    )
    np.savez_compressed(
        _output_dir(args) / f"checkpoint_{method}.npz",
        discounted_count=policy.discounted_count,
        discounted_cost=policy.discounted_cost,
        discounted_sq_cost=policy.discounted_sq_cost,
        time=policy.time,
    )
    return summary


def train_once(args: argparse.Namespace) -> dict[str, object]:
    if args.method in {QECO, QECO_LOAD_TOPK}:
        return _run_qeco(args)
    if args.method in {LYAPUNOV, LYAPUNOV_LOCAL_STALE, LYAPUNOV_LOCAL_STALE_TUNED, LYAPUNOV_FRESH, LYAPUNOV_STALE_TUNED, LYAPUNOV_TAIL}:
        return _run_lyapunov(args)
    if args.method in {MAB, MAB_TAIL}:
        return _run_mab(args)
    if args.method == SAFETAIL:
        return _run_safetail(args)
    raise ValueError(f"Unknown method: {args.method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recent MEC baselines in the common comparison environment.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", default="KAIST")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--distribution", default="poisson_bursty")
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-root", default=str(TRACE_ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "results_recent_mec_baselines"))
    parser.add_argument(
        "--canonical-schedule-root",
        default=str(ROOT / "results_qla_canonical_v1" / "shared_schedules"),
    )
    parser.add_argument("--canonical-max-feedback-delay", type=int, default=8)
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
    parser.add_argument("--score-resource-weight", type=float, default=1.0)
    parser.add_argument("--score-capacity-weight", type=float, default=0.5)
    parser.add_argument("--score-task-count-weight", type=float, default=0.5)
    parser.add_argument("--score-wait-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-threads", type=int, default=1)

    # QECO author-code defaults.
    parser.add_argument("--qeco-hidden", type=int, default=20)
    parser.add_argument("--qeco-lstm-hidden", type=int, default=20)
    parser.add_argument("--history-steps", type=int, default=10)
    parser.add_argument("--qeco-episode-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--replay-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--epsilon-max", type=float, default=0.99)
    parser.add_argument("--epsilon-increment", type=float, default=0.00025)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--deadline-s", type=float, default=10.0)
    parser.add_argument("--qeco-topk", type=int, default=4)
    parser.add_argument("--qeco-topk-temperature", type=float, default=0.5)
    parser.add_argument("--qeco-load-mask", action="store_true")
    parser.add_argument("--qeco-mask-quantile", type=float, default=0.75)

    # LyFOTO-inspired drift-plus-penalty adapter.
    parser.add_argument("--lyapunov-v", type=float, default=5.0)
    parser.add_argument("--lyapunov-temperature", type=float, default=1.0)
    parser.add_argument("--lyapunov-tuned-v", type=float, default=1.0)
    parser.add_argument("--lyapunov-tuned-temperature", type=float, default=2.5)
    parser.add_argument("--lyapunov-tail-weight", type=float, default=1.0)
    parser.add_argument("--lyapunov-tail-discount", type=float, default=0.99)
    parser.add_argument("--lyapunov-local-candidates", type=int, default=3)
    parser.add_argument("--lyapunov-local-v", type=float, default=1.0)
    parser.add_argument("--lyapunov-local-temperature", type=float, default=1.0)

    # TODAuction-MAB non-stationarity and auction parameters.
    parser.add_argument("--mab-discount", type=float, default=0.99)
    parser.add_argument("--mab-exploration", type=float, default=0.25)
    parser.add_argument("--mab-min-effective-count", type=float, default=0.1)
    parser.add_argument("--auction-bid-increment", type=float, default=0.01)
    parser.add_argument("--mab-tail-weight", type=float, default=1.0)
    parser.add_argument("--mab-stale-weight", type=float, default=1.0)
    parser.add_argument("--mab-capacity-weight", type=float, default=0.2)

    # SafeTail-inspired controlled redundancy adapter.
    parser.add_argument("--safetail-target-latency-s", type=float, default=1.4)
    parser.add_argument("--safetail-max-redundancy", type=int, default=4)
    parser.add_argument("--safetail-temperature", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(train_once(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
