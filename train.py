from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


@dataclass
class TrainingConfig:
    episodes: int = 10
    max_time_slots: int = 100
    replay_capacity: int = 5000
    batch_size: int = 32
    warmup_steps: int = 32
    exploration_noise: float = 0.1


class ReplayBuffer:
    """Lightweight replay buffer used by the outer training loop."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: Deque[Tuple[List[float], List[float], float, List[float], float]] = deque(maxlen=capacity)

    def add(
        self,
        state: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_state: Sequence[float],
        done: float,
    ) -> None:
        item = (list(state), list(action), float(reward), list(next_state), float(done))
        self._data.append(item)

    def sample(self, batch_size: int) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._data):
            raise ValueError("not enough samples in replay buffer")
        picked = random.sample(self._data, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*picked)
        return list(states), list(actions), [[r] for r in rewards], list(next_states), [[d] for d in dones]

    def __len__(self) -> int:
        return len(self._data)


def state_to_vector(state: Dict[str, Any]) -> List[float]:
    """Converts env state dict to a numeric vector for the agent."""
    vec: List[float] = [float(state.get("time_slot", 0))]

    for row in state.get("bandwidth", []):
        for bw in row:
            vec.append(float(bw))

    for node in state.get("nodes", []):
        vec.extend(
            [
                float(len(node.get("AQ", []))),
                float(len(node.get("CQ", []))),
                1.0 if node.get("PZ") is not None else 0.0,
                1.0 if node.get("SZ") is not None else 0.0,
            ]
        )

    for link in state.get("links", []):
        vec.append(float(len(link.get("queue_task_ids", []))))
        vec.append(float(sum(link.get("queue_remaining_sizes", []))))

    return vec


def _normalize_action(action: Sequence[float], action_dim: int) -> List[float]:
    a = [max(0.0, float(x)) for x in list(action)[:action_dim]]
    if len(a) < action_dim:
        a.extend([0.0 for _ in range(action_dim - len(a))])
    total = sum(a)
    if total <= 0:
        return [1.0 / action_dim for _ in range(action_dim)]
    return [x / total for x in a]


def train(
    env: Any,
    agent: Any,
    config: TrainingConfig,
) -> Dict[str, Any]:
    episode_rewards: List[float] = []
    total_steps = 0
    total_train_steps = 0
    replay = ReplayBuffer(config.replay_capacity)
    last_train_stats: Optional[Dict[str, float]] = None

    for _ in range(config.episodes):
        state, _ = env.reset()
        state_vec = state_to_vector(state)
        ep_reward = 0.0

        for _slot in range(config.max_time_slots):
            action_dim = int(getattr(env, "num_nodes", len(state.get("nodes", [])) or 1))
            raw_action = agent.select_action(state_vec, noise_std=config.exploration_noise)
            action_vec = _normalize_action(raw_action, action_dim)

            next_state, reward, done, _info = env.step([action_vec])
            next_state_vec = state_to_vector(next_state)

            replay.add(state_vec, action_vec, reward, next_state_vec, 1.0 if done else 0.0)
            state_vec = next_state_vec
            state = next_state
            ep_reward += float(reward)
            total_steps += 1

            if len(replay) >= config.batch_size and total_steps >= config.warmup_steps:
                maybe_stats = agent.train_step(replay, batch_size=config.batch_size)
                if maybe_stats is not None:
                    last_train_stats = dict(maybe_stats)
                    total_train_steps += 1

            if done:
                break

        episode_rewards.append(ep_reward)

    return {
        "episode_rewards": episode_rewards,
        "total_steps": total_steps,
        "train_steps": total_train_steps,
        "replay_size": len(replay),
        "last_train_stats": last_train_stats,
    }


def _build_demo_components() -> Tuple[Any, Any]:
    from p2td3 import P2TD3Agent
    from p2ts import PriorityWeights, QueueTask
    from rl_env import P2MEnvironment, RewardWeights

    env = P2MEnvironment(
        num_nodes=2,
        bandwidth_matrix=[[0.0, 2.0], [2.0, 0.0]],
        priority_weights=PriorityWeights(0.25, 0.25, 0.25, 0.25),
        reward_weights=RewardWeights(1.0, 0.1, 0.1),
        arrivals_by_slot=[
            [QueueTask("demo-0", 0.5, 10.0, 3.0, 0, 10.0)],
            [QueueTask("demo-1", 0.8, 10.0, 2.0, 0, 10.0)],
            [],
        ],
    )

    state, _ = env.reset()
    state_dim = len(state_to_vector(state))
    agent = P2TD3Agent(state_dim=state_dim, action_dim=env.num_nodes, hidden_dim=64)
    return env, agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2TD3 with Episodes -> Time Slots loop")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-time-slots", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=16)
    args = parser.parse_args()

    env, agent = _build_demo_components()
    stats = train(
        env,
        agent,
        TrainingConfig(
            episodes=args.episodes,
            max_time_slots=args.max_time_slots,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
        ),
    )
    print("Training finished:", stats)


if __name__ == "__main__":
    main()
