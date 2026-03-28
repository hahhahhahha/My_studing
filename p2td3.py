from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self._buffer.append((state.astype(np.float32), action.astype(np.float32), float(reward), next_state.astype(np.float32), float(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        idx = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self._buffer[i] for i in idx))
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.stack(next_states),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class P2TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2


class P2TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        config: P2TD3Config | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or P2TD3Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)

        self.total_it = 0

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        self.actor.train()

        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape)
            action = np.clip(action, 1e-8, None)
            action = action / np.sum(action)
        return action.astype(np.float32)

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Dict[str, float] | None:
        if len(replay_buffer) < batch_size:
            return None

        self.total_it += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_states_t)
            noise = torch.clamp(
                torch.randn_like(next_action) * self.config.policy_noise,
                -self.config.noise_clip,
                self.config.noise_clip,
            )
            next_action = torch.clamp(next_action + noise, min=1e-8)
            next_action = next_action / next_action.sum(dim=1, keepdim=True)

            target_q1 = self.critic1_target(next_states_t, next_action)
            target_q2 = self.critic2_target(next_states_t, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * target_q

        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)

        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % self.config.policy_delay == 0:
            actor_actions = self.actor(states_t)
            actor_loss = -self.critic1(states_t, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": float(critic1_loss.detach().cpu().item()),
            "critic2_loss": float(critic2_loss.detach().cpu().item()),
            "actor_loss": actor_loss_value,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.config.tau * sp.data + (1.0 - self.config.tau) * tp.data)
