from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .config import PPOConfig, SystemConfig


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_edges: int, hidden: int = 128) -> None:
        super().__init__()
        self.actor_mean = mlp(obs_dim, n_edges, hidden)
        self.actor_log_std = nn.Parameter(torch.full((n_edges,), -0.5))
        self.critic = mlp(obs_dim, 1, hidden)

    def gaussian(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return Normal(mean, std)

    def portfolio(self, latent_action: torch.Tensor) -> torch.Tensor:
        return torch.softmax(latent_action, dim=-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    latent_action: np.ndarray
    actions: list[int]
    log_prob: float
    value: float
    reward: float
    done: bool
    next_obs: np.ndarray


class PPOPolicy:
    def __init__(self, sys_cfg: SystemConfig, ppo_cfg: PPOConfig, seed: int = 42, device: str | None = None) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.sys_cfg = sys_cfg
        self.cfg = ppo_cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo_cfg.learning_rate)
        self.buffer: list[Transition] = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.gaussian(obs_t)
            latent = dist.mean if deterministic else dist.sample()
            value = self.net.value(obs_t).item()
            portfolio = self.net.portfolio(latent)
            log_prob = dist.log_prob(latent).sum(dim=-1).item()
        self.last_latent_action = latent.squeeze(0).cpu().numpy().astype(np.float32)
        return portfolio.squeeze(0).cpu().numpy(), float(log_prob), float(value)

    def log_prob_for_edges(self, obs: np.ndarray, edges: Iterable[int]) -> float:
        # Kept for compatibility with older experiment outputs. PPO now uses the
        # Gaussian latent portfolio log-prob stored by `act`, not edge log-probs.
        edges = list(edges)
        if not edges:
            return 0.0
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.net.actor_mean(obs_t)
            dist = Categorical(logits=logits)
            action_t = torch.as_tensor(edges, dtype=torch.long, device=self.device)
            return float(dist.log_prob(action_t).sum().item())

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.net.value(obs_t).item())

    def add(self, tr: Transition) -> None:
        self.buffer.append(tr)
        if len(self.buffer) > self.cfg.buffer_size:
            self.buffer = self.buffer[-self.cfg.buffer_size :]

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs = torch.as_tensor(np.stack([t.obs for t in self.buffer]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in self.buffer]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([t.log_prob for t in self.buffer], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t.reward for t in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in self.buffer], dtype=torch.float32, device=self.device)
        values = torch.as_tensor([t.value for t in self.buffer], dtype=torch.float32, device=self.device)
        latent_actions = torch.as_tensor(
            np.stack([t.latent_action for t in self.buffer]), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_values = self.net.value(next_obs)
            deltas = rewards + self.cfg.gamma * next_values * (1.0 - dones) - values
            adv = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, device=self.device)
            for idx in reversed(range(len(self.buffer))):
                gae = deltas[idx] + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[idx]) * gae
                adv[idx] = gae
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(self.buffer)
        actor_losses = []
        critic_losses = []
        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                dist = self.net.gaussian(obs[idx])
                new_logp = dist.log_prob(latent_actions[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy
                value_pred = self.net.value(obs[idx])
                critic_loss = ((returns[idx] - value_pred) ** 2).mean()
                loss = actor_loss + self.cfg.value_coef * critic_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))

        self.buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
        }


class PGPolicy:
    """Vanilla Policy Gradient (REINFORCE with learned baseline).

    Uses the same Gaussian→softmax actor as PPO but updates with vanilla
    policy gradient: single pass, no importance-sampling ratio, no clipping.
    """

    def __init__(self, sys_cfg: SystemConfig, ppo_cfg: PPOConfig, seed: int = 42, device: str | None = None) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.sys_cfg = sys_cfg
        self.cfg = ppo_cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo_cfg.learning_rate)
        self.buffer: list[Transition] = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.net.gaussian(obs_t)
            latent = dist.mean if deterministic else dist.sample()
            value = self.net.value(obs_t).item()
            portfolio = self.net.portfolio(latent)
            log_prob = dist.log_prob(latent).sum(dim=-1).item()
        self.last_latent_action = latent.squeeze(0).cpu().numpy().astype(np.float32)
        return portfolio.squeeze(0).cpu().numpy(), float(log_prob), float(value)

    def add(self, tr: Transition) -> None:
        self.buffer.append(tr)
        if len(self.buffer) > self.cfg.buffer_size:
            self.buffer = self.buffer[-self.cfg.buffer_size :]

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}

        obs = torch.as_tensor(np.stack([t.obs for t in self.buffer]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in self.buffer]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([t.reward for t in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in self.buffer], dtype=torch.float32, device=self.device)
        values = torch.as_tensor([t.value for t in self.buffer], dtype=torch.float32, device=self.device)
        latent_actions = torch.as_tensor(
            np.stack([t.latent_action for t in self.buffer]), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            next_values = self.net.value(next_obs)
            deltas = rewards + self.cfg.gamma * next_values * (1.0 - dones) - values
            adv = torch.zeros_like(rewards)
            gae = torch.tensor(0.0, device=self.device)
            for idx in reversed(range(len(self.buffer))):
                gae = deltas[idx] + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[idx]) * gae
                adv[idx] = gae
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Vanilla PG: log_prob × advantage, no ratio, no clipping, single pass
        dist = self.net.gaussian(obs)
        log_probs = dist.log_prob(latent_actions).sum(dim=-1)
        actor_loss = -(log_probs * adv).mean()
        value_pred = self.net.value(obs)
        critic_loss = ((returns - value_pred) ** 2).mean()
        loss = actor_loss + self.cfg.value_coef * critic_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        self.buffer.clear()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }


class CategoricalQ(nn.Module):
    def __init__(self, obs_dim: int, n_edges: int, hidden: int = 128, recurrent: bool = False) -> None:
        super().__init__()
        self.recurrent = recurrent
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True) if recurrent else None
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_edges))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.body(obs)
        if self.lstm is not None:
            x, _ = self.lstm(x.unsqueeze(1))
            x = x.squeeze(1)
        return self.head(x)


class DQNPolicy:
    def __init__(
        self,
        sys_cfg: SystemConfig,
        ppo_cfg: PPOConfig,
        seed: int = 42,
        recurrent: bool = False,
        soft_actor: bool = False,
    ) -> None:
        torch.manual_seed(seed)
        self.sys_cfg = sys_cfg
        self.cfg = ppo_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden, recurrent).to(self.device)
        self.target = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden, recurrent).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo_cfg.learning_rate)
        self.replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.gamma = ppo_cfg.gamma
        self.eps = 0.2
        self.soft_actor = soft_actor

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs_t).squeeze(0)
        if self.soft_actor:
            probs = torch.softmax(q, dim=-1).cpu().numpy()
        elif deterministic or np.random.random() > self.eps:
            probs = np.zeros(self.sys_cfg.n_edges, dtype=np.float32)
            probs[int(torch.argmax(q).item())] = 1.0
        else:
            edge_idx = int(np.random.randint(self.sys_cfg.n_edges))
            probs = np.zeros(self.sys_cfg.n_edges, dtype=np.float32)
            probs[edge_idx] = 1.0
        return probs, 0.0, 0.0

    def add(self, obs: np.ndarray, actions: list[int], reward: float, next_obs: np.ndarray, done: bool) -> None:
        edge = actions[0] if actions else 0
        self.replay.append((obs, edge, reward, next_obs, done))
        if len(self.replay) > self.cfg.buffer_size:
            self.replay = self.replay[-self.cfg.buffer_size :]

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"loss": 0.0}
        idx = np.random.choice(len(self.replay), self.cfg.batch_size, replace=False)
        batch = [self.replay[i] for i in idx]
        obs = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        q = self.net(obs).gather(1, acts[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = ((q - target) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.target.load_state_dict(self.net.state_dict())
        self.eps = max(0.02, self.eps * 0.995)
        return {"loss": float(loss.item())}


class DiscreteSACPolicy:
    def __init__(self, sys_cfg: SystemConfig, ppo_cfg: PPOConfig, seed: int = 42, alpha: float = 0.2) -> None:
        torch.manual_seed(seed)
        self.sys_cfg = sys_cfg
        self.cfg = ppo_cfg
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = mlp(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.q1 = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.q2 = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.tq1 = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.tq2 = CategoricalQ(sys_cfg.obs_dim, sys_cfg.n_edges, ppo_cfg.hidden).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ppo_cfg.learning_rate)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=ppo_cfg.learning_rate)
        self.replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.actor(obs_t), dim=-1).squeeze(0).cpu().numpy()
        if deterministic:
            one_hot = np.zeros_like(probs)
            one_hot[int(np.argmax(probs))] = 1.0
            return one_hot, 0.0, 0.0
        edge_idx = int(np.random.choice(len(probs), p=probs))
        one_hot = np.zeros_like(probs)
        one_hot[edge_idx] = 1.0
        return one_hot, 0.0, 0.0

    def add(self, obs: np.ndarray, actions: list[int], reward: float, next_obs: np.ndarray, done: bool) -> None:
        edge = actions[0] if actions else 0
        self.replay.append((obs, edge, reward, next_obs, done))
        if len(self.replay) > self.cfg.buffer_size:
            self.replay = self.replay[-self.cfg.buffer_size :]

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0}
        idx = np.random.choice(len(self.replay), self.cfg.batch_size, replace=False)
        batch = [self.replay[i] for i in idx]
        obs = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        acts = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_logits = self.actor(next_obs)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log_softmax(next_logits, dim=-1)
            next_min_q = torch.minimum(self.tq1(next_obs), self.tq2(next_obs))
            next_v = (next_probs * (next_min_q - self.alpha * next_log_probs)).sum(dim=1)
            target = rewards + self.cfg.gamma * next_v * (1.0 - dones)

        q1 = self.q1(obs).gather(1, acts[:, None]).squeeze(1)
        q2 = self.q2(obs).gather(1, acts[:, None]).squeeze(1)
        critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        self.q_opt.zero_grad()
        critic_loss.backward()
        self.q_opt.step()

        logits = self.actor(obs)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        min_q = torch.minimum(self.q1(obs), self.q2(obs))
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        tau = 0.01
        with torch.no_grad():
            for target_param, param in zip(self.tq1.parameters(), self.q1.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)
            for target_param, param in zip(self.tq2.parameters(), self.q2.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)
        return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}
