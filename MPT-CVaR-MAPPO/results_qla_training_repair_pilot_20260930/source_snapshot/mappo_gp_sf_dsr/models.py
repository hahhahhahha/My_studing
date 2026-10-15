from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .buffer import MAPPOBuffer
from .config import MAPPOConfig, MAPPOSystemConfig


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SharedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, n_edges: int, hidden: int, log_std_min: float, log_std_max: float) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = mlp(obs_dim, 2 * n_edges, hidden)

    def distribution(self, obs: torch.Tensor) -> Normal:
        params = self.net(obs)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    @staticmethod
    def portfolio(latent_action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(latent_action, dim=-1)


class FlatPriceBiasGaussianActor(nn.Module):
    """Flattened price-augmented actor with the same logit-bias mechanism.

    Observation layout is expected to be:
      [loc(2), distance(M), request(3), score(M), age(1), price(M), price_scale(1)].

    Unlike :class:`EdgeWisePriceGaussianActor`, this actor uses a conventional
    flattened MLP.  It still subtracts ``lambda_pi * price_ij`` from the output
    Gaussian mean so the ablation isolates edge-wise parameter sharing rather
    than changing the explicit price-to-logit channel.
    """

    def __init__(
        self,
        obs_dim: int,
        n_edges: int,
        hidden: int,
        log_std_min: float,
        log_std_max: float,
        price_logit_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.price_logit_bias = float(max(price_logit_bias, 0.0))
        self.base_dim = 2 + n_edges + 3 + n_edges + 1
        expected_dim = self.base_dim + n_edges + 1
        if obs_dim != expected_dim:
            raise ValueError(f"FlatPriceBiasGaussianActor expected obs_dim={expected_dim}, got {obs_dim}")
        self.net = mlp(obs_dim, 2 * n_edges, hidden)

    def distribution(self, obs: torch.Tensor) -> Normal:
        params = self.net(obs)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        if self.price_logit_bias > 0.0:
            price = obs[..., self.base_dim : self.base_dim + self.n_edges]
            mean = mean - self.price_logit_bias * price
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    @staticmethod
    def portfolio(latent_action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(latent_action, dim=-1)


class AgeGatedGaussianActor(nn.Module):
    """Gaussian portfolio actor with explicit stale-feedback score gating.

    Observation layout:
      [Loc(2), Dis(m), Req(3), Score(m), feedback_age_norm(1)]

    The actor keeps local/request information and server-score information in
    separate branches, then gates the score branch by exp(-eta * age_slots).
    This lets fresh feedback be used strongly while old feedback fades toward a
    more stable local policy.
    """

    def __init__(
        self,
        obs_dim: int,
        n_edges: int,
        hidden: int,
        log_std_min: float,
        log_std_max: float,
        gate_eta: float,
        age_norm: float,
    ) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.local_dim = 2 + n_edges + 3
        self.score_start = self.local_dim
        self.score_end = self.score_start + n_edges
        expected_dim = self.local_dim + n_edges + 1
        if obs_dim != expected_dim:
            raise ValueError(f"AgeGatedGaussianActor expected obs_dim={expected_dim}, got {obs_dim}")
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gate_eta = float(gate_eta)
        self.age_norm = float(max(age_norm, 1e-8))
        self.local_net = mlp(self.local_dim, 2 * n_edges, hidden)
        self.score_net = mlp(n_edges, 2 * n_edges, hidden)

    def distribution(self, obs: torch.Tensor) -> Normal:
        local = obs[..., : self.local_dim]
        score = obs[..., self.score_start : self.score_end]
        age_norm = obs[..., self.score_end : self.score_end + 1].clamp(min=0.0)
        age_slots = age_norm * self.age_norm
        gate = torch.exp(-self.gate_eta * age_slots).clamp(min=0.0, max=1.0)
        params = self.local_net(local) + gate * self.score_net(score)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    @staticmethod
    def portfolio(latent_action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(latent_action, dim=-1)


class ResidualGaussianActor(nn.Module):
    """Gaussian actor for residual-prior control.

    The first M latent dimensions parameterize a residual direction. The final
    latent dimension is an amplitude logit. The executed portfolio is built
    outside the network from the stale Lyapunov prior, so ``portfolio`` is only
    used for entropy diagnostics during PPO updates.
    """

    def __init__(
        self,
        obs_dim: int,
        n_edges: int,
        hidden: int,
        log_std_min: float,
        log_std_max: float,
        amplitude_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.action_dim = n_edges + 1
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = mlp(obs_dim, 2 * self.action_dim, hidden)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            with torch.no_grad():
                final.bias[self.n_edges].fill_(float(amplitude_init_bias))

    def distribution(self, obs: torch.Tensor) -> Normal:
        params = self.net(obs)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    def portfolio(self, latent_action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(latent_action[..., : self.n_edges], dim=-1)


class EdgeWisePriceGaussianActor(nn.Module):
    """Permutation-structured actor for price-augmented LCC observations.

    Observation layout is expected to be:
      [loc(2), distance(M), request(3), score(M), age(1), price(M), price_scale(1)].

    The same edge encoder and scorer are shared across all servers so the
    relationship ``price_ij -> logit_ij -> w_ij`` is explicit rather than
    learned only through a flattened MLP.  Optionally, a fixed negative
    price-to-logit bias makes high prices directly reduce the Gaussian mean,
    turning prices from mere context features into a monotone action prior that
    the learned residual logit can still override.
    """

    def __init__(
        self,
        obs_dim: int,
        n_edges: int,
        hidden: int,
        log_std_min: float,
        log_std_max: float,
        price_logit_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.price_logit_bias = float(max(price_logit_bias, 0.0))
        self.base_dim = 2 + n_edges + 3 + n_edges + 1
        expected_dim = self.base_dim + n_edges + 1
        if obs_dim != expected_dim:
            raise ValueError(f"EdgeWisePriceGaussianActor expected obs_dim={expected_dim}, got {obs_dim}")
        self.user_net = nn.Sequential(
            nn.Linear(2 + 3 + 1 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.edge_net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def distribution(self, obs: torch.Tensor) -> Normal:
        m = self.n_edges
        loc = obs[..., :2]
        distances = obs[..., 2 : 2 + m]
        req = obs[..., 2 + m : 2 + m + 3]
        score = obs[..., 2 + m + 3 : 2 + 2 * m + 3]
        age = obs[..., 2 + 2 * m + 3 : 2 + 2 * m + 4]
        price = obs[..., self.base_dim : self.base_dim + m]
        price_scale = obs[..., self.base_dim + m : self.base_dim + m + 1]

        user_feat = torch.cat([loc, req, age, price_scale], dim=-1)
        h_user = self.user_net(user_feat)
        edge_feat = torch.stack(
            [
                distances,
                score,
                price,
                age.expand_as(price),
                price_scale.expand_as(price),
            ],
            dim=-1,
        )
        h_edge = self.edge_net(edge_feat)
        h_user_expanded = h_user.unsqueeze(-2).expand_as(h_edge)
        params = self.score_net(torch.cat([h_user_expanded, h_edge], dim=-1))
        mean = params[..., 0]
        if self.price_logit_bias > 0.0:
            mean = mean - self.price_logit_bias * price
        log_std = torch.clamp(params[..., 1], self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    @staticmethod
    def portfolio(latent_action: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(latent_action, dim=-1)


class AgentCentralizedCritic(nn.Module):
    def __init__(
        self,
        critic_obs_dim: int,
        max_users: int,
        user_embedding_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(max_users, user_embedding_dim)
        self.value_net = mlp(critic_obs_dim + user_embedding_dim, 1, hidden)

    def forward(self, critic_obs: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        emb = self.user_embedding(user_ids)
        return self.value_net(torch.cat([critic_obs, emb], dim=-1)).squeeze(-1)


@dataclass
class MAPPOUpdateStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    portfolio_entropy: float = 0.0
    auto_entropy_coef: float = 0.0
    auto_entropy_loss: float = 0.0
    residual_amplitude_mean: float = 0.0
    residual_amplitude_penalty: float = 0.0
    residual_aux_loss: float = 0.0
    residual_aux_proposal_loss: float = 0.0
    residual_aux_amplitude_loss: float = 0.0
    residual_beta_penalty: float = 0.0


class MAPPOPolicy:
    def __init__(
        self,
        sys_cfg: MAPPOSystemConfig,
        cfg: MAPPOConfig,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.sys_cfg = sys_cfg
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if cfg.edgewise_price_actor:
            self.actor = EdgeWisePriceGaussianActor(
                sys_cfg.actor_obs_dim,
                sys_cfg.n_edges,
                cfg.hidden,
                cfg.log_std_min,
                cfg.log_std_max,
                cfg.lcc_price_logit_bias,
            ).to(self.device)
        elif cfg.flat_price_bias_actor:
            self.actor = FlatPriceBiasGaussianActor(
                sys_cfg.actor_obs_dim,
                sys_cfg.n_edges,
                cfg.hidden,
                cfg.log_std_min,
                cfg.log_std_max,
                cfg.lcc_price_logit_bias,
            ).to(self.device)
        elif cfg.residual_actor:
            self.actor = ResidualGaussianActor(
                sys_cfg.actor_obs_dim,
                sys_cfg.n_edges,
                cfg.hidden,
                cfg.log_std_min,
                cfg.log_std_max,
                cfg.residual_amplitude_init_bias,
            ).to(self.device)
        elif cfg.feedback_age_gate:
            if not sys_cfg.include_feedback_age:
                raise ValueError("feedback_age_gate requires sys_cfg.include_feedback_age=True")
            self.actor = AgeGatedGaussianActor(
                sys_cfg.actor_obs_dim,
                sys_cfg.n_edges,
                cfg.hidden,
                cfg.log_std_min,
                cfg.log_std_max,
                cfg.feedback_gate_eta,
                sys_cfg.feedback_age_norm,
            ).to(self.device)
        else:
            self.actor = SharedGaussianActor(
                sys_cfg.actor_obs_dim,
                sys_cfg.n_edges,
                cfg.hidden,
                cfg.log_std_min,
                cfg.log_std_max,
            ).to(self.device)
        critic_count = max(1, int(cfg.critic_ensemble_count))
        self.critics = nn.ModuleList(
            [
                AgentCentralizedCritic(
                    sys_cfg.critic_obs_dim,
                    sys_cfg.max_supported_users,
                    cfg.user_embedding_dim,
                    cfg.hidden,
                )
                for _ in range(critic_count)
            ]
        ).to(self.device)
        self.critic = self.critics[0]
        self.critic2 = (
            AgentCentralizedCritic(
                sys_cfg.critic_obs_dim,
                sys_cfg.max_supported_users,
                cfg.user_embedding_dim,
                cfg.hidden,
            ).to(self.device)
            if cfg.twin_critic and critic_count == 1
            else None
        )
        trainable = list(self.actor.parameters()) + list(self.critics.parameters())
        if self.critic2 is not None:
            trainable += list(self.critic2.parameters())
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=cfg.learning_rate,
        )
        self.log_portfolio_entropy_coef = torch.tensor(
            float(np.log(max(cfg.auto_entropy_init, 1e-8))),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.entropy_optimizer = torch.optim.Adam([self.log_portfolio_entropy_coef], lr=cfg.auto_entropy_lr)

    def act(self, actor_obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.actor.distribution(obs_t)
            latent = dist.mean if deterministic else dist.sample()
            portfolios = self.actor.portfolio(latent, obs_t)
            log_prob = dist.log_prob(latent).sum(dim=-1)
        return (
            portfolios.cpu().numpy().astype(np.float32),
            latent.cpu().numpy().astype(np.float32),
            log_prob.cpu().numpy().astype(np.float32),
        )

    def value(self, critic_obs: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        ids_t = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if len(self.critics) > 1:
                values = torch.stack([critic(obs_t, ids_t) for critic in self.critics], dim=0).mean(dim=0)
            else:
                values = self.critic(obs_t, ids_t)
            if self.critic2 is not None:
                values = torch.minimum(values, self.critic2(obs_t, ids_t))
        return values.cpu().numpy().astype(np.float32)

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ensemble": [critic.state_dict() for critic in self.critics],
            "critic2": self.critic2.state_dict() if self.critic2 is not None else None,
            "log_portfolio_entropy_coef": float(self.log_portfolio_entropy_coef.detach().cpu().item()),
            "n_edges": self.sys_cfg.n_edges,
            "actor_obs_dim": self.sys_cfg.actor_obs_dim,
            "include_feedback_age": self.sys_cfg.include_feedback_age,
            "feedback_age_gate": self.cfg.feedback_age_gate,
            "residual_actor": self.cfg.residual_actor,
            "residual_actor_prior_conditioned": self.cfg.residual_actor_prior_conditioned,
            "edgewise_price_actor": self.cfg.edgewise_price_actor,
            "flat_price_bias_actor": self.cfg.flat_price_bias_actor,
            "lcc_price_observation": self.cfg.lcc_price_observation,
            "lcc_price_scale_observation": self.cfg.lcc_price_scale_observation,
            "lcc_edgewise_price_actor": self.cfg.lcc_edgewise_price_actor,
            "lcc_flat_price_bias_actor": self.cfg.lcc_flat_price_bias_actor,
            "lcc_price_logit_bias": self.cfg.lcc_price_logit_bias,
            "lcc_price_components": self.cfg.lcc_price_components,
            "lcc_local_critic": self.cfg.lcc_local_critic,
            "target_portfolio_entropy": self.cfg.target_portfolio_entropy,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, actor_only: bool = False) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        state = payload.get("actor", payload)
        self.actor.load_state_dict(state)
        if not actor_only:
            if "critic_ensemble" in payload:
                for critic, state in zip(self.critics, payload["critic_ensemble"]):
                    critic.load_state_dict(state)
            elif "critic" in payload:
                self.critic.load_state_dict(payload["critic"])
            if self.critic2 is not None and payload.get("critic2") is not None:
                self.critic2.load_state_dict(payload["critic2"])
            if "log_portfolio_entropy_coef" in payload:
                with torch.no_grad():
                    self.log_portfolio_entropy_coef.copy_(
                        torch.as_tensor(payload["log_portfolio_entropy_coef"], device=self.device)
                    )

    def update(self, buffer: MAPPOBuffer) -> MAPPOUpdateStats:
        if len(buffer) < 2:
            return MAPPOUpdateStats(0.0, 0.0, 0.0, 0.0)
        arrays = buffer.arrays()
        advantages, returns = self._compute_gae(arrays)

        actor_obs = torch.as_tensor(arrays["actor_obs"], dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(arrays["critic_obs"], dtype=torch.float32, device=self.device)
        user_ids = torch.as_tensor(arrays["user_id"], dtype=torch.long, device=self.device)
        latent_actions = torch.as_tensor(arrays["latent_action"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(arrays["log_prob"], dtype=torch.float32, device=self.device)
        aux_target = torch.as_tensor(arrays["aux_target"], dtype=torch.float32, device=self.device)
        aux_weight = torch.as_tensor(arrays["aux_weight"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n = len(buffer)
        batch_size = min(self.cfg.batch_size, n)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        portfolio_entropies: list[float] = []
        auto_entropy_losses: list[float] = []
        auto_entropy_coefs: list[float] = []
        residual_amplitudes: list[float] = []
        residual_amplitude_penalties: list[float] = []
        residual_aux_losses: list[float] = []
        residual_aux_proposal_losses: list[float] = []
        residual_aux_amplitude_losses: list[float] = []
        residual_beta_penalties: list[float] = []
        kls: list[float] = []
        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                dist = self.actor.distribution(actor_obs[idx])
                new_logp = dist.log_prob(latent_actions[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                entropy_latent = dist.rsample()
                entropy_portfolio = self.actor.portfolio(entropy_latent, actor_obs[idx])
                portfolio_entropy = (
                    -(entropy_portfolio * torch.log(entropy_portfolio + 1e-8)).sum(dim=-1)
                    / np.log(self.sys_cfg.n_edges)
                ).mean()
                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[idx]
                auto_entropy_coef = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                if self.cfg.auto_portfolio_entropy:
                    auto_entropy_coef = torch.exp(self.log_portfolio_entropy_coef).detach()
                residual_amplitude_penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                residual_amplitude_mean = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                if self.cfg.residual_actor and self.cfg.residual_amplitude_penalty > 0.0:
                    amplitude_logits = dist.mean[..., self.sys_cfg.n_edges]
                    residual_amplitude = torch.sigmoid(amplitude_logits)
                    residual_amplitude_mean = residual_amplitude.mean()
                    residual_amplitude_penalty = (
                        float(self.cfg.residual_amplitude_penalty) * (residual_amplitude ** 2).mean()
                    )
                residual_aux_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                residual_aux_proposal_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                residual_aux_amplitude_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                residual_beta_penalty = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                if (
                    self.cfg.residual_actor
                    and self.cfg.residual_actor_prior_conditioned
                    and (
                        self.cfg.residual_aux_kl_weight > 0.0
                        or self.cfg.residual_aux_proposal_kl_weight > 0.0
                        or self.cfg.residual_aux_amplitude_weight > 0.0
                        or self.cfg.residual_beta_penalty > 0.0
                    )
                    and torch.sum(aux_weight[idx]) > 0.0
                ):
                    n_edges = self.sys_cfg.n_edges
                    logits = dist.mean[..., :n_edges]
                    q = torch.softmax(logits / max(float(self.cfg.residual_proposal_temperature), 1e-8), dim=-1)
                    prior = torch.clamp(actor_obs[idx, -(n_edges + 1) : -1], min=1e-8)
                    prior = prior / torch.clamp(prior.sum(dim=-1, keepdim=True), min=1e-8)
                    volatility = torch.clamp(actor_obs[idx, -1:], min=0.0)
                    if self.sys_cfg.include_feedback_age:
                        base_dim = self.sys_cfg.actor_obs_dim - (n_edges + 1)
                        age_norm = torch.clamp(actor_obs[idx, base_dim - 1 : base_dim], min=0.0)
                    else:
                        age_norm = torch.zeros_like(volatility)
                    beta = (
                        float(self.cfg.residual_beta_max)
                        * torch.sigmoid(dist.mean[..., n_edges : n_edges + 1])
                        * torch.exp(
                            -float(self.cfg.residual_beta_gamma_delta) * age_norm
                            -float(self.cfg.residual_beta_gamma_volatility) * volatility
                        )
                    )
                    beta = torch.clamp(beta, min=0.0, max=1.0)
                    executed = torch.clamp((1.0 - beta) * prior + beta * q, min=1e-8)
                    executed = executed / torch.clamp(executed.sum(dim=-1, keepdim=True), min=1e-8)
                    target = torch.clamp(aux_target[idx, :n_edges], min=1e-8)
                    target = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=1e-8)
                    weight_sum = torch.clamp(aux_weight[idx].sum(), min=1e-8)

                    if self.cfg.residual_aux_kl_weight > 0.0:
                        kl_executed = torch.sum(target * (torch.log(target) - torch.log(executed)), dim=-1)
                        residual_aux_loss = float(self.cfg.residual_aux_kl_weight) * (
                            (kl_executed * aux_weight[idx]).sum() / weight_sum
                        )

                    if self.cfg.residual_aux_proposal_kl_weight > 0.0:
                        kl_q = torch.sum(target * (torch.log(target) - torch.log(torch.clamp(q, min=1e-8))), dim=-1)
                        residual_aux_proposal_loss = float(self.cfg.residual_aux_proposal_kl_weight) * (
                            (kl_q * aux_weight[idx]).sum() / weight_sum
                        )

                    if self.cfg.residual_aux_amplitude_weight > 0.0:
                        def js_divergence(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
                            p = torch.clamp(p, min=1e-8)
                            r = torch.clamp(r, min=1e-8)
                            p = p / torch.clamp(p.sum(dim=-1, keepdim=True), min=1e-8)
                            r = r / torch.clamp(r.sum(dim=-1, keepdim=True), min=1e-8)
                            m = 0.5 * (p + r)
                            return 0.5 * torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1) + 0.5 * torch.sum(
                                r * (torch.log(r) - torch.log(m)), dim=-1
                            )

                        js_prior = js_divergence(target, prior)
                        js_q = js_divergence(target, q)
                        gain = torch.clamp((js_prior - js_q) / torch.clamp(js_prior, min=1e-8), min=0.0, max=1.0)
                        amplitude = torch.sigmoid(dist.mean[..., n_edges])
                        amp_loss = (amplitude - gain.detach()) ** 2
                        residual_aux_amplitude_loss = float(self.cfg.residual_aux_amplitude_weight) * (
                            (amp_loss * aux_weight[idx]).sum() / weight_sum
                        )

                    if self.cfg.residual_beta_penalty > 0.0:
                        beta_loss = beta.squeeze(-1) ** 2
                        residual_beta_penalty = float(self.cfg.residual_beta_penalty) * (
                            (beta_loss * aux_weight[idx]).sum() / weight_sum
                        )
                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.cfg.entropy_coef * entropy
                    - auto_entropy_coef * portfolio_entropy
                    + residual_amplitude_penalty
                    + residual_aux_loss
                    + residual_aux_proposal_loss
                    + residual_aux_amplitude_loss
                    + residual_beta_penalty
                )
                if len(self.critics) > 1:
                    critic_losses_each = [
                        ((ret[idx] - critic(critic_obs[idx], user_ids[idx])) ** 2).mean() for critic in self.critics
                    ]
                    critic_loss = torch.stack(critic_losses_each).mean()
                else:
                    value_pred = self.critic(critic_obs[idx], user_ids[idx])
                    critic_loss = ((ret[idx] - value_pred) ** 2).mean()
                if self.critic2 is not None:
                    value_pred2 = self.critic2(critic_obs[idx], user_ids[idx])
                    critic_loss = 0.5 * (critic_loss + ((ret[idx] - value_pred2) ** 2).mean())
                loss = actor_loss + self.cfg.value_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                trainable = list(self.actor.parameters()) + list(self.critics.parameters())
                if self.critic2 is not None:
                    trainable += list(self.critic2.parameters())
                nn.utils.clip_grad_norm_(
                    trainable,
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                auto_entropy_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                if self.cfg.auto_portfolio_entropy:
                    auto_entropy_loss = self.log_portfolio_entropy_coef * (
                        portfolio_entropy.detach() - self.cfg.target_portfolio_entropy
                    )
                    self.entropy_optimizer.zero_grad()
                    auto_entropy_loss.backward()
                    self.entropy_optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                portfolio_entropies.append(float(portfolio_entropy.item()))
                auto_entropy_losses.append(float(auto_entropy_loss.item()))
                auto_entropy_coefs.append(float(torch.exp(self.log_portfolio_entropy_coef).detach().item()))
                residual_amplitudes.append(float(residual_amplitude_mean.detach().item()))
                residual_amplitude_penalties.append(float(residual_amplitude_penalty.detach().item()))
                residual_aux_losses.append(float(residual_aux_loss.detach().item()))
                residual_aux_proposal_losses.append(float(residual_aux_proposal_loss.detach().item()))
                residual_aux_amplitude_losses.append(float(residual_aux_amplitude_loss.detach().item()))
                residual_beta_penalties.append(float(residual_beta_penalty.detach().item()))
                with torch.no_grad():
                    kls.append(float((old_logp[idx] - new_logp).mean().item()))

        buffer.clear()
        return MAPPOUpdateStats(
            actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            approx_kl=float(np.mean(kls)) if kls else 0.0,
            portfolio_entropy=float(np.mean(portfolio_entropies)) if portfolio_entropies else 0.0,
            auto_entropy_coef=float(np.mean(auto_entropy_coefs)) if auto_entropy_coefs else 0.0,
            auto_entropy_loss=float(np.mean(auto_entropy_losses)) if auto_entropy_losses else 0.0,
            residual_amplitude_mean=float(np.mean(residual_amplitudes)) if residual_amplitudes else 0.0,
            residual_amplitude_penalty=float(np.mean(residual_amplitude_penalties)) if residual_amplitude_penalties else 0.0,
            residual_aux_loss=float(np.mean(residual_aux_losses)) if residual_aux_losses else 0.0,
            residual_aux_proposal_loss=float(np.mean(residual_aux_proposal_losses)) if residual_aux_proposal_losses else 0.0,
            residual_aux_amplitude_loss=float(np.mean(residual_aux_amplitude_losses)) if residual_aux_amplitude_losses else 0.0,
            residual_beta_penalty=float(np.mean(residual_beta_penalties)) if residual_beta_penalties else 0.0,
        )

    def _compute_gae(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rewards = arrays["reward"]
        values = arrays["value"]
        dones = arrays["done"]
        user_ids = arrays["user_id"]
        slots = arrays["slot"]
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        for user_id in np.unique(user_ids):
            idxs = np.where(user_ids == user_id)[0]
            idxs = idxs[np.argsort(slots[idxs])]
            gae = 0.0
            for pos in reversed(range(len(idxs))):
                idx = idxs[pos]
                next_value = values[idxs[pos + 1]] if pos + 1 < len(idxs) else 0.0
                nonterminal = 1.0 - dones[idx]
                delta = rewards[idx] + self.cfg.gamma * next_value * nonterminal - values[idx]
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
                advantages[idx] = gae
                returns[idx] = gae + values[idx]
        return advantages, returns
