"""Prepared real historical neural controller; runtime needs CUDA/model gates.

There is no CPU implementation, fake model, launch command or formal-ready claim.
Engineering checks parse this source but NEVER construct this class.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import sys
from types import MethodType
import numpy as np

from visual_learned_bindings import source_bindings
from visual_learning_buffer import map_historical_action_output, HistoricalMAPPOBatchFacade, PreparedVisualEpisode


@dataclass(frozen=True)
class VisualOptimizerSettings:
    learning_rate: float = .001
    epochs: int = 4
    batch_size: int = 128
    clip_epsilon: float = .2
    entropy_coefficient: float = .01
    value_coefficient: float = .5
    max_gradient_norm: float = .5

    def __post_init__(self):
        if (type(self.epochs) is not int or self.epochs < 1 or type(self.batch_size) is not int or self.batch_size < 1
                or not all(math.isfinite(x) and x >= 0 for x in (self.learning_rate, self.clip_epsilon,
                    self.entropy_coefficient, self.value_coefficient, self.max_gradient_norm))
                or self.learning_rate <= 0 or self.max_gradient_norm <= 0):
            raise ValueError('Declared fixed optimizer settings required')


class NeuralVisualController:
    def __init__(self, method, actual_model, settings=None):
        self.method, self.model = method, actual_model
        self.settings = settings or VisualOptimizerSettings()
        self._failed = False
        self._started_episode_tokens = set()
        binding = source_bindings()
        expected_module = {'QLA-MAPPO': ('mappo_gp_sf_dsr.models', 'MAPPOPolicy'),
                           'PPO-MV-A': ('mappo_gp_sf_dsr.paper_policy_baselines', 'SharedGaussianPPO')}
        if method not in expected_module:
            raise ValueError('Only actual declared learning methods')
        module_name, class_name = expected_module[method]
        module = sys.modules.get(module_name)
        if (module is None or type(actual_model) is not getattr(module, class_name, None)
                or str(Path(module.__file__).resolve()) not in binding):
            raise RuntimeError('Actual historical class/source required, no substitute/stub model')
        from qla_environment import guard
        spec = guard()
        if Path(sys.executable).resolve() != (Path(spec['isolated_environment']) / 'Scripts/python.exe').resolve():
            raise RuntimeError('Wrong main model environment')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        self._torch = torch
        if method == 'QLA-MAPPO':
            cfg = actual_model.cfg
            if (actual_model.device.type != 'cuda' or len(actual_model.critics) != 1 or actual_model.critic2 is not None
                    or not cfg.edgewise_price_actor or cfg.residual_actor or cfg.auto_portfolio_entropy):
                raise ValueError('Declared single-critic Full edge-wise visual adaptation required')
            for name, expected in {'learning_rate': self.settings.learning_rate, 'update_epochs': self.settings.epochs,
                    'batch_size': self.settings.batch_size, 'clip_eps': self.settings.clip_epsilon,
                    'entropy_coef': self.settings.entropy_coefficient, 'value_coef': self.settings.value_coefficient,
                    'max_grad_norm': self.settings.max_gradient_norm}.items():
                if getattr(cfg, name) != expected:
                    raise ValueError('Optimizer settings mismatch; no silent per-method alteration')
            self.optimizer = actual_model.optimizer
        else:
            if any(parameter.device.type != 'cuda' for parameter in actual_model.parameters()):
                raise RuntimeError('Actual PPO model must stay on CUDA')
            self.optimizer = torch.optim.Adam(actual_model.parameters(), lr=self.settings.learning_rate)

    def act(self, packed, deterministic=False):
        if self._failed:
            raise RuntimeError('Failed controller cannot auto-retry')
        try:
            users = packed.actor.shape[0]
            servers = packed.q_pressure.shape[0]
            actor = packed.actor.copy()  # private model input, not mutable current observation
            if self.method == 'QLA-MAPPO':
                result = self.model.act(actor, deterministic=deterministic)
                values = self.model.value(packed.critic.copy(), np.arange(users, dtype=np.int64))
                return map_historical_action_output(self.method, result, users, servers, values)
            result = self.model.act(actor, deterministic=deterministic)
            return map_historical_action_output(self.method, result, users, servers)
        except Exception:
            self._failed = True
            raise

    def update_closed_episode(self, prepared_episode):
        if self._failed:
            raise RuntimeError('Failed controller cannot append/retry updates')
        if (not isinstance(prepared_episode, PreparedVisualEpisode)
                or prepared_episode.episode_token in self._started_episode_tokens):
            raise RuntimeError('Prepared episode required; no repeated optimizer update on the same episode')
        self._started_episode_tokens.add(prepared_episode.episode_token)
        try:
            if self.method == 'QLA-MAPPO':
                facade = HistoricalMAPPOBatchFacade(prepared_episode)
                if len(facade) < 2:
                    raise ValueError('At least two prepared samples for a genuine update')
                sentinel = object()
                previous = self.model.__dict__.get('_compute_gae', sentinel)
                def supplied_gae(_model, arrays):
                    return facade.supplied_gae(arrays)
                # Instance-local hook only. The original package/class files
                # and PPO loss/optimizer remain untouched, even on failure.
                self.model._compute_gae = MethodType(supplied_gae, self.model)
                try:
                    stats = asdict(self.model.update(facade))
                finally:
                    if previous is sentinel:
                        del self.model._compute_gae
                    else:
                        self.model._compute_gae = previous
                if len(facade) != 0 or not all(math.isfinite(float(value)) for value in stats.values()):
                    raise RuntimeError('Actual MAPPO update did not consume a finite prepared episode')
                return stats
            return self._update_ppo(prepared_episode)
        except Exception:
            self._failed = True  # partial optimizer state is evidence, not an auto-retry
            raise

    def _update_ppo(self, episode):
        torch = self._torch
        data = episode.flat()
        device = next(self.model.parameters()).device
        tensors = {name: torch.tensor(np.asarray(data[name]), dtype=torch.float32, device=device)
                   for name in ('actor_obs', 'latent_action', 'log_prob', 'advantages', 'returns')}
        advantage = tensors['advantages']
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        n = len(advantage)
        if n < 2:
            raise ValueError('At least two behavior samples required')
        logs = []
        for _ in range(self.settings.epochs):
            permutation = torch.randperm(n, device=device)
            for start in range(0, n, self.settings.batch_size):
                index = permutation[start:start+self.settings.batch_size]
                distribution = self.model.distribution(tensors['actor_obs'][index])
                new_logp = distribution.log_prob(tensors['latent_action'][index]).sum(dim=-1)
                ratio = torch.exp(new_logp - tensors['log_prob'][index])
                surrogate = torch.minimum(ratio * advantage[index],
                    ratio.clamp(1-self.settings.clip_epsilon, 1+self.settings.clip_epsilon) * advantage[index])
                actor_loss = -surrogate.mean()
                values = self.model.critic(tensors['actor_obs'][index]).squeeze(-1)
                critic_loss = ((values - tensors['returns'][index]) ** 2).mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = actor_loss + self.settings.value_coefficient * critic_loss - self.settings.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; do not clip or silently retry a bad update')
                self.optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_gradient_norm)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient; preserve partial controller state')
                self.optimizer.step()
                logs.append([float(actor_loss.detach()), float(critic_loss.detach()), float(entropy.detach()),
                             float((tensors['log_prob'][index] - new_logp).detach().mean())])
        means = np.mean(logs, axis=0)
        if not np.isfinite(means).all():
            raise RuntimeError('Nonfinite optimizer report')
        return dict(zip(('actor_loss', 'critic_loss', 'entropy', 'approx_kl'), means.tolist()))
