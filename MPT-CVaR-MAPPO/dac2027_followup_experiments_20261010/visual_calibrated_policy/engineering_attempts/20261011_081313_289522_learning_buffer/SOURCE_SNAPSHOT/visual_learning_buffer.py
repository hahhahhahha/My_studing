"""Strict closed-episode learning buffer; never silently truncate user samples."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from uuid import uuid4
import numpy as np

from visual_temporal_gae import per_user_temporal_gae, flatten_after_gae


def owned_readonly(value):
    value = np.asarray(value, dtype=np.float32).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class VisualActionSample:
    portfolios: np.ndarray
    latent_actions: np.ndarray
    log_probabilities: np.ndarray
    values: np.ndarray


def map_historical_action_output(method, result, users, servers, qla_values=None):
    """Pure tuple-layout validation, not a model call or fallback implementation."""
    if method == 'QLA-MAPPO' and len(result) == 3 and qla_values is not None:
        portfolios, latent, logp = result
        values = qla_values
    elif method == 'PPO-MV-A' and len(result) == 4 and qla_values is None:
        latent, logp, values, portfolios = result
    else:
        raise ValueError('Historical act tuple layout/critic value mismatch')
    portfolios, latent, logp, values = [np.asarray(x) for x in (portfolios, latent, logp, values)]
    if (portfolios.shape != (users, servers) or latent.shape != portfolios.shape
            or logp.shape != (users,) or values.shape != (users,)
            or not all(np.isfinite(x).all() for x in (portfolios, latent, logp, values))
            or np.any(portfolios < 0) or not np.allclose(portfolios.sum(axis=1), 1., atol=1e-6, rtol=0.)):
        raise ValueError('Finite action/critic arrays with complete user/server correspondence required')
    stable = latent.astype(np.float64) - latent.max(axis=1, keepdims=True)
    expected = np.exp(stable)
    expected /= expected.sum(axis=1, keepdims=True)
    if not np.allclose(portfolios, expected, atol=1e-6, rtol=0.):
        raise ValueError('Executed portfolios do not correspond to sampled latent actions')
    return VisualActionSample(*(owned_readonly(x) for x in (portfolios, latent, logp, values)))


@dataclass(frozen=True)
class PreparedVisualEpisode:
    episode_token: str
    time_user_fields: dict
    advantages: np.ndarray
    returns: np.ndarray
    physical_durations: np.ndarray

    def flat(self):
        return flatten_after_gae(self.time_user_fields, self.advantages, self.returns)


class VisualEpisodeBuffer:
    def __init__(self, users, servers, actor_dim, critic_dim, max_episode_steps):
        if not all(type(x) is int and x > 0 for x in (users, servers, actor_dim, critic_dim, max_episode_steps)):
            raise ValueError('Explicit positive complete-episode dimensions required')
        self.users, self.servers, self.actor_dim, self.critic_dim = users, servers, actor_dim, critic_dim
        self.max_episode_steps = max_episode_steps
        self._rows, self._prepared = [], False

    def add_action(self, index, start_seconds, end_seconds, packed, sample):
        if self._prepared or len(self._rows) >= self.max_episode_steps:
            raise RuntimeError('Prepared/full episode cannot append or discard earlier samples')
        if index != len(self._rows) + 1 or not all(math.isfinite(x) for x in (start_seconds, end_seconds)) or end_seconds <= start_seconds:
            raise ValueError('Ordered physical action intervals required')
        if self._rows and not math.isclose(start_seconds, self._rows[-1]['end'], abs_tol=1e-12, rel_tol=0.):
            raise ValueError('Action intervals must be contiguous')
        if not isinstance(sample, VisualActionSample):
            raise TypeError('Validated historical action result required')
        if packed.actor.shape != (self.users, self.actor_dim) or packed.critic.shape != (self.users, self.critic_dim):
            raise ValueError('Pre-action actor and critic layout mismatch')
        # Revalidate even a manually constructed sample; do not trust its name.
        sample = map_historical_action_output('QLA-MAPPO',
            (sample.portfolios, sample.latent_actions, sample.log_probabilities), self.users, self.servers, sample.values)
        if not np.isfinite(packed.actor).all() or not np.isfinite(packed.critic).all():
            raise ValueError('Nonfinite pre-action features')
        self._rows.append({'index': index, 'start': start_seconds, 'end': end_seconds,
            'actor_obs': owned_readonly(packed.actor), 'critic_obs': owned_readonly(packed.critic),
            'latent_action': sample.latent_actions, 'log_prob': sample.log_probabilities,
            'value': sample.values, 'portfolio': sample.portfolios})

    def prepare_closed_episode(self, closed_transitions, gamma_per_second=.99, lambda_per_second=.95):
        if self._prepared or len(closed_transitions) != len(self._rows) or not self._rows:
            raise RuntimeError('Exactly one complete reward closure for all retained action samples required')
        rewards = []
        for stored, transition in zip(self._rows, closed_transitions):
            if (transition['index'] != stored['index'] or not math.isclose(transition['time_seconds'], stored['start'], abs_tol=1e-12)
                    or not math.isclose(transition['end_seconds'], stored['end'], abs_tol=1e-12)
                    or transition['pre_critic_sha256'] != hashlib.sha256(stored['critic_obs'].tobytes()).hexdigest()
                    or not np.allclose(transition['portfolios'], stored['portfolio'], atol=1e-6, rtol=0.)):
                raise ValueError('Closed rewards/actions/time do not match the stored behavior samples')
            reward = np.asarray(transition['reward']['per_user_rewards'], dtype=np.float64)
            if (reward.shape != (self.users,) or not np.isfinite(reward).all() or np.any(reward < -1) or np.any(reward > 0)
                    or not np.allclose(reward, transition['reward']['team_reward'], rtol=0., atol=1e-7)):
                raise ValueError('Finite shared bounded rewards required')
            rewards.append(reward)
        last = closed_transitions[-1]
        if (not last.get('terminal_closing_applied') or last.get('terminal_done') != [True] * self.users
                or last.get('terminal_bootstrap_value') != [0.] * self.users):
            raise ValueError('Actual fixed closed-episode terminal accounting required')
        fields = {name: np.stack([row[name] for row in self._rows]) for name in
                  ('actor_obs', 'critic_obs', 'latent_action', 'log_prob', 'value')}
        fields['reward'] = np.asarray(rewards)
        fields['user_id'] = np.tile(np.arange(self.users), (len(self._rows), 1))
        fields['slot'] = np.tile(np.arange(1, len(self._rows)+1)[:, None], (1, self.users))
        terminal = np.zeros((len(self._rows), self.users), dtype=bool)
        terminal[-1] = True
        fields['done'] = terminal
        next_values = np.zeros_like(fields['value'])
        next_values[:-1] = fields['value'][1:]  # same user, next pre-action value
        duration = np.asarray([row['end'] - row['start'] for row in self._rows])
        advantages, returns = per_user_temporal_gae(fields['reward'], fields['value'], next_values,
            terminal, duration, gamma_per_second, lambda_per_second)
        for value in fields.values():
            value.setflags(write=False)
        duration.setflags(write=False)
        self._prepared = True
        return PreparedVisualEpisode(uuid4().hex, fields, advantages, returns, duration)


class HistoricalMAPPOBatchFacade:
    """One-shot in-memory facade for the unchanged historical update method."""
    def __init__(self, episode):
        if not isinstance(episode, PreparedVisualEpisode):
            raise TypeError('Per-user GAE must be prepared before historical update')
        flat = episode.flat()
        self._arrays = {key: np.asarray(value).copy() for key, value in flat.items() if key not in ('advantages', 'returns')}
        n, m = self._arrays['latent_action'].shape
        self._arrays['aux_target'] = np.zeros((n, max(m-1, 1)), dtype=np.float32)
        self._arrays['aux_weight'] = np.zeros(n, dtype=np.float32)
        self.advantages = flat['advantages'].astype(np.float32)
        self.returns = flat['returns'].astype(np.float32)
        self._consumed = False

    def __len__(self):
        return 0 if self._consumed else len(self.advantages)

    def arrays(self):
        if self._consumed:
            raise RuntimeError('Consumed episode cannot update again')
        return self._arrays

    def supplied_gae(self, arrays):
        if self._consumed or arrays is not self._arrays:
            raise RuntimeError('Prepared GAE may be used only with its own behavior batch')
        return self.advantages.copy(), self.returns.copy()

    def clear(self):
        if self._consumed:
            raise RuntimeError('Episode was already consumed')
        self._consumed = True  # retain arrays for failure evidence, no silent deletion
