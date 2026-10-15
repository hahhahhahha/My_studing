"""Ordered persistent historical MV statistic, no model or raw-data mutation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
import math

from visual_mean_variance_reward import MeanVarianceRecurrence, finalize_mean_variance_rewards


@dataclass(frozen=True)
class MVStatisticState:
    mu_seconds: float
    count: int
    mean: float
    m2: float
    previous_ratio: float | None


class PersistentMVStream:
    """One statistic per training seed/method, not one per user/episode.

    Engineering validation-receipt streams are separate objects and cannot be
    installed in the real training collector. There is no auto-resume/restore
    or public reset method. A partial failure permanently stops this stream.
    """
    def __init__(self, users, mu_seconds=1.5, purpose='training'):
        if type(users) is not int or users < 1 or purpose not in ('training', 'engineering_validation_receipts'):
            raise ValueError('Fixed user count and explicit training/engineering purpose required')
        self._users, self._purpose = users, purpose
        self._statistic = MeanVarianceRecurrence(mu_seconds)
        self._completed_keys, self._started_keys = [], set()
        self._failed, self._failure_evidence = False, None

    @property
    def users(self):
        return self._users

    @property
    def purpose(self):
        return self._purpose

    @property
    def failed(self):
        return self._failed

    @property
    def failure_evidence(self):
        return deepcopy(self._failure_evidence)

    def state(self):
        value = self._statistic
        return MVStatisticState(value.mu, value.count, value.mean, value.m2,
                                None if value.prev_sr is None else float(value.prev_sr))

    @property
    def next_episode_index(self):
        return len(self._completed_keys) + 1

    def mark_failed(self, error, partial_state=None):
        self._failed = True
        if self._failure_evidence is None:
            self._failure_evidence = {'error': repr(error), 'committed_state': asdict(self.state()),
                'partial_state': asdict(partial_state) if partial_state is not None else None,
                'started_episode_keys': sorted(self._started_keys), 'automatic_retry_allowed': False}

    def close_episode(self, episode_index, episode_key, data_role, ordinary_transitions, ledger,
                      identities, drop_penalty_seconds=1.):
        if self.failed:
            raise RuntimeError('Failed MV stream cannot append/retry/reset')
        expected_role = 'calibration' if self.purpose == 'training' else 'validation'
        if (type(episode_index) is not int or episode_index != self.next_episode_index
                or not isinstance(episode_key, str) or not episode_key.strip()
                or episode_key in self._started_keys or data_role != expected_role):
            raise ValueError('Ordered unique episode and purpose-matched measurement role required')
        before = self.state()
        self._started_keys.add(episode_key)  # burn even a partially failed entry
        working = MeanVarianceRecurrence(before.mu_seconds)
        working.count, working.mean, working.m2, working.prev_sr = (
            before.count, before.mean, before.m2, before.previous_ratio)
        try:
            # Reuse the unchanged receipt/identity/terminal validator to get
            # the exact cost matrix. Its fresh-statistic rewards are NOT the
            # persistent training target and are replaced explicitly below.
            closed = finalize_mean_variance_rewards(ordinary_transitions, ledger, identities,
                self.users, before.mu_seconds, drop_penalty_seconds)
            for row in closed:
                reward = row['reward']
                costs = reward['per_user_observed_or_proxy_cost_mean_seconds']
                if len(costs) != self.users:
                    raise ValueError('One cost for every user in fixed ascending order required')
                rewards = [working(value) for value in costs]
                reward['per_episode_fresh_statistic_count_diagnostic_only'] = reward['shared_MV_count_after']
                reward.update(per_user_rewards=rewards, mean_reward_diagnostic_only=math.fsum(rewards)/self.users,
                    shared_MV_count_after=working.count, shared_MV_mean_after=working.mean,
                    shared_MV_m2_after=working.m2, shared_MV_ratio_after=float(working.prev_sr),
                    MV_stream_episode_index=episode_index, MV_stream_episode_key=episode_key,
                    MV_stream_count_before_episode=before.count,
                    MV_statistic_reset_at_episode_boundary=False)
            if working.count != before.count + len(closed)*self.users:
                raise RuntimeError('Persistent statistic lost or duplicated user-time calls')
            self._statistic = working  # commit only complete valid arithmetic
            self._completed_keys.append(episode_key)
            return {'closed_transitions': closed, 'state_before': before, 'state_after': self.state(),
                'episode_index': episode_index, 'episode_key': episode_key,
                'purpose': self.purpose, 'user_order': 'ascending user ID with one shared statistic',
                'reset_at_episode_boundary': False}
        except Exception as error:
            partial = MVStatisticState(working.mu, working.count, working.mean, working.m2,
                                      None if working.prev_sr is None else float(working.prev_sr))
            self.mark_failed(error, partial)
            raise
