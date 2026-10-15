"""Provisional continuous-time shared-reward receipt accounting, no learner."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CompletedReceipt:
    task_id: str
    user: int
    actual_arrival_seconds: float
    completed_seconds: float


@dataclass(frozen=True)
class RewardOutcomeBatch:
    start_seconds: float
    end_seconds: float
    predicted_queue_before_seconds: tuple[float, ...]
    completed: tuple[CompletedReceipt, ...] = ()
    dropped: tuple[tuple[str, float], ...] = ()


class VisualSharedReward:
    def __init__(self, scales, v=.5, migration_weight=.05, concentration_factor=2., drop_penalty_seconds=1.):
        values = (v, migration_weight, concentration_factor, drop_penalty_seconds)
        if any(not math.isfinite(x) or x < 0 for x in values) or drop_penalty_seconds <= 0:
            raise ValueError('Finite fixed reward settings required')
        self.scales = scales
        self.v, self.migration_weight = v, migration_weight
        self.concentration_factor, self.drop_penalty = concentration_factor, drop_penalty_seconds
        self.h_queue = 0.
        self._last_index, self._last_end = 0, None
        self._completed, self._dropped = set(), set()

    def compute(self, pre_observation, portfolios, outcome):
        if pre_observation.index != self._last_index + 1 or not math.isclose(
                outcome.start_seconds, pre_observation.physical_seconds, abs_tol=1e-12, rel_tol=0.):
            raise ValueError('Exactly one ordered reward update per observed action')
        if (not all(math.isfinite(x) for x in (outcome.start_seconds, outcome.end_seconds))
                or outcome.end_seconds <= outcome.start_seconds
                or (self._last_end is not None and not math.isclose(
                    outcome.start_seconds, self._last_end, abs_tol=1e-12, rel_tol=0.))):
            raise ValueError('Contiguous physical reward receipt intervals required')
        users, m = len(pre_observation.users), len(self.scales.profiles)
        if m < 2:
            raise ValueError('Cooperative concentration reward requires at least two servers')
        weights = np.asarray(portfolios, dtype=np.float64)
        if (weights.shape != (users, m) or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.allclose(weights.sum(axis=1), 1., atol=1e-7, rtol=0.)):
            raise ValueError('Valid current executed portfolios required')
        queues = np.asarray(outcome.predicted_queue_before_seconds, dtype=np.float64)
        if queues.shape != (m,) or not np.isfinite(queues).all() or np.any(queues < 0):
            raise ValueError('Pre-action predicted server queues required, not actual future work')
        fresh_completed, fresh_dropped, latencies = set(), set(), []
        for row in outcome.completed:
            if (row.task_id in self._completed | self._dropped | fresh_completed
                    or not 0 <= row.user < users
                    or not all(math.isfinite(x) for x in (row.actual_arrival_seconds, row.completed_seconds))
                    or row.actual_arrival_seconds < 0 or row.actual_arrival_seconds > row.completed_seconds
                    or not outcome.start_seconds < row.completed_seconds <= outcome.end_seconds):
                raise ValueError('Duplicate, invalid or future completion receipt')
            fresh_completed.add(row.task_id)
            latencies.append(row.completed_seconds - row.actual_arrival_seconds)
        for task, time in outcome.dropped:
            if (task in self._completed | self._dropped | fresh_completed | fresh_dropped
                    or not math.isfinite(time) or not outcome.start_seconds < time <= outcome.end_seconds):
                raise ValueError('Duplicate, invalid or future drop receipt')
            fresh_dropped.add(task)
        ref_index = self.scales.profiles.index('CPU-1T')
        demand = np.asarray([math.fsum(request.predicted_service_seconds[ref_index]
                        for request in user.requests) for user in pre_observation.users])
        joint = weights.T @ demand
        total = float(joint.sum())
        shares = joint / total if total > 0 else np.zeros(m)
        q = np.clip(queues / (self.scales.decision_seconds * self.scales.q_max), 0., 1.)
        q_term = float(q @ shares)
        concentration = max(0., min(1., (float(shares @ shares) - 1. / m) / (1. - 1. / m))) if total > 0 else 0.
        budget = max(.02, (self.concentration_factor - 1.) / max(m - 1., 1.))
        violation = concentration - budget
        previous = np.asarray([user.previous_portfolio for user in pre_observation.users])
        migration = float(np.mean(np.sum(np.abs(weights - previous), axis=1)))
        count = len(latencies) + len(fresh_dropped)
        penalty_mean = (math.fsum(latencies) + len(fresh_dropped) * self.drop_penalty) / count if count else 0.
        h_before = self.h_queue
        phi = max(0., q_term + math.tanh(h_before / 5.) * violation +
                  self.v * (penalty_mean / self.scales.latency_scale_seconds + self.migration_weight * migration))
        reward = -math.tanh(phi)
        # Preserve the canonical0.05 H-update rate per physical second, not
        # per shortened50ms decision; final rule remains a pilot setting.
        self.h_queue = max(0., h_before + .05 * (outcome.end_seconds - outcome.start_seconds) * violation)
        self._last_index, self._last_end = pre_observation.index, outcome.end_seconds
        self._completed.update(fresh_completed)
        self._dropped.update(fresh_dropped)
        return {'team_reward': reward, 'per_user_rewards': np.full(users, reward, dtype=np.float32),
                'q_term': q_term, 'expected_reference_demand_shares': shares.tolist(),
                'expected_concentration': concentration, 'concentration_budget': budget,
                'h_before': h_before, 'h_after': self.h_queue, 'migration': migration,
                'training_penalty_mean_seconds': penalty_mean,
                'completed_count': len(latencies), 'observed_drop_count': len(fresh_dropped),
                'completed_untruncated_mean_seconds': math.fsum(latencies) / len(latencies) if latencies else None}
