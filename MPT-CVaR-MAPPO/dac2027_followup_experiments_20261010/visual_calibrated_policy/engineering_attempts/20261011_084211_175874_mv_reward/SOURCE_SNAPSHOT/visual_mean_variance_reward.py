"""Historical MV recurrence with disclosed past-receipt visual input adapter."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from visual_terminal_accounting import ClosedEpisodeLedger, IssuedIdentity


class MeanVarianceRecurrence:
    """Exactly the frozen historical MVReward arithmetic and user ordering.

    One shared statistic is called sequentially for all users, NOT independent
    per-user statistics or a shared QLA team reward. The ratio is a training
    signal, not a normalized evaluation score.
    """
    def __init__(self, mu=1.5):
        if not math.isfinite(mu):
            raise ValueError('Fixed finite historical mu required')
        self.mu = float(mu)
        self.reset()

    def reset(self):
        self.count, self.mean, self.m2, self.prev_sr = 0, 0., 0., None

    def __call__(self, latency):
        if not math.isfinite(latency) or latency < 0:
            raise ValueError('Finite nonnegative observed latency/proxy cost required')
        ret = self.mu - float(latency)
        self.count += 1
        delta = ret - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (ret - self.mean)
        variance = self.m2 / self.count
        ratio = self.mean / max(np.sqrt(variance), 1e-8)
        reward = 0. if self.prev_sr is None else (1. if ratio >= self.prev_sr else 0.)
        self.prev_sr = ratio
        if not all(math.isfinite(x) for x in (self.mean, self.m2, ratio)):
            raise RuntimeError('Nonfinite historical MV state; preserve failure, no silent clipping')
        return float(reward)


def finalize_mean_variance_rewards(transitions, ledger, identities, users, mu=1.5, drop_penalty_seconds=1.):
    """Copy-only per-user terminal cost merge before the last N MV updates.

    Receipt inputs are a VISUAL adaptation, not strict historical slot-input
    reproduction. Empty users retain the historical zero-cost convention.
    """
    if not isinstance(ledger, ClosedEpisodeLedger) or type(users) is not int or users < 1 or not transitions:
        raise ValueError('Actual closed ledger and declared users required')
    if not math.isfinite(drop_penalty_seconds) or drop_penalty_seconds <= 0:
        raise ValueError('Fixed positive drop training penalty required')
    identities = tuple(identities)
    lookup = {row.task_id: row for row in identities}
    if len(lookup) != len(identities) or len(lookup) != ledger.generated or any(not isinstance(row, IssuedIdentity)
            or not 0 <= row.user < users for row in identities):
        raise ValueError('One known user identity per generated task required')
    output = deepcopy(transitions)
    if any(row.get('terminal_closing_applied') or row['reward'].get('reward_semantics') == 'PPO-MV-A' for row in output):
        raise ValueError('Use untouched ordinary transitions; do not credit closure twice')
    costs = [[[] for _ in range(users)] for _ in output]
    seen, completed_latency = set(), []
    for index, row in enumerate(output, start=1):
        if row['index'] != index or (index > 1 and not math.isclose(row['time_seconds'],
                output[index-2]['end_seconds'], rel_tol=0., abs_tol=1e-12)):
            raise ValueError('Ordered contiguous ordinary transitions required')
        for receipt in row['outcome_batch']['completed']:
            task = receipt['task_id']
            if task not in lookup or task in seen:
                raise ValueError('Unknown or duplicate completion identity')
            identity = lookup[task]
            time = receipt['completed_seconds']
            if (receipt['user'] != identity.user or not math.isclose(receipt['actual_arrival_seconds'],
                    identity.actual_arrival_seconds, abs_tol=1e-12, rel_tol=0.)
                    or not row['time_seconds'] < time <= row['end_seconds']):
                raise ValueError('Future/mismatched completion cannot enter MV reward')
            cost = time - identity.actual_arrival_seconds
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('Invalid observed completion latency')
            costs[index-1][identity.user].append(cost)
            completed_latency.append(cost)
            seen.add(task)
        for task, time in row['outcome_batch']['dropped']:
            if task not in lookup or task in seen or not row['time_seconds'] < time <= row['end_seconds']:
                raise ValueError('Invalid/duplicate actual drop receipt')
            costs[index-1][lookup[task].user].append(drop_penalty_seconds)
            seen.add(task)
    if not math.isclose(output[-1]['end_seconds'], ledger.last_action_settlement_seconds, abs_tol=1e-12, rel_tol=0.):
        raise ValueError('Last action and closed ledger disagree')
    for tail in ledger.tail_outcomes:
        if (tail.task_id not in lookup or tail.task_id in seen or lookup[tail.task_id].user != tail.user
                or not 0 <= tail.observed_seconds <= ledger.fixed_end_seconds
                or not math.isfinite(tail.training_penalty_seconds) or tail.training_penalty_seconds < 0):
            raise ValueError('Invalid/duplicate/future terminal identity')
        costs[-1][tail.user].append(tail.training_penalty_seconds)
        if tail.kind == 'completed':
            completed_latency.append(tail.completed_latency_seconds)
        seen.add(tail.task_id)
    if seen != set(lookup) or not math.isclose(math.fsum(completed_latency),
            ledger.all_completed_latency_sum_seconds, abs_tol=1e-10, rel_tol=0.):
        raise ValueError('MV receipt ledger is incomplete or alters uncapped completed latency')
    recurrence = MeanVarianceRecurrence(mu)
    for row, per_user_costs in zip(output, costs):
        means = [math.fsum(values) / len(values) if values else 0. for values in per_user_costs]
        # Preserve shared historical statistic and ascending user calls. Do not
        # secretly switch to per-user statistics or broadcast a team reward.
        rewards = [recurrence(value) for value in means]
        row['shared_QL_reward_diagnostics_not_PPO_target'] = row['reward']
        row['reward'] = {'reward_semantics': 'PPO-MV-A', 'team_reward': None,
            'per_user_rewards': rewards, 'mean_reward_diagnostic_only': math.fsum(rewards) / users,
            'per_user_observed_or_proxy_cost_mean_seconds': means,
            'per_user_outcome_counts': [len(values) for values in per_user_costs],
            'empty_user_cost_convention': 'zero, as historical code',
            'shared_MV_count_after': recurrence.count, 'shared_MV_mean_after': recurrence.mean,
            'shared_MV_m2_after': recurrence.m2, 'shared_MV_ratio_after': float(recurrence.prev_sr)}
    output[-1].update(terminal_closing_applied=True, terminal_done=[True] * users,
        terminal_bootstrap_value=[0.] * users, terminal_close_seconds=ledger.fixed_end_seconds,
        terminal_cost_merge_semantics='per-user costs merged before final ascending-user MV calls; no extra decision/statistic updates')
    return output
