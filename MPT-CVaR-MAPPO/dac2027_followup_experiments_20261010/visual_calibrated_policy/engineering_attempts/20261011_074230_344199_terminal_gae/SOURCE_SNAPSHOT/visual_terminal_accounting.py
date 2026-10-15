"""Provisional closed-episode training accounting; reporting truth unchanged.

Only after a fixed drain has actually ended may past outcomes close a training
episode. Unfinished penalties are explicit surrogates, never completion labels.
This engineering rule is not a frozen formal protocol or a learner invocation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IssuedIdentity:
    task_id: str
    user: int
    actual_arrival_seconds: float


@dataclass(frozen=True)
class TerminalOutcome:
    task_id: str
    user: int
    kind: str
    observed_seconds: float
    actual_arrival_seconds: float
    completed_latency_seconds: float | None
    elapsed_lower_bound_seconds: float | None
    training_penalty_seconds: float


@dataclass(frozen=True)
class ClosedEpisodeLedger:
    fixed_end_seconds: float
    last_action_settlement_seconds: float
    generated: int
    completed: int
    dropped: int
    unfinished: int
    tail_outcomes: tuple[TerminalOutcome, ...]
    all_completed_latency_sum_seconds: float
    tail_training_penalty_mean_seconds: float


def build_closed_ledger(final, identities, fixed_end_seconds, users,
                        drop_penalty_seconds=1., unfinished_penalty_seconds=1.):
    if (type(users) is not int or users < 1 or not all(math.isfinite(x) and x > 0 for x in
            (fixed_end_seconds, drop_penalty_seconds, unfinished_penalty_seconds))):
        raise ValueError('Declared positive fixed close time and training penalties required')
    identities = tuple(identities)
    lookup = {row.task_id: row for row in identities}
    if len(lookup) != len(identities) or any(not isinstance(row, IssuedIdentity)
            or not 0 <= row.user < users or not math.isfinite(row.actual_arrival_seconds)
            or not 0 <= row.actual_arrival_seconds <= fixed_end_seconds for row in identities):
        raise ValueError('Valid unique issued identities required; no invented outcomes')
    settled = final['controller_accounting']['last_action_settlement_seconds']
    if not math.isfinite(settled) or not 0 < settled <= fixed_end_seconds:
        raise ValueError('Fixed drain must have covered the last action settlement')
    completed, dropped, admitted, latency, tail = {}, {}, set(), [], []
    for row in final['controller_events_reporting_only']:
        task, kind, time = row['task_id'], row['kind'], row['physical_seconds']
        if task not in lookup or kind not in ('admitted', 'completed', 'dropped'):
            raise ValueError('Unknown task/event in controller history')
        identity = lookup[task]
        if not math.isfinite(time) or not identity.actual_arrival_seconds <= time <= fixed_end_seconds:
            raise ValueError('Invalid/future event cannot close an episode')
        if kind == 'admitted':
            if task in admitted:
                raise ValueError('Duplicate admission receipt')
            admitted.add(task)
            continue
        if task in completed or task in dropped:
            raise ValueError('Duplicate or contradictory task outcome')
        if kind == 'completed':
            completed[task] = time
            cost = time - identity.actual_arrival_seconds
            latency.append(cost)
        else:
            dropped[task] = time
            cost = drop_penalty_seconds
        if time > settled:
            tail.append(TerminalOutcome(task, identity.user, kind, time,
                identity.actual_arrival_seconds, cost if kind == 'completed' else None,
                None, cost))
    if not set(completed) <= admitted or set(dropped) & admitted:
        raise ValueError('Accepted/completed/drop receipt contradiction')
    missing = set(lookup) - set(completed) - set(dropped)
    for task in sorted(missing):
        identity = lookup[task]
        elapsed = fixed_end_seconds - identity.actual_arrival_seconds
        # A past elapsed lower bound, NOT actual future remaining service or a
        # fabricated completion time. Raw final reliability remains unfinished.
        tail.append(TerminalOutcome(task, identity.user, 'unfinished', fixed_end_seconds,
                    identity.actual_arrival_seconds, None, elapsed,
                    max(unfinished_penalty_seconds, elapsed)))
    tasks = final['tasks']
    if (tasks['generated'] != len(lookup) or tasks['completed'] != len(completed)
            or tasks['dropped'] != len(dropped) or tasks['unfinished'] != len(missing)
            or tasks['accepted'] != len(admitted)):
        raise ValueError('Closed receipt totals must match unchanged final task accounting')
    want = sorted(latency)
    got = sorted(final['uncapped_latency_seconds'])
    if len(want) != len(got) or any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=0.) for a, b in zip(want, got)):
        raise ValueError('Observed complete latencies disagree with raw final metrics')
    tail.sort(key=lambda row: (row.observed_seconds, row.task_id, row.kind))
    return ClosedEpisodeLedger(fixed_end_seconds, settled, len(lookup), len(completed),
        len(dropped), len(missing), tuple(tail), math.fsum(latency),
        math.fsum(row.training_penalty_seconds for row in tail) / len(tail) if tail else 0.)


def finalize_training_rewards(transitions, ledger, latency_scale_seconds=1.,
                              v=.5, migration_weight=.05, terminal_weight=1.):
    """Copy-only terminal cost fold; no past observations/critic/H mutation.

    Completed/drop outcomes after the last action interval and censored tasks
    are one bounded closing reward on the final transition. This pooling rule
    is a disclosed candidate, not an exact continuous discounted objective.
    """
    if (not isinstance(ledger, ClosedEpisodeLedger) or not transitions
            or not all(math.isfinite(x) and x >= 0 for x in (v, migration_weight, terminal_weight))
            or not math.isfinite(latency_scale_seconds) or latency_scale_seconds <= 0):
        raise ValueError('Valid provisional closing-rule settings required')
    output = deepcopy(transitions)
    if any(row.get('terminal_closing_applied') for row in output):
        raise ValueError('Terminal outcomes cannot be credited twice')
    seen = set()
    for index, row in enumerate(output, start=1):
        if row['index'] != index or (index > 1 and not math.isclose(row['time_seconds'],
                output[index-2]['end_seconds'], rel_tol=0., abs_tol=1e-12)):
            raise ValueError('Ordered contiguous action transitions required')
        batch = row['outcome_batch']
        for receipt in batch['completed']:
            task, time = receipt['task_id'], receipt['completed_seconds']
            if task in seen or not row['time_seconds'] < time <= row['end_seconds']:
                raise ValueError('Duplicate or out-of-interval action receipt')
            seen.add(task)
        for task, time in batch['dropped']:
            if task in seen or not row['time_seconds'] < time <= row['end_seconds']:
                raise ValueError('Duplicate or out-of-interval action drop receipt')
            seen.add(task)
    if any(row.task_id in seen for row in ledger.tail_outcomes):
        raise ValueError('Action and terminal receipt sets overlap')
    if len(seen) + len(ledger.tail_outcomes) != ledger.generated:
        raise ValueError('Every generated task needs exactly one action outcome or terminal entry')
    last = output[-1]
    if not math.isclose(last['end_seconds'], ledger.last_action_settlement_seconds, abs_tol=1e-12, rel_tol=0.):
        raise ValueError('Closing ledger and final action boundary differ')
    reward = last['reward']
    action_phi = max(0., reward['q_term'] + math.tanh(reward['h_before'] / 5.) *
        (reward['expected_concentration'] - reward['concentration_budget']) +
        v * (reward['training_penalty_mean_seconds'] / latency_scale_seconds + migration_weight * reward['migration']))
    if not math.isclose(-math.tanh(action_phi), reward['team_reward'], abs_tol=1e-12, rel_tol=0.):
        raise ValueError('Closing coefficients do not match the original action reward')
    extra_phi = terminal_weight * v * ledger.tail_training_penalty_mean_seconds / latency_scale_seconds
    closing_reward = -math.tanh(action_phi + extra_phi)
    users = len(reward['per_user_rewards'])
    if users < 1:
        raise ValueError('Shared team reward dimension missing')
    last['terminal_closing_applied'] = True
    last['terminal_done'] = [True] * users
    last['terminal_bootstrap_value'] = [0.] * users
    last['terminal_close_seconds'] = ledger.fixed_end_seconds
    last['terminal_raw_training_penalty_mean_seconds'] = ledger.tail_training_penalty_mean_seconds
    last['terminal_training_phi_addition'] = extra_phi
    last['unmodified_action_team_reward'] = reward['team_reward']
    reward['team_reward'] = closing_reward
    reward['per_user_rewards'] = [closing_reward] * users
    return output
