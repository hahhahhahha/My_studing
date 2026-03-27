from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PriorityWeights:
    omega1: float
    omega2: float
    omega3: float
    omega4: float

    def validate(self) -> None:
        total = self.omega1 + self.omega2 + self.omega3 + self.omega4
        if abs(total - 1.0) > 1e-9:
            raise ValueError("omega1 + omega2 + omega3 + omega4 must equal 1.")


@dataclass
class QueueTask:
    task_id: str
    task_type_score: float  # e_{n,j,τ}
    tolerable_delay: float  # D_tlr
    compute_delay: float  # D_comp
    arrival_slot: int  # τ
    left_delay: float  # D_left at current time slot


@dataclass
class RunningTask:
    task_id: str
    remaining_compute_delay: float  # D_rem^comp
    left_delay: float  # D_left


@dataclass
class PreemptionResult:
    preemptive_task_ids: List[str]
    regular_task_ids: List[str]
    discarded_target: bool
    target_task_id: Optional[str]
    situation: str


def execution_priority(
    task: QueueTask, current_slot: int, weights: PriorityWeights
) -> float:
    """
    Priority score from Eq. (7) in the paper:
    p = ω1*e + ω2*s + ω3*g + ω4*u
    where:
    s = 1 / D_tlr, g = 1 / o (here modeled as compute_delay), u = (t-τ) / D_tlr.
    """
    if task.tolerable_delay <= 0 or task.compute_delay <= 0:
        raise ValueError("tolerable_delay and compute_delay must be positive.")
    weights.validate()

    s = 1.0 / task.tolerable_delay
    g = 1.0 / task.compute_delay
    u = (current_slot - task.arrival_slot) / task.tolerable_delay
    return (
        weights.omega1 * task.task_type_score
        + weights.omega2 * s
        + weights.omega3 * g
        + weights.omega4 * u
    )


def reorder_cq_by_priority(
    cq_tasks: Sequence[QueueTask], current_slot: int, weights: PriorityWeights
) -> List[QueueTask]:
    return sorted(
        cq_tasks,
        key=lambda t: execution_priority(t, current_slot=current_slot, weights=weights),
        reverse=True,
    )


def _sum_compute_delay(cq_tasks: Sequence[QueueTask], k: int) -> float:
    # k is 1-based sequence number in the paper.
    if k <= 0:
        return 0.0
    return sum(t.compute_delay for t in cq_tasks[:k])


def _upsilon_a(x: float, k: int, pz_remaining: float, cq_tasks: Sequence[QueueTask]) -> float:
    # Eq. (9)
    return x - pz_remaining - _sum_compute_delay(cq_tasks, k)


def _upsilon_b(
    x: float, k: int, pz_remaining: float, sz_remaining: float, cq_tasks: Sequence[QueueTask]
) -> float:
    # Eq. (10)
    return x - pz_remaining - sz_remaining - _sum_compute_delay(cq_tasks, k)


def preemptive_evaluation(
    cq_tasks_desc_priority: Sequence[QueueTask],
    pz_task: Optional[RunningTask],
    sz_task: Optional[RunningTask] = None,
) -> PreemptionResult:
    """
    Reproduces Algorithm 1 decision flow in Section IV-B for one CPN node.
    Input CQ must already be ordered from highest to lowest priority.
    """
    regular_ids = [t.task_id for t in cq_tasks_desc_priority]
    if pz_task is None or not cq_tasks_desc_priority:
        return PreemptionResult([], regular_ids, False, None, "none")

    preemptive_ids: List[str] = []

    if sz_task is None:
        # Situation A: target is task -2 in PZ.
        for i, uncertain in enumerate(cq_tasks_desc_priority, start=1):
            if _upsilon_a(
                uncertain.left_delay, i, pz_task.remaining_compute_delay, cq_tasks_desc_priority
            ) >= 0:
                regular_from_here = [t.task_id for t in cq_tasks_desc_priority[i - 1 :]]
                return PreemptionResult(
                    preemptive_ids,
                    regular_from_here,
                    False,
                    pz_task.task_id,
                    "A",
                )

            if _upsilon_a(
                pz_task.left_delay, i, pz_task.remaining_compute_delay, cq_tasks_desc_priority
            ) >= 0:
                preemptive_ids.append(uncertain.task_id)
            else:
                # Target discarded; all CQ tasks become regular tasks.
                return PreemptionResult(
                    [],
                    [t.task_id for t in cq_tasks_desc_priority],
                    True,
                    pz_task.task_id,
                    "A",
                )

        return PreemptionResult(preemptive_ids, [], False, pz_task.task_id, "A")

    # Situation B: target is task -1 in SZ, task -2 in PZ is not preempted.
    for i, uncertain in enumerate(cq_tasks_desc_priority, start=1):
        if _upsilon_b(
            uncertain.left_delay,
            i,
            pz_task.remaining_compute_delay,
            sz_task.remaining_compute_delay,
            cq_tasks_desc_priority,
        ) >= 0:
            regular_from_here = [t.task_id for t in cq_tasks_desc_priority[i - 1 :]]
            return PreemptionResult(
                preemptive_ids,
                regular_from_here,
                False,
                sz_task.task_id,
                "B",
            )

        if _upsilon_b(
            sz_task.left_delay,
            i,
            pz_task.remaining_compute_delay,
            sz_task.remaining_compute_delay,
            cq_tasks_desc_priority,
        ) >= 0:
            preemptive_ids.append(uncertain.task_id)
        else:
            return PreemptionResult(
                [],
                [t.task_id for t in cq_tasks_desc_priority],
                True,
                sz_task.task_id,
                "B",
            )

    return PreemptionResult(preemptive_ids, [], False, sz_task.task_id, "B")


def execution_order_after_evaluation(
    result: PreemptionResult,
    cq_tasks_desc_priority: Sequence[QueueTask],
    pz_task: Optional[RunningTask],
    sz_task: Optional[RunningTask] = None,
) -> Tuple[str, ...]:
    """
    Returns task execution order IDs after evaluation, following Section IV-B:
    - Situation A: preemptive -> target(-2) -> regular
    - Situation B: task(-2) -> preemptive -> target(-1) -> regular
    """
    id_to_task = {t.task_id: t for t in cq_tasks_desc_priority}
    regular_ids = [tid for tid in result.regular_task_ids if tid in id_to_task]

    if result.situation == "A":
        if result.discarded_target or pz_task is None:
            return tuple(regular_ids)
        return tuple(result.preemptive_task_ids + [pz_task.task_id] + regular_ids)

    if result.situation == "B":
        order: List[str] = []
        if pz_task is not None:
            order.append(pz_task.task_id)
        order.extend(result.preemptive_task_ids)
        if not result.discarded_target and sz_task is not None:
            order.append(sz_task.task_id)
        order.extend(regular_ids)
        return tuple(order)

    # No trigger.
    order = []
    if pz_task is not None:
        order.append(pz_task.task_id)
    if sz_task is not None:
        order.append(sz_task.task_id)
    order.extend(t.task_id for t in cq_tasks_desc_priority)
    return tuple(order)