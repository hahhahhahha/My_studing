from __future__ import annotations

from dataclasses import dataclass
from statistics import pvariance
from typing import Dict, List, Optional, Sequence, Tuple

from p2ts import (
    PriorityWeights,
    QueueTask,
    RunningTask,
    execution_priority,
    preemptive_evaluation,
    reorder_cq_by_priority,
)


@dataclass(frozen=True)
class RewardWeights:
    mu1: float
    mu2: float
    mu3: float


@dataclass
class NodeState:
    aq: List[QueueTask]
    cq: List[QueueTask]
    pz: Optional[RunningTask]
    sz: Optional[RunningTask]


class P2MEnvironment:
    """A lightweight Gym-like wrapper around P2M-style scheduling logic."""

    def __init__(
        self,
        num_nodes: int,
        bandwidth_matrix: Sequence[Sequence[float]],
        priority_weights: PriorityWeights,
        reward_weights: RewardWeights,
        arrivals_by_slot: Optional[Sequence[Sequence[QueueTask]]] = None,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if len(bandwidth_matrix) != num_nodes:
            raise ValueError("bandwidth_matrix row count must equal num_nodes")
        if any(len(row) != num_nodes for row in bandwidth_matrix):
            raise ValueError("bandwidth_matrix must be square num_nodes x num_nodes")

        self.num_nodes = num_nodes
        self.bandwidth_matrix = [list(row) for row in bandwidth_matrix]
        self.priority_weights = priority_weights
        self.reward_weights = reward_weights
        self.arrivals_by_slot = [list(slot) for slot in (arrivals_by_slot or [])]

        self.current_slot = 0
        self.nodes: List[NodeState] = []
        self._task_meta: Dict[str, QueueTask] = {}

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        self.current_slot = 0
        self.nodes = [NodeState(aq=[], cq=[], pz=None, sz=None) for _ in range(self.num_nodes)]
        self._task_meta = {}
        return self._build_state(), {}

    def step(self, action: Sequence[object]) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if not self.nodes:
            self.reset()

        incoming = self.arrivals_by_slot[self.current_slot] if self.current_slot < len(self.arrivals_by_slot) else []
        assignments = self._decode_action(action, len(incoming))

        for task, node_idx in zip(incoming, assignments):
            task.arrival_slot = self.current_slot
            task.left_delay = task.tolerable_delay
            self._task_meta[task.task_id] = task
            self.nodes[node_idx].aq.append(task)

        completed_ids: List[str] = []
        dropped_ids: List[str] = []

        for node in self.nodes:
            comp, drop = self._advance_node(node)
            completed_ids.extend(comp)
            dropped_ids.extend(drop)

        success_rate = self._success_rate(completed_ids, dropped_ids)
        avg_delay = self._average_delay(completed_ids)
        load_variance = self._load_variance()

        reward = (
            self.reward_weights.mu1 * success_rate
            - self.reward_weights.mu2 * avg_delay
            - self.reward_weights.mu3 * load_variance
        )

        self.current_slot += 1
        done = self.current_slot >= len(self.arrivals_by_slot) and self._system_empty()

        info = {
            "E": success_rate,
            "D": avg_delay,
            "W": load_variance,
            "completed_task_ids": completed_ids,
            "dropped_task_ids": dropped_ids,
        }
        return self._build_state(), reward, done, info

    def _decode_action(self, action: Sequence[object], n_tasks: int) -> List[int]:
        if n_tasks == 0:
            return []
        if len(action) != n_tasks:
            raise ValueError("action length must match number of incoming AQ tasks")

        assignments: List[int] = []
        for a in action:
            if isinstance(a, int):
                node_idx = a
            else:
                probs = list(a)
                if len(probs) != self.num_nodes:
                    raise ValueError("probability action dimension must equal num_nodes")
                node_idx = max(range(self.num_nodes), key=lambda i: probs[i])

            if node_idx < 0 or node_idx >= self.num_nodes:
                raise ValueError("assigned node index out of range")
            assignments.append(node_idx)
        return assignments

    def _advance_node(self, node: NodeState) -> Tuple[List[str], List[str]]:
        completed_ids: List[str] = []
        dropped_ids: List[str] = []

        if node.sz is not None and node.pz is None:
            node.pz, node.sz = node.sz, None

        if node.aq:
            node.cq.extend(node.aq)
            node.aq = []

        if node.cq:
            node.cq = reorder_cq_by_priority(node.cq, self.current_slot, self.priority_weights)

        result = preemptive_evaluation(
            node.cq,
            pz_task=node.pz,
            sz_task=node.sz,
            current_slot=self.current_slot,
            weights=self.priority_weights,
        )

        if result.discarded_target:
            if result.target_task_id == (node.pz.task_id if node.pz else None):
                dropped_ids.append(node.pz.task_id)
                node.pz = None
            elif result.target_task_id == (node.sz.task_id if node.sz else None):
                dropped_ids.append(node.sz.task_id)
                node.sz = None

        if result.preemptive_task_ids:
            preempt_id = result.preemptive_task_ids[0]
            candidate = self._pop_cq_task(node.cq, preempt_id)
            if candidate is not None:
                if node.pz is not None:
                    node.sz = node.pz
                node.pz = RunningTask(
                    task_id=candidate.task_id,
                    remaining_compute_delay=candidate.compute_delay,
                    left_delay=candidate.left_delay,
                    priority_score=execution_priority(candidate, self.current_slot, self.priority_weights),
                )

        if node.pz is None and node.cq:
            candidate = node.cq.pop(0)
            node.pz = RunningTask(
                task_id=candidate.task_id,
                remaining_compute_delay=candidate.compute_delay,
                left_delay=candidate.left_delay,
                priority_score=execution_priority(candidate, self.current_slot, self.priority_weights),
            )

        if node.pz is not None:
            node.pz.remaining_compute_delay -= 1.0
            node.pz.left_delay -= 1.0
            if node.pz.remaining_compute_delay <= 0:
                completed_ids.append(node.pz.task_id)
                node.pz = None

        for t in node.cq:
            t.left_delay -= 1.0
        if node.sz is not None:
            node.sz.left_delay -= 1.0

        surviving_cq: List[QueueTask] = []
        for t in node.cq:
            if t.left_delay < 0:
                dropped_ids.append(t.task_id)
            else:
                surviving_cq.append(t)
        node.cq = surviving_cq

        if node.sz is not None and node.sz.left_delay < 0:
            dropped_ids.append(node.sz.task_id)
            node.sz = None
        if node.pz is not None and node.pz.left_delay < 0:
            dropped_ids.append(node.pz.task_id)
            node.pz = None

        return completed_ids, dropped_ids

    @staticmethod
    def _pop_cq_task(cq: List[QueueTask], task_id: str) -> Optional[QueueTask]:
        for i, t in enumerate(cq):
            if t.task_id == task_id:
                return cq.pop(i)
        return None

    def _success_rate(self, completed_ids: Sequence[str], dropped_ids: Sequence[str]) -> float:
        denom = len(completed_ids) + len(dropped_ids)
        if denom == 0:
            return 0.0
        return len(completed_ids) / denom

    def _average_delay(self, completed_ids: Sequence[str]) -> float:
        if not completed_ids:
            return 0.0
        completion_slot = self.current_slot + 1  # step() completes one slot of execution before accounting delay
        delays: List[float] = []
        for tid in completed_ids:
            meta = self._task_meta.get(tid)
            if meta is None:
                continue
            delays.append(float(completion_slot - meta.arrival_slot))
        return sum(delays) / len(delays) if delays else 0.0

    def _load_variance(self) -> float:
        loads = [
            len(node.aq) + len(node.cq) + (1 if node.pz is not None else 0) + (1 if node.sz is not None else 0)
            for node in self.nodes
        ]
        return float(pvariance(loads)) if loads else 0.0

    def _system_empty(self) -> bool:
        return all(not node.aq and not node.cq and node.pz is None and node.sz is None for node in self.nodes)

    def _build_state(self) -> Dict[str, object]:
        return {
            "time_slot": self.current_slot,
            "bandwidth": [list(row) for row in self.bandwidth_matrix],
            "nodes": [
                {
                    "AQ": [t.task_id for t in node.aq],
                    "CQ": [t.task_id for t in node.cq],
                    "PZ": node.pz.task_id if node.pz else None,
                    "SZ": node.sz.task_id if node.sz else None,
                }
                for node in self.nodes
            ],
        }
