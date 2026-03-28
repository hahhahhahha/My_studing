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


@dataclass
class TransferItem:
    task: QueueTask
    src_node: int
    dst_node: int
    remaining_size: float


class P2MEnvironment:
    """A Gym-like wrapper around P2M scheduling with transfer-queue modeling."""

    def __init__(
        self,
        num_nodes: int,
        bandwidth_matrix: Sequence[Sequence[float]],
        priority_weights: PriorityWeights,
        reward_weights: RewardWeights,
        arrivals_by_slot: Optional[Sequence[Sequence[QueueTask]]] = None,
        arrivals_source_by_slot: Optional[Sequence[Sequence[int]]] = None,
        transfer_size_by_task: Optional[Dict[str, float]] = None,
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
        self.arrivals_source_by_slot = [list(slot) for slot in (arrivals_source_by_slot or [])]
        self.transfer_size_by_task = dict(transfer_size_by_task or {})

        self.current_slot = 0
        self.nodes: List[NodeState] = []
        self.link_queues: Dict[Tuple[int, int], List[TransferItem]] = {}
        self._task_meta: Dict[str, QueueTask] = {}

    def reset(self) -> Tuple[Dict[str, object], Dict[str, object]]:
        self.current_slot = 0
        self.nodes = [NodeState(aq=[], cq=[], pz=None, sz=None) for _ in range(self.num_nodes)]
        self.link_queues = {
            (src, dst): []
            for src in range(self.num_nodes)
            for dst in range(self.num_nodes)
            if src != dst
        }
        self._task_meta = {}
        return self._build_state(), {}

    def step(self, action: Sequence[object]) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if not self.nodes:
            self.reset()

        incoming = self.arrivals_by_slot[self.current_slot] if self.current_slot < len(self.arrivals_by_slot) else []
        assignments = self._decode_action(action, len(incoming))
        source_nodes = self._resolve_source_nodes(self.current_slot, len(incoming))

        for task, source_node, target_node in zip(incoming, source_nodes, assignments):
            task.arrival_slot = self.current_slot
            task.left_delay = task.tolerable_delay
            self._task_meta[task.task_id] = task
            if source_node == target_node:
                self.nodes[target_node].aq.append(task)
            else:
                transfer_size = self.transfer_size_by_task.get(task.task_id, 1.0)
                if transfer_size <= 0:
                    raise ValueError("transfer size must be positive")
                self.link_queues[(source_node, target_node)].append(
                    TransferItem(
                        task=task,
                        src_node=source_node,
                        dst_node=target_node,
                        remaining_size=float(transfer_size),
                    )
                )

        completed_ids: List[str] = []
        dropped_ids: List[str] = []

        for node in self.nodes:
            comp, drop = self._advance_node(node)
            completed_ids.extend(comp)
            dropped_ids.extend(drop)

        transferred_ids, transfer_dropped = self._advance_transfers()
        dropped_ids.extend(transfer_dropped)

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
            "transferred_task_ids": transferred_ids,
        }
        return self._build_state(), reward, done, info

    def _decode_action(self, action: Sequence[object], n_tasks: int) -> List[int]:
        if n_tasks == 0:
            return []

        normalized_actions = self._normalize_batch(action, n_tasks, default=0)
        assignments: List[int] = []
        for a in normalized_actions:
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

    def _normalize_batch(self, values: Sequence[object], n_tasks: int, default: object) -> List[object]:
        if n_tasks == 0:
            return []
        if not values:
            return [default for _ in range(n_tasks)]
        if len(values) == n_tasks:
            return list(values)
        if len(values) == 1:
            return [values[0] for _ in range(n_tasks)]
        if len(values) < n_tasks:
            return [values[i % len(values)] for i in range(n_tasks)]
        return list(values[:n_tasks])

    def _resolve_source_nodes(self, slot: int, n_tasks: int) -> List[int]:
        if n_tasks == 0:
            return []

        if slot < len(self.arrivals_source_by_slot):
            provided = self.arrivals_source_by_slot[slot]
            normalized = self._normalize_batch(provided, n_tasks, default=0)
            source_nodes = [int(x) for x in normalized]
        else:
            source_nodes = [(slot + i) % self.num_nodes for i in range(n_tasks)]

        for n in source_nodes:
            if n < 0 or n >= self.num_nodes:
                raise ValueError("source node index out of range")
        return source_nodes

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

    def _advance_transfers(self) -> Tuple[List[str], List[str]]:
        transferred_ids: List[str] = []
        dropped_ids: List[str] = []

        for (src, dst), queue in self.link_queues.items():
            if not queue:
                continue

            for item in queue:
                item.task.left_delay -= 1.0

            alive_queue: List[TransferItem] = []
            for item in queue:
                if item.task.left_delay < 0:
                    dropped_ids.append(item.task.task_id)
                else:
                    alive_queue.append(item)
            queue[:] = alive_queue

            bandwidth = float(self.bandwidth_matrix[src][dst])
            if bandwidth <= 0:
                continue

            while queue and bandwidth > 0:
                head = queue[0]
                sent = min(head.remaining_size, bandwidth)
                head.remaining_size -= sent
                bandwidth -= sent

                if head.remaining_size <= 1e-12:
                    delivered = queue.pop(0)
                    self.nodes[dst].aq.append(delivered.task)
                    transferred_ids.append(delivered.task.task_id)

        return transferred_ids, dropped_ids

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
        completion_slot = self.current_slot + 1  # current slot index starts at 0, so end-of-slot completion time is (current_slot + 1)
        delays: List[float] = []
        for tid in completed_ids:
            meta = self._task_meta.get(tid)
            if meta is None:
                continue
            delays.append(float(completion_slot - meta.arrival_slot))
        return sum(delays) / len(delays) if delays else 0.0

    def _load_variance(self) -> float:
        loads = []
        for node in self.nodes:
            cq_load = sum(t.compute_delay for t in node.cq)
            pz_load = node.pz.remaining_compute_delay if node.pz is not None else 0.0
            sz_load = node.sz.remaining_compute_delay if node.sz is not None else 0.0
            loads.append(float(cq_load + pz_load + sz_load))
        return float(pvariance(loads)) if loads else 0.0

    def _system_empty(self) -> bool:
        nodes_empty = all(not node.aq and not node.cq and node.pz is None and node.sz is None for node in self.nodes)
        links_empty = all(not q for q in self.link_queues.values())
        return nodes_empty and links_empty

    def _build_state(self) -> Dict[str, object]:
        link_state = []
        for (src, dst), queue in self.link_queues.items():
            link_state.append(
                {
                    "src": src,
                    "dst": dst,
                    "bandwidth": float(self.bandwidth_matrix[src][dst]),
                    "queue_task_ids": [item.task.task_id for item in queue],
                    "queue_remaining_sizes": [item.remaining_size for item in queue],
                }
            )

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
            "links": link_state,
        }
