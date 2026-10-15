"""Past controller acknowledgments; actual service remains in execution only."""
from __future__ import annotations

from dataclasses import dataclass

from predicted_fifo_feedback import PredictedFIFOFeedback


@dataclass(frozen=True)
class ControllerEvent:
    kind: str
    task_id: str
    server: int
    physical_seconds: float
    predicted_service_seconds: float


@dataclass(frozen=True)
class AdmissionAcknowledgment:
    task_id: str
    user: int
    server: int
    admitted_seconds: float
    predicted_service_seconds: float


class ControllerReceiptFeedback(PredictedFIFOFeedback):
    """Additive event surface, without changing original FIFO/broadcast rules.

    Journaling reads observed status/timestamps only after execution advances.
    No actual service requirement or scheduled future completion is exported.
    Completion/drop events are reward/report inputs, not policy observations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller_events = []
        self._journaled = set()

    def advance(self, until):
        super().advance(until)
        additions = []
        for task, record in self._execution.records.items():
            # A dropped request is rejected at its upload-ready event. The
            # ready time is past only when the status actually became dropped.
            candidates = [('admitted', record['admitted']), ('completed', record['completed'])]
            if record['status'] == 'dropped':
                candidates.append(('dropped', record['ready']))
            for kind, time in candidates:
                if time is None or (task, kind) in self._journaled:
                    continue
                if time > self.time:
                    raise RuntimeError('Future controller event cannot be exported')
                additions.append(ControllerEvent(kind, task, record['request'].server,
                                  time, self._predicted[task]))
                self._journaled.add((task, kind))
        order = {'completed': 0, 'admitted': 1, 'dropped': 2}
        additions.sort(key=lambda row: (row.physical_seconds, order[row.kind], row.task_id))
        self._controller_events.extend(additions)

    def controller_events_since(self, cursor):
        if type(cursor) is not int or not 0 <= cursor <= len(self._controller_events):
            raise ValueError('Valid monotonic event cursor required')
        return tuple(self._controller_events[cursor:]), len(self._controller_events)

    def predicted_queue_for_training_reward(self):
        """Fresh PRE-action prediction; never supplied to actor/QL/critic."""
        return self._snapshot().predicted_backlog_seconds
