"""Additive strict observe/apply/settle protocol with real past-event rewards."""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
import math

from visual_controller_receipts import AdmissionAcknowledgment, ControllerReceiptFeedback
from visual_environment_adapter import DecisionObservation, VisualEnvironmentAdapter
from visual_shared_reward import CompletedReceipt, RewardOutcomeBatch


@dataclass(frozen=True)
class ControllerDecisionObservation(DecisionObservation):
    # Identical availability for all four methods. Only JSEQ chooses to use
    # this controller's own admission history in its queue reconstruction.
    admission_acknowledgments: tuple[AdmissionAcknowledgment, ...] = ()


class VisualReceiptEnvironment(VisualEnvironmentAdapter):
    def __init__(self, schedule, catalog, mobility, edge_positions, profiles,
                 mobility_start_seconds, broadcast_seconds=.1, nominal_delay_seconds=.1,
                 waiting_capacity=32, link=None):
        super().__init__(schedule, catalog, mobility, edge_positions, profiles,
                         mobility_start_seconds, broadcast_seconds, nominal_delay_seconds,
                         waiting_capacity, link)
        self._feedback = ControllerReceiptFeedback(profiles, waiting_capacity,
                                                  broadcast_seconds, nominal_delay_seconds)
        self._receipt_cursor = 0
        self._admissions = []
        self._unsettled = None
        self._settled_receipts = []
        self._last_settlement = None

    def _capture_events(self):
        rows, cursor = self._feedback.controller_events_since(self._receipt_cursor)
        for row in rows:
            issued = self._issued[row.task_id]
            if row.kind == 'admitted':
                self._admissions.append(AdmissionAcknowledgment(row.task_id, issued.user,
                        row.server, row.physical_seconds, row.predicted_service_seconds))
        self._receipt_cursor = cursor
        return rows

    def observe(self):
        if self._unsettled is not None:
            raise RuntimeError('Settle the last executed action before observing another decision')
        return super().observe()

    def _build_observation(self):
        base = super()._build_observation()
        new_events = self._capture_events()
        if any(row.kind != 'admitted' for row in new_events):
            raise RuntimeError('Uncredited completion/drop between ordered settlement intervals')
        values = {field.name: getattr(base, field.name) for field in fields(DecisionObservation)}
        self._pending = ControllerDecisionObservation(**values,
                                admission_acknowledgments=tuple(self._admissions))
        return self._pending

    def apply(self, portfolios):
        if self._unsettled is not None:
            raise RuntimeError('An action interval is already awaiting settlement')
        if self._failed or self._finished or self._pending is None:
            return super().apply(portfolios)  # original fail-closed guard
        pre = self._pending
        queue_before = self._feedback.predicted_queue_for_training_reward()
        super().apply(portfolios)
        self._unsettled = pre, queue_before

    def settle_action(self):
        if self._failed or self._finished or self._unsettled is None:
            raise RuntimeError('Exactly one settlement per successfully executed action')
        pre, queue_before = self._unsettled
        end = float((pre.index + 1) * Fraction(str(self._schedule.decision_seconds)))
        try:
            self._feedback.advance(end)
            events = self._capture_events()
            if any(not pre.physical_seconds < row.physical_seconds <= end for row in events):
                raise RuntimeError('Receipt outside its physical action interval; no silent filtering')
            completed = tuple(CompletedReceipt(row.task_id, self._issued[row.task_id].user,
                        self._issued[row.task_id].arrival_seconds, row.physical_seconds)
                        for row in events if row.kind == 'completed')
            dropped = tuple((row.task_id, row.physical_seconds) for row in events if row.kind == 'dropped')
            batch = RewardOutcomeBatch(pre.physical_seconds, end, queue_before, completed, dropped)
            self._settled_receipts.extend(events)
            self._last_settlement = end
            self._unsettled = None
            return batch
        except Exception:
            self._failed = True
            raise

    def finish(self, drain_seconds):
        if self._unsettled is not None:
            raise RuntimeError('Final action must be settled before the fixed drain')
        # Last decision is at H, last learning interval ends at H+Delta. Drain
        # is still measured from H, not silently extended by an extra Delta.
        if (not math.isfinite(drain_seconds) or drain_seconds < self._schedule.decision_seconds
                or self._last_settlement is None):
            raise ValueError('Fixed drain must include the final physical action interval')
        result = super().finish(drain_seconds)
        tail = self._capture_events()
        result['controller_accounting'] = {
            'last_action_settlement_seconds': self._last_settlement,
            'settled_completed': sum(row.kind == 'completed' for row in self._settled_receipts),
            'settled_dropped': sum(row.kind == 'dropped' for row in self._settled_receipts),
            'drain_only_completed': sum(row.kind == 'completed' for row in tail),
            'drain_only_dropped': sum(row.kind == 'dropped' for row in tail),
            'admission_acknowledgments': len(self._admissions),
            'drain_outcomes_used_for_learning_reward': False,
            'unfinished_training_terminal_rule_frozen': False}
        result['controller_events_reporting_only'] = tuple(self._settled_receipts) + tail
        return result
