"""NumPy-only delayed-information rule adaptations; never read execution truth."""
from __future__ import annotations

import math
import numpy as np

from visual_environment_adapter import DecisionObservation
from visual_receipt_environment import ControllerDecisionObservation


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite user-by-server logits required')
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60., 0.))
    result = exponent / exponent.sum(axis=1, keepdims=True)
    result.setflags(write=False)
    return result


class ScoreSoftmaxPolicy:
    name = 'Score-softmax'

    def __init__(self, temperature=.25):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Fixed positive temperature required')
        self.temperature = temperature

    def portfolios(self, observation, packed):
        if not isinstance(observation, DecisionObservation):
            raise TypeError('Observable input only; no environment access')
        m = len(observation.feedback.predicted_backlog_seconds)
        return softmax(packed.actor[:, 5+m:5+2*m] / self.temperature)


class JSEQVisualPolicy:
    """Advance stale predicted work using only own past admission acknowledgments.

    Continuous service-work seconds are reduced at one second per server-second,
    event by event. Admissions at the snapshot timestamp are already in that
    snapshot and must not be counted twice. This is a visual workload adaptation,
    not an exact reproduction of the original queue-length JSEQ algorithm.
    """
    name = 'JSEQ-A'

    def __init__(self, temperature=.25, cost_scale_floor_seconds=.001):
        if not all(math.isfinite(x) and x > 0 for x in (temperature, cost_scale_floor_seconds)):
            raise ValueError('Fixed positive cost/temperature settings required')
        self.temperature, self.scale_floor = temperature, cost_scale_floor_seconds

    def estimated_backlog(self, observation):
        if not isinstance(observation, ControllerDecisionObservation):
            raise TypeError('Only the shared observable controller surface is allowed')
        feedback, now = observation.feedback, observation.physical_seconds
        source = 0. if feedback.missing else feedback.source_seconds
        if source is None or not 0 <= source <= now:
            raise ValueError('Causal feedback source required')
        estimate = np.asarray(feedback.predicted_backlog_seconds, dtype=np.float64).copy()
        acknowledgments = observation.admission_acknowledgments
        if len({row.task_id for row in acknowledgments}) != len(acknowledgments):
            raise ValueError('Duplicate own admission acknowledgment')
        previous = source
        for row in sorted(acknowledgments, key=lambda row: (row.admitted_seconds, row.task_id)):
            if (not 0 <= row.admitted_seconds <= now or not 0 <= row.server < len(estimate)
                    or not math.isfinite(row.predicted_service_seconds) or row.predicted_service_seconds <= 0):
                raise ValueError('Invalid/future admission acknowledgment')
            if row.admitted_seconds <= source:
                continue
            estimate = np.maximum(estimate - (row.admitted_seconds - previous), 0.)
            estimate[row.server] += row.predicted_service_seconds
            previous = row.admitted_seconds
        estimate = np.maximum(estimate - (now - previous), 0.)
        estimate.setflags(write=False)
        return estimate

    def portfolios(self, observation, packed):
        estimate = self.estimated_backlog(observation)
        stale = np.asarray(observation.feedback.predicted_backlog_seconds)
        # The same observable upload/service proxy as every other adapter;
        # replace ONLY its stale waiting term with the acknowledged estimate.
        cost = packed.expected_latency_seconds.astype(np.float64) - stale[None, :] + estimate[None, :]
        scale = np.maximum(cost.std(axis=1, keepdims=True), self.scale_floor)
        return softmax(-cost / (self.temperature * scale))
