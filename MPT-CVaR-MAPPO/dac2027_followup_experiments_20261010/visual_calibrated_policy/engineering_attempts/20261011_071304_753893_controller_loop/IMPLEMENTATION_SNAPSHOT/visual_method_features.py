"""Common observable features for frozen-structure QLA/PPO visual adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from execution_service_catalog import SESSIONS
from visual_environment_adapter import DecisionObservation
from visual_upload_link import UploadLinkConfig


@dataclass(frozen=True)
class VisualFeatureScales:
    location_low: tuple[float, float]
    location_span: tuple[float, float]
    edge_positions: tuple[tuple[float, float], ...]
    profiles: tuple[str, ...]
    mean_payload_bytes: float
    mean_profile_service_seconds: tuple[float, ...]
    reference_service_seconds: float
    decision_seconds: float = .05
    request_scale: float = 7.
    q_max: float = 3.
    latency_scale_seconds: float = 1.
    age_scale_seconds: float = .5
    calibration_role: str = 'calibration'

    def __post_init__(self):
        values = (*self.location_span, self.mean_payload_bytes, *self.mean_profile_service_seconds,
                  self.reference_service_seconds, self.decision_seconds, self.request_scale,
                  self.q_max, self.latency_scale_seconds, self.age_scale_seconds)
        if self.calibration_role != 'calibration' or not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('Positive, calibration-only feature scales required')
        if len(self.edge_positions) != len(self.profiles) or len(self.profiles) != len(self.mean_profile_service_seconds):
            raise ValueError('Action-aligned server scales required')


def calibration_scales(catalog, mobility, training_window, edge_positions, profiles,
                       decision_seconds=.05, latency_scale_seconds=1., age_scale_seconds=.5):
    if catalog.role != 'calibration':
        raise ValueError('Validation/test timings cannot fit feature normalization')
    images = catalog.image_ids()
    profile_values = {'CPU-1T': [], 'CPU-4T': []}
    payloads = []
    for image in images:
        payloads.append(catalog.observable_metadata(image).payload_bytes)
        for session in sorted(SESSIONS['calibration']):
            pair = catalog.measured_pair_seconds(image, session)
            for profile in profile_values:
                profile_values[profile].append(pair[profile])
    means = {profile: math.fsum(values) / len(values) for profile, values in profile_values.items()}
    low, high = mobility.training_bounds(training_window)
    return VisualFeatureScales(low, tuple(max(b - a, 1.) for a, b in zip(low, high)),
            tuple(tuple(edge) for edge in edge_positions), tuple(profiles),
            math.fsum(payloads) / len(payloads), tuple(means[profile] for profile in profiles),
            means['CPU-1T'], decision_seconds=decision_seconds,
            latency_scale_seconds=latency_scale_seconds, age_scale_seconds=age_scale_seconds)


@dataclass(frozen=True)
class PackedVisualFeatures:
    actor: np.ndarray
    critic: np.ndarray
    q_pressure: np.ndarray
    expected_latency_seconds: np.ndarray
    reference_demand_seconds: np.ndarray
    previous_portfolios: np.ndarray

    def actor_for(self, method):
        if method not in ('QLA-MAPPO', 'PPO-MV-A'):
            raise ValueError('Only the declared matched-input learning methods')
        return self.actor  # same19-dimensional matrix; no privileged QLA metadata


def readonly(value):
    value = np.asarray(value, dtype=np.float32)
    value.setflags(write=False)
    return value


def pack_features(observation, scales, prior_h_queue=0., v_weight=.5):
    if not isinstance(observation, DecisionObservation) or not isinstance(scales, VisualFeatureScales):
        raise TypeError('Only the pre-action observation and declared scales are inputs')
    if not math.isfinite(prior_h_queue) or prior_h_queue < 0 or not math.isfinite(v_weight) or v_weight < 0:
        raise ValueError('Valid prior virtual state and fixed QL coefficient required')
    m = len(scales.profiles)
    if len(observation.feedback.predicted_backlog_seconds) != m:
        raise ValueError('Server-action correspondence mismatch')
    backlog = np.asarray(observation.feedback.predicted_backlog_seconds, dtype=np.float64)
    q = np.clip(backlog / (scales.decision_seconds * scales.q_max), 0., 1.)
    age = -1. if observation.feedback.missing else observation.feedback.age_seconds / scales.age_scale_seconds
    link = UploadLinkConfig()
    actors, costs, demands, previous = [], [], [], []
    for user in observation.users:
        requests = user.requests
        count = len(requests)
        reference = math.fsum(request.predicted_service_seconds[scales.profiles.index('CPU-1T')]
                              for request in requests)
        if requests:
            service = np.mean([request.predicted_service_seconds for request in requests], axis=0)
            transmission = np.mean([request.transmission_seconds for request in requests], axis=0)
        else:
            service = np.asarray(scales.mean_profile_service_seconds)
            transmission = np.asarray([link.transmission_seconds(int(round(scales.mean_payload_bytes)),
                              user.location_meters, edge) for edge in scales.edge_positions])
        # Candidate all-to-one-server average of the present user's batch;
        # known current metadata only, no current action or future duration.
        cost = user.existing_upload_wait_seconds + .5 * (max(count, 1) + 1) * transmission + service + backlog
        load_norm = reference / (scales.request_scale * scales.reference_service_seconds)
        prices = load_norm * q + v_weight * cost / scales.latency_scale_seconds
        scale = min(10., math.log1p(float(np.mean(np.maximum(prices, 0.)))))
        span = max(float(np.max(prices) - np.min(prices)), 1e-8)
        normalized_prices = (prices - np.min(prices)) / span
        loc = [(value - low) / width for value, low, width in
               zip(user.location_meters, scales.location_low, scales.location_span)]
        diagonal = math.hypot(*scales.location_span)
        distances = [math.dist(user.location_meters, edge) / diagonal for edge in scales.edge_positions]
        req = [count / scales.request_scale,
               math.fsum(request.metadata.payload_bytes for request in requests) /
                    (scales.request_scale * scales.mean_payload_bytes), load_norm]
        scores = 1. / (1. + cost / scales.latency_scale_seconds)
        actors.append([*loc, *distances, *req, *scores, age, *normalized_prices, scale])
        costs.append(cost)
        demands.append(reference)
        previous.append(user.previous_portfolio)
    actor = np.asarray(actors, dtype=np.float32)
    if actor.shape != (len(observation.users), 3 * m + 7) or not np.isfinite(actor).all():
        raise RuntimeError('Actor feature layout/unit failure')
    global_state = np.concatenate([actor.reshape(-1), np.asarray(previous).reshape(-1), q,
        np.asarray(observation.feedback.waiting_counts) / 32.,
        np.asarray(observation.feedback.busy, dtype=np.float64),
        [math.tanh(prior_h_queue / 5.), math.fsum(demands) /
            (len(observation.users) * scales.request_scale * scales.reference_service_seconds),
         float(observation.feedback.missing)]])
    critic = np.tile(global_state[None, :], (len(observation.users), 1))
    return PackedVisualFeatures(readonly(actor), readonly(critic), readonly(q), readonly(costs),
                                readonly(demands), readonly(previous))
