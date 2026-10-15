"""Explicit-power simulated wireless link and serialized per-user uplink."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UploadLinkConfig:
    bandwidth_hz: float = 2e6
    noise_power_w: float = 1e-10
    antenna_gain: float = 4.11
    carrier_hz: float = 915e6
    tx_power_w: float = .25
    light_speed_m_s: float = 3e8
    min_distance_m: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError('Positive finite physical link constants required')

    def bits_per_second(self, location, edge_position):
        if len(location) != 2 or len(edge_position) != 2 or not all(
                math.isfinite(x) for x in (*location, *edge_position)):
            raise ValueError('Finite meter coordinates required')
        distance = max(math.dist(location, edge_position), self.min_distance_m)
        gain = self.antenna_gain * (self.light_speed_m_s /
                (4 * math.pi * self.carrier_hz * distance)) ** 2
        snr = self.tx_power_w * gain / self.noise_power_w
        return self.bandwidth_hz * math.log1p(snr) / math.log(2.)

    def transmission_seconds(self, payload_bytes, location, edge_position):
        if type(payload_bytes) is not int or payload_bytes <= 0:
            raise ValueError('Actual positive JPEG byte count required')
        return payload_bytes * 8. / self.bits_per_second(location, edge_position)




class SerializedUserUploads:
    def __init__(self, users):
        if type(users) is not int or users < 1:
            raise ValueError('Positive user count required')
        self._available = [0.] * users

    def existing_wait(self, user, now):
        return max(0., self._available[user] - now)

    def enqueue(self, user, decision_seconds, transmission_seconds):
        if (not 0 <= user < len(self._available) or not math.isfinite(decision_seconds)
                or decision_seconds < 0 or not math.isfinite(transmission_seconds)
                or transmission_seconds <= 0):
            raise ValueError('Valid physical upload timing required')
        start = max(decision_seconds, self._available[user])
        ready = start + transmission_seconds
        self._available[user] = ready
        return ready - decision_seconds
