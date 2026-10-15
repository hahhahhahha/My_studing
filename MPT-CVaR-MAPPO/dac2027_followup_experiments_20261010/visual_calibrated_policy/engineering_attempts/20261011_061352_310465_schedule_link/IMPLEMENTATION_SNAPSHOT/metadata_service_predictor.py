"""Calibration-only service prediction from observable JPEG metadata.

No measurement/test CSV is imported or opened here. Units returned are seconds.
Only this predictor, not the execution catalog, may enter actor/QL features.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'dac_visual_calibration_20261010_v2/model/CONDITIONAL_MODEL.json'
MODEL_SHA = 'fc5a5311c6f61572480987ccb485724b3a3eeb78a028b3ffc7885d495edc295e'
PROFILES = ('CPU-1T', 'CPU-4T')


@dataclass(frozen=True)
class ObservableImage:
    width: int
    height: int
    payload_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.width, self.height, self.payload_bytes)):
            raise ValueError('Positive integer pixel dimensions and actual JPEG bytes required')


def size_stratum(metadata):
    pixels = metadata.width * metadata.height
    return next((index for index, bound in enumerate((250000, 500000, 1000000, 2000000)) if pixels <= bound), 4)


class MetadataServicePredictor:
    def __init__(self, model_path=MODEL):
        model_path = Path(model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != MODEL_SHA:
            raise RuntimeError('Calibration model changed; no automatic refit')
        model = json.loads(model_path.read_text(encoding='utf-8'))
        if (model['k'] != 20 or model['support_distance'] != 2.0 or
            model['strata_upper_bounds'] != [250000, 500000, 1000000, 2000000] or
            model['fitted_from'] != 'V1 calibration only; V1 validation used for design diagnosis, not timing fit'):
            raise RuntimeError('Frozen metadata prediction protocol changed')
        self.model_sha256 = digest
        self._entries = {}
        for entry in model['training_entries']:
            if entry['image_split'] != 'calibration' or entry['image_id'] in self._entries:
                raise RuntimeError('Non-calibration or duplicate predictor entry')
            metadata = ObservableImage(int(entry['width']), int(entry['height']), int(entry['payload_bytes']))
            self._entries[entry['image_id']] = metadata
        if len(self._entries) != 2000 or len(model['paired_rows']) != 10000:
            raise RuntimeError('Incomplete frozen calibration model')
        self._calibrated = {image: {} for image in self._entries}
        for row in model['paired_rows']:
            image, session = row['image_id'], int(row['session'])
            if image not in self._calibrated or session in self._calibrated[image] or set(row['profiles']) != set(PROFILES):
                raise RuntimeError('Invalid paired calibration row')
            pair = {profile: float(row['profiles'][profile]['service_ms']) * .001 for profile in PROFILES}
            if not all(math.isfinite(value) and value > 0 for value in pair.values()):
                raise RuntimeError('Invalid calibration durations')
            self._calibrated[image][session] = pair
        if any(set(sessions) != {1, 2, 3, 4, 5} for sessions in self._calibrated.values()):
            raise RuntimeError('Calibration sessions are incomplete')
        self._cache = {}

    def neighbors(self, metadata):
        if not isinstance(metadata, ObservableImage):
            raise TypeError('Only observable image metadata is permitted')
        target = math.log2(metadata.width * metadata.height), math.log2(metadata.payload_bytes)
        distances = []
        for image, known in self._entries.items():
            if size_stratum(known) == size_stratum(metadata):
                distance = math.hypot(target[0] - math.log2(known.width * known.height),
                                      target[1] - math.log2(known.payload_bytes))
                distances.append((distance, image))
        distances.sort()
        if not distances or distances[0][0] > 2.0:
            raise RuntimeError('Metadata outside frozen calibration support; no silent fallback')
        return tuple(image for _, image in distances[:20]), distances[0][0]

    def predict_seconds(self, metadata):
        if not isinstance(metadata, ObservableImage):
            raise TypeError('Actual durations/classes are not predictor inputs')
        if metadata not in self._cache:
            neighbors, _ = self.neighbors(metadata)
            count = len(neighbors) * 5
            self._cache[metadata] = {
                profile: math.fsum(self._calibrated[image][session][profile]
                                  for image in neighbors for session in range(1, 6)) / count
                for profile in PROFILES}
        return dict(self._cache[metadata])  # caller cannot mutate the prediction cache
