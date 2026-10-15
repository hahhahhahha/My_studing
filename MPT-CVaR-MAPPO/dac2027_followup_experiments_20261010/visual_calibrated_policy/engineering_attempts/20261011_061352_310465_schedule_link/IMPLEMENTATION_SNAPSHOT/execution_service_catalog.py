"""Measured paired durations for FIFO execution, never for actor/QL inputs.

This engineering-stage catalog allows calibration and validation only. Test
activation requires a later separately reviewed frozen-policy execution entry.
No stochastic session selection is fixed by this module: schedule generation
must specify one common image/session pair per task before comparing methods.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from metadata_service_predictor import ObservableImage, PROFILES

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = {
    'calibration': ROOT / 'dac_visual_calibration_20261009/measurements/calibration',
    'validation': ROOT / 'dac_visual_calibration_20261010_v2/measurements/validation'}
SESSIONS = {'calibration': {1, 2, 3, 4, 5}, 'validation': {1, 2, 3}}
ROW_COUNTS = {'calibration': 20000, 'validation': 12000}


class ExecutionServiceCatalog:
    def __init__(self, role):
        if role not in DIRECTORIES:
            raise ValueError('Engineering catalog cannot access retained test data')
        self.role = role
        self.source_sha256 = {}
        self._pairs, self._metadata = {}, {}
        rows = 0
        files = sorted(DIRECTORIES[role].glob('*.csv'))
        for path in files:
            self.source_sha256[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8', newline='') as handle:
                for row in csv.DictReader(handle):
                    if row['image_split'] != role or row['error_status'] != 'ok':
                        raise RuntimeError('Wrong measured role or service error; no row filtering')
                    image, session, profile = row['image_id'], int(row['session_id']), row['profile_id']
                    if session not in SESSIONS[role] or profile not in PROFILES:
                        raise RuntimeError('Unknown measured session/profile')
                    metadata = ObservableImage(int(row['width']), int(row['height']), int(row['payload_bytes']))
                    signature = (metadata, row['image_sha256'])
                    if image in self._metadata and self._metadata[image] != signature:
                        raise RuntimeError('Image metadata/hash changed between measured profiles/sessions')
                    self._metadata[image] = signature
                    key = (image, session)
                    pair = self._pairs.setdefault(key, {})
                    if profile in pair:
                        raise RuntimeError('Duplicate measured image/session/profile')
                    value = float(row['service_ms']) * .001
                    stages = [float(row[field]) * .001 for field in ('decode_ms', 'preprocess_ms', 'infer_ms', 'postprocess_ms')]
                    if not math.isfinite(value) or value <= 0 or any(not math.isfinite(stage) or stage < 0 for stage in stages):
                        raise RuntimeError('Invalid service durations')
                    if not math.isclose(value, math.fsum(stages), rel_tol=0., abs_tol=1e-10):
                        raise RuntimeError('Measured service boundary/stage sum mismatch')
                    pair[profile] = value
                    rows += 1
        if rows != ROW_COUNTS[role] or any(set(pair) != set(PROFILES) for pair in self._pairs.values()):
            raise RuntimeError('Incomplete paired measured pool')
        for image in self._metadata:
            if {session for known, session in self._pairs if known == image} != SESSIONS[role]:
                raise RuntimeError('Measured image lacks a registered session')
        self.row_count = rows

    def image_ids(self):
        return tuple(sorted(self._metadata))

    def observable_metadata(self, image):
        return self._metadata[image][0]

    def measured_pair_seconds(self, image, session):
        """Execution-only lookup. The same pair must be used across policies."""
        return dict(self._pairs[(image, session)])
