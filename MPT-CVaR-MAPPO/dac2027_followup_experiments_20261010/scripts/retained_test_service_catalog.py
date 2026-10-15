"""SOURCE ONLY: gated paired retained service truth, execution-private.

No module-level timing reads. The real constructor requires the completed
frozen-weight contract and current full source review. Never pass this catalog
to a policy/predictor/scaler; only immutable observable metadata crosses out.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

from metadata_service_predictor import ObservableImage, PROFILES
from visual_episode_collector import BASE
from visual_test_contract import FrozenTestExecutionContract, OriginalRetainedMetadataPool
from visual_training_coordinator import require_reviewed_runtime
from visual_training_transactions import file_sha, require_sha

SERVICE_ROOT = BASE.parent/'dac_visual_calibration_20261010_v2'
RAW_AUDIT = SERVICE_ROOT/'analysis/retained_test_continuation/STRICT_TEST_AUDIT.json'
FINAL_REVIEW = SERVICE_ROOT/'analysis/retained_test_continuation/result_review/EXPERIMENT_AUDIT.json'
SERVICE_EVIDENCE_BINDINGS = {
    str(RAW_AUDIT): '8c256aad126f3631986bcb99b68bdbb19e494836f990ad9155f20fe6ef9a9d4f',
    str(FINAL_REVIEW): 'e689c1fe565ed0692d6118b59815be92721bcf7bb88dac6a9b961890827826c8'}
SESSIONS = (1, 2, 3, 4, 5)
STAGES = ('decode_ms', 'preprocess_ms', 'infer_ms', 'postprocess_ms')


def _registered_inventory(measurement_sha256):
    """Pure original-file identity check, no timing parse or path discovery."""
    expected = {f'measurements/test/test_s{s}_b{b}_{p}.{ext}'
                for s in SESSIONS for b in range(1, 6) for p in PROFILES for ext in ('csv', 'json')}
    normalized = {}
    for name, sha in measurement_sha256.items():
        win = PureWindowsPath(name)
        if win.is_absolute() or win.drive or '..' in win.parts:
            raise ValueError('Original relative measured file paths only')
        key = '/'.join(win.parts)
        if key in normalized:
            raise ValueError('Duplicate normalized measured file identity')
        normalized[key] = require_sha(sha)
    if set(normalized) != expected:
        raise RuntimeError('Exactly original 50 CSV and 50 metadata files, no extra or missing evidence')
    return {str(SERVICE_ROOT/key): sha for key, sha in sorted(normalized.items())}


class _PairedDurationIndex:
    """Pure parser; isolated artificial-row checks are NOT real catalog access."""
    def __init__(self, metadata, image_sha, sessions=SESSIONS):
        if (not metadata or set(metadata) != set(image_sha)
                or any(type(value) is not ObservableImage for value in metadata.values())
                or type(sessions) is not tuple or not sessions
                or any(type(s) is not int or s < 1 for s in sessions) or len(set(sessions)) != len(sessions)):
            raise ValueError('Explicit metadata/image hashes/unique measured sessions required')
        self._metadata, self._image_sha = dict(metadata), dict(image_sha)
        for value in self._image_sha.values():
            require_sha(value)
        self._sessions, self._pairs, self._rows = sessions, {}, 0
        self._failed, self._sealed = False, False

    def add_row(self, row, expected_session, expected_profile):
        if self._failed or self._sealed:
            raise RuntimeError('Paired parser failed/sealed; no append or retry')
        try:
            image = row['image_id']
            session, profile = int(row['session_id']), row['profile_id']
            if (row['image_split'] != 'test' or row['error_status'] != 'ok' or image not in self._metadata
                    or session not in self._sessions or session != expected_session or profile not in PROFILES
                    or profile != expected_profile or row['image_sha256'] != self._image_sha[image]
                    or ObservableImage(*(int(row[key]) for key in ('width', 'height', 'payload_bytes'))) != self._metadata[image]):
                raise ValueError('Measured test image/role/hash/JPEG/session/profile identity differs')
            value = float(row['service_ms'])*.001
            stages = tuple(float(row[key])*.001 for key in STAGES)
            if (not math.isfinite(value) or value <= 0 or any(not math.isfinite(x) or x < 0 for x in stages)
                    or not math.isclose(value, math.fsum(stages), rel_tol=0., abs_tol=1e-10)):
                raise ValueError('Finite positive paired seconds and exact measured stage boundary required')
            pair = self._pairs.setdefault((image, session), {})
            if profile in pair:
                raise ValueError('Duplicate measured image/session/profile, never average replacements')
            pair[profile] = value
            self._rows += 1
        except Exception:
            self._failed = True
            raise

    def seal(self):
        if self._failed or self._sealed:
            raise RuntimeError('Paired parser failed/sealed; no resume')
        expected = {(image, session) for image in self._metadata for session in self._sessions}
        if (self._rows != len(expected)*len(PROFILES) or set(self._pairs) != expected
                or any(set(pair) != set(PROFILES) for pair in self._pairs.values())):
            self._failed = True
            raise RuntimeError('Every original image/session must have both profiles; no gap filling/filtering')
        self._sealed = True

    def pair_seconds(self, image, session):
        if self._failed or not self._sealed:
            raise RuntimeError('Complete paired index required')
        if type(session) is not int or session not in self._sessions:
            raise ValueError('Exactly predeclared measured session required')
        return dict(self._pairs[(image, session)])


class RetainedExecutionServiceCatalog:
    """SOURCE PREPARED ONLY. No real constructor or timing lookup yet executed.

    Runtime checks occur before loading, after loading and at explicit cell
    boundaries, not on every task (no repeated heavy full-file hashing).
    A full test entry must call verify_boundary before and after each cell.
    """
    def __init__(self, contract):
        if type(contract) is not FrozenTestExecutionContract:
            raise TypeError('Completed exact frozen-weight common test contract required before timing read')
        self._contract, self._failed = contract, False
        self._ready, self._index, self._source_sha256 = False, None, {}
        self._pool = None
        try:
            self._check_authority()
            audit = json.loads(RAW_AUDIT.read_text(encoding='utf-8'))
            if audit['passed'] is not True or audit['rows'] != 10000 or audit['blocks'] != 50:
                raise RuntimeError('Original complete paired retained-test audit required')
            inventory = _registered_inventory(audit['measurement_sha256'])
            self._source_sha256 = dict(SERVICE_EVIDENCE_BINDINGS) | inventory
            self._verify_files()
            self._pool = OriginalRetainedMetadataPool()
            if self._pool.identity_sha256 != self._contract._pool_identity:
                raise RuntimeError('Original retained metadata pool differs from frozen weights/matrix')
            self._index = _PairedDurationIndex(self._pool._metadata, self._pool._image_sha)
            for name in sorted(inventory):
                path = Path(name)
                if path.suffix != '.csv':
                    continue
                # Original immutable name and audit identity bind each file;
                # do not scan nearby directories for substitutes.
                _, session, _, profile = path.stem.split('_')
                with path.open(encoding='utf-8', newline='') as stream:
                    count = 0
                    for row in csv.DictReader(stream):
                        self._index.add_row(row, int(session[1:]), profile)
                        count += 1
                if count != 200:
                    raise RuntimeError('Original measured block must have all 200 rows')
            self._index.seal()
            self._check_authority()
            self._verify_files()
            self._ready = True
        except Exception:
            self._failed = True
            raise

    def _check_authority(self):
        self._contract.cell_descriptors()
        protocol = require_reviewed_runtime(self._contract._bindings)
        required = (Path(__file__), BASE/'scripts/visual_test_contract.py')
        if any(protocol['source_files_sha256'].get(str(path)) != file_sha(path) for path in required):
            raise RuntimeError('Execution adapter and test contract must be in the fresh complete source review')
        if any(file_sha(Path(path)) != sha for path, sha in SERVICE_EVIDENCE_BINDINGS.items()):
            raise RuntimeError('Original raw/final audit evidence changed; never remeasure or silently rebind')
        review = json.loads(FINAL_REVIEW.read_text(encoding='utf-8'))
        if review['record_integrity_status'] != 'PASS' or review['blocking_findings'] != []:
            raise RuntimeError('Completed original service record-integrity audit required')

    def _verify_files(self):
        root = SERVICE_ROOT.resolve()
        for name, sha in self._source_sha256.items():
            path = Path(name)
            if (path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root)
                    or file_sha(path) != sha):
                raise RuntimeError('Original retained bytes changed/path substituted; fail-stop execution')

    def verify_boundary(self):
        if self._failed or not self._ready:
            raise RuntimeError('Catalog unavailable/failed; no partial reuse')
        try:
            self._check_authority()
            self._verify_files()
            if OriginalRetainedMetadataPool().identity_sha256 != self._pool.identity_sha256:
                raise RuntimeError('Original observable metadata identity changed')
        except Exception:
            self._failed = True
            raise

    @property
    def role(self):
        return 'test'

    @property
    def source_sha256(self):
        return MappingProxyType(self._source_sha256)

    @property
    def row_count(self):
        if self._failed or not self._ready:
            raise RuntimeError('Incomplete or failed catalog')
        return self._index._rows

    def image_ids(self):
        if self._failed or not self._ready:
            raise RuntimeError('Incomplete or failed catalog')
        return self._pool.image_ids()

    def observable_metadata(self, image):
        if self._failed or not self._ready:
            raise RuntimeError('Incomplete or failed catalog')
        return self._pool.observable_metadata(image)

    def measured_pair_seconds(self, image, session):
        """PRIVILEGED EXECUTION ONLY, never policy/predictor/QL/scaler input."""
        if self._failed or not self._ready:
            raise RuntimeError('Incomplete or failed catalog')
        try:
            return self._index.pair_seconds(image, session)
        except Exception:
            self._failed = True
            raise

    def fitting_rows(self):
        raise RuntimeError('Retained service cannot fit normalization, prediction or checkpoint selection')
