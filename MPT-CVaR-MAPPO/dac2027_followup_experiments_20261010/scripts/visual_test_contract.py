"""Prepared fixed-checkpoint test metadata/matrix contract, no service truth.

No test timing CSV, queue, model, optimizer or final evaluation is loaded here.
Actual execution permit requires completed frozen weights and external review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import csv
import json
import math
from pathlib import Path

from metadata_service_predictor import ObservableImage
from visual_episode_collector import BASE, jsonable
from visual_training_coordinator import FrozenRuntimeArtifactBindings, require_reviewed_runtime
from visual_training_transactions import digest, file_sha, require_sha

ROOT = BASE.parent
SPLITS = ROOT/'dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json'
MANIFEST = ROOT/'dac_visual_calibration_20261009/protocol/IMAGE_MANIFEST.csv'
SERVICE_PLAN = ROOT/'dac_visual_calibration_20261010_v2/protocol/PLAN.json'
ORIGINAL = {str(SPLITS): '8d6a3701fc3e046f8a2c898aaaf243ceb40449d2eb4d0fd7dfc2e0d8dcfb3b48',
    str(MANIFEST): 'ab761a6ea28e5447312eb981e5bcc08287fe57c57d9191897d8f0ab9bb021e7f',
    str(SERVICE_PLAN): 'bbddc7b6e15e68625deb41992707d64bf32faa184910de8d8c01fbab1a64f282'}
TEST_METHODS = ('QLA-MAPPO', 'PPO-MV-A', 'JSEQ-A', 'Score-softmax')
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


class OriginalRetainedMetadataPool:
    """Original 1000 image identities/JPEG metadata only, never timing labels."""
    def __init__(self):
        if not all(file_sha(Path(path)) == sha for path, sha in ORIGINAL.items()):
            raise RuntimeError('Original retained image pool/plan changed; no replacement')
        plan = json.loads(SERVICE_PLAN.read_text(encoding='utf-8'))
        if (plan['old_readonly_sha256']['protocol/IMAGE_SPLITS.json'] != ORIGINAL[str(SPLITS)]
                or plan['old_readonly_sha256']['protocol/IMAGE_MANIFEST.csv'] != ORIGINAL[str(MANIFEST)]):
            raise RuntimeError('Retained image provenance must match the unchanged service plan')
        splits = json.loads(SPLITS.read_text(encoding='utf-8'))
        ids, hashes = set(), set()
        for role, rows in splits.items():
            for row in rows:
                if row['image_split'] != role or row['image_id'] in ids or row['image_sha256'] in hashes:
                    raise RuntimeError('Image identities/hashes overlap across frozen roles')
                ids.add(row['image_id'])
                hashes.add(require_sha(row['image_sha256']))
        rows = splits['test']
        if len(rows) != 1000 or any(row['official_split'] != 'val' for row in rows):
            raise RuntimeError('Exactly original 1000 official-val retained images required')
        with MANIFEST.open(encoding='utf-8', newline='') as stream:
            original_rows = {row['image_id']: row for row in csv.DictReader(stream)}
        self._metadata, self._image_sha = {}, {}
        for row in rows:
            original = original_rows[row['image_id']]
            if (original['image_sha256'] != row['image_sha256'] or
                    any(int(original[key]) != row[key] for key in ('width', 'height', 'payload_bytes'))):
                raise RuntimeError('Retained split metadata differs from original whole-pool manifest')
            self._metadata[row['image_id']] = ObservableImage(row['width'], row['height'], row['payload_bytes'])
            self._image_sha[row['image_id']] = row['image_sha256']
        self.identity_sha256 = digest(dict(original_files_sha256=ORIGINAL,
            ordered_images=[dict(image_id=image, image_sha256=self._image_sha[image],
                metadata=asdict(self._metadata[image])) for image in sorted(self._metadata)]))

    def image_ids(self):
        return tuple(sorted(self._metadata))

    def observable_metadata(self, image):
        return self._metadata[image]

    def measured_pair_seconds(self, *args):
        raise RuntimeError('Metadata contract cannot reveal retained service truth; execution adapter not enabled')

    def fitting_rows(self):
        raise RuntimeError('Retained metadata is not a fitting/normalization/selection API')


@dataclass(frozen=True)
class FixedTestMatrixCandidate:
    load_labels: tuple[str, str]
    total_rates_per_second: tuple[float, float]
    generation_decisions: int
    fixed_drain_seconds: float
    common_schedule_sha256_by_load_seed: tuple[tuple[str, ...], tuple[str, ...]]
    retained_pool_identity_sha256: str
    decision_seconds: float = .05
    broadcast_seconds: float = .1
    users: int = 30
    profiles: tuple[str, ...] = PROFILES
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        if (type(self.load_labels) is not tuple or len(self.load_labels) != 2
                or len(set(self.load_labels)) != 2 or any(not isinstance(label, str) or not label for label in self.load_labels)
                or type(self.total_rates_per_second) is not tuple or len(self.total_rates_per_second) != 2
                or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in self.total_rates_per_second)
                or not self.total_rates_per_second[0] < self.total_rates_per_second[1]
                or type(self.generation_decisions) is not int or self.generation_decisions < 1
                or type(self.decision_seconds) not in (int, float) or not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0
                or type(self.fixed_drain_seconds) not in (int, float) or not math.isfinite(self.fixed_drain_seconds)
                or self.fixed_drain_seconds < self.decision_seconds
                or type(self.broadcast_seconds) not in (int, float) or not math.isfinite(self.broadcast_seconds) or self.broadcast_seconds <= 0
                or type(self.users) is not int or self.users != 30 or self.profiles != PROFILES
                or type(self.seeds) is not tuple or any(type(seed) is not int for seed in self.seeds)
                or self.seeds != (1, 2, 3, 4, 5)):
            raise ValueError('Explicit common physical horizon/drain/two loads/30 users/four profiles/five seeds required')
        if (type(self.common_schedule_sha256_by_load_seed) is not tuple
                or len(self.common_schedule_sha256_by_load_seed) != 2
                or any(type(row) is not tuple or len(row) != 5 for row in self.common_schedule_sha256_by_load_seed)):
            raise ValueError('Exactly ten common exogenous schedule bindings, not per-method or per-delay draws')
        for row in self.common_schedule_sha256_by_load_seed:
            for sha in row:
                require_sha(sha)
        require_sha(self.retained_pool_identity_sha256)

    def definition(self):
        return jsonable(asdict(self))

    def cells(self):
        generation = Fraction(str(self.decision_seconds))*self.generation_decisions
        drain = Fraction(str(self.fixed_drain_seconds))
        period = Fraction(str(self.broadcast_seconds))
        rows = []
        for load_index, label in enumerate(self.load_labels):
            for seed_index, seed in enumerate(self.seeds):
                common = self.common_schedule_sha256_by_load_seed[load_index][seed_index]
                # No method/delay-specific request/service/session/uniform stream.
                for delay_multiple in (0, 1, 4):
                    for method in TEST_METHODS:
                        rows.append(dict(method=method, seed=seed, load_label=label,
                            nominal_delay_seconds=float(period*delay_multiple),
                            common_exogenous_schedule_sha256=common,
                            total_requests_per_second=self.total_rates_per_second[load_index],
                            generation_decisions=self.generation_decisions, users=self.users,
                            profiles=list(self.profiles), decision_seconds=self.decision_seconds,
                            broadcast_seconds=self.broadcast_seconds, generation_seconds=float(generation),
                            fixed_drain_seconds=self.fixed_drain_seconds, fixed_end_seconds=float(generation+drain),
                            stop_on_completed_task_count=False, role='test',
                            retained_pool_identity_sha256=self.retained_pool_identity_sha256))
        return rows

    def prepared_record(self):
        return dict(status='prepared_metadata_contract_not_frozen_test_entry', definition=self.definition(),
            contract_semantic_sha256=digest(self.definition()), cells=self.cells(),
            actual_test_schedules_generated=False, actual_test_execution_cells=0,
            test_launch_authorized=False, network_load_horizon_pilot_frozen=False)


def verify_cell_matrix(rows, candidate):
    if type(candidate) is not FixedTestMatrixCandidate:
        raise TypeError('Registered common matrix candidate required')
    expected = candidate.cells()
    key = lambda row: (row['method'], row['seed'], row['load_label'], row['nominal_delay_seconds'])
    actual_by_key = {key(row): row for row in rows}
    expected_by_key = {key(row): row for row in expected}
    if (len(rows) != 120 or len(actual_by_key) != 120 or set(actual_by_key) != set(expected_by_key)
            or any(digest(actual_by_key[k]) != digest(expected_by_key[k]) for k in expected_by_key)):
        raise ValueError('Exact 120 cells, same horizon/drain/exogenous draws and no hidden early stop or field changes')
    return dict(descriptor_cells=120, paired_seeds=5, common_exogenous_bindings=10,
        actual_test_execution_cells=0, descriptor_check_not_results=True)


def fixed_generation_decisions_candidate(low_rate, decision_seconds, minimum_seconds=120., target_generated=10000):
    """Plan's suggested start, not test-outcome stopping or a frozen duration."""
    if (any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in
            (low_rate, decision_seconds, minimum_seconds)) or type(target_generated) is not int or target_generated < 1):
        raise ValueError('Positive calibration/pilot-derived scalar inputs required')
    duration = max(Fraction(str(minimum_seconds)), Fraction(target_generated)/Fraction(str(low_rate)))
    return math.ceil(duration/Fraction(str(decision_seconds)))


class FrozenTestExecutionContract:
    """SOURCE ONLY; never constructed successfully or used to enable truth."""
    def __init__(self, candidate, bindings, registered_weights_path, registered_weights_sha):
        if type(candidate) is not FixedTestMatrixCandidate or type(bindings) is not FrozenRuntimeArtifactBindings:
            raise TypeError('Common matrix and exact completed runtime bindings required')
        protocol = require_reviewed_runtime(bindings)
        if (protocol.get('test_matrix_definition') != candidate.definition()
                or protocol.get('test_matrix_registered_before_test') is not True
                or protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__))):
            raise RuntimeError('This exact matrix/source must be pilot-frozen and fully independently reviewed')
        path, root = Path(registered_weights_path), (BASE/'visual_calibrated_policy').resolve()
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
            raise ValueError('Externally bound whole final-checkpoint registration inside visual output required')
        if file_sha(path) != require_sha(registered_weights_sha) or (path.parent/'FAILURE.json').exists():
            raise RuntimeError('Complete final-weight registration changed/failed')
        registered = json.loads(path.read_text(encoding='utf-8'))
        if (registered['status'] != 'all_final_checkpoints_registered_pending_test_entry'
                or registered['bindings'] != asdict(bindings)
                or registered['test_access_used_for_selection'] is not False
                or registered['rule'] != protocol['checkpoint_selection_rule']):
            raise RuntimeError('All learned weights fixed after common training/validation, before test access')
        from visual_checkpoint_registration import check_complete_paired_grid, FinalCheckpointRuleCandidate
        from visual_training_transactions import CommonBudgetCandidate
        budget = dict(protocol['common_budget'])
        budget['episode_decisions'] = tuple(budget['episode_decisions'])
        check_complete_paired_grid(registered['entries'], FinalCheckpointRuleCandidate(CommonBudgetCandidate(**budget)))
        if not all(file_sha(Path(name)) == sha and not (Path(name).parent/'FAILURE.json').exists()
                   for name, sha in registered['input_files_sha256'].items()):
            raise RuntimeError('Registered learned weights/validation evidence changed before test entry')
        pool = OriginalRetainedMetadataPool()
        if pool.identity_sha256 != candidate.retained_pool_identity_sha256:
            raise RuntimeError('Current original retained metadata pool differs from the frozen matrix')
        self._candidate, self._bindings, self._weights = candidate, bindings, registered
        self._pool_identity = pool.identity_sha256
        self._registration_path, self._registration_sha = path, registered_weights_sha
        # This contract intentionally exposes no measured_pair_seconds method.
        # A separately integrated execution-only adapter remains required.

    def cell_descriptors(self):
        require_reviewed_runtime(self._bindings)
        if (file_sha(self._registration_path) != self._registration_sha
                or (self._registration_path.parent/'FAILURE.json').exists()
                or not all(file_sha(Path(name)) == sha and not (Path(name).parent/'FAILURE.json').exists()
                    for name, sha in self._weights['input_files_sha256'].items())
                or OriginalRetainedMetadataPool().identity_sha256 != self._pool_identity):
            raise RuntimeError('Frozen weight registration/metadata changed before descriptor retrieval')
        return self._candidate.cells()
