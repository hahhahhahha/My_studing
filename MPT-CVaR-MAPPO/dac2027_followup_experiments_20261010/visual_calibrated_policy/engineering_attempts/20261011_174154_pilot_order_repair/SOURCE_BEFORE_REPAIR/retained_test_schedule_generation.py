"""SOURCE ONLY: paired retained request metadata, before formal entry.

No measured timing reads, fitting, policies, models or queues. The real writer
requires completed cal/validation design, actual smoke and preparation review.
Persist ten schedules BEFORE the immutable formal training/test protocol so
its byte/semantic bindings need not change after weights are selected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path
import sys
import numpy as np

from retained_test_evaluator import _write_once
from retained_test_grid_registration import _safe_path, _read_actual, _claim_grid_once
from retained_test_schedule import SCHEMA, ARRIVAL_RULE, _PreparedRetainedSchedulePayload, _decode_registered_document
from visual_episode_collector import BASE, jsonable
from visual_request_schedule import ScheduledRequest
from visual_test_contract import OriginalRetainedMetadataPool, ORIGINAL, PROFILES, FixedTestMatrixCandidate
from visual_training_transactions import digest, file_sha, require_sha

RECIPE = 'numpy_2_2_6_PCG64_SeedSequence_seed_spawn5_step_user_order_v1'
STREAM_NAMES = ('counts_and_burst', 'arrival_offsets', 'images', 'measured_sessions', 'assignment_uniforms')
REVIEW_SCOPE = 'retained_metadata_schedule_generation_only_after_cal_validation_pilot_and_actual_smoke'


@dataclass(frozen=True)
class RetainedGenerationDesign:
    load_labels: tuple[str, str]
    total_rates_per_second: tuple[float, float]
    generation_decisions: int
    fixed_drain_seconds: float
    retained_pool_identity_sha256: str
    burst_probability: float
    burst_multiplier: float
    decision_seconds: float = .05
    broadcast_seconds: float = .1
    users: int = 30
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        scalars = (self.decision_seconds, self.broadcast_seconds, self.fixed_drain_seconds,
                   *self.total_rates_per_second, self.burst_multiplier)
        if (type(self.load_labels) is not tuple or len(self.load_labels) != 2
                or len(set(self.load_labels)) != 2 or any(type(x) is not str or not x for x in self.load_labels)
                or type(self.total_rates_per_second) is not tuple or len(self.total_rates_per_second) != 2
                or not self.total_rates_per_second[0] < self.total_rates_per_second[1]
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in scalars)
                or type(self.generation_decisions) is not int or self.generation_decisions < 1
                or self.fixed_drain_seconds < self.decision_seconds or self.burst_multiplier < 1
                or type(self.burst_probability) not in (int, float) or not math.isfinite(self.burst_probability)
                or not 0 <= self.burst_probability <= 1 or type(self.users) is not int or self.users != 30
                or type(self.seeds) is not tuple or any(type(x) is not int for x in self.seeds)
                or self.seeds != (1, 2, 3, 4, 5)):
            raise ValueError('Explicit cal/validation-frozen two loads/full horizon/drain/physical grids/30 users/five seeds required')
        require_sha(self.retained_pool_identity_sha256)

    def definition(self):
        return dict(**jsonable(asdict(self)), arrival_rule=ARRIVAL_RULE, generator_recipe=RECIPE,
            independent_stream_names=list(STREAM_NAMES), profiles=list(PROFILES),
            image_rule='uniform_full_original_retained_pool_with_replacement',
            paired_service_rule='uniform_original_measured_session_1_to_5_shared_across_configurations',
            request_cap=None, outcome_driven_horizon_or_resampling=False,
            seed_recipe_shared_across_loads=True, draw_order='step_then_ascending_user_then_task',
            persisted_order='true_arrival_then_user_then_task_id')


def _generate_descriptor_payload(design, load_index, seed, ordered_image_ids):
    """Pure sampling on explicit IDs. Artificial IDs do not enable execution.

    Same five PCG64 streams and unconditional Poisson-bursty recipe as the
    existing calibration generator. Role gate on that old generator unchanged.
    Low/high loads share seed recipes, but each has its own complete artifact;
    no independent-load claim and no method/delay-dependent random stream.
    """
    if (type(design) is not RetainedGenerationDesign or type(load_index) is not int or load_index not in (0, 1)
            or type(seed) is not int or seed not in design.seeds or type(ordered_image_ids) is not tuple
            or not ordered_image_ids or any(type(x) is not str or not x for x in ordered_image_ids)
            or tuple(sorted(set(ordered_image_ids))) != ordered_image_ids):
        raise ValueError('Explicit unique sorted immutable metadata IDs/load index/paired integer seed required')
    if np.__version__ != '2.2.6':
        raise RuntimeError('Registered NumPy 2.2.6 binary RNG recipe required')
    streams = [np.random.Generator(np.random.PCG64(child)) for child in np.random.SeedSequence(seed).spawn(5)]
    before = {name: digest(rng.bit_generator.state) for name, rng in zip(STREAM_NAMES, streams)}
    counts_rng, offsets_rng, images_rng, sessions_rng, uniform_rng = streams
    rate = design.total_rates_per_second[load_index]
    base = rate/design.users/(1+design.burst_probability*(design.burst_multiplier-1))
    period, rows = Fraction(str(design.decision_seconds)), []
    for step in range(design.generation_decisions):
        for user in range(design.users):
            burst = counts_rng.random() < design.burst_probability
            count = int(counts_rng.poisson(base*design.decision_seconds*(design.burst_multiplier if burst else 1)))
            # No per-user seven-request ceiling or outcome-based row discard.
            for _ in range(count):
                arrival = float(step*period)+float(offsets_rng.random())*design.decision_seconds
                # Preserve the old exclusive interval rounding guard, not a
                # latency/service cap or a fabricated replacement request.
                arrival = min(arrival, math.nextafter(float((step+1)*period), -math.inf))
                rows.append(ScheduledRequest(f'task-{len(rows):08d}', user, arrival,
                    ordered_image_ids[int(images_rng.integers(len(ordered_image_ids)))],
                    1+int(sessions_rng.integers(5)), float(uniform_rng.random())))
    rows.sort(key=lambda row: (row.arrival_seconds, row.user, row.task_id))
    payload = _PreparedRetainedSchedulePayload('test', design.load_labels[load_index], design.users,
        design.decision_seconds, design.generation_decisions, seed, rate, design.burst_probability,
        design.burst_multiplier, design.retained_pool_identity_sha256, tuple(rows))
    witness = dict(generator_recipe=RECIPE, paired_seed=seed, stream_names=list(STREAM_NAMES),
        state_sha256_before=before,
        state_sha256_after={name: digest(rng.bit_generator.state) for name, rng in zip(STREAM_NAMES, streams)},
        global_numpy_rng_used=False, methods_delays_or_service_truth_used=False)
    return payload, witness


@dataclass(frozen=True)
class RetainedGenerationArtifactBindings:
    pilot_path: str
    pilot_sha256: str
    preparation_review_path: str
    preparation_review_sha256: str


def _require_generation_preparation(design, bindings):
    """Preparation permit ONLY; not FrozenRuntimeArtifactBindings/launch."""
    if type(design) is not RetainedGenerationDesign or type(bindings) is not RetainedGenerationArtifactBindings:
        raise TypeError('Completed cal/validation generation design and external preparation review required')
    from qla_environment import guard
    spec = guard()  # No GPU probe/allocation in this metadata-only gate.
    if (sys.version.split()[0] != '3.13.5' or np.__version__ != '2.2.6'
            or Path(sys.executable).resolve() != (Path(spec['isolated_environment'])/'Scripts/python.exe').resolve()):
        raise RuntimeError('Own declared Python/NumPy runtime; no global install or neural CPU fallback')
    pilot = _read_actual(bindings.pilot_path, bindings.pilot_sha256, BASE/'protocol')
    review = _read_actual(bindings.preparation_review_path, bindings.preparation_review_sha256, BASE.parent/'.aris/traces')
    if (pilot.get('status') != 'cal_validation_generation_design_frozen_not_formal_entry'
            or pilot.get('generation_design') != design.definition()
            or pilot.get('test_labels_or_outcomes_used_for_design') is not False
            or pilot.get('cal_validation_engineering_pilot_completed') is not True
            or pilot.get('formal_training_or_test_launch_authorized') is not False
            or review.get('review_type') != 'Type-A' or review.get('model') != 'gpt-6-astra'
            or review.get('reasoning_effort') != 'xhigh' or review.get('scope') != REVIEW_SCOPE
            or review.get('status') not in ('PASS', 'WARN') or review.get('blockers') != []
            or review.get('pilot_sha256') != bindings.pilot_sha256
            or review.get('permitted_actions') != ['generate_common_request_metadata_only']
            or review.get('formal_training_or_test_launch_authorized') is not False):
        raise RuntimeError('Uncompleted/artificial pilot or review is not actual metadata-generation permission')
    from visual_model_smoke import source_lock, protected_bindings
    sources, protected = source_lock(), protected_bindings()
    if (pilot.get('source_files_sha256') != sources or review.get('source_files_sha256') != sources
            or pilot.get('protected_input_and_source_sha256') != protected):
        raise RuntimeError('Exact current complete source/preparation/raw lock required, never reuse stale smoke plan')
    smoke = _read_actual(pilot['actual_model_smoke_path'], pilot['actual_model_smoke_sha256'], BASE/'visual_calibrated_policy')
    if (smoke.get('status') != 'actual_cuda_two_arm_smoke_pass'
            or smoke.get('methods') != ['QLA-MAPPO', 'PPO-MV-A']
            or smoke.get('actual_model_forward_and_optimizer_executed') is not True
            or smoke.get('source_files_sha256') != sources
            or smoke.get('protected_input_and_source_sha256') != protected):
        raise RuntimeError('Actual current two-arm CUDA smoke required, not an artificial certification')
    evidence = pilot.get('cal_validation_pilot_evidence_sha256', {})
    if not evidence:
        raise RuntimeError('Actual completed cal/validation engineering pilot evidence required')
    for name, sha in evidence.items():
        path = _safe_path(name, BASE/'visual_calibrated_policy')
        if not path.is_file() or file_sha(path) != require_sha(sha):
            raise RuntimeError('Actual cal/validation pilot evidence changed')
    files = dict(sources) | protected | ORIGINAL | evidence | {
        str(Path(bindings.pilot_path)): bindings.pilot_sha256,
        str(Path(bindings.preparation_review_path)): bindings.preparation_review_sha256,
        str(Path(pilot['actual_model_smoke_path'])): require_sha(pilot['actual_model_smoke_sha256'])}
    return pilot, files


class RealRetainedScheduleGenerator:
    """SOURCE PREPARED ONLY. Actual constructor/generate/register count zero."""
    def __init__(self, design, bindings, new_directory):
        self._pilot, self._inputs = _require_generation_preparation(design, bindings)
        pool = OriginalRetainedMetadataPool()
        if len(pool.image_ids()) != 1000 or pool.identity_sha256 != design.retained_pool_identity_sha256:
            raise RuntimeError('Exactly the original 1000 metadata identities, never selected/substituted images')
        self.design, self.bindings, self._ids = design, bindings, pool.image_ids()
        self.path = _safe_path(new_directory, BASE/'visual_calibrated_policy/test_schedules')
        if self.path.exists():
            raise ValueError('New owned common schedule directory only, no append/replacement')
        self.identity_sha256 = digest(design.definition())
        self._lock = _claim_grid_once(BASE/'visual_calibrated_policy/test_schedule_generation_locks',
            self.identity_sha256, dict(at=datetime.now().astimezone().isoformat(), output_path=str(self.path),
                request_generation_not_policy_execution=True, automatic_resampling_append_or_retry_allowed=False))
        self._owned, self._started, self._failed, self._finished = False, False, False, False
        self._artifacts, self._witnesses = {}, {}
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            self._owned = True
            _write_once(self.path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(),
                design=design.definition(), bindings=asdict(bindings), identity_sha256=self.identity_sha256,
                actual_training_and_test_cells=0, measured_service_access_allowed=False))
        except Exception as error:
            self._fail(error)
            raise

    def _verify_boundary(self):
        _, current = _require_generation_preparation(self.design, self.bindings)
        if current != self._inputs or OriginalRetainedMetadataPool().identity_sha256 != self.design.retained_pool_identity_sha256:
            raise RuntimeError('Protected pilot/source/image pool changed during generation')
        for label in self._artifacts:
            for row in self._artifacts[label].values():
                if file_sha(Path(row['path'])) != row['file_sha256']:
                    raise RuntimeError('Already written common requests changed; preserve partial generation')

    def _fail(self, error):
        self._failed = True
        evidence = dict(error=repr(error), at=datetime.now().astimezone().isoformat(),
            identity_sha256=self.identity_sha256, partial_artifacts=self._artifacts, partial_rng_witnesses=self._witnesses,
            input_files_sha256=self._inputs, automatic_resampling_append_or_retry_allowed=False,
            partial_evidence_preserved=True, actual_training_and_test_cells=0)
        if self._owned and not (self.path/'FAILURE.json').exists():
            _write_once(self.path/'FAILURE.json', evidence)
        failed_lock = self._lock.with_name(self._lock.stem+'_FAILURE.json')
        if not failed_lock.exists():
            _write_once(failed_lock, evidence)

    def generate_and_register(self):
        if self._started or self._failed or self._finished:
            raise RuntimeError('One-shot generation, no partial resampling/append/resume/retry')
        self._started = True
        try:
            self._verify_boundary()
            common = []
            counts = {}
            for load_index, label in enumerate(self.design.load_labels):
                self._artifacts[label], self._witnesses[label], counts[label] = {}, {}, {}
                hashes = []
                for seed in self.design.seeds:
                    payload, witness = _generate_descriptor_payload(self.design, load_index, seed, self._ids)
                    path = self.path/f'L{load_index}_s{seed}.json'
                    document = dict(schema=SCHEMA, payload=payload.to_dict(), semantic_sha256=payload.fingerprint())
                    _decode_registered_document(document, frozenset(self._ids), self.design.retained_pool_identity_sha256)
                    _write_once(path, document)
                    self._artifacts[label][str(seed)] = dict(path=str(path), file_sha256=file_sha(path),
                        semantic_sha256=payload.fingerprint())
                    self._witnesses[label][str(seed)] = witness
                    counts[label][str(seed)] = len(payload.requests)
                    hashes.append(payload.fingerprint())
                common.append(tuple(hashes))
            self._verify_boundary()
            candidate = FixedTestMatrixCandidate(self.design.load_labels, self.design.total_rates_per_second,
                self.design.generation_decisions, self.design.fixed_drain_seconds, tuple(common),
                self.design.retained_pool_identity_sha256, self.design.decision_seconds,
                self.design.broadcast_seconds, self.design.users)
            record = dict(status='ten_common_request_artifacts_generated_pending_formal_test_entry',
                at=datetime.now().astimezone().isoformat(), generation_design=self.design.definition(),
                generation_identity_sha256=self.identity_sha256, preparation_bindings=asdict(self.bindings),
                test_schedule_schema=SCHEMA, test_arrival_design=dict(arrival_rule=ARRIVAL_RULE,
                    burst_probability=self.design.burst_probability, burst_multiplier=self.design.burst_multiplier),
                test_schedule_artifacts_by_load_seed=self._artifacts, rng_stream_witnesses=self._witnesses,
                realized_request_counts_by_load_seed=counts, matrix_definition=candidate.definition(),
                input_files_sha256=self._inputs, actual_policy_cells=0, actual_training_cells=0,
                test_labels_or_outcomes_used=False, formal_runtime_protocol_frozen=False,
                formal_training_or_test_launch_authorized=False, independent_final_result_audit_done=False,
                generation_precedes_immutable_formal_training_test_protocol_and_weight_selection=True)
            _write_once(self.path/'GENERATED_COMMON_REQUESTS_PENDING_TEST_ENTRY.json', record)
            self._finished = True
            return record
        except Exception as error:
            self._fail(error)
            raise
