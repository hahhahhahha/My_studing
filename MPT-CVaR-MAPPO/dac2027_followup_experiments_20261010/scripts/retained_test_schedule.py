"""SOURCE ONLY: immutable, pre-registered common retained request schedule.

No generator, policy, service truth or timing CSV access. Actual loader requires
completed frozen-weight contract and current integrated pilot/source review.
Pure artificial payloads are not registered schedules or launch authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import json
import math
from pathlib import Path
from types import MappingProxyType

from visual_episode_collector import BASE, jsonable
from visual_request_schedule import ScheduledRequest
from visual_test_contract import FrozenTestExecutionContract, OriginalRetainedMetadataPool
from visual_training_coordinator import require_reviewed_runtime
from visual_training_transactions import digest, file_sha, require_sha

SCHEMA = 'retained-common-requests-v1'
ARRIVAL_RULE = 'per_user_interval_poisson_bursty_unconditional_total_rate_no_request_cap_v1'
ROW_FIELDS = {field.name for field in fields(ScheduledRequest)}


@dataclass(frozen=True)
class _PreparedRetainedSchedulePayload:
    """Pure descriptor only, NEVER proof of registration or execution access."""
    role: str
    load_label: str
    users: int
    decision_seconds: float
    decisions: int
    seed: int
    total_requests_per_second: float
    burst_probability: float
    burst_multiplier: float
    retained_pool_identity_sha256: str
    requests: tuple[ScheduledRequest, ...]
    arrival_rule: str = ARRIVAL_RULE

    def __post_init__(self):
        scalars = (self.decision_seconds, self.total_requests_per_second, self.burst_multiplier)
        if (self.role != 'test' or type(self.load_label) is not str or not self.load_label
                or type(self.users) is not int or self.users < 1 or type(self.decisions) is not int or self.decisions < 1
                or type(self.seed) is not int or self.seed not in (1, 2, 3, 4, 5)
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in scalars)
                or type(self.burst_probability) not in (int, float) or not math.isfinite(self.burst_probability)
                or not 0 <= self.burst_probability <= 1 or self.burst_multiplier < 1
                or type(self.requests) is not tuple or self.arrival_rule != ARRIVAL_RULE):
            raise ValueError('Explicit registered test/seed/physical grid/unconditional arrival design required')
        require_sha(self.retained_pool_identity_sha256)
        seen = {}
        for row in self.requests:
            if (type(row) is not ScheduledRequest or type(row.task_id) is not str or not row.task_id
                    or row.task_id in seen or type(row.user) is not int or not 0 <= row.user < self.users
                    or type(row.image_id) is not str or not row.image_id
                    or type(row.measured_session) is not int or row.measured_session not in (1, 2, 3, 4, 5)
                    or type(row.arrival_seconds) not in (int, float) or not math.isfinite(row.arrival_seconds)
                    or not 0 <= row.arrival_seconds < self.horizon_seconds
                    or type(row.assignment_uniform) not in (int, float) or not math.isfinite(row.assignment_uniform)
                    or not 0 <= row.assignment_uniform < 1):
                raise ValueError('Complete unique tasks with original sessions, true arrival and exclusive uniform required')
            seen[row.task_id] = row
        if tuple(sorted(self.requests, key=lambda row: (row.arrival_seconds, row.user, row.task_id))) != self.requests:
            raise ValueError('Persisted deterministic true-arrival/user/task order required; no hidden reordering')
        object.__setattr__(self, '_by_task', MappingProxyType(seen))

    @property
    def horizon_seconds(self):
        return float(self.decisions*Fraction(str(self.decision_seconds)))

    def decision_time(self, index):
        if type(index) is not int or not 1 <= index <= self.decisions:
            raise ValueError('Decision index outside fixed common physical grid')
        return float(index*Fraction(str(self.decision_seconds)))

    def decision_index(self, request):
        if type(request) is not ScheduledRequest or self._by_task.get(request.task_id) != request:
            raise ValueError('Only own unchanged persisted task may enter decision batches')
        return max(1, math.ceil(Fraction(str(request.arrival_seconds))/Fraction(str(self.decision_seconds))))

    def to_dict(self):
        # Dynamic immutable task-index cache is excluded from dataclass fields.
        return jsonable(asdict(self))

    def fingerprint(self):
        return digest(self.to_dict())


PAYLOAD_FIELDS = {field.name for field in fields(_PreparedRetainedSchedulePayload)}


def _strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON fields cannot hide altered request definitions')
            value[key] = item
        return value
    def finite_constant(value):
        raise ValueError('Nonfinite JSON constants refused: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=finite_constant)


def _decode_registered_document(document, allowed_image_ids, retained_pool_sha):
    """Pure strict parser; artificial checks do not enable the real loader."""
    if (type(document) is not dict or set(document) != {'schema', 'payload', 'semantic_sha256'}
            or document['schema'] != SCHEMA or type(document['payload']) is not dict
            or set(document['payload']) != PAYLOAD_FIELDS):
        raise ValueError('Exact complete schema and field set, no hidden method/delay/service/outcome fields')
    raw = dict(document['payload'])
    if type(raw['requests']) is not list or any(type(row) is not dict or set(row) != ROW_FIELDS for row in raw['requests']):
        raise ValueError('Exact per-task fields, no service truth/labels/extra branch data')
    raw['requests'] = tuple(ScheduledRequest(**row) for row in raw['requests'])
    payload = _PreparedRetainedSchedulePayload(**raw)
    if (payload.retained_pool_identity_sha256 != require_sha(retained_pool_sha)
            or any(row.image_id not in allowed_image_ids for row in payload.requests)
            or payload.fingerprint() != require_sha(document['semantic_sha256'])):
        raise RuntimeError('Original image-pool identity and exact complete request semantic SHA required')
    return payload


@dataclass(frozen=True, init=False)
class RegisteredRetainedRequestSchedule:
    """SOURCE ONLY; no successful real constructor or registered rows loaded."""
    _contract: FrozenTestExecutionContract
    _payload: _PreparedRetainedSchedulePayload
    _path: Path
    _file_sha256: str
    _common_sha256: str

    def __init__(self, contract, load_label, seed, registered_path, external_file_sha256):
        if type(contract) is not FrozenTestExecutionContract:
            raise TypeError('Exact completed frozen-weight contract before actual retained schedule access')
        rows = contract.cell_descriptors()
        protocol = require_reviewed_runtime(contract._bindings)
        if (protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__))
                or protocol.get('test_schedule_schema') != SCHEMA
                or protocol.get('test_schedules_registered_before_policy_tests') is not True):
            raise RuntimeError('This current source/schema and all common schedule artifacts must be fully reviewed/registered')
        if type(seed) is not int or type(load_label) is not str:
            raise ValueError('Exact registered load label and integer paired seed')
        matched = [row for row in rows if row['load_label'] == load_label and row['seed'] == seed]
        if len(matched) != 12 or len({row['common_exogenous_schedule_sha256'] for row in matched}) != 1:
            raise ValueError('One common schedule for all four methods and three delays of this load/seed')
        common_sha = matched[0]['common_exogenous_schedule_sha256']
        registration = protocol['test_schedule_artifacts_by_load_seed'][load_label][str(seed)]
        if type(registration) is not dict or set(registration) != {'path', 'file_sha256', 'semantic_sha256'}:
            raise ValueError('Exact external path/byte/semantic registration fields required')
        path = Path(registered_path)
        root = (BASE/'visual_calibrated_policy').resolve()
        if (path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root)
                or str(path) != registration['path'] or (path.parent/'FAILURE.json').exists()
                or require_sha(external_file_sha256) != registration['file_sha256']
                or file_sha(path) != external_file_sha256 or registration['semantic_sha256'] != common_sha):
            raise RuntimeError('Externally registered unfailed schedule path/bytes/whole semantic SHA required')
        pool = OriginalRetainedMetadataPool()
        payload = _decode_registered_document(_strict_json(path.read_text(encoding='utf-8')),
            frozenset(pool.image_ids()), pool.identity_sha256)
        burst = protocol['test_arrival_design']
        expected = matched[0]
        if (payload.load_label != load_label or payload.seed != seed or payload.users != expected['users']
                or payload.decision_seconds != expected['decision_seconds']
                or payload.decisions != expected['generation_decisions']
                or payload.horizon_seconds != expected['generation_seconds']
                or payload.total_requests_per_second != expected['total_requests_per_second']
                or payload.retained_pool_identity_sha256 != contract._pool_identity
                or payload.fingerprint() != common_sha
                or payload.arrival_rule != burst['arrival_rule']
                or payload.burst_probability != burst['burst_probability']
                or payload.burst_multiplier != burst['burst_multiplier']):
            raise RuntimeError('Unchanged common pilot-frozen time/load/pool/arrival/burst design required')
        for key, value in dict(_contract=contract, _payload=payload, _path=path,
                _file_sha256=external_file_sha256, _common_sha256=common_sha).items():
            object.__setattr__(self, key, value)

    def verify_boundary(self):
        rows = self._contract.cell_descriptors()
        protocol = require_reviewed_runtime(self._contract._bindings)
        registration = protocol['test_schedule_artifacts_by_load_seed'][self.load_label][str(self.seed)]
        if (protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__))
                or file_sha(self._path) != self._file_sha256 or (self._path.parent/'FAILURE.json').exists()
                or registration != dict(path=str(self._path), file_sha256=self._file_sha256, semantic_sha256=self._common_sha256)
                or self._payload.fingerprint() != self._common_sha256
                or OriginalRetainedMetadataPool().identity_sha256 != self._payload.retained_pool_identity_sha256
                or any(row['common_exogenous_schedule_sha256'] != self._common_sha256
                    for row in rows if row['load_label'] == self.load_label and row['seed'] == self.seed)):
            raise RuntimeError('Schedule/pool/weights/source registration changed at cell boundary')

    @property
    def role(self):
        return self._payload.role

    @property
    def load_label(self):
        return self._payload.load_label

    @property
    def users(self):
        return self._payload.users

    @property
    def decisions(self):
        return self._payload.decisions

    @property
    def decision_seconds(self):
        return self._payload.decision_seconds

    @property
    def seed(self):
        return self._payload.seed

    @property
    def requests(self):
        return self._payload.requests

    @property
    def arrival_rule(self):
        return self._payload.arrival_rule

    @property
    def horizon_seconds(self):
        return self._payload.horizon_seconds

    def decision_time(self, index):
        return self._payload.decision_time(index)

    def decision_index(self, request):
        return self._payload.decision_index(request)

    def fingerprint(self):
        return self._common_sha256

    def to_dict(self):
        return self._payload.to_dict()
