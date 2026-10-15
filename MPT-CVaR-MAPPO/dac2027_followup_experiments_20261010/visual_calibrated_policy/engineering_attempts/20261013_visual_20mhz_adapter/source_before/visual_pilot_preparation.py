"""SOURCE ONLY: a pre-pilot gate, distinct from completed-pilot formal entry.

No launch, model, optimizer, sampling, fitting, pilot freeze or test API. The
real preparation object requires current actual smoke, independent repair and
engineering-entry reviews, and externally verified actual human direction.
Neither a candidate nor a local 'user' field is proof of human authorization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from pathlib import Path
import sys
import numpy as np

from retained_test_grid_registration import _read_actual, _safe_path
from visual_episode_collector import BASE, EpisodeClosureCandidate
from visual_method_features import VisualFeatureScales
from visual_protocol_assembly import _complete_dataclass, merge_frozen_file_bindings, validate_environment_descriptor
from visual_test_contract import PROFILES
from visual_training_transactions import METHODS, digest, file_sha, require_sha
from visual_upload_link import UploadLinkConfig

STATUS = 'prepared_cal_validation_engineering_candidate_not_frozen'
ENTRY_SCOPE = 'cal_validation_closed_engineering_no_optimizer_after_actual_smoke_and_verified_user_direction'
REPAIR_SCOPE = 'visual_smoke_source_repair_after_preserved_FAIL'
NETWORK_CHOICE = 'retain_registered_2MHz_adverse_condition'
FAILED_REVIEW_SHA = 'e70537fd0a330631d6ce0f42c2c3fb80115dcec0af91f109ebcf12a5b36ae5ab'
FAILED_REVIEW = BASE/'visual_calibrated_policy/entry_source_reviews/20261011_115016_650790_independent_typeA_source/REVIEW_RESPONSE.md'
FIELDS = frozenset(('status', 'users', 'profiles', 'decision_seconds', 'broadcast_seconds',
    'waiting_capacity', 'upload_link', 'virtual_H_episode_rule', 'feature_scales',
    'closure_settings', 'engineering_conditions', 'placement_calibration_window_seconds',
    'placement_seed', 'mobility_role_gap_seconds'))
CONDITION_FIELDS = frozenset(('condition_id', 'role', 'seed', 'decisions',
    'schedule_artifact', 'environment_descriptor'))
PERMITTED_ACTIONS = ['collect_closed_calibration_engineering_without_optimizer',
                     'readonly_validation_engineering_without_optimizer']


def pilot_role_physical_windows(candidate, actual_physical_support=None):
    """Pure full physical windows, including drain and fitting geometry.

    An artificial support is only an arithmetic fixture. The real caller uses
    the original thirty-file PhysicalMobility support, without wrap or hold.
    Same-role repeated conditions may overlap; held-out validation cannot
    overlap either calibration execution/drain OR the geometry fitting window.
    No fitting, trajectory reads, candidate selection or result access here.
    """
    window = candidate['placement_calibration_window_seconds']
    gap, seed = candidate['mobility_role_gap_seconds'], candidate['placement_seed']
    if (type(window) is not list or len(window) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in window)
            or not 0 <= window[0] < window[1] or type(seed) is not int or seed < 0
            or type(gap) not in (int, float) or not math.isfinite(gap) or gap < 0):
        raise ValueError('Explicit calibration geometry window/native placement seed/finite physical role gap required')
    dt, drain = candidate['decision_seconds'], candidate['closure_settings']['fixed_drain_seconds']
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (dt, drain)):
        raise ValueError('Physical full generation and fixed drain, not index or implicit duration')
    rows = [dict(role='calibration', purpose='placement_and_location_scale_fit',
        start_seconds=window[0], end_seconds=window[1])]
    exact = [('calibration', Fraction(str(window[0])), Fraction(str(window[1])))]
    exact_gap = Fraction(str(gap))
    for condition in candidate['engineering_conditions']:
        role, n = condition['role'], condition['decisions']
        start = condition['environment_descriptor']['mobility_start_seconds']
        if (role not in ('calibration', 'validation') or type(n) is not int or n < 1
                or type(start) not in (int, float) or not math.isfinite(start) or start < 0):
            raise ValueError('Declared cal/validation whole interval and native decisions required')
        end_exact = Fraction(str(start))+n*Fraction(str(dt))+Fraction(str(drain))
        try:
            end = float(end_exact)
        except OverflowError as error:
            raise ValueError('Finite whole engineering physical window required') from error
        if not math.isfinite(end):
            raise ValueError('Finite whole engineering physical window required')
        rows.append(dict(role=role, purpose='whole_engineering_generation_and_fixed_drain',
            start_seconds=start, end_seconds=end))
        exact.append((role, Fraction(str(start)), end_exact))
    # Same registered decimal-second semantics as request/broadcast indices;
    # no epsilon tolerance or silent window shift to overcome float drift.
    for calibration in (row for row in exact if row[0] == 'calibration'):
        for validation in (row for row in exact if row[0] == 'validation'):
            if not (calibration[2]+exact_gap <= validation[1]
                    or validation[2]+exact_gap <= calibration[1]):
                raise ValueError('Validation overlaps calibration execution/drain/geometry fitting or explicit role gap')
    if actual_physical_support is not None:
        if (type(actual_physical_support) not in (list, tuple) or len(actual_physical_support) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in actual_physical_support)
                or actual_physical_support[0] >= actual_physical_support[1]):
            raise ValueError('Complete original common physical support required')
        low, high = (Fraction(str(value)) for value in actual_physical_support)
        if any(not low <= row[1] < row[2] <= high for row in exact):
            raise ValueError('Full geometry/role generation+drain exceeds original support; no wrap/hold/extrapolation')
    return deepcopy(rows)


def validate_cal_validation_pilot_candidate(candidate):
    """Pure explicit candidate descriptors, not actual conditions or authority.

    Every condition is shared by both methods. This first engineering phase
    does not optimize/choose checkpoints, freeze the final design or generate
    retained requests. Its evidence must not be relabeled as the full pilot.
    """
    if type(candidate) is not dict or set(candidate) != FIELDS:
        raise ValueError('Complete provisional engineering schema only; no outcome/authority/future fields')
    if (candidate['status'] != STATUS or type(candidate['users']) is not int or candidate['users'] != 30
            or candidate['profiles'] != list(PROFILES) or type(candidate['waiting_capacity']) is not int
            or candidate['waiting_capacity'] != 32
            or type(candidate['decision_seconds']) not in (int, float) or candidate['decision_seconds'] != .05
            or type(candidate['broadcast_seconds']) not in (int, float) or candidate['broadcast_seconds'] != .1
            or candidate['upload_link'] != asdict(UploadLinkConfig())
            or candidate['virtual_H_episode_rule'] not in ('carry_across_training_episodes', 'reset_each_independent_episode')):
        raise ValueError('Explicit current 30-user/four-profile/32/.05/.1 candidate; nondefault link unsupported')
    scales = _complete_dataclass(VisualFeatureScales, candidate['feature_scales'],
        ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
    closure = _complete_dataclass(EpisodeClosureCandidate, candidate['closure_settings'])
    if (scales.profiles != PROFILES or scales.decision_seconds != .05
            or scales.calibration_role != 'calibration' or closure.fixed_drain_seconds < .05):
        raise ValueError('Cal-only normalization and explicit shared physical closure required')
    scalar_scales = [value for name, value in candidate['feature_scales'].items()
        if name not in ('location_low', 'location_span', 'edge_positions', 'profiles',
                        'mean_profile_service_seconds', 'calibration_role')]
    numeric = (*scalar_scales, *scales.location_low, *scales.location_span,
        *scales.mean_profile_service_seconds, *candidate['closure_settings'].values())
    if (len(scales.location_low) != 2 or len(scales.location_span) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in numeric)):
        raise ValueError('Explicit native finite scale/closure numbers and two physical coordinates, not booleans')
    rows = candidate['engineering_conditions']
    if type(rows) is not list or not rows:
        raise ValueError('Explicit whole cal/validation engineering conditions required')
    identifiers, roles, descriptors = set(), set(), []
    for row in rows:
        if type(row) is not dict or set(row) != CONDITION_FIELDS:
            raise ValueError('Common condition, no method-specific settings or outcome selection')
        role, registration = row['role'], row['schedule_artifact']
        if (type(row['condition_id']) is not str or not row['condition_id'] or row['condition_id'] in identifiers
                or role not in ('calibration', 'validation') or type(row['seed']) is not int or row['seed'] < 0
                or type(row['decisions']) is not int or row['decisions'] < 1
                or type(registration) is not dict or set(registration) != {'path', 'file_sha256', 'semantic_sha256'}
                or type(registration['path']) is not str or not registration['path']):
            raise ValueError('Unique complete cal/validation schedule bindings, native seeds and full decisions required')
        require_sha(registration['file_sha256'])
        env = validate_environment_descriptor(row['environment_descriptor'], role,
            registration['semantic_sha256'], candidate['broadcast_seconds'])
        if (env['upload_link'] != candidate['upload_link']
                or env['edge_positions'] != [list(point) for point in scales.edge_positions]):
            raise ValueError('Same candidate link and calibration-derived placement for every method/role')
        identifiers.add(row['condition_id'])
        roles.add(role)
        for method in METHODS:
            descriptors.append(dict(condition_id=row['condition_id'], method=method, role=role,
                seed=row['seed'], decisions=row['decisions'], schedule_sha256=registration['semantic_sha256'],
                common_condition_sha256=digest(row), optimizer_updates_allowed=False,
                formal_training_or_test_cell=False))
    if roles != {'calibration', 'validation'}:
        raise ValueError('Both original roles required; test role cannot enter this engineering preparation')
    windows = pilot_role_physical_windows(candidate)
    # digest rejects NaN anywhere, including unknown numeric dataclass values.
    identity = digest(candidate)
    return dict(candidate_scientific_identity_sha256=identity,
        common_conditions=len(rows), paired_engineering_descriptors=deepcopy(descriptors),
        declared_role_physical_windows=windows,
        actual_pilot_or_model_execution_proven=0, final_pilot_or_protocol_frozen=False)


def _require_actual_calibration_geometry_and_scales(candidate, catalog, mobility):
    """SOURCE ONLY real provenance check, after actual preparation authority.

    Recompute ONLY common calibration-derived fields from complete original
    calibration data/explicit geometry window. It neither reads test truth nor
    changes any candidate, original predictor, measurement or frozen protocol.
    Engineering tuning fields stay explicit candidates, not inferred defaults.
    """
    from execution_service_catalog import ExecutionServiceCatalog
    from physical_mobility import PhysicalMobility, TrainingWindow
    from visual_method_features import calibration_scales
    from visual_episode_collector import jsonable
    if type(catalog) is not ExecutionServiceCatalog or catalog.role != 'calibration' or type(mobility) is not PhysicalMobility:
        raise TypeError('Actual full original calibration catalog and true physical mobility only, no validation fitting/stubs')
    windows = pilot_role_physical_windows(candidate, (mobility.common_start_seconds, mobility.common_end_seconds))
    window = TrainingWindow(*candidate['placement_calibration_window_seconds'])
    edges = mobility.server_positions(window, candidate['placement_seed'], 4)
    declared = candidate['feature_scales']
    derived = jsonable(asdict(calibration_scales(catalog, mobility, window, edges, tuple(candidate['profiles']),
        decision_seconds=candidate['decision_seconds'], latency_scale_seconds=declared['latency_scale_seconds'],
        age_scale_seconds=declared['age_scale_seconds'])))
    fields = ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_payload_bytes',
        'mean_profile_service_seconds', 'reference_service_seconds', 'decision_seconds', 'calibration_role')
    if any(declared[name] != derived[name] for name in fields):
        raise RuntimeError('Declared placement/location/payload/service scales do not come from this whole calibration input/window')
    for row in windows:
        mobility.positions(row['start_seconds'])
        mobility.positions(row['end_seconds'])
    return dict(calibration_catalog_source_sha256=dict(catalog.source_sha256),
        mobility_source_sha256=dict(mobility.source_hashes), explicit_geometry_window=asdict(window),
        placement_seed=candidate['placement_seed'], recomputed_calibration_only_field_names=list(fields),
        complete_original_support_seconds=[mobility.common_start_seconds, mobility.common_end_seconds],
        full_role_windows=windows, candidate_mutated_or_frozen=False, validation_or_test_labels_used_for_fitting=False)


@dataclass(frozen=True)
class PilotPreparationArtifactBindings:
    candidate_path: str
    candidate_sha256: str
    actual_smoke_path: str
    actual_smoke_sha256: str
    source_repair_review_path: str
    source_repair_review_sha256: str
    pilot_entry_review_path: str
    pilot_entry_review_sha256: str
    human_direction_record_path: str
    human_direction_record_sha256: str


def _require_review(review, scope, sources, protected):
    if (review.get('review_type') != 'Type-A' or review.get('model') != 'gpt-6-astra'
            or review.get('reasoning_effort') != 'xhigh' or review.get('scope') != scope
            or review.get('status') not in ('PASS', 'WARN') or review.get('blockers') != []
            or review.get('source_files_sha256') != sources
            or review.get('protected_input_and_source_sha256') != protected):
        raise RuntimeError('Fresh exact independent review, not an author fixture or old source FAIL')


class RealCalValidationPilotPreparation:
    """SOURCE ONLY. Actual successful construction count zero; no run API.

    A future engineering executor must consume this exact live object and call
    verify_boundary before/after each condition, then separately check actual
    controller/environment/receipt provenance. This is not completed evidence.
    """
    def __init__(self, bindings):
        if type(bindings) is not PilotPreparationArtifactBindings:
            raise TypeError('Actual external smoke, repair review, candidate, entry review and human direction required')
        from qla_environment import guard
        spec = guard()  # Version/spec only. No additional GPU probe/allocation.
        if (sys.version.split()[0] != '3.13.5' or np.__version__ != '2.2.6'
                or Path(sys.executable).resolve() != (Path(spec['isolated_environment'])/'Scripts/python.exe').resolve()):
            raise RuntimeError('Declared isolated runtime only, no CPU neural fallback/global rebuild')
        from visual_model_smoke import source_lock, protected_bindings
        sources, protected = source_lock(), protected_bindings()
        traces, output = BASE.parent/'.aris/traces', BASE/'visual_calibrated_policy'
        candidate = _read_actual(bindings.candidate_path, bindings.candidate_sha256, BASE/'protocol')
        summary = validate_cal_validation_pilot_candidate(candidate)
        smoke = _read_actual(bindings.actual_smoke_path, bindings.actual_smoke_sha256, output)
        repair = _read_actual(bindings.source_repair_review_path, bindings.source_repair_review_sha256, traces)
        review = _read_actual(bindings.pilot_entry_review_path, bindings.pilot_entry_review_sha256, traces)
        direction = _read_actual(bindings.human_direction_record_path, bindings.human_direction_record_sha256,
                                 BASE/'protocol/human_direction')
        _require_review(repair, REPAIR_SCOPE, sources, protected)
        _require_review(review, ENTRY_SCOPE, sources, protected)
        if (repair.get('preserved_failed_review_sha256') != FAILED_REVIEW_SHA
                or file_sha(_safe_path(FAILED_REVIEW, output)) != FAILED_REVIEW_SHA):
            raise RuntimeError('Preserved original FAIL and bounded additive repair review must remain explicit')
        if (smoke.get('status') != 'actual_cuda_two_arm_smoke_pass' or smoke.get('methods') != list(METHODS)
                or smoke.get('actual_model_forward_and_optimizer_executed') is not True
                or smoke.get('actual_roundtrip_performed') is not True
                or smoke.get('source_files_sha256') != sources
                or smoke.get('protected_input_and_source_sha256') != protected):
            raise RuntimeError('Actual current two-arm forward/update/roundtrip evidence required')
        if (type(direction) is not dict or set(direction) != {'trusted_user_message_reference',
                'verbatim_user_message', 'network_choice', 'upload_link_sha256',
                'local_record_alone_is_not_authorization'}
                or type(direction['trusted_user_message_reference']) is not str or not direction['trusted_user_message_reference']
                or type(direction['verbatim_user_message']) is not str or not direction['verbatim_user_message'].strip()
                or direction['local_record_alone_is_not_authorization'] is not True
                or direction['network_choice'] != NETWORK_CHOICE
                or direction['upload_link_sha256'] != digest(candidate['upload_link'])):
            raise RuntimeError('Actual explicit network direction required; a local user/origin flag is not authority')
        if (review.get('candidate_sha256') != bindings.candidate_sha256
                or review.get('candidate_scientific_identity_sha256') != summary['candidate_scientific_identity_sha256']
                or review.get('actual_smoke_sha256') != bindings.actual_smoke_sha256
                or review.get('source_repair_review_sha256') != bindings.source_repair_review_sha256
                or review.get('human_direction_record_sha256') != bindings.human_direction_record_sha256
                or review.get('trusted_user_message_reference') != direction['trusted_user_message_reference']
                or review.get('human_direction_verified_against_actual_user_message') is not True
                or review.get('authorized_network_choice') != NETWORK_CHOICE
                or review.get('permitted_actions') != PERMITTED_ACTIONS
                or review.get('final_pilot_freeze_authorized') is not False
                or review.get('metadata_schedule_generation_authorized') is not False
                or review.get('formal_training_or_test_launch_authorized') is not False):
            raise RuntimeError('Externally verified human provenance and narrowly scoped engineering permission only')
        files = merge_frozen_file_bindings(sources, protected)
        files = merge_frozen_file_bindings(files, {str(FAILED_REVIEW): FAILED_REVIEW_SHA})
        for prefix in ('candidate', 'actual_smoke', 'source_repair_review', 'pilot_entry_review', 'human_direction_record'):
            files = merge_frozen_file_bindings(files, {str(Path(getattr(bindings, prefix+'_path'))):
                require_sha(getattr(bindings, prefix+'_sha256'))})
        # Bind full actual original cal/validation schedules and catalogs, not
        # only their declarations. Parse requests without any test access.
        from visual_protocol_assembly import _read_engineering_schedule
        from execution_service_catalog import DIRECTORIES, ExecutionServiceCatalog
        from physical_mobility import PhysicalMobility
        original_catalogs = {role: {str(path): file_sha(path) for path in DIRECTORIES[role].glob('*.csv')}
            for role in ('calibration', 'validation')}
        mobility = PhysicalMobility(users=30)
        mobility_sources = mobility.source_hashes
        for row in candidate['engineering_conditions']:
            schedule = _read_engineering_schedule(row['schedule_artifact'], row['role'], row['decisions'], .05)
            if type(schedule.seed) is not int or schedule.seed != row['seed']:
                raise RuntimeError('Actual common exogenous schedule seed must match its declared engineering seed')
            env = row['environment_descriptor']
            if (env['catalog_source_sha256'] != original_catalogs[row['role']]
                    or env['mobility_source_sha256'] != mobility_sources):
                raise RuntimeError('Whole original role catalog and thirty true physical mobility files required')
            for mapping in (original_catalogs[row['role']], mobility_sources,
                    {row['schedule_artifact']['path']: row['schedule_artifact']['file_sha256']}):
                files = merge_frozen_file_bindings(files, mapping)
        calibration = ExecutionServiceCatalog('calibration')
        if calibration.source_sha256 != original_catalogs['calibration']:
            raise RuntimeError('Original calibration catalog changed before provenance verification')
        geometry_provenance = _require_actual_calibration_geometry_and_scales(candidate, calibration, mobility)
        if review.get('engineering_input_files_sha256') != {
                name: sha for name, sha in files.items() if name not in sources and name not in protected
                and Path(name).resolve() != Path(bindings.pilot_entry_review_path).resolve()}:
            raise RuntimeError('Entry review must bind all actual candidate/smoke/direction/role inputs')
        self.bindings, self._candidate, self._summary, self._inputs = bindings, deepcopy(candidate), summary, files
        self._geometry_provenance = geometry_provenance
        self._failed = False
        self.verify_boundary()

    def verify_boundary(self):
        if self._failed:
            raise RuntimeError('Failed preparation cannot be reused; preserve evidence/new reviewed additive attempt')
        try:
            from visual_model_smoke import source_lock, protected_bindings
            current = merge_frozen_file_bindings(source_lock(), protected_bindings())
            if any(self._inputs.get(name) != sha for name, sha in current.items()):
                raise RuntimeError('Current complete source/protected lock changed after preparation')
            for name, sha in self._inputs.items():
                path = _safe_path(name, BASE.parent)
                if not path.is_file() or file_sha(path) != sha:
                    raise RuntimeError('Externally bound preparation/actual role evidence changed')
            candidate = _read_actual(self.bindings.candidate_path, self.bindings.candidate_sha256, BASE/'protocol')
            if candidate != self._candidate or validate_cal_validation_pilot_candidate(candidate) != self._summary:
                raise RuntimeError('Candidate changed in memory or on disk; no outcome-selected replacement')
        except Exception:
            self._failed = True
            raise

    def paired_condition_descriptors(self):
        self.verify_boundary()
        return deepcopy(self._summary['paired_engineering_descriptors'])
