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
from visual_upload_link import (UploadLinkConfig, REGISTERED_20MHZ_ORTHOGONAL_LINK,
    is_registered_link_configuration)

STATUS = 'prepared_cal_validation_engineering_candidate_not_frozen'
ENTRY_SCOPE = 'cal_validation_closed_engineering_no_optimizer_after_actual_smoke_and_verified_user_direction'
REPAIR_SCOPE = 'visual_smoke_source_repair_after_preserved_FAIL'
NETWORK_CHOICE = 'retain_2MHz_adverse_evidence_exclude_2MHz_full_grid_register_20MHz_per_user_orthogonal_simulation_for_one_formal_grid'
NETWORK_UPLOAD_LINK = asdict(REGISTERED_20MHZ_ORTHOGONAL_LINK)
NETWORK_DIRECTION_PATH = BASE/'protocol/human_direction/20261013_visual_network_choice.json'
NETWORK_DIRECTION_SHA256 = 'b0635013bd3d07dc330847c03baca313612813c836dbca174f266c050e270b32'
NETWORK_CONDITION_ID = '20MHz_per_user_orthogonal_simulation'
SMALL_SMOKE_SETTINGS = dict(users=30, decisions=8, decision_seconds=.05, fixed_drain_seconds=30.,
    total_requests_per_second=30., nominal_delay_seconds=.1, broadcast_seconds=.1, waiting_capacity=32,
    mobility_start_seconds=0., training_geometry_window=[0., 500.], geometry_seed=123,
    schedule_seed=209, model_seed=1, profiles=['CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'],
    upload_link=NETWORK_UPLOAD_LINK, network_condition_id=NETWORK_CONDITION_ID, initial_H=0.,
    purpose='Low-load 20 MHz per-user orthogonal calibration-only functional model smoke, not final scenario or budget')
SMALL_SMOKE_ADDITIONAL_BINDINGS = {
    'visual_model_smoke_small.py': '1e5c6a112caed73bc2235969fef3badfdb2af23f5dcce9fb28deee502dd115fa',
    'reward_smoke_small.py': '99f917dfc77ba9d552d1b824e5f04470b9045ae64201aaa964f9be965409be4c',
    'reward_input_integrity.py': '06a4b49ef92ee744664ac8277732adf7b727f9ea52621f3cc4397789f9e7a7c7',
    'FROZEN_BASELINE.json': '742d529b7f02301734a24f70de42e540a4b4ad5080870179762846174cf1387d',
    'README_SMALL_SMOKE_中文.md': 'fa14beb5e58e60643173cd15d7f57f1dd756d028a4ee25c5e12ce763aadd2c2c',
    'USER_REQUEST.json': 'd3abf88c574c57083c25f74b9adeaac62368a277fec3bb3cdb832e3a4d8acbcd',
}
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


def _small_smoke_source_lock():
    """Exact A4 receiver: no arbitrary subset/superset or path relabeling."""
    from visual_model_smoke import source_lock
    base = source_lock()
    if len(base) != 203:
        raise RuntimeError('The reviewed baseline import/source closure changed; bind and review a new smoke identity')
    root = BASE/'visual_calibrated_policy/engineering_attempts/20261012_s4'
    additions = {str(root/name): expected for name, expected in SMALL_SMOKE_ADDITIONAL_BINDINGS.items()}
    if set(additions) & set(base):
        raise RuntimeError('The six additive smoke bindings must remain disjoint from the 203-source baseline')
    if any(file_sha(Path(path)) != expected for path, expected in additions.items()):
        raise RuntimeError('A4 small-smoke source input changed; no old result may be relabeled')
    return base | additions


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
            or candidate['upload_link'] != NETWORK_UPLOAD_LINK
            or candidate['virtual_H_episode_rule'] not in ('carry_across_training_episodes', 'reset_each_independent_episode')):
        raise ValueError('Explicit current 30-user/four-profile/32/.05/.1 candidate; nondefault link unsupported')
    scales = _complete_dataclass(VisualFeatureScales, candidate['feature_scales'],
        ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
    closure = _complete_dataclass(EpisodeClosureCandidate, candidate['closure_settings'])
    if (scales.profiles != PROFILES or scales.decision_seconds != .05
            or not is_registered_link_configuration(scales.upload_link)
            or asdict(scales.upload_link) != candidate['upload_link']
            or scales.calibration_role != 'calibration' or closure.fixed_drain_seconds < .05):
        raise ValueError('Cal-only normalization and explicit shared physical closure required')
    scalar_scales = [value for name, value in candidate['feature_scales'].items()
        if name not in ('location_low', 'location_span', 'edge_positions', 'profiles',
                        'mean_profile_service_seconds', 'calibration_role', 'upload_link')]
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
        age_scale_seconds=declared['age_scale_seconds'],
        upload_link=UploadLinkConfig(**candidate['upload_link']))))
    fields = ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_payload_bytes',
        'mean_profile_service_seconds', 'reference_service_seconds', 'decision_seconds', 'calibration_role', 'upload_link')
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


def _validate_small_smoke_plan_and_result(plan, smoke, plan_path, plan_sha, sources, protected):
    required_plan_fields = {'status', 'methods', 'source_files_sha256',
        'protected_input_and_source_sha256', 'smoke_settings', 'formal_protocol_frozen',
        'test_access_allowed', 'actual_model_forward_and_optimizer_executed', 'actual_roundtrip_performed'}
    if (type(plan) is not dict or set(plan) != required_plan_fields
            or plan.get('status') != 'prepared_unexecuted_functional_smoke_only'
            or plan.get('methods') != list(METHODS)
            or plan.get('source_files_sha256') != sources
            or plan.get('protected_input_and_source_sha256') != protected
            or plan.get('smoke_settings') != SMALL_SMOKE_SETTINGS
            or plan.get('formal_protocol_frozen') is not False
            or plan.get('test_access_allowed') is not False
            or plan.get('actual_model_forward_and_optimizer_executed') is not False
            or plan.get('actual_roundtrip_performed') is not False):
        raise RuntimeError('Exact 8-decision, calibration-only 20 MHz smoke plan and source identity required')
    plan_sha = require_sha(plan_sha)
    plan_identity = digest(plan)
    if (type(smoke) is not dict or smoke.get('status') != 'actual_cuda_two_arm_smoke_pass'
            or smoke.get('methods') != list(METHODS)
            or smoke.get('actual_model_forward_and_optimizer_executed') is not True
            or smoke.get('actual_roundtrip_performed') is not True
            or smoke.get('independent_complete_entry_review') is not False
            or smoke.get('formal_protocol_frozen') is not False
            or smoke.get('policy_training_cells') != 0 or smoke.get('formal_policy_test_cells') != 0
            or smoke.get('source_files_sha256') != sources
            or smoke.get('protected_input_and_source_sha256') != protected
            or smoke.get('smoke_plan_path') != str(Path(plan_path).resolve())
            or smoke.get('plan_sha256') != plan_sha
            or smoke.get('semantic_identity_sha256') != plan_identity
            or smoke.get('smoke_settings') != SMALL_SMOKE_SETTINGS):
        raise RuntimeError('Actual smoke must consume this exact byte/semantic-bound 8-decision 20 MHz plan')
    reports = smoke.get('method_reports')
    if (type(reports) is not list or len(reports) != len(METHODS)
            or [row.get('method') if type(row) is dict else None for row in reports] != list(METHODS)):
        raise RuntimeError('Two method-specific actual smoke receipts required')
    from execution_service_catalog import DIRECTORIES
    from physical_mobility import KAIST
    expected_calibration = {path: sha for path, sha in protected.items()
        if Path(path).parent.resolve() == Path(DIRECTORIES['calibration']).resolve()
        and Path(path).suffix.lower() == '.csv'}
    expected_mobility = {path: sha for path, sha in protected.items()
        if Path(path).parent.resolve() == Path(KAIST).resolve() and Path(path).suffix.lower() == '.txt'}
    if not expected_calibration or len(expected_mobility) != SMALL_SMOKE_SETTINGS['users']:
        raise RuntimeError('Exact protected calibration pool and thirty mobility inputs are required')
    common_condition = None
    common_schedule = None
    common_paired_design = None
    for method in METHODS:
        row = reports[METHODS.index(method)]
        input_condition = row.get('input_condition')
        input_fields = {'calibration_source_sha256', 'mobility_source_sha256', 'schedule_sha256',
            'server_positions', 'calibration_scales', 'smoke_settings'}
        if type(input_condition) is not dict or set(input_condition) != input_fields:
            raise RuntimeError(f'{method} receipt lacks the exact producer input-condition schema')
        schedule_sha = require_sha(row.get('schedule_sha256'))
        condition_sha = require_sha(row.get('input_condition_sha256'))
        if (digest(input_condition) != condition_sha
                or input_condition.get('schedule_sha256') != schedule_sha
                or input_condition.get('smoke_settings') != SMALL_SMOKE_SETTINGS
                or input_condition.get('calibration_source_sha256') != expected_calibration
                or input_condition.get('mobility_source_sha256') != expected_mobility
                or row.get('calibration_source_sha256') != expected_calibration
                or row.get('mobility_source_sha256') != expected_mobility
                or type(input_condition.get('server_positions')) is not list
                or len(input_condition['server_positions']) != 4
                or type(input_condition.get('calibration_scales')) is not dict
                or input_condition['calibration_scales'].get('upload_link') != NETWORK_UPLOAD_LINK):
            raise RuntimeError(f'{method} receipt raw membership or input-condition identity is invalid')
        ledger = row.get('ledger')
        ledger_fields = {'method', 'seed', 'phase', 'candidate_budget', 'accounting',
            'paired_design_sha256', 'committed_episodes', 'current_episode', 'events', 'failure',
            'checkpoint_selection_done', 'formal_protocol_frozen',
            'model_execution_proven_by_this_ledger', 'automatic_resume_supported'}
        expected_budget = dict(episode_decisions=[8], users=30, decision_seconds=.05,
            fixed_drain_seconds=30.)
        if (type(ledger) is not dict or set(ledger) != ledger_fields
                or ledger.get('method') != method or type(ledger.get('seed')) is not int or ledger['seed'] != 1
                or ledger.get('phase') != 'budget_complete_pending_validation'
                or ledger.get('candidate_budget') != expected_budget
                or ledger.get('failure') is not None or ledger.get('current_episode') is not None
                or ledger.get('checkpoint_selection_done') is not False
                or ledger.get('formal_protocol_frozen') is not False
                or ledger.get('model_execution_proven_by_this_ledger') is not False
                or ledger.get('automatic_resume_supported') is not False):
            raise RuntimeError(f'{method} ledger is not a successful one-episode completed smoke transaction')
        accounting = ledger.get('accounting')
        if (type(accounting) is not dict or accounting.get('environment_decisions') != 8
                or accounting.get('user_behavior_samples') != 240):
            raise RuntimeError(f'{method} receipt budget accounting is inconsistent with 8×30')
        paired_design_sha = require_sha(ledger.get('paired_design_sha256'))
        committed = ledger.get('committed_episodes')
        if type(committed) is not list or len(committed) != 1 or type(committed[0]) is not dict:
            raise RuntimeError(f'{method} must have exactly one actually committed episode')
        commit = committed[0]
        commit_fields = {'episode_index', 'schedule_sha256', 'decisions', 'data_role', 'episode_token',
            'users', 'behavior_samples', 'action_interval_seconds', 'reward_semantics',
            'update_statistics_sha256', 'checkpoint_sha256'}
        token = commit.get('episode_token')
        expected_reward = 'QLA-shared-receipt' if method == 'QLA-MAPPO' else 'PPO-MV-A'
        expected_counts = dict(episode_token=token, decisions=8, users=30, behavior_samples=240,
            action_interval_seconds=.4, reward_semantics=expected_reward)
        stats = row.get('update_statistics')
        stats_sha = require_sha(commit.get('update_statistics_sha256'))
        checkpoint_sha = require_sha(commit.get('checkpoint_sha256'))
        checkpoint = row.get('checkpoint_archive')
        if (set(commit) != commit_fields or type(commit.get('episode_index')) is not int
                or commit['episode_index'] != 1 or commit.get('schedule_sha256') != schedule_sha
                or commit.get('decisions') != 8 or commit.get('data_role') != 'calibration'
                or not isinstance(token, str) or not token.strip()
                or type(commit.get('users')) is not int or commit['users'] != 30
                or type(commit.get('behavior_samples')) is not int or commit['behavior_samples'] != 240
                or type(commit.get('action_interval_seconds')) not in (int, float)
                or not math.isfinite(commit['action_interval_seconds'])
                or commit['action_interval_seconds'] != .4 or commit.get('reward_semantics') != expected_reward
                or type(stats) is not dict or not stats
                or any(not isinstance(key, str) or type(value) not in (int, float) or not math.isfinite(value)
                    for key, value in stats.items())
                or digest(stats) != stats_sha
                or row.get('counts') != expected_counts
                or type(checkpoint) is not dict or checkpoint.get('artifact_sha256') != checkpoint_sha):
            raise RuntimeError(f'{method} committed episode/update/checkpoint does not match actual counts and hashes')
        descriptor = dict(method=method, seed=1, episode_index=1,
            committed_environment_decisions=8, committed_behavior_samples=240,
            paired_design_sha256=paired_design_sha, data_role='calibration',
            episode_token=token, snapshot_only_no_resume=True)
        events = ledger.get('events')
        expected_events = [
            dict(event='begin_collect', episode_index=1, schedule_sha256=schedule_sha,
                decisions=8, data_role='calibration'),
            dict(event='collection_closed', counts=expected_counts),
            dict(event='begin_update', episode_token=token),
            dict(event='update_committed', episode_token=token),
            dict(event='checkpoint_attached', **descriptor, artifact_sha256=checkpoint_sha),
        ]
        if events != expected_events or checkpoint.get('descriptor') != descriptor:
            raise RuntimeError(f'{method} ledger event chain/checkpoint attachment is incomplete or reordered')
        if (row.get('source_files_sha256') != sources
                or row.get('protected_input_and_source_sha256') != protected
                or row.get('plan_sha256') != plan_sha
                or row.get('semantic_identity_sha256') != plan_identity
                or row.get('actual_model_forward_and_optimizer_executed') is not True
                or row.get('actual_roundtrip_performed') is not True
                or row.get('exact_loaded_parameters_and_deterministic_outputs') is not True
                or row.get('deterministic_Torch_RNG_unchanged') is not True
                or require_sha(row.get('before_parameter_sha256')) == require_sha(row.get('after_parameter_sha256'))):
            raise RuntimeError(f'{method} receipt does not prove actual update and exact registered 8×30 roundtrip')
        witnesses = row.get('roundtrip_witness_files_sha256')
        if (type(witnesses) is not dict or set(witnesses) !=
                {'ACTUAL_ROUNDTRIP_INPUT_OUTPUT.npz', 'ACTUAL_READONLY_TORCH_RNG.npz'}):
            raise RuntimeError(f'{method} roundtrip witness files are incomplete')
        for value in witnesses.values():
            require_sha(value)
        if common_condition is None:
            common_condition, common_schedule, common_paired_design = input_condition, schedule_sha, paired_design_sha
        elif (input_condition != common_condition or schedule_sha != common_schedule
                or paired_design_sha != common_paired_design):
            raise RuntimeError('Both methods must share one exact schedule, source map, and input condition')
    if smoke.get('input_condition_sha256') != digest(common_condition):
        raise RuntimeError('Two-arm summary must carry the SHA256 of the shared input-condition bytes')


def _validate_human_direction(direction, path, sha256, candidate_upload_link):
    required = {'schema', 'recorded_at', 'timezone', 'trusted_user_message_reference',
        'verbatim_user_message', 'network_choice', 'link_configuration', 'upload_link_sha256',
        'interpretation', 'local_record_alone_is_not_authorization'}
    if (Path(path).resolve() != NETWORK_DIRECTION_PATH.resolve()
            or require_sha(sha256) != NETWORK_DIRECTION_SHA256
            or type(direction) is not dict or set(direction) != required
            or direction.get('schema') != 'visual-network-direction-v1'
            or direction.get('recorded_at') != '2026-10-13'
            or direction.get('timezone') != 'Asia/Shanghai'
            or type(direction.get('trusted_user_message_reference')) is not str
            or not direction['trusted_user_message_reference'].strip()
            or type(direction.get('verbatim_user_message')) is not str
            or not direction['verbatim_user_message'].strip()
            or direction.get('local_record_alone_is_not_authorization') is not True
            or direction.get('network_choice') != NETWORK_CHOICE
            or direction.get('link_configuration') != NETWORK_UPLOAD_LINK
            or candidate_upload_link != NETWORK_UPLOAD_LINK
            or direction.get('upload_link_sha256') != digest(candidate_upload_link)):
        raise RuntimeError('Exact externally authorized direction and registered 20 MHz link required')
    return direction


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
        from visual_model_smoke import protected_bindings
        sources, protected = _small_smoke_source_lock(), protected_bindings()
        traces, output = BASE.parent/'.aris/traces', BASE/'visual_calibrated_policy'
        candidate = _read_actual(bindings.candidate_path, bindings.candidate_sha256, BASE/'protocol')
        summary = validate_cal_validation_pilot_candidate(candidate)
        if candidate.get('upload_link') != NETWORK_UPLOAD_LINK:
            raise RuntimeError('The one formal grid is separately registered only for the exact 20 MHz per-user link')
        smoke = _read_actual(bindings.actual_smoke_path, bindings.actual_smoke_sha256, output)
        smoke_plan_path = smoke.get('smoke_plan_path') if type(smoke) is dict else None
        smoke_plan_sha = smoke.get('plan_sha256') if type(smoke) is dict else None
        plan = _read_actual(smoke_plan_path, smoke_plan_sha, output)
        _validate_small_smoke_plan_and_result(plan, smoke, smoke_plan_path, smoke_plan_sha, sources, protected)
        repair = _read_actual(bindings.source_repair_review_path, bindings.source_repair_review_sha256, traces)
        review = _read_actual(bindings.pilot_entry_review_path, bindings.pilot_entry_review_sha256, traces)
        direction = _read_actual(bindings.human_direction_record_path, bindings.human_direction_record_sha256,
                                 BASE/'protocol/human_direction')
        _require_review(repair, REPAIR_SCOPE, sources, protected)
        _require_review(review, ENTRY_SCOPE, sources, protected)
        if (repair.get('preserved_failed_review_sha256') != FAILED_REVIEW_SHA
                or file_sha(_safe_path(FAILED_REVIEW, output)) != FAILED_REVIEW_SHA):
            raise RuntimeError('Preserved original FAIL and bounded additive repair review must remain explicit')
        _validate_human_direction(direction, bindings.human_direction_record_path,
            bindings.human_direction_record_sha256, candidate['upload_link'])
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
        files = merge_frozen_file_bindings(files, {str(Path(smoke_plan_path).resolve()): require_sha(smoke_plan_sha)})
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
        self._smoke_sources, self._smoke_protected = deepcopy(sources), deepcopy(protected)
        self._smoke_plan_path, self._smoke_plan_sha = str(Path(smoke_plan_path).resolve()), require_sha(smoke_plan_sha)
        self._smoke_plan_identity = digest(plan)
        self._geometry_provenance = geometry_provenance
        self._failed = False
        self.verify_boundary()

    def verify_boundary(self):
        if self._failed:
            raise RuntimeError('Failed preparation cannot be reused; preserve evidence/new reviewed additive attempt')
        try:
            from visual_model_smoke import protected_bindings
            current = merge_frozen_file_bindings(_small_smoke_source_lock(), protected_bindings())
            if any(self._inputs.get(name) != sha for name, sha in current.items()):
                raise RuntimeError('Current complete source/protected lock changed after preparation')
            for name, sha in self._inputs.items():
                path = _safe_path(name, BASE.parent)
                if not path.is_file() or file_sha(path) != sha:
                    raise RuntimeError('Externally bound preparation/actual role evidence changed')
            candidate = _read_actual(self.bindings.candidate_path, self.bindings.candidate_sha256, BASE/'protocol')
            if candidate != self._candidate or validate_cal_validation_pilot_candidate(candidate) != self._summary:
                raise RuntimeError('Candidate changed in memory or on disk; no outcome-selected replacement')
            output = BASE/'visual_calibrated_policy'
            smoke = _read_actual(self.bindings.actual_smoke_path, self.bindings.actual_smoke_sha256, output)
            plan = _read_actual(self._smoke_plan_path, self._smoke_plan_sha, output)
            if digest(plan) != self._smoke_plan_identity:
                raise RuntimeError('The exact byte-bound 20 MHz smoke plan changed after preparation')
            _validate_small_smoke_plan_and_result(plan, smoke, self._smoke_plan_path,
                self._smoke_plan_sha, self._smoke_sources, self._smoke_protected)
        except Exception:
            self._failed = True
            raise

    def paired_condition_descriptors(self):
        self.verify_boundary()
        return deepcopy(self._summary['paired_engineering_descriptors'])
