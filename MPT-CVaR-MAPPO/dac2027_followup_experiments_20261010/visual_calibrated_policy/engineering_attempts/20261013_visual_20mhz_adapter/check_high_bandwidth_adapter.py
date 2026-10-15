"""Bounded, non-neural calibration/validation link-adapter checks."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE/'scripts'))

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility, TrainingWindow
from visual_environment_adapter import VisualEnvironmentAdapter
from visual_episode_collector import EpisodeClosureCandidate
from visual_method_features import VisualFeatureScales, calibration_scales, pack_features
from visual_pilot_preparation import (NETWORK_CHOICE, NETWORK_UPLOAD_LINK, STATUS,
    NETWORK_DIRECTION_PATH, SMALL_SMOKE_SETTINGS, _small_smoke_source_lock,
    _validate_human_direction, _validate_small_smoke_plan_and_result,
    validate_cal_validation_pilot_candidate)
from visual_protocol_assembly import validate_environment_descriptor
from visual_request_schedule import generate_schedule
from visual_test_contract import PROFILES
from visual_training_transactions import digest, file_sha
from visual_upload_link import (UploadLinkConfig, REGISTERED_20MHZ_ORTHOGONAL_LINK,
    is_registered_link_configuration)


def main():
    calibration = ExecutionServiceCatalog('calibration')
    mobility = PhysicalMobility(users=30)
    window = TrainingWindow(0., 500.)
    edges = mobility.server_positions(window, 123, 4)
    default_link = UploadLinkConfig()
    high_link = REGISTERED_20MHZ_ORTHOGONAL_LINK
    assert is_registered_link_configuration(default_link)
    assert is_registered_link_configuration(high_link)
    assert not is_registered_link_configuration(UploadLinkConfig(bandwidth_hz=10e6))

    schedule = generate_schedule(calibration, 30, 8, .05, 123, 30.)
    env_default = VisualEnvironmentAdapter(schedule, calibration, mobility, edges,
        ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'), 0., link=default_link)
    env_high = VisualEnvironmentAdapter(schedule, calibration, mobility, edges,
        ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'), 0., link=high_link)
    default_view, high_view = env_default.observe(), env_high.observe()
    for user_low, user_high in zip(default_view.users, high_view.users):
        assert [r.task_id for r in user_low.requests] == [r.task_id for r in user_high.requests]
        for request_low, request_high in zip(user_low.requests, user_high.requests):
            assert request_low.metadata.payload_bytes == request_high.metadata.payload_bytes
            assert np.allclose(np.asarray(request_high.transmission_seconds),
                np.asarray(request_low.transmission_seconds) / 10., rtol=1e-12, atol=0.)

    scales_default = calibration_scales(calibration, mobility, window, edges,
        ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'), upload_link=default_link)
    scales_high = calibration_scales(calibration, mobility, window, edges,
        ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'), upload_link=high_link)
    empty_view = replace(high_view, users=tuple(replace(user, requests=()) for user in high_view.users))
    low_empty_features = pack_features(empty_view, scales_default)
    high_empty_features = pack_features(empty_view, scales_high)
    assert low_empty_features.actor.shape == high_empty_features.actor.shape == (30, 19)
    assert np.isfinite(high_empty_features.actor).all()
    assert not np.array_equal(low_empty_features.actor, high_empty_features.actor)

    descriptor = dict(schedule_sha256=schedule.fingerprint(), role='calibration',
        profiles=list(('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')),
        edge_positions=[list(point) for point in edges], mobility_start_seconds=0.,
        broadcast_seconds=.1, nominal_delay_seconds=.1, waiting_capacity=32,
        upload_link=asdict(high_link), catalog_source_sha256=dict(calibration.source_sha256),
        mobility_source_sha256=dict(mobility.source_hashes))
    accepted = validate_environment_descriptor(descriptor, 'calibration', schedule.fingerprint(), .1)
    assert accepted['upload_link'] == asdict(high_link)
    bad = dict(descriptor, upload_link=asdict(UploadLinkConfig(bandwidth_hz=10e6)))
    try:
        validate_environment_descriptor(bad, 'calibration', schedule.fingerprint(), .1)
    except ValueError:
        pass
    else:
        raise AssertionError('Unregistered bandwidth must be rejected')

    # Pure candidate checks prove only that the separately registered network
    # and the all-method common descriptor are wired consistently.
    profile_edges = [list(point) for point in edges]
    feature_scales = json.loads(json.dumps(asdict(VisualFeatureScales(
        (0., 0.), (1., 1.), tuple(edges), PROFILES, 100., (.03, .03, .01, .01), .03,
        upload_link=high_link))))
    conditions = []
    for i, role in enumerate(('calibration', 'validation')):
        sha = f'{i+1:064x}'
        conditions.append(dict(condition_id='ARTIFICIAL_'+role, role=role, seed=1, decisions=8,
            schedule_artifact=dict(path='ARTIFICIAL_NO_SCHEDULE_'+role,
                file_sha256=sha, semantic_sha256=sha),
            environment_descriptor=dict(schedule_sha256=sha, role=role, profiles=list(PROFILES),
                edge_positions=profile_edges, mobility_start_seconds=1000.*(i+1),
                broadcast_seconds=.1, nominal_delay_seconds=.1, waiting_capacity=32,
                upload_link=asdict(high_link), catalog_source_sha256={'ARTIFICIAL_'+role: sha},
                mobility_source_sha256={f'ARTIFICIAL_MOBILITY_{u}': sha for u in range(30)})))
    candidate = dict(status=STATUS, users=30, profiles=list(PROFILES), decision_seconds=.05,
        broadcast_seconds=.1, waiting_capacity=32, upload_link=NETWORK_UPLOAD_LINK,
        virtual_H_episode_rule='reset_each_independent_episode', feature_scales=feature_scales,
        closure_settings=asdict(EpisodeClosureCandidate()), engineering_conditions=conditions,
        placement_calibration_window_seconds=[0., 500.], placement_seed=123,
        mobility_role_gap_seconds=20.)
    summary = validate_cal_validation_pilot_candidate(candidate)
    assert summary['actual_pilot_or_model_execution_proven'] == 0
    bad_candidate = json.loads(json.dumps(candidate))
    bad_candidate['upload_link'] = asdict(default_link)
    try:
        validate_cal_validation_pilot_candidate(bad_candidate)
    except ValueError:
        pass
    else:
        raise AssertionError('The registered 2 MHz adverse condition cannot silently become this pilot candidate')
    bound_sources = _small_smoke_source_lock()
    assert len(bound_sources) == 209
    protected = __import__('visual_model_smoke').protected_bindings()
    assert protected and all(file_sha(Path(path)) == sha for path, sha in protected.items())

    # Exact external direction record hash/schema/link are checked by the same
    # pure validator the future calibration/validation entry will consume.
    direction = json.loads(NETWORK_DIRECTION_PATH.read_text(encoding='utf-8'))
    direction_sha = file_sha(NETWORK_DIRECTION_PATH)
    _validate_human_direction(direction, str(NETWORK_DIRECTION_PATH), direction_sha, NETWORK_UPLOAD_LINK)
    for mutation in ('sha', 'link', 'field'):
        bad_direction = json.loads(json.dumps(direction))
        bad_sha = direction_sha
        if mutation == 'sha':
            bad_sha = '0'*64
        elif mutation == 'link':
            bad_direction['link_configuration'] = asdict(default_link)
        else:
            bad_direction['unreviewed_extra'] = True
        try:
            _validate_human_direction(bad_direction, str(NETWORK_DIRECTION_PATH), bad_sha, NETWORK_UPLOAD_LINK)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f'Changed human direction {mutation} must be rejected')

    # No-GPU fixture: the downstream receiver accepts one exact eight-decision
    # semantic plan and refuses a changed budget or source-map superset.
    source_fixture, protected_fixture = {'fixture.py': 'a'*64}, {'input.csv': 'b'*64}
    plan = dict(status='prepared_unexecuted_functional_smoke_only', methods=['QLA-MAPPO', 'PPO-MV-A'],
        source_files_sha256=source_fixture, protected_input_and_source_sha256=protected_fixture,
        smoke_settings=SMALL_SMOKE_SETTINGS, formal_protocol_frozen=False, test_access_allowed=False,
        actual_model_forward_and_optimizer_executed=False, actual_roundtrip_performed=False)
    plan_sha, plan_path = 'f'*64, Path('C:/fixture/SMOKE_CANDIDATE_PLAN.json')
    identity = digest(plan)
    receipt_rows = [dict(method=method, source_files_sha256=source_fixture,
        protected_input_and_source_sha256=protected_fixture, plan_sha256=plan_sha,
        semantic_identity_sha256=identity, input_condition={'smoke_settings': SMALL_SMOKE_SETTINGS},
        input_condition_sha256='c'*64, schedule_sha256='d'*64,
        actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
        ledger={'accounting': {'environment_decisions': 8, 'user_behavior_samples': 240}})
        for method in ('QLA-MAPPO', 'PPO-MV-A')]
    smoke = dict(status='actual_cuda_two_arm_smoke_pass', methods=['QLA-MAPPO', 'PPO-MV-A'],
        actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
        source_files_sha256=source_fixture, protected_input_and_source_sha256=protected_fixture,
        smoke_plan_path=str(plan_path.resolve()), plan_sha256=plan_sha,
        semantic_identity_sha256=identity, smoke_settings=SMALL_SMOKE_SETTINGS,
        input_condition_sha256='c'*64, method_reports=receipt_rows)
    _validate_small_smoke_plan_and_result(plan, smoke, plan_path, plan_sha, source_fixture, protected_fixture)
    altered = json.loads(json.dumps(plan))
    altered['smoke_settings']['decisions'] = 16
    try:
        _validate_small_smoke_plan_and_result(altered, smoke, plan_path, plan_sha, source_fixture, protected_fixture)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed smoke decision budget must be rejected')
    superset = dict(source_fixture, **{'unregistered.py': '0'*64})
    superset_plan = dict(plan, source_files_sha256=superset)
    superset_smoke = dict(smoke, source_files_sha256=superset)
    try:
        _validate_small_smoke_plan_and_result(superset_plan, superset_smoke, plan_path,
            plan_sha, source_fixture, protected_fixture)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Arbitrary source-map supersets must be rejected')
    assert 'torch' not in sys.modules
    report = dict(status='non_neural_link_adapter_checks_passed', role='calibration_only',
        same_task_and_payload_ids=True, populated_request_transmission_scales_exactly_with_bandwidth=True,
        empty_request_features_use_the_selected_link=True, actor_shape=[30, 19],
        stale_protected_sources_are_rebound_only_by_the_current_complete_source_lock=True,
        actual_smoke_receiver_checks_exact_plan_semantics_and_8x30_receipts=True,
        exact_human_direction_file_sha_schema_and_link_configuration_checked=True,
        registered_conditions=['2MHz_adverse_preserved', '20MHz_per_user_orthogonal_simulation'],
        unregistered_link_rejected=True, torch_imported=False, model_or_optimizer_calls=0,
        test_role_opened=False, formal_policy_cells=0,
        sources_sha256={str(path): file_sha(path) for path in (
            Path(__file__), BASE/'scripts/visual_upload_link.py', BASE/'scripts/visual_method_features.py',
            BASE/'scripts/visual_protocol_assembly.py')})
    path = Path(__file__).parent/'ADAPTER_CHECKS.json'
    path.write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(), **report),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('VISUAL_20MHZ_ADAPTER_CHECKS_PASS', path)


if __name__ == '__main__':
    main()
