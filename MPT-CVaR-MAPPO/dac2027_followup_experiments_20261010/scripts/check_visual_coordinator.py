"""New compact-H assembly checks with EXPLICIT ARTIFICIAL behavior only."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility, TrainingWindow
from visual_episode_collector import EpisodeClosureCandidate, jsonable, write_evidence
from visual_learning_buffer import map_historical_action_output
from visual_method_features import calibration_scales
from visual_mv_stream import PersistentMVStream
from visual_receipt_environment import VisualReceiptEnvironment
from visual_request_schedule import generate_schedule
from visual_training_coordinator import (CompactCoordinatorAssembly, closed_episode_counts,
    environment_binding, FrozenRuntimeArtifactBindings, require_reviewed_runtime,
    RealMultiepisodeTrainingCoordinator, RealReadonlyValidationEvaluator)
from visual_training_transactions import file_sha, digest

BASE = Path(__file__).resolve().parents[1]
PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_104326_421558_training_transactions/RESULT.json'
PRIOR_SHA = '068978b44780097dbf3fe1ccd1466245bd3d877be3140cb91c365d86b36ef765'
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Integrated runtime guard did not reject invalid entry')


def compact_boundary_check(attempt):
    catalog, mobility = ExecutionServiceCatalog('calibration'), PhysicalMobility(users=30)
    window = TrainingWindow(0., 500.)
    edges = mobility.server_positions(window, 123, 4)
    scales = calibration_scales(catalog, mobility, window, edges, PROFILES)
    summaries = []
    for method in ('QLA-MAPPO', 'PPO-MV-A'):
        stream = PersistentMVStream(30, purpose='training') if method == 'PPO-MV-A' else None
        # Explicit artificial boundary-H fixture, not trained/selected state.
        h = .1
        for index in (1, 2):
            schedule = generate_schedule(catalog, 30, 8, .05, 200+index, 30.)
            env = VisualReceiptEnvironment(schedule, catalog, mobility, edges, PROFILES, float(index-1))
            original_binding = digest(jsonable(environment_binding(env)))
            changed_config = jsonable(environment_binding(env))
            changed_config['nominal_delay_seconds'] += .1
            assert digest(changed_config) != original_binding  # descriptor-only counterexample, not executed
            assembly = CompactCoordinatorAssembly(method, 30, 4, 8, scales, EpisodeClosureCandidate(),
                stream, index, f'artificial-calibration-{method}-{index}', initial_H=h)
            original_histories = []
            for _ in range(8):
                pre = env.observe()
                original_histories.append(jsonable([asdict(row) for row in pre.admission_acknowledgments]))
                packed = assembly.features_for(pre)
                latent = np.tile([4., -2., -2., -2.], (30, 1))
                ports = np.exp(latent-latent.max(axis=1, keepdims=True))
                ports /= ports.sum(axis=1, keepdims=True)
                logp, values = np.full(30, -2.), np.arange(30)*.01+pre.index*.001
                sample = (map_historical_action_output(method, (ports, latent, logp), 30, 4, values)
                    if method == 'QLA-MAPPO' else map_historical_action_output(method, (latent, logp, values, ports), 30, 4))
                assembly.execute_validated_sample(env, pre, packed, sample)
            result = assembly.close_after_fixed_drain(env)
            assert result['virtual_H_before'] == h
            assert result['ordinary_transitions'][0]['reward']['h_before'] == h
            h = result['virtual_H_after']
            assert np.isfinite(h) and h >= 0
            for saved, expected in zip(result['observations'], original_histories):
                n = saved['admission_history_archive_prefix_length']
                assert jsonable(result['ACK_ARCHIVE'][:n]) == expected
                assert 'admission_acknowledgments' not in saved
            assert len({row['task_id'] for row in result['ACK_ARCHIVE']}) == len(result['ACK_ARCHIVE'])
            prepared = result['prepared_episode']
            checked = closed_episode_counts(prepared, 30, 8, .05)
            assert checked.behavior_samples == 240 and checked.decisions == 8
            refused(lambda: closed_episode_counts(prepared, 30, 9, .05))
            refused(lambda: closed_episode_counts(prepared, 29, 8, .05))
            refused(lambda: closed_episode_counts(prepared, 30, 8, .1))
            refused(lambda: closed_episode_counts(replace(prepared, physical_durations=np.full(8, .1)), 30, 8, .05))
            assert digest(jsonable(environment_binding(env))) == original_binding
            if stream is not None:
                assert stream.state().count == index*240
            summary = dict(method=method, episode=index, counts=asdict(checked),
                generated=result['ledger'].generated, completed=result['ledger'].completed,
                dropped=result['ledger'].dropped, unfinished=result['ledger'].unfinished,
                artificial_H_before=result['virtual_H_before'], artificial_H_after=h,
                ordinary_cumulative_ack_entries=sum(map(len, original_histories)),
                exact_unique_retained_ack_entries=len(result['ACK_ARCHIVE']),
                observation_history_reconstructed_exactly=True, actual_model_calls=0,
                actions_logp_values_and_initial_H_are_explicit_artificial=True)
            summaries.append(summary)
            write_evidence(attempt/f'{method}_{index}_EXPLICIT_ARTIFICIAL_TRACE.json',
                {key: value for key, value in result.items() if key != 'prepared_episode'} | summary)
    assert summaries[1]['artificial_H_before'] == summaries[0]['artificial_H_after']
    assert summaries[3]['artificial_H_before'] == summaries[2]['artificial_H_after']
    assert summaries[0]['ordinary_cumulative_ack_entries'] > summaries[0]['exact_unique_retained_ack_entries'] > 0
    return dict(cases=summaries, source_sha256=catalog.source_sha256 | mobility.source_hashes,
        full_input_history_unchanged_and_exact_archive_prefix_reconstruction=True,
        common_shapes_and_physical_time_checks_passed=True,
        initial_H_point_one_is_literal_fixture_not_training_state=True)


def static_and_rejection_checks(attempt):
    refused(lambda: require_reviewed_runtime(object()))
    nonexistent = FrozenRuntimeArtifactBindings(str(BASE/'protocol/NO_REGISTERED_FROZEN_VISUAL_PROTOCOL.json'),
        '0'*64, str(BASE.parent/'.aris/traces/NO_COMPLETE_VISUAL_ENTRY_REVIEW.json'), '0'*64)
    refused(lambda: require_reviewed_runtime(nonexistent))
    # Source constructors stop at missing reviewed artifacts BEFORE any model.
    refused(lambda: RealMultiepisodeTrainingCoordinator(object(), None, None, None, nonexistent))
    refused(lambda: RealReadonlyValidationEvaluator(object(), None, None, nonexistent, None, {}, '0'*64, '0'*64))
    # Explicit NON-CERTIFICATE fixture outside protocol/review roots must fail.
    fixture = attempt/'EXPLICIT_NONCERTIFICATE.json'
    write_evidence(fixture, dict(scope='EXPLICIT_SYNTHETIC_NONCERTIFICATE', status='NOT_A_PASS_RECORD'))
    outside = FrozenRuntimeArtifactBindings(str(fixture), file_sha(fixture), str(fixture), file_sha(fixture))
    refused(lambda: require_reviewed_runtime(outside))
    tree = ast.parse((BASE/'scripts/visual_training_coordinator.py').read_text(encoding='utf-8'))
    train = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RealMultiepisodeTrainingCoordinator')
    run = next(node for node in train.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    text = ast.unparse(run)
    assert text.index('close_collection(') < text.index('begin_update(') < text.index('update_closed_episode(') < text.index('commit_update(') < text.index('write_actual_controller_checkpoint(')
    act = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'act']
    assert len(act) == 1 and ast.unparse(act[0].args[0]) == 'packed'
    validation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RealReadonlyValidationEvaluator')
    evaluate = next(node for node in validation.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    calls = [node for node in ast.walk(evaluate) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in ('update', 'update_closed_episode', 'step', 'load_checkpoint') for node in calls)
    assert not any(isinstance(node.func, ast.Name) and node.func.id in ('PersistentMVStream', 'build_cuda_model') for node in calls)
    return dict(external_review_pilot_smoke_registration_required=True,
        missing_or_out_of_root_noncertificate_before_model_rejections_passed=True,
        source_order_collect_close_burn_update_commit_export_checked=True,
        validation_source_has_no_optimizer_or_train_state_restore_or_MV_construction=True,
        actual_train_or_validation_constructor_succeeded=False)


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_coordinator')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = prior['protected_input_and_source_sha256'] | prior['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
        integration = compact_boundary_check(attempt)
        guards = static_and_rejection_checks(attempt)
        protected |= integration['source_sha256']
        assert all(file_sha(Path(path)) == value for path, value in protected.items()) and 'torch' not in sys.modules
        sources = [Path(__file__), BASE/'scripts/visual_training_coordinator.py', BASE/'README_VISUAL_COORDINATOR_中文.md']
        write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(), passed=True,
            runtime_python=sys.version.split()[0], numpy=np.__version__, torch_imported=False,
            compact_H_artificial_assembly=integration, source_layout_and_runtime_refusals=guards,
            protected_input_and_source_sha256=protected,
            new_implementation_sha256={str(path): file_sha(path) for path in sources},
            actual_model_or_optimizer_or_real_checkpoint_calls=0, actual_training_coordinator_executed=False,
            actual_readonly_policy_validation_executed=False, independent_complete_entry_review=False,
            prior_success_and_source_snapshot_preserved='visual_calibrated_policy/engineering_attempts/20261011_111536_798293_coordinator',
            final_checkpoint_selection_implemented=False, formal_protocol_frozen=False,
            network_parameters_changed=False, test_measurements_accessed=False,
            policy_training_cells=0, formal_policy_test_cells=0))
        print('COORDINATOR_SOURCE_AND_ARTIFICIAL_ASSEMBLY_OK_NOT_MODEL_EXECUTION', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), at=datetime.now().astimezone().isoformat(),
            actual_model_or_optimizer_or_real_checkpoint_calls=0, policy_training_cells=0, formal_policy_test_cells=0))
        raise


if __name__ == '__main__':
    main()
