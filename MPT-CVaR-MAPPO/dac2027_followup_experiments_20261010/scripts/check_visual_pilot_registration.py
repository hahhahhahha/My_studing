"""Explicit artificial primitive arrays/metrics and AST only, not runtime."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from check_visual_pilot_episode import fixture, refused
from retained_test_evaluator import _check_reporting_final
from visual_episode_collector import BASE, write_evidence
from visual_pilot_driver import pilot_driver_job_descriptors, validate_pilot_driver_job_coverage
from visual_pilot_registration import (BEHAVIOR, check_pilot_behavior_arrays, pilot_registration_identity,
    RealPairedPilotEvidenceRegistrar)
from visual_temporal_gae import per_user_temporal_gae
from visual_training_transactions import file_sha

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_195042_126370_pilot_driver/RESULT.json'
PRIOR_SHA = 'a3bd6aa7805103d80923b24d3d2c18d018bb4cb128fb4141dc75180bbe319868'
SAVED = BASE/'visual_calibrated_policy/engineering_attempts/20261011_201538_registry_source_interface_precheck'
SAVED_SOURCE_SHA = '71b45557d6cdd38faa883d55bbcc94acfa7b290312db7357efee7a2174b4088d'
FAILED = BASE/'visual_calibrated_policy/engineering_attempts/20261011_202045_628878_pilot_registry'
FAILED_BINDINGS = {
    str(FAILED/'FAILURE.json'): 'c2b578663774f82c46d93e0c4fd1795f228a10de9b67763030756881f4869141',
    str(FAILED/'SOURCE_BEFORE_ARTIFICIAL_ANALYTIC_DTYPE_FIX/visual_pilot_registration.py'): 'a80c370d793bdb82812c184c3c5d00dd130be5f0caf23ded74ba267b805a605f',
    str(FAILED/'SOURCE_BEFORE_ARTIFICIAL_ANALYTIC_DTYPE_FIX/check_visual_pilot_registration.py'): '0cca562f5d4fdcd55cc0d54a9bdd2a96a79b1a5a198b8f17c87d3f65963ced8b',
    str(FAILED/'SOURCE_BEFORE_ARTIFICIAL_ANALYTIC_DTYPE_FIX/README_VISUAL_PILOT_REGISTRATION_中文.md'): '86d3a63239541d6ad6aa1755cba7cfe42c61270aebdc94e8ff3f9d77912b3b50'}


def artificial_arrays(method):
    # Two times/three users/two servers, NOT actual 30-user model outputs.
    arrays = dict(actor_obs=np.zeros((2, 3, 13), dtype=np.float32), critic_obs=np.zeros((2, 3, 54), dtype=np.float32),
        latent_action=np.zeros((2, 3, 2), dtype=np.float32), log_prob=np.zeros((2, 3), dtype=np.float32),
        value=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        reward=np.asarray([[-.2]*3, [-.4]*3]) if method == 'QLA-MAPPO' else np.asarray([[0., 1., 0.], [1., 0., 1.]]),
        user_id=np.tile(np.arange(3), (2, 1)), slot=np.tile(np.arange(1, 3)[:, None], (1, 3)),
        done=np.asarray([[False]*3, [True]*3]), physical_durations=np.full(2, .05))
    next_value = np.zeros_like(arrays['value'])
    next_value[:-1] = arrays['value'][1:]
    arrays['advantages'], arrays['returns'] = per_user_temporal_gae(arrays['reward'], arrays['value'], next_value,
        arrays['done'], arrays['physical_durations'], .99, .95)
    return arrays


def consumers(tree, variable):
    return {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == variable
        and isinstance(node.slice, ast.Constant) and type(node.slice.value) is str}


def fields_with_status(tree, status):
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'dict' and any(kw.arg == 'status' and isinstance(kw.value, ast.Constant)
            and kw.value.value == status for kw in node.keywords)]
    assert len(found) == 1
    return {kw.arg for kw in found[0].keywords}


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_registry')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        protected.update({str(SAVED/'visual_pilot_registration.py'): SAVED_SOURCE_SHA,
            str(SAVED/'STATIC_INTERFACE_DEFECT_中文.md'): file_sha(SAVED/'STATIC_INTERFACE_DEFECT_中文.md')})
        protected.update(FAILED_BINDINGS)
        protected[str(FAILED/'ARTIFICIAL_ANALYTIC_DTYPE_DIAGNOSTIC_中文.md')] = file_sha(FAILED/'ARTIFICIAL_ANALYTIC_DTYPE_DIAGNOSTIC_中文.md')
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No test timing parsing in artificial registration check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            summaries = []
            for method in ('QLA-MAPPO', 'PPO-MV-A'):
                arrays = artificial_arrays(method)
                summary = check_pilot_behavior_arrays(arrays, 3, 2, 2, .05, .99, .95, method)
                assert summary['behavior_samples'] == 6 and summary['actual_neural_behavior_proven_by_pure_arrays'] is False
                gamma, trace = .99**.05, .95**.05
                # Same explicit float64 definition as physical-time GAE;
                # do not accidentally round gamma*float32 before summation.
                value64 = arrays['value'].astype(np.float64)
                last = arrays['reward'][1]-value64[1]
                first = arrays['reward'][0]+gamma*value64[1]-value64[0]+gamma*trace*last
                assert np.allclose(arrays['advantages'], np.stack((first, last)), atol=1e-15, rtol=0)
                assert np.array_equal(arrays['returns'], arrays['advantages']+arrays['value'])
                summaries.append(dict(method=method, explicit_artificial_primitive_arrays_not_neural_outputs=True, **summary))
            base = artificial_arrays('QLA-MAPPO')
            bad = []
            missing = deepcopy(base); missing.pop('value'); bad.append(missing)
            extra = deepcopy(base); extra['future_service'] = np.ones((2, 3)); bad.append(extra)
            for field, value in (('actor_obs', np.zeros((1, 3, 13))), ('done', np.ones((2, 3), dtype=bool)),
                    ('done', base['done'].astype(float)), ('user_id', np.roll(base['user_id'], 1, axis=1)),
                    ('slot', np.zeros((2, 3), dtype=int)), ('physical_durations', np.full(2, .1)),
                    ('critic_obs', np.full((2, 3, 54), np.nan)), ('log_prob', np.full((2, 3), 'ART', dtype=object))):
                row = deepcopy(base); row[field] = value; bad.append(row)
            for field in ('advantages', 'returns', 'reward'):
                row = deepcopy(base); row[field] = row[field].copy(); row[field][0, 1] += .01; bad.append(row)
            for row in bad:
                refused(lambda: check_pilot_behavior_arrays(row, 3, 2, 2, .05, .99, .95, 'QLA-MAPPO'))
            refused(lambda: check_pilot_behavior_arrays(base, True, 2, 2, .05, .99, .95, 'QLA-MAPPO'))
            refused(lambda: check_pilot_behavior_arrays(base, 3, 2, 2, .05, .99, .95, 'PPO-MV-A'))
            cell = dict(generation_decisions=4, generation_seconds=.2, decision_seconds=.05, fixed_end_seconds=30.2, profiles=['ART_A','ART_B'])
            final = dict(tasks=dict(generated=2, accepted=0, completed=0, dropped=2, unfinished=0, inflight_upload=0, latency_seconds=[]),
                uncapped_latency_seconds=[], controller_accounting=dict(admission_acknowledgments=0, settled_completed=0,
                    drain_only_completed=0, settled_dropped=2, drain_only_dropped=0, last_action_settlement_seconds=.25),
                work=[dict(offered=2., inflight=0., dropped=2., admitted=0., executed=0., admitted_remaining=0.) for _ in range(2)])
            zero = _check_reporting_final(final, 2, 4, cell)
            assert zero['no_completed_samples'] is True and zero['mean_seconds'] is None and zero['drops_and_unfinished_retained'] is True
            candidate = fixture()
            jobs = pilot_driver_job_descriptors(candidate)
            assert validate_pilot_driver_job_coverage(jobs, candidate)['actual_closed_episodes_proven'] == 0
            refused(lambda: validate_pilot_driver_job_coverage(jobs[:-1], candidate))
            assert pilot_registration_identity('1'*64) != pilot_registration_identity('2'*64)
            refused(lambda: pilot_registration_identity('NOT_ACTUAL_DRIVER_OR_SHA'))
            refused(lambda: RealPairedPilotEvidenceRegistrar(None, '1'*64))
            source = (BASE/'scripts/visual_pilot_registration.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else '' for node in calls}
            assert not names & {'build_cuda_model','witness','act','collect_once','update_closed_episode','step','train',
                'manual_seed','load_checkpoint','load_state_dict','generate_schedule','require_reviewed_runtime','assemble_once','seal_all'}
            assert any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np'
                and node.func.attr == 'load' and any(kw.arg == 'allow_pickle' and isinstance(kw.value, ast.Constant) and kw.value.value is False
                    for kw in node.keywords) for node in calls)
            episode_tree = ast.parse((BASE/'scripts/visual_pilot_episode.py').read_text(encoding='utf-8'))
            driver_tree = ast.parse((BASE/'scripts/visual_pilot_driver.py').read_text(encoding='utf-8'))
            episode_keys = fields_with_status(episode_tree, 'actual_closed_no_optimizer_pilot_episode_pending_independent_check')
            driver_keys = fields_with_status(driver_tree, 'actual_paired_readonly_closed_episodes_pending_independent_engineering_check')
            assert consumers(tree, 'episode') <= episode_keys and consumers(tree, 'marker') <= driver_keys
            loader = next(node for node in driver_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_own_smoke_model_for_readonly_pilot')
            return_node = next(node for node in ast.walk(loader) if isinstance(node, ast.Return))
            witness_keys = {kw.arg for kw in return_node.value.elts[1].keywords}
            assert consumers(tree, 'witness') <= witness_keys
            transactions = ast.parse((BASE/'scripts/visual_training_transactions.py').read_text(encoding='utf-8'))
            cls = next(node for node in transactions.body if isinstance(node, ast.ClassDef) and node.name == 'ClosedEpisodeCounts')
            count_fields = {node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)}
            assert {'decisions','users','behavior_samples','action_interval_seconds'} <= count_fields and 'physical_seconds' not in count_fields
            saved_source = (SAVED/'visual_pilot_registration.py').read_text(encoding='utf-8')
            assert "('decisions', 'users', 'behavior_samples', 'physical_seconds')" in saved_source
            assert "('decisions', 'users', 'behavior_samples', 'action_interval_seconds')" in source
            for text in ('_actual_smoke_weight_bindings', 'build_closed_ledger', '_check_reporting_final',
                    '_check_actual_execution_records', 'measured_pair_seconds', 'payload_bytes', 'oracle.advance(end)',
                    'per_user_temporal_gae', 'ACTUAL_READONLY_RNG.npz', 'pre_critic_sha256',
                    'no_adverse_or_zero_completion_outcomes_filtered=True', 'full_pilot_completed=False',
                    'metadata_generation_training_test_or_figure_permission=False', 'current_descriptor_and_stage', 'checked_summaries_before_failure'):
                assert text in source
            assert 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_registration.py',
                BASE/'README_VISUAL_PILOT_REGISTRATION_中文.md')}
            write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(),
                artificial_primitive_behavior_GAE_and_static_registration_checks_passed=True,
                artificial_primitive_array_cases=summaries, artificial_bad_array_cases_rejected=len(bad)+2,
                artificial_zero_completion_and_all_dropped_diagnostics_not_filtered=True,
                artificial_candidate_descriptors=4, actual_registered_episodes_proven=0,
                static_same_live_driver_full_pair_original_own_load_outputs_ledger_GAE_RNG_FIFO_pipeline_checked=True,
                static_episode_driver_load_witness_and_counts_producer_consumer_fields_checked=True,
                initial_new_source_counts_field_mismatch_and_source_snapshot_preserved=True,
                artificial_analytic_dtype_failure_and_three_sources_preserved_original_tolerance_not_relaxed=True,
                missing_actual_driver_early_type_rejection_exercised=True,
                actual_registrar_constructor_register_role_catalog_NPZ_FIFO_or_serialization_calls=0,
                actual_permission_driver_model_path_raw_archive_or_runtime_guards_exercised=False,
                actual_user_smoke_MODEL_checkpoint_completed_driver_episode_or_PASS_fixtures_fabricated=False,
                actual_test_timing_CSV_opens=timing_opens, torch_imported=False,
                source_only_not_launch_authority=True, independent_audit_or_pilot_complete_or_frozen=False,
                original_protected_files_sha256=protected, new_implementation_sha256=sources))
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_PRIMITIVE_ARRAYS_STATIC_PILOT_REGISTRATION_ONLY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_registrar_driver_catalog_model_NPZ_FIFO_or_optimizer_calls=0, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
