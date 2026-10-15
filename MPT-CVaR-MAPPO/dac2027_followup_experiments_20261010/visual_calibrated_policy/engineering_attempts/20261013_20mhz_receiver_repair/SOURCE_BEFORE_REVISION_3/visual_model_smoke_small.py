"""Prepared separate-process two-method CUDA smoke and safe readonly check.

Today only `check` may run while the original GPU gate fails. No implicit
fallback, automatic retry, formal training, final test or fake PASS record.
"""
from __future__ import annotations
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]/'scripts'))
from windows_runtime import initialize
initialize()

import argparse
import ast
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable, write_evidence
from visual_learned_bindings import source_bindings, VisualPolicyDimensions, SOURCE
from visual_training_transactions import (file_sha, digest, CommonBudgetCandidate,
    TrainingTransactionLedger, inspect_checkpoint_archive, require_sha)
from visual_upload_link import REGISTERED_20MHZ_ORTHOGONAL_LINK, UploadLinkConfig

METHODS = ('QLA-MAPPO', 'PPO-MV-A')
NETWORK_CONDITION_ID = '20MHz_per_user_orthogonal_simulation'
NETWORK_UPLOAD_LINK = asdict(REGISTERED_20MHZ_ORTHOGONAL_LINK)
SMOKE_UPLOAD_LINK = UploadLinkConfig(**NETWORK_UPLOAD_LINK)
PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_111741_173552_coordinator/RESULT.json'
PRIOR_SHA = '4ed5a34d5b55b8465813a162c87e947eef27d73041e3b42706998cbe65359b5a'
SMOKE_SETTINGS = dict(users=30, decisions=8, decision_seconds=.05, fixed_drain_seconds=30.,
    total_requests_per_second=30., nominal_delay_seconds=.1, broadcast_seconds=.1, waiting_capacity=32,
    mobility_start_seconds=0., training_geometry_window=[0., 500.], geometry_seed=123,
    schedule_seed=209, model_seed=1, profiles=['CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'],
    upload_link=NETWORK_UPLOAD_LINK, network_condition_id=NETWORK_CONDITION_ID,
    initial_H=0., purpose='Low-load 20 MHz per-user orthogonal calibration-only functional model smoke, not final scenario or budget')
INDEPENDENT_REVIEW_DIR = BASE/'visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/independent_repair_rereview'


def source_lock():
    # Bind followup scripts AND the historical local Python import tree,
    # rather than claiming that only three selected model files are its
    # entire import closure. Installed packages remain environment-bound.
    paths = sorted((BASE/'scripts').glob('*.py')) + sorted(SOURCE.rglob('*.py'))
    additive = tuple(Path(__file__).parent/name for name in
        ('visual_model_smoke_small.py', 'reward_smoke_small.py', 'reward_input_integrity.py',
         'FROZEN_BASELINE.json', 'README_SMALL_SMOKE_中文.md', 'USER_REQUEST.json'))
    return {str(path): file_sha(path) for path in paths} | source_bindings() | {str(path): file_sha(path) for path in additive}


def protected_bindings():
    if file_sha(PRIOR) != PRIOR_SHA:
        raise RuntimeError('Pinned preparation evidence changed; no replacement or reseal')
    old = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (type(old.get('protected_input_and_source_sha256')) is not dict
            or type(old.get('new_implementation_sha256')) is not dict):
        raise RuntimeError('Exact prior preparation input/source maps required')
    current_sources = source_lock()
    historical = (old['protected_input_and_source_sha256'] | old['new_implementation_sha256']
        | {str(PRIOR): PRIOR_SHA})
    protected = {}
    for path, value in historical.items():
        require_sha(value)
        # Current source bytes are already bound by the complete source lock;
        # do not retain a second stale implementation hash as a data binding.
        if path in current_sources:
            continue
        if path in old['new_implementation_sha256'] and Path(path).suffix.lower() == '.py':
            raise RuntimeError('Historical implementation closure is not fully covered by the current source lock')
        if file_sha(Path(path)) != value:
            raise RuntimeError('Protected preparation/raw input bytes changed; stop before resources/models')
        protected[path] = value
    if str(PRIOR) not in protected:
        raise RuntimeError('Pinned prior preparation evidence must remain a protected non-source input')
    return protected


def expected_plan():
    return dict(status='prepared_unexecuted_functional_smoke_only', methods=list(METHODS),
        source_files_sha256=source_lock(), protected_input_and_source_sha256=protected_bindings(),
        smoke_settings=SMOKE_SETTINGS, formal_protocol_frozen=False, test_access_allowed=False,
        actual_model_forward_and_optimizer_executed=False, actual_roundtrip_performed=False)


def semantic_plan_identity(plan):
    expected = expected_plan()
    # Exact complete schema and canonical values: extra fields and altered
    # numeric/boolean encodings are not a newly registered experiment.
    if digest(plan) != digest(expected):
        raise RuntimeError('Exact complete prepared source/input/settings schema required')
    return digest(expected)


def assert_runtime_bindings(plan):
    identity = semantic_plan_identity(plan)
    # The above checks raw/prior evidence as well as local sources each time.
    return identity


def assert_consumed_input_maps(plan, calibration_sources, mobility_sources):
    """Pin consumed membership, not only the bytes of previously known files.

    Exact expected sets come from the original pinned preparation map, never
    today's directory listing or the two arms' mutual match. No input is added
    to the reviewed set and no labels are fitted here.
    """
    from execution_service_catalog import DIRECTORIES
    from physical_mobility import KAIST
    protected = plan['protected_input_and_source_sha256']
    expected_calibration = {path: value for path, value in protected.items()
        if Path(path).parent == DIRECTORIES['calibration'] and Path(path).suffix == '.csv'}
    expected_mobility = {path: value for path, value in protected.items()
        if Path(path).parent == KAIST and Path(path).suffix == '.txt'}
    if not expected_calibration or len(expected_mobility) != SMOKE_SETTINGS['users']:
        raise RuntimeError('Original calibration pool and exact 30 mobility inputs must remain registered')
    if calibration_sources != expected_calibration or mobility_sources != expected_mobility:
        raise RuntimeError('Consumed calibration/mobility path and hash sets differ from registered inputs')
    return dict(calibration_source_sha256=expected_calibration, mobility_source_sha256=expected_mobility)


def server_positions_record(edges):
    """JSON-safe copy of the producer's tuple-of-tuples, without resampling."""
    return [list(point) for point in edges]


def read_plan(path, expected_sha):
    path = Path(path)
    root = (BASE/'visual_calibrated_policy/engineering_attempts').resolve()
    if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
        raise ValueError('Exact new check-generated smoke plan inside its intended output required')
    if file_sha(path) != require_sha(expected_sha):
        raise RuntimeError('Smoke candidate plan bytes changed')
    plan = json.loads(path.read_text(encoding='utf-8'))
    assert_runtime_bindings(plan)
    return plan


def check():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_model_smoke_entry')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        protected = protected_bindings()
        locked = source_lock()
        for name in ('visual_model_smoke.py', 'visual_checkpoint_load.py'):
            ast.parse((BASE/'scripts'/name).read_text(encoding='utf-8'))
        loader = ast.parse((BASE/'scripts/visual_checkpoint_load.py').read_text(encoding='utf-8'))
        fun = next(row for row in loader.body if isinstance(row, ast.FunctionDef))
        calls = [row for row in ast.walk(fun) if isinstance(row, ast.Call)]
        assert not any(isinstance(row.func, ast.Name) and row.func.id in ('build_cuda_model', 'witness') for row in calls)
        assert not any(isinstance(row.func, ast.Attribute) and row.func.attr in ('step', 'update', 'update_closed_episode', 'load_optimizer') for row in calls)
        from visual_checkpoint_load import load_own_snapshot_in_verified_cuda_process
        try:
            load_own_snapshot_in_verified_cuda_process(object(), None, {}, '0'*64, '0'*64, 1)
        except TypeError:
            pass
        else:
            raise AssertionError('Wrong runtime object reached loading')
        assert 'torch' not in sys.modules
        plan = expected_plan()
        identity = semantic_plan_identity(plan)
        assert semantic_plan_identity(json.loads(json.dumps(plan, indent=7))) == identity
        assert semantic_plan_identity(dict(reversed(list(plan.items())))) == identity
        try:
            semantic_plan_identity(dict(plan, ignored_extra_field='not a new attempt'))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Extra fields could alter one-shot identity')
        from unittest.mock import patch
        module = sys.modules[__name__]
        true_sha = file_sha
        witness_path = next(path for path in protected if path.endswith('.csv'))
        def simulated_changed_input(path):
            return '0'*64 if str(path) == witness_path else true_sha(path)
        with patch.object(module, 'file_sha', side_effect=simulated_changed_input):
            try:
                assert_runtime_bindings(plan)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Changed raw bytes did not stop launch validation')
        write_evidence(attempt/'SMOKE_CANDIDATE_PLAN.json', plan)
        report = dict(at=datetime.now().astimezone().isoformat(), source_entry_check_passed=True,
            actual_cuda_smoke_pass=False, torch_imported=False, actual_model_or_optimizer_or_roundtrip_calls=0,
            protected_input_and_source_sha256=protected, source_files_sha256=locked,
            smoke_plan_path=str(attempt/'SMOKE_CANDIDATE_PLAN.json'),
            smoke_plan_sha256=file_sha(attempt/'SMOKE_CANDIDATE_PLAN.json'),
            smoke_semantic_identity_sha256=identity,
            protected_raw_and_prior_evidence_revalidated_at_launch_and_completion_source_prepared=True,
            equivalent_JSON_encodings_share_one_shot_identity_checked=True,
            extra_plan_fields_rejected_checked=True,
            changed_raw_hash_rejection_uses_explicit_mock_without_editing_inputs=True,
            historical_local_python_import_tree_locked=True,
            deterministic_roundtrip_arrays_and_rng_witness_export_source_prepared=True,
            loader_requires_bound_actual_controller_and_no_optimizer_restore=True,
            factory_free_GPU_check_not_repeated_for_same_process_clone=True,
            separate_process_per_method_source_prepared=True, automatic_retry_allowed=False,
            policy_training_cells=0, formal_policy_test_cells=0,
            independent_complete_entry_review=False)
        write_evidence(attempt/'RESULT.json', report)
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
        assert_runtime_bindings(plan)
        print('MODEL_SMOKE_ENTRY_PREPARED_ONLY_NOT_EXECUTED', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_model_or_optimizer_or_roundtrip_calls=0))
        raise


def worker(method, plan_path, plan_sha, new_directory):
    from reward_input_integrity import require_review
    require_review(INDEPENDENT_REVIEW_DIR)
    """Actual method worker SOURCE ONLY: never invoked in the blocked wake."""
    plan = read_plan(plan_path, plan_sha)
    identity = semantic_plan_identity(plan)
    if method not in METHODS:
        raise ValueError('Only the declared two real methods')
    from qla_environment import guard
    spec = guard()
    if Path(sys.executable).resolve() != (Path(spec['isolated_environment'])/'Scripts/python.exe').resolve():
        raise RuntimeError('Declared isolated environment required')
    path = Path(new_directory).resolve()
    root = (BASE/'visual_calibrated_policy').resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('New worker evidence must remain inside visual outputs')
    lock_path = BASE/'visual_calibrated_policy/smoke_locks'/f'{identity}.json'
    parent = json.loads(lock_path.read_text(encoding='utf-8'))
    if (parent['plan_sha256'] != plan_sha or parent['semantic_identity_sha256'] != identity
            or Path(parent['output']).resolve() != path.parent
            or path.name != method):
        raise RuntimeError('Private worker requires its live recorded parent; no direct/replacement worker invocation')
    from reward_input_integrity import require_worker_parent
    canonical_worker_command = [str(Path(spec['isolated_environment'])/'Scripts/python.exe'), '-B', '-X', 'utf8',
        str(Path(__file__)), '_worker', '--method', method, '--plan', str(plan_path), '--plan-sha', plan_sha, '--output', str(path)]
    parent_receipt = require_worker_parent(parent, canonical_worker_command)
    path.mkdir(parents=True, exist_ok=False)
    controller, stream, assembly, ledger = None, None, None, None
    try:
        write_evidence(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
            method=method, plan_sha256=plan_sha, semantic_identity_sha256=identity,
            parent_chain=parent_receipt,
            source_files_sha256=plan['source_files_sha256'],
            protected_input_and_source_sha256=plan['protected_input_and_source_sha256']))
        from execution_service_catalog import ExecutionServiceCatalog
        from physical_mobility import PhysicalMobility, TrainingWindow
        from visual_method_features import calibration_scales
        from visual_receipt_environment import VisualReceiptEnvironment
        from visual_request_schedule import generate_schedule
        from visual_neural_controller import NeuralVisualController
        from visual_learned_bindings import build_cuda_model
        from visual_mv_stream import PersistentMVStream
        from visual_training_coordinator import CompactCoordinatorAssembly, closed_episode_counts, actual_parameter_sha256
        from visual_training_transactions import write_actual_controller_checkpoint
        from visual_checkpoint_load import load_own_snapshot_in_verified_cuda_process
        catalog, mobility = ExecutionServiceCatalog('calibration'), PhysicalMobility(users=30)
        assert_consumed_input_maps(plan, catalog.source_sha256, mobility.source_hashes)
        window = TrainingWindow(0., 500.)
        profiles = tuple(SMOKE_SETTINGS['profiles'])
        edges = mobility.server_positions(window, 123, 4)
        scales = calibration_scales(catalog, mobility, window, edges, profiles,
                                    upload_link=SMOKE_UPLOAD_LINK)
        decisions = SMOKE_SETTINGS['decisions']
        schedule = generate_schedule(catalog, 30, decisions, .05, 209, 30.)
        input_condition = dict(calibration_source_sha256=catalog.source_sha256,
            mobility_source_sha256=mobility.source_hashes, schedule_sha256=schedule.fingerprint(),
            server_positions=server_positions_record(edges), calibration_scales=asdict(scales), smoke_settings=SMOKE_SETTINGS)
        env = VisualReceiptEnvironment(schedule, catalog, mobility, edges, profiles, 0.,
                                       link=SMOKE_UPLOAD_LINK)
        settings = EpisodeClosureCandidate()
        dimensions = VisualPolicyDimensions(30, 4, 19, 705, 30)
        # This is the ONE initial free-GPU/kernel gate before this worker
        # creates CUDA models. Clone loading later reuses this own context.
        assert_runtime_bindings(plan)  # including raw bytes, immediately before resources/models
        assert_consumed_input_maps(plan, catalog.source_sha256, mobility.source_hashes)
        model = build_cuda_model(method, dimensions, {}, 1)
        controller = NeuralVisualController(method, model)
        budget = CommonBudgetCandidate((decisions,))
        ledger = TrainingTransactionLedger(method, 1, budget, (schedule.fingerprint(),), digest(SMOKE_SETTINGS))
        ledger.begin_episode(1, schedule.fingerprint(), 'calibration')
        stream = PersistentMVStream(30, purpose='training') if method == 'PPO-MV-A' else None
        assembly = CompactCoordinatorAssembly(method, 30, 4, decisions, scales, settings, stream, 1, 'functional-smoke-only', initial_H=0.)
        before_parameters = actual_parameter_sha256(controller)
        first_packed = None
        for _ in range(decisions):
            pre = env.observe()
            packed = assembly.features_for(pre)
            if first_packed is None:
                first_packed = packed
            sample = controller.act(packed, deterministic=False)
            assembly.execute_validated_sample(env, pre, packed, sample)
        result = assembly.close_after_fixed_drain(env)
        prepared = result['prepared_episode']
        if actual_parameter_sha256(controller) != before_parameters:
            raise RuntimeError('Model changed during behavior collection before optimizer')
        write_evidence(path/'CALIBRATION_SMOKE_CLOSED_EPISODE.json', {key: value for key, value in result.items() if key != 'prepared_episode'})
        np.savez_compressed(path/'ACTUAL_BEHAVIOR_AND_GAE.npz', **prepared.time_user_fields,
            advantages=prepared.advantages, returns=prepared.returns, physical_durations=prepared.physical_durations)
        ledger.close_collection(closed_episode_counts(prepared, 30, decisions, .05))
        ledger.begin_update(prepared.episode_token)
        stats = controller.update_closed_episode(prepared)
        ledger.commit_update(prepared.episode_token, stats)
        after_parameters = actual_parameter_sha256(controller)
        if after_parameters == before_parameters:
            raise RuntimeError('No parameter update witnessed; retain the numerical outcome, never fake a smoke pass')
        descriptor = ledger.checkpoint_descriptor()
        archive = write_actual_controller_checkpoint(controller, ledger, path/'checkpoint', stream)
        loaded, load_report = load_own_snapshot_in_verified_cuda_process(controller, path/'checkpoint', descriptor,
            archive['artifact_sha256'], after_parameters, 1001)
        # Fixed saved PRE-action observation, no future data or invented value.
        action_inputs = first_packed  # actual immutable first PRE-action bundle, never invented diagnostic inputs
        original_action = controller.act(action_inputs, deterministic=True)
        loaded_action = loaded.act(action_inputs, deterministic=True)
        witness_arrays = {f'input_{name}': getattr(action_inputs, name) for name in
            ('actor', 'critic', 'q_pressure', 'expected_latency_seconds', 'reference_demand_seconds', 'previous_portfolios')}
        for label, sample in (('saved_model', original_action), ('loaded_model', loaded_action)):
            witness_arrays.update({f'{label}_{name}': getattr(sample, name) for name in
                ('portfolios', 'latent_actions', 'log_probabilities', 'values')})
        np.savez_compressed(path/'ACTUAL_ROUNDTRIP_INPUT_OUTPUT.npz', **witness_arrays)
        if not all(np.array_equal(getattr(original_action, name), getattr(loaded_action, name))
                   for name in ('portfolios', 'latent_actions', 'log_probabilities', 'values')):
            raise RuntimeError('Loaded model deterministic outputs do not exactly roundtrip')
        torch = controller._torch
        rng_before = [torch.get_rng_state().clone(), *[item.clone() for item in torch.cuda.get_rng_state_all()]]
        loaded.act(action_inputs, deterministic=True)
        rng_after = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
        np.savez_compressed(path/'ACTUAL_READONLY_TORCH_RNG.npz',
            **{f'before_{i}': value.cpu().numpy() for i, value in enumerate(rng_before)},
            **{f'after_{i}': value.cpu().numpy() for i, value in enumerate(rng_after)})
        if len(rng_before) != len(rng_after) or not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
            raise RuntimeError('Deterministic loaded act changed Torch RNG')
        if actual_parameter_sha256(loaded) != after_parameters or actual_parameter_sha256(controller) != after_parameters:
            raise RuntimeError('Loaded model changed during readonly roundtrip')
        inspect_checkpoint_archive(path/'checkpoint', descriptor, archive['artifact_sha256'])
        assert_runtime_bindings(plan)
        assert_consumed_input_maps(plan, catalog.source_sha256, mobility.source_hashes)
        write_evidence(path/'METHOD_SMOKE_PASS.json', dict(status='actual_cuda_method_smoke_pass', method=method,
            source_files_sha256=plan['source_files_sha256'], schedule_sha256=schedule.fingerprint(),
            protected_input_and_source_sha256=plan['protected_input_and_source_sha256'],
            input_condition=input_condition, input_condition_sha256=digest(input_condition),
            semantic_identity_sha256=identity, plan_sha256=plan_sha,
            actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
            exact_loaded_parameters_and_deterministic_outputs=True, deterministic_Torch_RNG_unchanged=True,
            before_parameter_sha256=before_parameters, after_parameter_sha256=after_parameters,
            update_statistics=stats, counts=asdict(closed_episode_counts(prepared, 30, decisions, .05)),
            checkpoint_archive=archive, loader=load_report, calibration_source_sha256=catalog.source_sha256,
            roundtrip_witness_files_sha256={name: file_sha(path/name) for name in
                ('ACTUAL_ROUNDTRIP_INPUT_OUTPUT.npz', 'ACTUAL_READONLY_TORCH_RNG.npz')},
            mobility_source_sha256=mobility.source_hashes, ledger=ledger.snapshot(),
            policy_training_cells=0, formal_policy_test_cells=0, scope='Functional smoke only, not performance or frozen formal scenario'))
    except Exception as error:
        if controller is not None:
            controller._failed = True
        if stream is not None:
            stream.mark_failed(error)
        if ledger is not None:
            ledger.mark_failed(error)
        write_evidence(path/'FAILURE.json', dict(error=repr(error), automatic_retry_allowed=False,
            partial_behavior_rows=assembly.buffer._rows if assembly else [],
            partial_transaction_ledger=ledger.snapshot() if ledger else None,
            policy_training_cells=0, formal_policy_test_cells=0, actual_smoke_pass=False))
        raise


def run(plan_path, plan_sha, new_directory):
    """Prepared parent, never run while GPU occupied; one-shot no append."""
    from reward_input_integrity import require_review
    require_review(INDEPENDENT_REVIEW_DIR)
    plan = read_plan(plan_path, plan_sha)
    identity = semantic_plan_identity(plan)
    from qla_environment import guard
    spec = guard()
    path = Path(new_directory).resolve()
    root = (BASE/'visual_calibrated_policy').resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('New parent run must remain inside visual outputs')
    # Permanent canonical semantic lock, independent of JSON whitespace,
    # key order and output name. Byte SHA remains a separate artifact binding.
    lock_root = BASE/'visual_calibrated_policy/smoke_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root/f'{identity}.json'
    from reward_input_integrity import parent_creation_time, validate_worker_parent_receipt
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(plan_sha256=plan_sha, semantic_identity_sha256=identity, output=str(path), parent_pid=os.getpid(),
                       parent_creation_time=parent_creation_time(),
                       at=datetime.now().astimezone().isoformat()), stream)
    path.mkdir(parents=True, exist_ok=False)
    try:
        write_evidence(path/'STARTED.json', dict(parent_pid=os.getpid(), plan_path=str(plan_path),
            plan_sha256=plan_sha, semantic_identity_sha256=identity,
            at=datetime.now().astimezone().isoformat()))
        reports = []
        python = Path(spec['isolated_environment'])/'Scripts/python.exe'
        for method in METHODS:
            assert_runtime_bindings(plan)
            child = path/method
            command = [str(python), '-B', '-X', 'utf8', str(Path(__file__)), '_worker',
                '--method', method, '--plan', str(plan_path), '--plan-sha', plan_sha, '--output', str(child)]
            with (path/f'{method}_stdout.log').open('xb') as out, (path/f'{method}_stderr.log').open('xb') as err:
                process = subprocess.Popen(command, stdout=out, stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                write_evidence(path/f'{method}_WORKER_STARTED.json', dict(pid=process.pid, command=command,
                    at=datetime.now().astimezone().isoformat()))
                write_evidence(path/'STATE.json', dict(phase='running_method_worker', method=method,
                    parent_pid=os.getpid(), worker_pid=process.pid, automatic_retry_allowed=False))
                if process.wait() != 0:
                    raise RuntimeError(f'{method} worker failed; no second arm, replacement or retry')
            validate_worker_parent_receipt(json.loads((child/'STARTED.json').read_text(encoding='utf-8'))['parent_chain'], process.pid,
                json.loads(lock.read_text(encoding='utf-8')))
            reports.append(json.loads((child/'METHOD_SMOKE_PASS.json').read_text(encoding='utf-8')))
        if reports[0]['schedule_sha256'] != reports[1]['schedule_sha256']:
            raise RuntimeError('Two arms did not use the same actual calibration schedule')
        if (reports[0]['input_condition'] != reports[1]['input_condition']
                or reports[0]['input_condition_sha256'] != reports[1]['input_condition_sha256']):
            raise RuntimeError('Two arms did not use exactly the same bound raw/physical condition')
        for report in reports:
            assert_consumed_input_maps(plan, report['calibration_source_sha256'], report['mobility_source_sha256'])
            assert_consumed_input_maps(plan, report['input_condition']['calibration_source_sha256'],
                report['input_condition']['mobility_source_sha256'])
            if (report['protected_input_and_source_sha256'] != plan['protected_input_and_source_sha256']
                    or report['semantic_identity_sha256'] != identity):
                raise RuntimeError('Worker result lost the exact reviewed input/one-shot identity')
        assert_runtime_bindings(plan)
        write_evidence(path/'ACTUAL_TWO_ARM_SMOKE_PASS.json', dict(status='actual_cuda_two_arm_smoke_pass',
            methods=list(METHODS), source_files_sha256=plan['source_files_sha256'],
            protected_input_and_source_sha256=plan['protected_input_and_source_sha256'],
            input_condition_sha256=reports[0]['input_condition_sha256'], semantic_identity_sha256=identity,
            actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
            method_reports=reports, smoke_plan_path=str(Path(plan_path).resolve()), plan_sha256=plan_sha,
            smoke_settings=SMOKE_SETTINGS,
            formal_protocol_frozen=False,
            independent_complete_entry_review=False, policy_training_cells=0, formal_policy_test_cells=0))
        write_evidence(path/'STATE.json', dict(phase='actual_functional_smoke_complete_pending_independent_review',
            parent_pid=os.getpid(), formal_protocol_frozen=False))
    except Exception as error:
        write_evidence(path/'FAILURE.json', dict(error=repr(error), partial_worker_evidence_preserved=True,
            automatic_retry_allowed=False, actual_two_arm_smoke_pass=False))
        write_evidence(path/'STATE.json', dict(phase='failed', parent_pid=os.getpid(), automatic_retry_allowed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'run', '_worker'])
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha')
    parser.add_argument('--output')
    parser.add_argument('--method')
    args = parser.parse_args()
    if args.action == 'check':
        if any((args.plan, args.plan_sha, args.output, args.method)):
            parser.error('Readonly check takes no run arguments')
        check()
    elif args.action == 'run':
        if not args.plan or not args.plan_sha or not args.output or args.method:
            parser.error('Run requires exact plan/SHA and new output, no method override')
        run(args.plan, args.plan_sha, args.output)
    else:
        if args.method not in METHODS or not args.plan or not args.plan_sha or not args.output:
            parser.error('Private worker requires declared method and exact parent-bound inputs')
        worker(args.method, args.plan, args.plan_sha, args.output)
