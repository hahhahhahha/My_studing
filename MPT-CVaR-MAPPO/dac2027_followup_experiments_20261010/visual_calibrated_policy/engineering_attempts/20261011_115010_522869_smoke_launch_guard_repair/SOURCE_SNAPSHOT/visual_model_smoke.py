"""Prepared separate-process two-method CUDA smoke and safe readonly check.

Today only `check` may run while the original GPU gate fails. No implicit
fallback, automatic retry, formal training, final test or fake PASS record.
"""
from __future__ import annotations
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
from visual_learned_bindings import source_bindings, VisualPolicyDimensions
from visual_training_transactions import (file_sha, digest, CommonBudgetCandidate,
    TrainingTransactionLedger, inspect_checkpoint_archive, require_sha)

METHODS = ('QLA-MAPPO', 'PPO-MV-A')
PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_111741_173552_coordinator/RESULT.json'
PRIOR_SHA = '4ed5a34d5b55b8465813a162c87e947eef27d73041e3b42706998cbe65359b5a'
SMOKE_SETTINGS = dict(users=30, decisions=16, decision_seconds=.05, fixed_drain_seconds=30.,
    total_requests_per_second=30., nominal_delay_seconds=.1, broadcast_seconds=.1, waiting_capacity=32,
    mobility_start_seconds=0., training_geometry_window=[0., 500.], geometry_seed=123,
    schedule_seed=209, model_seed=1, profiles=['CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T'],
    upload_link='unchanged_default_2MHz', initial_H=0., purpose='Low-load calibration-only functional model smoke, not final scenario or budget')


def source_lock():
    # Caller/check commands and every current dependency are locked; do not
    # silently substitute an earlier successful report after source edits.
    return {str(path): file_sha(path) for path in sorted((BASE/'scripts').glob('*.py'))} | source_bindings()


def read_plan(path, expected_sha):
    path = Path(path)
    root = (BASE/'visual_calibrated_policy/engineering_attempts').resolve()
    if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
        raise ValueError('Exact new check-generated smoke plan inside its intended output required')
    if file_sha(path) != require_sha(expected_sha):
        raise RuntimeError('Smoke candidate plan bytes changed')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan.get('status') != 'prepared_unexecuted_functional_smoke_only'
            or plan.get('methods') != list(METHODS) or plan.get('smoke_settings') != SMOKE_SETTINGS
            or plan.get('source_files_sha256') != source_lock()
            or plan.get('formal_protocol_frozen') is not False or plan.get('test_access_allowed') is not False):
        raise RuntimeError('Prepared exact current source/settings required; no reseal, test access or fake completion')
    return plan


def check():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_model_smoke_entry')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['protected_input_and_source_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
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
        plan = dict(status='prepared_unexecuted_functional_smoke_only', methods=list(METHODS),
            source_files_sha256=locked, smoke_settings=SMOKE_SETTINGS,
            formal_protocol_frozen=False, test_access_allowed=False,
            actual_model_forward_and_optimizer_executed=False, actual_roundtrip_performed=False)
        write_evidence(attempt/'SMOKE_CANDIDATE_PLAN.json', plan)
        report = dict(at=datetime.now().astimezone().isoformat(), source_entry_check_passed=True,
            actual_cuda_smoke_pass=False, torch_imported=False, actual_model_or_optimizer_or_roundtrip_calls=0,
            protected_input_and_source_sha256=protected, source_files_sha256=locked,
            smoke_plan_path=str(attempt/'SMOKE_CANDIDATE_PLAN.json'),
            smoke_plan_sha256=file_sha(attempt/'SMOKE_CANDIDATE_PLAN.json'),
            loader_requires_bound_actual_controller_and_no_optimizer_restore=True,
            factory_free_GPU_check_not_repeated_for_same_process_clone=True,
            separate_process_per_method_source_prepared=True, automatic_retry_allowed=False,
            policy_training_cells=0, formal_policy_test_cells=0,
            independent_complete_entry_review=False)
        write_evidence(attempt/'RESULT.json', report)
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
        print('MODEL_SMOKE_ENTRY_PREPARED_ONLY_NOT_EXECUTED', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_model_or_optimizer_or_roundtrip_calls=0))
        raise


def worker(method, plan_path, plan_sha, new_directory):
    """Actual method worker SOURCE ONLY: never invoked in the blocked wake."""
    plan = read_plan(plan_path, plan_sha)
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
    path.mkdir(parents=True, exist_ok=False)
    write_evidence(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
        method=method, plan_sha256=plan_sha, source_files_sha256=plan['source_files_sha256']))
    lock_path = BASE/'visual_calibrated_policy/smoke_locks'/f'{plan_sha}.json'
    parent = json.loads(lock_path.read_text(encoding='utf-8'))
    if (parent['plan_sha256'] != plan_sha or Path(parent['output']).resolve() != path.parent
            or path.name != method or parent['parent_pid'] != os.getppid()):
        raise RuntimeError('Private worker requires its live recorded parent; no direct/replacement worker invocation')
    controller, stream, assembly, ledger = None, None, None, None
    try:
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
        window = TrainingWindow(0., 500.)
        profiles = tuple(SMOKE_SETTINGS['profiles'])
        edges = mobility.server_positions(window, 123, 4)
        scales = calibration_scales(catalog, mobility, window, edges, profiles)
        schedule = generate_schedule(catalog, 30, 16, .05, 209, 30.)
        env = VisualReceiptEnvironment(schedule, catalog, mobility, edges, profiles, 0.)
        settings = EpisodeClosureCandidate()
        dimensions = VisualPolicyDimensions(30, 4, 19, 705, 30)
        # This is the ONE initial free-GPU/kernel gate before this worker
        # creates CUDA models. Clone loading later reuses this own context.
        model = build_cuda_model(method, dimensions, {}, 1)
        controller = NeuralVisualController(method, model)
        budget = CommonBudgetCandidate((16,))
        ledger = TrainingTransactionLedger(method, 1, budget, (schedule.fingerprint(),), digest(SMOKE_SETTINGS))
        ledger.begin_episode(1, schedule.fingerprint(), 'calibration')
        stream = PersistentMVStream(30, purpose='training') if method == 'PPO-MV-A' else None
        assembly = CompactCoordinatorAssembly(method, 30, 4, 16, scales, settings, stream, 1, 'functional-smoke-only', initial_H=0.)
        before_parameters = actual_parameter_sha256(controller)
        first_packed = None
        for _ in range(16):
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
        ledger.close_collection(closed_episode_counts(prepared, 30, 16, .05))
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
        if not all(np.array_equal(getattr(original_action, name), getattr(loaded_action, name))
                   for name in ('portfolios', 'latent_actions', 'log_probabilities', 'values')):
            raise RuntimeError('Loaded model deterministic outputs do not exactly roundtrip')
        torch = controller._torch
        rng_before = [torch.get_rng_state().clone(), *[item.clone() for item in torch.cuda.get_rng_state_all()]]
        loaded.act(action_inputs, deterministic=True)
        rng_after = [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]
        if len(rng_before) != len(rng_after) or not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
            raise RuntimeError('Deterministic loaded act changed Torch RNG')
        if actual_parameter_sha256(loaded) != after_parameters:
            raise RuntimeError('Loaded model changed during readonly roundtrip')
        inspect_checkpoint_archive(path/'checkpoint', descriptor, archive['artifact_sha256'])
        if source_lock() != plan['source_files_sha256']:
            raise RuntimeError('Sources changed mid-smoke')
        write_evidence(path/'METHOD_SMOKE_PASS.json', dict(status='actual_cuda_method_smoke_pass', method=method,
            source_files_sha256=plan['source_files_sha256'], schedule_sha256=schedule.fingerprint(),
            actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
            exact_loaded_parameters_and_deterministic_outputs=True, deterministic_Torch_RNG_unchanged=True,
            before_parameter_sha256=before_parameters, after_parameter_sha256=after_parameters,
            update_statistics=stats, counts=asdict(closed_episode_counts(prepared, 30, 16, .05)),
            checkpoint_archive=archive, loader=load_report, calibration_source_sha256=catalog.source_sha256,
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
    plan = read_plan(plan_path, plan_sha)
    from qla_environment import guard
    spec = guard()
    path = Path(new_directory).resolve()
    root = (BASE/'visual_calibrated_policy').resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('New parent run must remain inside visual outputs')
    # A permanent exclusive plan-specific lock: existing partial attempt
    # cannot be retried by inventing another output directory.
    lock_root = BASE/'visual_calibrated_policy/smoke_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root/f'{plan_sha}.json'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(plan_sha256=plan_sha, output=str(path), parent_pid=os.getpid(),
                       at=datetime.now().astimezone().isoformat()), stream)
    path.mkdir(parents=True, exist_ok=False)
    try:
        write_evidence(path/'STARTED.json', dict(parent_pid=os.getpid(), plan_path=str(plan_path),
            plan_sha256=plan_sha, at=datetime.now().astimezone().isoformat()))
        reports = []
        python = Path(spec['isolated_environment'])/'Scripts/python.exe'
        for method in METHODS:
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
            reports.append(json.loads((child/'METHOD_SMOKE_PASS.json').read_text(encoding='utf-8')))
        if reports[0]['schedule_sha256'] != reports[1]['schedule_sha256']:
            raise RuntimeError('Two arms did not use the same actual calibration schedule')
        if source_lock() != plan['source_files_sha256']:
            raise RuntimeError('Sources changed between method workers')
        write_evidence(path/'ACTUAL_TWO_ARM_SMOKE_PASS.json', dict(status='actual_cuda_two_arm_smoke_pass',
            methods=list(METHODS), source_files_sha256=plan['source_files_sha256'],
            actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
            method_reports=reports, plan_sha256=plan_sha, formal_protocol_frozen=False,
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
