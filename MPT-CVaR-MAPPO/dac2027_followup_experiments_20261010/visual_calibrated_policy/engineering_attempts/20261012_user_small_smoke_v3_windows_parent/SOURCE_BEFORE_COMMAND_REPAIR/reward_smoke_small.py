"""Additive 32/8 smoke with verified frozen closure/raw/schedules; no formal API."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

from reward_input_integrity import (HERE, BASE, ROOT, OLD, SCHEDULE_ROOT, SOURCE,
    sha, digest, read, save, source_guard, own_source_map, require_review,
    load_frozen, schedule_records, build_schedules_once, parent_creation_time,
    require_worker_parent, validate_worker_parent_receipt)

sys.path.insert(0, str(BASE/'scripts'))
from windows_runtime import initialize
initialize()
TRAIN_STEPS, VALIDATION_STEPS = 32, 8
ARMS = ('Full', 'NoJointDemandReward')
OUTPUT_ROOT = BASE/'reward_smoke_attempts'


def expected_plan():
    baseline, _ = source_guard()
    return {'schema': 'user-small-reward-smoke-v2', 'arms': list(ARMS),
        'train_steps': TRAIN_STEPS, 'validation_steps': VALIDATION_STEPS,
        'seed': 1, 'nominal_delay': 8, 'role': 'validation',
        'source_files_sha256': own_source_map(),
        'baseline_sha256': sha(HERE/'FROZEN_BASELINE.json'),
        'source_review_sha256': require_review(),
        'schedule_registry_sha256': sha(HERE/'SCHEDULE_REGISTRY.json'),
        'schedule_registry': schedule_records(), 'original_output': str(OLD),
        'consumed_raw_sha256': baseline['consumed_KAIST_first30_file_sha256'],
        'GPU_memory_used_max_MiB_exclusive': 1800,
        'formal_training_or_test_access': False, 'automatic_resume': False}


def read_plan(path, expected_sha):
    if Path(path).resolve() != (HERE/'SMALL_REWARD_PLAN.json').resolve() or sha(path) != expected_sha:
        raise RuntimeError('Exact new plan path/byte SHA required')
    plan = read(path)
    if digest(plan) != digest(expected_plan()):
        raise RuntimeError('Exact complete current plan/source/raw/schedule/review identity required')
    return plan


def prepare():
    source_guard()
    require_review()
    if (HERE/'SMALL_REWARD_PLAN.json').exists():
        raise FileExistsError('New plan already exists; do not reseal')
    build_schedules_once()
    save(HERE/'SMALL_REWARD_PLAN.json', expected_plan())
    print('SMALL_REWARD_V2_PREPARED_ONLY', sha(HERE/'SMALL_REWARD_PLAN.json'), flush=True)


def original_module():
    # The original runner imports its reward core; full closure is verified
    # before it, then again before its command-builder dynamic import.
    source_guard()
    sys.path.insert(0, str(ROOT/'paper_doc_tools'))
    return load_frozen(ROOT/'paper_doc_tools/run_joint_reward_ablation.py', 'small_v2_original_reward_runner')


def command(out, arm):
    source_guard()
    schedule_records()
    module = original_module()
    design = read(OLD/'DESIGN.json')
    cmd = module.command(out, {'variant': arm, 'seed': 1, 'age': 8}, OLD, design, 'validation', smoke=True)
    changes = (('--train-steps', '72', '32'), ('--test-steps', '32', '8'),
        ('--canonical-schedule-root', str(ROOT/'results_qla_canonical_v1/shared_schedules'), str(SCHEDULE_ROOT)))
    for flag, old, new in changes:
        if cmd.count(flag) != 1 or cmd[cmd.index(flag)+1] != old:
            raise RuntimeError(f'Frozen command flag mismatch: {flag}')
        cmd[cmd.index(flag)+1] = new
    source_guard()
    schedule_records()
    return cmd


def validate(out, arm):
    source_guard()
    registry = schedule_records()
    module = original_module()
    design = read(OLD/'DESIGN.json')
    summary = module.summary_path(out)
    row = module.csv_row(summary)
    manifest = read(out/'diagnostics/intervention_manifest.json')
    repair = read(out/'diagnostics/repair_manifest.json')
    lock = design['source_lock']
    if repair['worker_sha256'] != lock['repair_worker_sha256'] or repair['helper_sha256'] != lock['repair_helper_sha256']:
        raise RuntimeError('Required training wrapper changed')
    if manifest['variant'] != arm or manifest['source_sha256'] != lock['source_sha256']:
        raise RuntimeError('Intervention provenance mismatch')
    if manifest['intervention_worker_sha256'] != lock['intervention_worker_sha256'] or manifest['intervention_core_sha256'] != lock['intervention_core_sha256']:
        raise RuntimeError('Reward intervention source changed')
    if sha(manifest['effective_reward_file']) != manifest['effective_reward_sha256']:
        raise RuntimeError('Effective reward artifact changed')
    if row['evaluation_split_role'] != 'validation' or manifest['evaluation_split_role'] != 'validation':
        raise RuntimeError('Wrong evaluation role')
    if not repair['critic_action_features_zeroed'] or not repair['rollout_bootstrap']:
        raise RuntimeError('Required critic/bootstrap repair absent')
    if any(not math.isfinite(float(row[k])) for k in module.PRIMARY):
        raise RuntimeError('Nonfinite metric; retain numerical failure')
    if abs(float(row['episode_mass_conservation_error'])) > 1e-6:
        raise RuntimeError('Original mass conservation tolerance failed')
    if row['drain_finished'].lower() != 'true' or int(float(row['n_unfinished_tasks'])):
        raise RuntimeError('Drain failed')
    config = read(summary.with_name('config_LCC-MAPPO.json'))
    if config['train_steps'] != 32 or config['test_steps'] != 8 or manifest['evaluation_slots'] != 8:
        raise RuntimeError('Actual reduced budget mismatch')
    with (out/'diagnostics/reward_slots.csv').open(encoding='utf-8', newline='') as handle:
        rewards = list(csv.DictReader(handle))
    if sum(r['phase'] == 'train' for r in rewards) != 32 or sum(r['phase'] != 'train' for r in rewards) != 8:
        raise RuntimeError('Incomplete per-slot reward evidence')
    contents = {r['role']: r['content_sha256'] for r in registry['records']}
    if row['train_schedule_sha256'] != contents['train'] or row['evaluation_schedule_sha256'] != contents['validation']:
        raise RuntimeError('Actual consumed schedules differ from registered original-formula schedules')
    source_guard()
    schedule_records()
    return {'arm': arm, 'summary': str(summary), 'summary_sha256': sha(summary),
        'checkpoint_sha256': sha(summary.with_name('checkpoint_LCC-MAPPO.pt')),
        'train_schedule_sha256': row['train_schedule_sha256'],
        'validation_schedule_sha256': row['evaluation_schedule_sha256'], 'config': config,
        'intervention_manifest_sha256': sha(out/'diagnostics/intervention_manifest.json')}


def worker(plan_path, plan_sha, output, arm, remaining):
    plan = read_plan(plan_path, plan_sha)
    identity = digest(plan)
    output = Path(output).resolve()
    lock = read(OUTPUT_ROOT/'locks'/f'{identity}.json')
    if arm not in ARMS or output != Path(lock['output']).resolve()/arm or lock['plan_sha256'] != plan_sha:
        raise RuntimeError('Private worker requires live recorded parent and exact arm')
    if output.exists():
        raise FileExistsError('Partial arm cannot be reused')
    if remaining and remaining[0] == '--':
        remaining = remaining[1:]
    if remaining != command(output, arm)[3:]:
        raise RuntimeError('Worker arguments changed')
    parent_receipt = require_worker_parent(lock, [sys.executable, '-B', '-X', 'utf8', *sys.argv])
    if read(output.parent/f'{arm}_COMMAND.json') != parent_receipt['worker_command']:
        raise RuntimeError('Actual worker command differs from parent artifact')
    save(output.parent/f'{arm}_PARENT_CHAIN.json', parent_receipt)
    from qla_environment import witness
    witness()  # fresh process-local resource/kernel check before old worker imports Torch
    read_plan(plan_path, plan_sha)
    source_guard()
    schedule_records()
    # The separate witness is not evidence of this process's runtime. Import
    # and verify actual modules only after its resource/kernel gate passed.
    spec = read(BASE/'protocol/QLA_ENVIRONMENT_SPEC.json')
    from reward_input_integrity import actual_environment
    runtime = actual_environment(spec)
    import numpy as np
    import torch
    if np.__version__ != spec['numpy'] or torch.__version__ != spec['torch'] or not Path(np.__file__).resolve().is_relative_to(Path(spec['isolated_environment']).resolve()):
        raise RuntimeError('Actual imported NumPy/Torch do not match the declared runtime')
    torch.set_num_threads(spec['cpu_threads'])
    if torch.get_num_interop_threads() != spec['interop_threads']:
        torch.set_num_interop_threads(spec['interop_threads'])
    if torch.get_num_threads() != spec['cpu_threads'] or torch.get_num_interop_threads() != spec['interop_threads'] or not torch.cuda.is_available() or torch.cuda.get_device_name(0) != spec['gpu']:
        raise RuntimeError('Actual worker threads/device differ; no fallback')
    runtime.update(numpy_module_file=str(Path(np.__file__).resolve()),
        torch_module_file=str(Path(torch.__file__).resolve()), gpu=torch.cuda.get_device_name(0),
        cpu_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        runtime_is_actual_worker=True, model_started=False)
    save(output.parent/f'{arm}_ACTUAL_RUNTIME.json', runtime)
    source_guard()
    old_worker = ROOT/'paper_doc_tools/joint_reward_ablation_worker.py'
    sys.argv = [str(old_worker), *remaining]
    runpy.run_path(str(old_worker), run_name='__main__')
    read_plan(plan_path, plan_sha)


def run(plan_path, plan_sha, output):
    plan = read_plan(plan_path, plan_sha)
    identity = digest(plan)
    output = Path(output).resolve()
    if output == OUTPUT_ROOT.resolve() or not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Separate output root required')
    save(OUTPUT_ROOT/'locks'/f'{identity}.json', {'identity': identity, 'plan_sha256': plan_sha,
        'parent_pid': os.getpid(), 'parent_creation_time': parent_creation_time(),
        'output': str(output), 'at': datetime.now().astimezone().isoformat()})
    output.mkdir(parents=True, exist_ok=False)
    try:
        save(output/'STARTED.json', {'identity': identity, 'plan_sha256': plan_sha, 'actual_smoke_complete': False})
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        reports = []
        for arm in ARMS:
            read_plan(plan_path, plan_sha)
            out = output/arm
            frozen_cmd = command(out, arm)
            cmd = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)), '_worker',
                '--plan', str(plan_path), '--plan-sha', plan_sha, '--output', str(out), '--arm', arm, '--', *frozen_cmd[3:]]
            save(output/f'{arm}_COMMAND.json', cmd)
            with (output/f'{arm}_stdout.log').open('xb') as stdout, (output/f'{arm}_stderr.log').open('xb') as stderr:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                save(output/f'{arm}_STARTED.json', {'pid': process.pid, 'arm': arm})
                if process.wait() != 0:
                    raise RuntimeError(f'{arm} failed; preserve partial evidence, no retry/merge')
            validate_worker_parent_receipt(read(output/f'{arm}_PARENT_CHAIN.json'), process.pid,
                read(OUTPUT_ROOT/'locks'/f'{identity}.json'))
            reports.append(validate(out, arm))
        for field in ('train_schedule_sha256', 'validation_schedule_sha256'):
            if reports[0][field] != reports[1][field]:
                raise RuntimeError('Common schedule mismatch')
        ignored = {'output_dir'}
        if {k:v for k,v in reports[0]['config'].items() if k not in ignored} != {k:v for k,v in reports[1]['config'].items() if k not in ignored}:
            raise RuntimeError('Nonreward config differs')
        read_plan(plan_path, plan_sha)
        save(output/'SMALL_TWO_ARM_COMPLETE.json', {'identity': identity, 'plan_sha256': plan_sha,
            'reports': reports, 'actual_smoke_complete': True, 'formal_training_or_test_started': False,
            'scope': '32 train/8 validation per arm; actual functional only, pending independent evidence/adoption'})
        print('SMALL_REWARD_V2_ACTUAL_TWO_ARM_COMPLETE', output, flush=True)
    except BaseException as error:
        save(output/'FAILURE.json', {'error': repr(error), 'automatic_resume': False,
            'partial_evidence_preserved': True, 'actual_smoke_complete': False, 'formal_training_started': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', '_worker'))
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha')
    parser.add_argument('--output')
    parser.add_argument('--arm', choices=ARMS)
    args, remaining = parser.parse_known_args()
    if args.action == 'prepare':
        if any((args.plan, args.plan_sha, args.output, args.arm, remaining)):
            parser.error('Prepare takes no runtime arguments')
        prepare()
    elif args.action == '_worker':
        if not all((args.plan, args.plan_sha, args.output, args.arm)):
            parser.error('Worker requires private parent binding')
        worker(args.plan, args.plan_sha, args.output, args.arm, remaining)
    else:
        if not all((args.plan, args.plan_sha, args.output)) or args.arm or remaining:
            parser.error('Run needs exact plan/SHA/new output')
        run(args.plan, args.plan_sha, args.output)
