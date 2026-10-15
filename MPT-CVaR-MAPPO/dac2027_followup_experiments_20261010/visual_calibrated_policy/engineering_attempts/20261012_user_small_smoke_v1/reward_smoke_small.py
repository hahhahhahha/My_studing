"""Separately bound 32/8 two-arm functional smoke; no formal or resume API.

Source-only until independent review and the unchanged original resource gate.
Historical entry, bindings, design, source and outputs are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path.insert(0, str(BASE/'scripts'))
from windows_runtime import initialize
initialize()
import reward_followup_v2 as prior

TRAIN_STEPS, VALIDATION_STEPS = 32, 8
ARMS = ('Full', 'NoJointDemandReward')
OUTPUT_ROOT = BASE/'reward_smoke_attempts'


def sha(path):
    return prior.original.module.sha(Path(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def expected_plan():
    prior.check()  # read-only original/V2/design/checkpoint continuity
    files = {str(HERE/name): sha(HERE/name) for name in
        ('visual_model_smoke_small.py', 'reward_smoke_small.py', 'README_SMALL_SMOKE_中文.md', 'USER_REQUEST.json')}
    for name in ('ADDITIVE_ENTRY_BINDING.json', 'ADDITIVE_V2_BINDING.json', 'DESIGN.json'):
        path = prior.OUT/name
        files[str(path)] = sha(path)
    return dict(schema='user-small-reward-smoke-v1', arms=list(ARMS), train_steps=TRAIN_STEPS,
        validation_steps=VALIDATION_STEPS, seed=1, nominal_delay=8, role='validation',
        files_sha256=files, original_output=str(prior.OUT), formal_training_or_test_access=False,
        actual_smoke_executed=False, gpu_gate='memory.used <500MiB', automatic_resume=False)


def prepare():
    plan = expected_plan()
    save(HERE/'SMALL_REWARD_PLAN.json', plan)
    print('SMALL_REWARD_PREPARED_ONLY', sha(HERE/'SMALL_REWARD_PLAN.json'), flush=True)


def command(out, design):
    command = prior.original.module.command(out, dict(variant='Full', seed=1, age=8),
        prior.OUT, design, 'validation', smoke=True)
    # Only the requested functional budget changes, not formal config or source.
    for flag, old, new in (('--train-steps', '72', str(TRAIN_STEPS)),
                           ('--test-steps', '32', str(VALIDATION_STEPS))):
        if command.count(flag) != 1:
            raise RuntimeError('Original smoke budget flag is ambiguous')
        index = command.index(flag)+1
        if command[index] != old:
            raise RuntimeError('Original smoke budget changed')
        command[index] = new
    return command


def validate(out, arm, design):
    module = prior.original.module
    summary = module.summary_path(out)
    row = module.csv_row(summary)
    manifest = json.loads((out/'diagnostics/intervention_manifest.json').read_text(encoding='utf-8'))
    repair = json.loads((out/'diagnostics/repair_manifest.json').read_text(encoding='utf-8'))
    lock = design['source_lock']
    if repair['worker_sha256'] != lock['repair_worker_sha256'] or repair['helper_sha256'] != lock['repair_helper_sha256']:
        raise RuntimeError('Required training wrapper hash changed')
    if manifest['variant'] != arm or manifest['source_sha256'] != lock['source_sha256']:
        raise RuntimeError('Intervention provenance mismatch')
    if manifest['intervention_worker_sha256'] != lock['intervention_worker_sha256'] or manifest['intervention_core_sha256'] != lock['intervention_core_sha256']:
        raise RuntimeError('Reward intervention source changed')
    if sha(manifest['effective_reward_file']) != manifest['effective_reward_sha256']:
        raise RuntimeError('Effective reward source artifact changed')
    if row['evaluation_split_role'] != 'validation' or manifest['evaluation_split_role'] != 'validation':
        raise RuntimeError('Wrong held-out split')
    if not repair['critic_action_features_zeroed'] or not repair['rollout_bootstrap']:
        raise RuntimeError('Required training semantics missing')
    if any(not math.isfinite(float(row[key])) for key in module.PRIMARY):
        raise RuntimeError('Nonfinite performance metric')
    if abs(float(row['episode_mass_conservation_error'])) > 1e-6:
        raise RuntimeError('Mass conservation failed')
    if row['drain_finished'].lower() != 'true' or int(float(row['n_unfinished_tasks'])):
        raise RuntimeError('Drain failed')
    config = json.loads(summary.with_name('config_LCC-MAPPO.json').read_text(encoding='utf-8'))
    if config['train_steps'] != TRAIN_STEPS or config['test_steps'] != VALIDATION_STEPS or manifest['evaluation_slots'] != VALIDATION_STEPS:
        raise RuntimeError('Actual reduced budget mismatch; no 72/32 relabelling')
    with (out/'diagnostics/reward_slots.csv').open(encoding='utf-8', newline='') as handle:
        rewards = list(csv.DictReader(handle))
    if sum(item['phase'] == 'train' for item in rewards) != TRAIN_STEPS or sum(item['phase'] != 'train' for item in rewards) != VALIDATION_STEPS:
        raise RuntimeError('Incomplete per-slot reward evidence')
    checkpoint = summary.with_name('checkpoint_LCC-MAPPO.pt')
    return dict(arm=arm, summary=str(summary), summary_sha256=sha(summary),
        checkpoint_sha256=sha(checkpoint), train_schedule_sha256=row['train_schedule_sha256'],
        validation_schedule_sha256=row['evaluation_schedule_sha256'], config=config,
        intervention_manifest_sha256=sha(out/'diagnostics/intervention_manifest.json'))


def run(plan_path, plan_sha, output):
    plan_path, output = Path(plan_path), Path(output).resolve()
    if plan_path.resolve() != (HERE/'SMALL_REWARD_PLAN.json').resolve() or sha(plan_path) != plan_sha:
        raise RuntimeError('Exact separate plan and byte SHA required')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if digest(plan) != digest(expected_plan()):
        raise RuntimeError('Exact complete current source/budget/binding identity required')
    if output == OUTPUT_ROOT.resolve() or not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('New small-smoke output must remain in its separate root')
    identity = digest(plan)
    save(OUTPUT_ROOT/'locks'/f'{identity}.json', dict(identity=identity, plan_sha256=plan_sha,
        parent_pid=os.getpid(), output=str(output), at=datetime.now().astimezone().isoformat()))
    output.mkdir(parents=True, exist_ok=False)
    try:
        save(output/'STARTED.json', dict(identity=identity, plan_sha256=plan_sha, actual_smoke_complete=False))
        from qla_environment import witness
        witness()  # before Torch/model allocation; lock retained on resource failure
        design = prior.original.read(prior.OUT/'DESIGN.json')
        reports = []
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        for arm in ARMS:
            if digest(expected_plan()) != identity:
                raise RuntimeError('Current binding changed between arms')
            cell = output/arm
            cmd = command(cell, design)
            index = cmd.index('--variant')+1
            cmd[index] = arm
            save(output/f'{arm}_COMMAND.json', cmd)
            with (output/f'{arm}_stdout.log').open('xb') as stdout, (output/f'{arm}_stderr.log').open('xb') as stderr:
                process = subprocess.Popen(cmd, cwd=prior.original.ROOT, env=env, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                save(output/f'{arm}_STARTED.json', dict(pid=process.pid, arm=arm))
                if process.wait() != 0:
                    raise RuntimeError(f'{arm} failed; no retry, second-arm replacement or merge')
            reports.append(validate(cell, arm, design))
        for field in ('train_schedule_sha256', 'validation_schedule_sha256'):
            if reports[0][field] != reports[1][field]:
                raise RuntimeError('Two arms did not use common actual schedules')
        ignored = {'output_dir'}
        if {k:v for k,v in reports[0]['config'].items() if k not in ignored} != {k:v for k,v in reports[1]['config'].items() if k not in ignored}:
            raise RuntimeError('Nonreward configuration differs between arms')
        if digest(expected_plan()) != identity:
            raise RuntimeError('Source/binding changed during execution')
        save(output/'SMALL_TWO_ARM_COMPLETE.json', dict(identity=identity, plan_sha256=plan_sha,
            reports=reports, actual_smoke_complete=True, formal_training_or_test_started=False,
            scope='32 train/8 validation slots per arm; functional only, pending independent actual-evidence audit'))
    except BaseException as error:
        save(output/'FAILURE.json', dict(error=repr(error), automatic_resume=False,
            partial_evidence_preserved=True, actual_smoke_complete=False, formal_training_started=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--plan')
    parser.add_argument('--plan-sha')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'prepare':
        if any((args.plan, args.plan_sha, args.output)):
            parser.error('Prepare takes no execution arguments')
        prepare()
    else:
        if not all((args.plan, args.plan_sha, args.output)):
            parser.error('Run needs exact plan/SHA and a new output')
        run(args.plan, args.plan_sha, args.output)
