"""Additive control entry: historical Full replay BEFORE reward ablation training.

No original source/protocol/output is modified. No automatic resume or CPU fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

from windows_runtime import initialize
initialize()

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
OUT = ROOT / 'results_qla_joint_reward_ablation_20261010_repaired'
DOC = BASE / 'README_REWARD_EXECUTION_中文.md'
ORIGINAL = ROOT / 'paper_doc_tools/run_joint_reward_ablation.py'
sys.path.insert(0, str(ROOT / 'paper_doc_tools'))
loader = importlib.util.spec_from_file_location('original_reward_protocol_entry', ORIGINAL)
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def stamp():
    return datetime.now().astimezone().isoformat()


def require_environment():
    import importlib.metadata
    from qla_environment import guard
    spec = guard()
    expected = Path(spec['isolated_environment']) / 'Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('Use the declared isolated interpreter; no base-environment fallback')
    if '.'.join(map(str, sys.version_info[:3])) != spec['python'] or importlib.metadata.version('numpy') != spec['numpy'] or importlib.metadata.version('torch') != spec['torch']:
        raise RuntimeError('Historical software versions not matched')


def bindings():
    paths = [Path(__file__), DOC, ORIGINAL,
             ROOT / 'paper_doc_tools/joint_reward_ablation_worker.py',
             ROOT / 'paper_doc_tools/joint_reward_ablation_core.py',
             ROOT / 'paper_doc_tools/test_joint_reward_ablation.py',
             BASE / 'scripts/windows_runtime.py', BASE / 'scripts/qla_environment.py',
             BASE / 'protocol/QLA_ENVIRONMENT_SPEC.json',
             ROOT / 'results_qla_joint_reward_ablation_20261009/PROTOCOL_中文.md']
    return {str(path): module.sha(path) for path in paths}


def prepare():
    require_environment()
    if OUT.exists():
        raise FileExistsError('New output exists; preserve it and use check, not repeated prepare')
    design = module.prepare(OUT)
    save(OUT / 'ADDITIVE_ENTRY_BINDING.json', dict(at=stamp(), files=bindings(), design_sha256=module.sha(OUT / 'DESIGN.json'),
         order='two-arm smoke, all25 historical Full test replays,25 variant training/validation, freeze,25 variant tests, independent result audit',
         Full_replay_for_implementation_identity_only=True, test_used_for_hyperparameter_selection=False,
         original_tolerances_unchanged=True, training_started=False))
    module.save(OUT / 'STATUS.json', dict(status='prepared_not_started', training_started=False, at=stamp()))
    print('REWARD_FOLLOWUP_PREPARED_NOT_STARTED', len(design['Full_references']), flush=True)


def check():
    require_environment()
    binding = read(OUT / 'ADDITIVE_ENTRY_BINDING.json')
    if binding['files'] != bindings() or binding['design_sha256'] != module.sha(OUT / 'DESIGN.json'):
        raise RuntimeError('Entry/doc/design source binding changed')
    design = read(OUT / 'DESIGN.json')
    if len(design['Full_references']) != 25:
        raise RuntimeError('Incomplete Full checkpoint registry')
    for reference in design['Full_references']:
        if module.sha(reference['checkpoint']) != reference['checkpoint_sha256']:
            raise RuntimeError('Historical Full checkpoint changed')
    print('REWARD_FOLLOWUP_READONLY_CHECK_OK', flush=True)
    return design


def smoke():
    design = check()
    from qla_environment import witness
    witness()  # resource guard before any CUDA allocation
    module.preflight(OUT)
    reference = next(item for item in design['Full_references'] if item['seed'] == 1 and item['age'] == 8)
    for variant in ('Full', module.VARIANT):
        module.run_cell(OUT, dict(variant=variant, seed=1, age=8), 'validation', design, reference, smoke=True)
    save(OUT / 'TWO_ARM_SMOKE_COMPLETE.json', dict(at=stamp(), binding_sha256=module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'),
         scope='72 training and32 validation slots/arm; not formal evidence', formal_training_started=False))
    module.save(OUT / 'STATUS.json', dict(status='smoke_complete_pending_fresh_entry_review', training_started=False, at=stamp()))


def entry_check():
    design = check()
    if read(OUT / 'TWO_ARM_SMOKE_COMPLETE.json')['binding_sha256'] != module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'):
        raise RuntimeError('Smoke binding changed')
    reference = next(item for item in design['Full_references'] if item['seed'] == 1 and item['age'] == 8)
    for variant in ('Full', module.VARIANT):
        cell = dict(variant=variant, seed=1, age=8)
        cell_out = OUT / 'smoke' / variant / 'seed_1/delay_8'
        if not (cell_out / 'COMPLETE.json').exists():
            raise RuntimeError('Missing actual two-arm smoke result')
        module.validate(cell_out, 'validation', cell, reference, design, smoke=True)
    print('REWARD_FOLLOWUP_ACTUAL_ENTRY_CHECK_OK', flush=True)


def formal():
    design = check()
    entry_check()
    review = read(OUT / 'ENTRY_REVIEW.json')
    if review.get('passed') is not True or review.get('binding_sha256') != module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'):
        raise RuntimeError('Fresh independent source/doc/actual-smoke review required')
    from qla_environment import witness
    witness()
    module.preflight(OUT)
    save(OUT / 'RUN_STARTED.json', dict(pid=os.getpid(), at=stamp(), binding_sha256=module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'),
         entry_review_sha256=module.sha(OUT / 'ENTRY_REVIEW.json')))
    try:
        rows = []
        precheck = []
        # These are fixed historical Full checkpoints. No variant exists yet and
        # no hyperparameter is selected from their test outcomes.
        for reference in design['Full_references']:
            cell = dict(variant='Full', seed=reference['seed'], age=reference['age'])
            checkpoint = Path(reference['checkpoint'])
            summary, row, manifest = module.run_cell(OUT, cell, 'test', design, reference, checkpoint=checkpoint)
            precheck.append(dict(seed=reference['seed'], age=reference['age'], summary=str(summary), summary_sha256=module.sha(summary)))
            rows.append(result_row(cell, row, manifest))
        save(OUT / 'ALL_FULL_REPLAYS_MATCH.json', dict(at=stamp(), abs_tolerance=1e-8, cells=precheck,
             variant_training_started=False, criterion='all four historical metrics and exact schedules/checkpoints'))
        frozen = []
        for reference in design['Full_references']:
            cell = dict(variant=module.VARIANT, seed=reference['seed'], age=reference['age'])
            summary, _, _ = module.run_cell(OUT, cell, 'validation', design, reference)
            checkpoint = summary.with_name('checkpoint_LCC-MAPPO.pt')
            frozen.append(dict(**cell, checkpoint=str(checkpoint), checkpoint_sha256=module.sha(checkpoint)))
        save(OUT / 'FROZEN_BEFORE_TEST.json', dict(source_lock=design['source_lock'], checkpoints=frozen, frozen_at=stamp()))
        for reference in design['Full_references']:
            cell = dict(variant=module.VARIANT, seed=reference['seed'], age=reference['age'])
            item = next(r for r in frozen if r['seed'] == cell['seed'] and r['age'] == cell['age'])
            checkpoint = Path(item['checkpoint'])
            if module.sha(checkpoint) != item['checkpoint_sha256']:
                raise RuntimeError('Frozen variant changed')
            _, row, manifest = module.run_cell(OUT, cell, 'test', design, reference, checkpoint=checkpoint)
            rows.append(result_row(cell, row, manifest))
        if len(rows) != 50:
            raise RuntimeError('Incomplete formal grid')
        module.analyze(OUT, rows)
        module.save(OUT / 'STATUS.json', dict(status='complete_pending_independent_audit', training_started=True,
             new_training_conditions=25, Full_replays=25, test_conditions=50, at=stamp()))
    except Exception as exc:
        module.save(OUT / 'STATUS.json', dict(status='failed_preserved_no_auto_resume', error=repr(exc), at=stamp()))
        raise


def result_row(cell, row, manifest):
    return {**cell, **{key: float(row[key]) for key in module.PRIMARY},
            **{key: manifest[key] for key in module.SECONDARY},
            **{key: float(row[key]) for key in ('n_generated_tasks', 'n_accepted_tasks', 'n_dropped_tasks', 'n_unfinished_tasks')},
            'h_active_slot_fraction': manifest['h_active_slot_fraction'],
            'reward_mean': manifest['reward_mean'], 'reward_std': manifest['reward_std']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'smoke', 'entry-check', 'run'])
    {'prepare': prepare, 'check': check, 'smoke': smoke, 'entry-check': entry_check, 'run': formal}[parser.parse_args().action]()
