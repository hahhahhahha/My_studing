"""One-shot retained-test continuation; frozen measurements, isolated statistics.

This additive driver does not restart the interrupted confirmation runner.
Its parent never imports torch or numpy, and has no resume/refit/retry action.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v2_common import BASE, OLD, PROFILES, read, save, sha, csv_write, stem

PLAN_SHA = 'bbddc7b6e15e68625deb41992707d64bf32faa184910de8d8c01fbab1a64f282'
VALIDATION_SHA = '3827883cb3e74020890c54b0324ce667e76085e21e9331c0fbe3d001f077a181'
REVIEW_SHA = '46d8f31bc753fcfac77961346f4a6761baf88152a0c034c2a0b8fba43cf6f79d'
FREEZE = BASE / 'protocol/RETAINED_TEST_CONTINUATION.json'
STATE = BASE / 'protocol/RETAINED_TEST_STATE.json'
LOCK = BASE / 'protocol/RETAINED_TEST_RUN_STARTED.json'
ENTRY = BASE / 'protocol/RETAINED_TEST_ENTRY_REVIEW.json'
OUTPUT = BASE / 'analysis/retained_test_continuation'
README = BASE / 'README_RETAINED_TEST_中文.md'


def now():
    return datetime.now().astimezone().isoformat()


def safe_runtime():
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').upper() not in ('', 'FALSE', '0'):
        raise RuntimeError('Refuse unsafe duplicate-OpenMP override')
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch must stay outside orchestration/statistics process')


def prerequisites():
    safe_runtime()
    if sha(BASE / 'protocol/PLAN.json') != PLAN_SHA:
        raise RuntimeError('Original PLAN changed')
    if sha(BASE / 'analysis/VALIDATION_REPORT.json') != VALIDATION_SHA:
        raise RuntimeError('Confirmation result changed')
    review_path = BASE / 'analysis/recovery_20261010/reviewer/AUDIT_REPORT.json'
    if sha(review_path) != REVIEW_SHA:
        raise RuntimeError('Independent confirmation review changed')
    report, review = read(BASE / 'analysis/VALIDATION_REPORT.json'), read(review_path)
    if report['accepted'] is not True or len(report['criteria']) != 6 or not all(x['passed'] for x in report['criteria']):
        raise RuntimeError('Confirmation is not accepted')
    if review['record_integrity_status'] != 'PASS' or review['frozen_six_metric_acceptance'] != 'PASS':
        raise RuntimeError('Confirmation integrity/criteria review is not PASS')
    plan = read(BASE / 'protocol/PLAN.json')
    for key, root in [('source_sha256', BASE / 'scripts'), ('artifact_sha256', BASE), ('old_readonly_sha256', OLD)]:
        for name, expected in plan[key].items():
            if sha(root / name) != expected:
                raise RuntimeError('Original frozen fingerprint changed: ' + name)
    return plan


def measurement_day(plan):
    if datetime.now().astimezone().date() != datetime.fromisoformat(plan['frozen_at']).date():
        raise RuntimeError('Same-day measurement cannot be silently moved to another day')


def assert_unstarted():
    paths = [LOCK, STATE, BASE / 'analysis/TEST_REPORT.json',
             BASE / 'analysis/test_errors.csv', BASE / 'analysis/V2_保留测试结果_中文.md',
             BASE / 'logs/retained_test_stdout.log', BASE / 'logs/retained_test_stderr.log']
    paths += list((BASE / 'measurements/test').glob('*'))
    paths += list((BASE / 'protocol/jobs').glob('test_*.json'))
    paths += list((BASE / 'logs').glob('test_*.log'))
    paths += list((BASE / 'protocol').glob('idle_retained_test_*.json'))
    paths += list((BASE / 'protocol').glob('retained_test_session_*.json'))
    if any(p.exists() for p in paths):
        raise FileExistsError('Retained test already started or has evidence; no restart')


def prepare(repaired=False):
    plan = prerequisites()
    measurement_day(plan)
    assert_unstarted()
    if FREEZE.exists() and not repaired:
        raise FileExistsError('Continuation freeze already exists; no resealing')
    previous = None
    if repaired:
        old_review = read(ENTRY)
        if old_review.get('passed') is not False or old_review['continuation_sha256'] != sha(FREEZE):
            raise RuntimeError('Entry revision requires a fingerprint-bound failed independent review')
        previous = sha(FREEZE)
        save(BASE / 'protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json', read(FREEZE), exclusive=True)
        save(BASE / 'protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json', old_review, exclusive=True)
    # Recover the archived guard without calling the confirmation-only snapshot.
    protected = dict(read(BASE / 'analysis/recovery_20261010/AFTER_ANALYSIS_SNAPSHOT.json')['protected_files'])
    for folder in ['measurements/validation', 'measurements/reference', 'analysis/recovery_20261010']:
        for p in sorted((BASE / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                protected[p.relative_to(BASE).as_posix()] = sha(p)
    for name in ['analysis/VALIDATION_REPORT.json', 'analysis/validation_errors.csv',
                 'analysis/reference_drift.csv', 'analysis/V2_确认验证结果_中文.md',
                 'analysis/v2_service_requests.csv', 'figures/fig_visual_service_v2.png',
                 'figures/fig_visual_service_v2.pdf', 'analysis/RUNTIME_INTERRUPTION_20261010.json',
                 'analysis/V2_运行时中断归档_中文.md', 'protocol/STATE.json', 'protocol/RUN_STARTED.json',
                 'logs/runner_stdout.log', 'logs/runner_stderr.log']:
        protected[name] = sha(BASE / name)
    if repaired:
        for name in ['protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json',
                     'protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json',
                     'analysis/retained_test_continuation/run_retained_test_isolated_pre_review.py',
                     'analysis/retained_test_continuation/README_RETAINED_TEST_PRE_REVIEW_中文.md']:
            protected[name] = sha(BASE / name)
    value = dict(created_at=now(), original_plan_sha256=PLAN_SHA,
                 validation_report_sha256=VALIDATION_SHA, confirmation_review_sha256=REVIEW_SHA,
                 entry_source_sha256=sha(__file__), readme_sha256=sha(README), protected_sha256=protected,
                 sessions=[x for x in plan['sessions'] if x['kind'] == 'retained_test'],
                 expected_test_rows=10000, expected_total_new_rows=23200,
                 authorization='User: 现在是不是可以开始保留测试 和完整实验了; continue original service measurement only',
                 scope='No vision/QLA training, simulator work or paper edits; no refitting or gate changes')
    if repaired:
        value['prelaunch_entry_repair'] = dict(previous_continuation_sha256=previous,
            failed_review_sha256=sha(BASE / 'protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json'),
            reason='Restrict date guard to measurement; require accepted test report before complete export',
            original_experimental_plan_changed=False, test_rows_before_repair=0)
    save(FREEZE, value, exclusive=not repaired)
    print('CONTINUATION_FROZEN ' + sha(FREEZE), flush=True)


def frozen_guard():
    plan = prerequisites()
    freeze = read(FREEZE)
    if freeze['entry_source_sha256'] != sha(__file__) or freeze['readme_sha256'] != sha(README):
        raise RuntimeError('Continuation source/document changed after freeze')
    for name, expected in freeze['protected_sha256'].items():
        if sha(BASE / name) != expected:
            raise RuntimeError('Protected confirmation/failure evidence changed: ' + name)
    sessions = [x for x in plan['sessions'] if x['kind'] == 'retained_test']
    if freeze['sessions'] != sessions or [x['session'] for x in sessions] != list(range(1, 6)):
        raise RuntimeError('Original five test sessions were changed')
    splits = read(BASE / 'protocol/SPLITS.json')
    if splits['test'] != read(OLD / 'protocol/IMAGE_SPLITS.json')['test'] or len(splits['test']) != 1000:
        raise RuntimeError('Original retained test pool changed')
    expected_ids = {e['image_id'] for e in splits['test']}
    jobs = [j for x in sessions for j in x['jobs']]
    if len(jobs) != 50 or sum(len(j['image_ids']) for j in jobs) != 10000:
        raise RuntimeError('Wrong original test grid')
    for item in sessions:
        for profile in PROFILES:
            ids = [i for j in item['jobs'] if j['profile'] == profile for i in j['image_ids']]
            if len(ids) != 1000 or set(ids) != expected_ids:
                raise RuntimeError('Incomplete/duplicated session-profile test identities')
        if any(j['role'] != 'test' or j['session'] != item['session'] for j in item['jobs']):
            raise RuntimeError('Invalid test job role/session')
    return plan, freeze


def check():
    plan, _ = frozen_guard()
    measurement_day(plan)
    assert_unstarted()
    from runner import audit
    info, _ = audit(False)
    result = subprocess.run([sys.executable, '-X', 'utf8', str(BASE / 'scripts/runner.py'), 'check'], check=False)
    if result.returncode:
        raise RuntimeError('Isolated original protocol/CPU witness failed')
    print('RETAINED_TEST_READONLY_CHECK_OK ' + json.dumps(info), flush=True)


def resources():
    safe_runtime()
    path = Path('C:/Users/CKX/.codex/skills/get-available-resources/scripts/detect_resources.py')
    spec = importlib.util.spec_from_file_location('resource_skill', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    disk = shutil.disk_usage(BASE)
    result = dict(at=now(), cpu=module.get_cpu_info(), memory=module.get_memory_info(),
                  nvidia_probe=module.detect_nvidia_gpus(), gpu_used=False,
                  disk=dict(free_bytes=disk.free, total_bytes=disk.total),
                  strategy='Serial fixed CPU profiles; existing unchanged environment; no GPU measurement')
    save(BASE / 'protocol/RETAINED_TEST_RESOURCES.json', result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def child(args, label):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / (label + '_stdout.log')).open('x', encoding='utf-8') as out, \
         (OUTPUT / (label + '_stderr.log')).open('x', encoding='utf-8') as err:
        result = subprocess.run([sys.executable, '-X', 'utf8', *map(str, args)], stdout=out, stderr=err, check=False)
    if result.returncode:
        raise RuntimeError(f'Isolated child {label} failed: exit {result.returncode}; evidence retained')


def statistics_worker():
    safe_runtime()
    frozen_guard()
    from runner import audit, evaluate
    info, rows = audit(True)
    measured_protocol_guard(rows)
    save(OUTPUT / 'STRICT_TEST_AUDIT.json', info, exclusive=True)
    result = evaluate(True)
    safe_runtime()
    frozen_guard()
    print('TEST_STATISTICS_COMPLETE ' + json.dumps(result['criteria']), flush=True)


def measured_protocol_guard(rows):
    plan, _ = frozen_guard()
    expected_day = datetime.fromisoformat(plan['frozen_at']).date()
    if any(datetime.fromisoformat(r['measurement_timestamp']).date() != expected_day for r in rows):
        raise RuntimeError('Measured test timestamps violate the same-day protocol')
    previous = None
    for session in range(1, 6):
        event = read(BASE / f'protocol/retained_test_session_{session}.json')
        start = datetime.fromisoformat(event['started_at'])
        if start.date() != expected_day or event['session'] != session:
            raise RuntimeError('Test session start violates frozen date/identity')
        if previous is not None and (start - previous).total_seconds() < plan['spacing_seconds']:
            raise RuntimeError('Test session start spacing violated')
        previous = start
        stamps = [datetime.fromisoformat(r['measurement_timestamp']) for r in rows if int(r['session_id']) == session]
        if not stamps or min(stamps) < start:
            raise RuntimeError('Test row precedes its recorded session start')
        for p in (BASE / 'measurements/test').glob(f'test_s{session}_*.json'):
            meta = read(p)
            if any(datetime.fromisoformat(meta[k]['wall_clock']).date() != expected_day for k in ['sensor_before', 'sensor_after']):
                raise RuntimeError('Measured test block crosses the protocol day')


def audit_all():
    frozen_guard()
    from runner import audit
    confirmation, c_rows = audit(False)
    test, t_rows = audit(True)
    measured_protocol_guard(t_rows)
    report = read(BASE / 'analysis/TEST_REPORT.json')
    if report['integrity'] != test or report['plan_sha256'] != PLAN_SHA:
        raise RuntimeError('Test report does not match actual records/plan')
    if len(c_rows) + len(t_rows) != 23200:
        raise RuntimeError('Full V2 record count mismatch')
    print('FULL_V2_RECORD_AUDIT_OK ' + json.dumps(dict(confirmation=confirmation, test=test)), flush=True)


def export_worker():
    safe_runtime()
    audit_all()
    test_report = read(BASE / 'analysis/TEST_REPORT.json')
    if (test_report.get('accepted') is not True or len(test_report['criteria']) != 6
        or not all(x['passed'] is True for x in test_report['criteria'])
        or test_report['unsupported_inputs']
        or test_report['model_sha256'] != sha(BASE / 'model/CONDITIONAL_MODEL.json')):
        raise RuntimeError('Refuse success-like complete export for an unaccepted test')
    from runner import audit
    _, c_rows = audit(False); _, t_rows = audit(True)
    targets = [BASE / 'analysis/v2_complete_service_requests.csv',
               BASE / 'figures/fig_visual_service_v2_complete.png', BASE / 'figures/fig_visual_service_v2_complete.pdf']
    if any(p.exists() for p in targets):
        raise FileExistsError('Do not overwrite full exports')
    csv_write(targets[0], c_rows + t_rows, exclusive=True)
    config = BASE / '.runtime_cache/matplotlib'
    config.mkdir(parents=True, exist_ok=True); os.environ.setdefault('MPLCONFIGDIR', str(config))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for row_index, role in enumerate(['validation', 'test']):
        report = read(BASE / ('analysis/' + role.upper() + '_REPORT.json'))
        for ax, profile in zip(axes[row_index], PROFILES):
            s = report['summary'][profile]; keys = ['mean', 'p95', 'p99']; x = np.arange(3)
            actual = [s['measured_ms'][k] for k in keys]
            ax.bar(x - .18, [s['predicted_ms'][k] for k in keys], .36, label='Frozen prediction', color='#0072B2')
            ax.bar(x + .18, actual, .36, label='Measured full service', color='#D55E00')
            for i, v in enumerate(actual):
                ax.text(i + .18, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x, ['Mean', 'P95', 'P99']); ax.set_ylabel('Service time (ms)')
            ax.set_title(('Target-weighted confirmation: ' if role == 'validation' else 'Retained test: ') + profile)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Single machine, two CPU profiles; same-day repeated service measurements', fontsize=10)
    fig.tight_layout(); fig.savefig(targets[1], dpi=180); fig.savefig(targets[2]); plt.close(fig)
    save(OUTPUT / 'COMPLETE_EXPORT.json', dict(at=now(), rows=23200,
         artifacts={p.relative_to(BASE).as_posix(): sha(p) for p in targets}, independent_result_review='pending'), exclusive=True)
    frozen_guard(); safe_runtime()


def run():
    plan, freeze = frozen_guard()
    measurement_day(plan)
    # External launch redirects already opened these two logs; every other evidence
    # path must still be absent. A one-shot lock prevents any concurrent relaunch.
    for p in [LOCK, STATE, BASE / 'analysis/TEST_REPORT.json']:
        if p.exists():
            raise FileExistsError('Continuation has already started')
    if any((BASE / 'measurements/test').glob('*')) or any((BASE / 'protocol/jobs').glob('test_*.json')):
        raise FileExistsError('Existing test evidence; no restart')
    review = read(ENTRY)
    if review.get('passed') is not True or review['continuation_sha256'] != sha(FREEZE) or review['entry_source_sha256'] != sha(__file__):
        raise RuntimeError('Missing independent, fingerprint-bound continuation entry review')
    save(LOCK, dict(pid=os.getpid(), at=now(), continuation_sha256=sha(FREEZE), entry_review_sha256=sha(ENTRY)), exclusive=True)
    state = dict(status='preflight', pid=os.getpid(), started_at=now(), completed_blocks=0, completed_rows=0,
                 independent_result_review='pending', original_runner_restarted=False)
    save(STATE, state)
    try:
        child([BASE / 'scripts/runner.py', 'check'], 'launch_frozen_check')
        from runner import idle_gate
        last_start = None
        for item in freeze['sessions']:
            frozen_guard()
            state.update(status='waiting_for_session', session=item['session']); save(STATE, state)
            if last_start is not None:
                while time.monotonic() - last_start < plan['spacing_seconds']:
                    time.sleep(min(30, plan['spacing_seconds'] - (time.monotonic() - last_start)))
            idle_gate(item)
            frozen_guard()
            measurement_day(plan)
            last_start = time.monotonic()
            started_at = now()
            save(BASE / f"protocol/retained_test_session_{item['session']}.json",
                 dict(session=item['session'], started_at=started_at, continuation_sha256=sha(FREEZE)), exclusive=True)
            state.update(status='measuring', session_started_at=started_at); save(STATE, state)
            for job in item['jobs']:
                prerequisites()
                measurement_day(plan)
                job_path = BASE / 'protocol/jobs' / (stem(job) + '.json')
                save(job_path, job, exclusive=True)
                for suffix in ['.csv', '.json']:
                    if (BASE / 'measurements/test' / (stem(job) + suffix)).exists():
                        raise FileExistsError('No measurement overwrite')
                with (BASE / 'logs' / (stem(job) + '.log')).open('x', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, '-X', 'utf8', str(BASE / 'scripts/measure_block.py'), '--job', str(job_path)], stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f'Test measurement failed: {stem(job)}, exit {result.returncode}')
                state['completed_blocks'] += 1; state['completed_rows'] += len(job['image_ids'])
                state['last_block'] = stem(job); save(STATE, state)
                print('TEST_BLOCK_COMPLETE', stem(job), state['completed_rows'], flush=True)
        state['status'] = 'auditing_and_analyzing_test'; save(STATE, state)
        child([__file__, 'statistics-worker'], 'isolated_test_statistics')
        child([BASE / 'scripts/runner.py', 'check'], 'final_frozen_check')
        state['status'] = 'exporting_full_measurements'; save(STATE, state)
        child([__file__, 'export-worker'], 'isolated_complete_export')
        frozen_guard()
        state.update(status='complete_pending_independent_audit', ended_at=now(),
                     test_report_sha256=sha(BASE / 'analysis/TEST_REPORT.json'), total_new_rows=23200,
                     test_gate_passed=True, independent_result_review='pending')
        save(STATE, state); print('TEST_COMPLETE_PENDING_INDEPENDENT_AUDIT', flush=True)
    except Exception as error:
        state.update(status='failed', error=repr(error), ended_at=now()); save(STATE, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'prepare-repaired-entry', 'resources', 'check', 'run', 'statistics-worker', 'audit-all', 'export-worker'])
    action = parser.parse_args().action
    {'prepare': prepare, 'prepare-repaired-entry': lambda: prepare(repaired=True), 'resources': resources, 'check': check, 'run': run,
     'statistics-worker': statistics_worker, 'audit-all': audit_all, 'export-worker': export_worker}[action]()
