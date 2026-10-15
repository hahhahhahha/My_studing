"""Recover confirmation analysis only, with PyTorch and NumPy in separate processes.

Never launches measurement or retained tests. Frozen runner, model, rules and raw
records are read-only. No duplicate-OpenMP workaround or package change is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from v2_common import BASE, OLD, PROFILES, read, save, sha, csv_write

RECOVERY = BASE / 'analysis/recovery_20261010'
EXPECTED_PLAN = 'bbddc7b6e15e68625deb41992707d64bf32faa184910de8d8c01fbab1a64f282'


def runtime_paths():
    import psutil
    return sorted({m.path for m in psutil.Process().memory_maps()
                   if any(k in m.path.lower() for k in ('iomp', 'mkl', 'libomp'))})


def safe_environment():
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').upper() not in ('', 'FALSE', '0'):
        raise RuntimeError('Unsafe duplicate-OpenMP override is set; refuse analysis')


def snapshot():
    if sha(BASE / 'protocol/PLAN.json') != EXPECTED_PLAN:
        raise RuntimeError('Frozen plan does not match reviewed fingerprint')
    plan = read(BASE / 'protocol/PLAN.json')
    result = {'plan_sha256': EXPECTED_PLAN, 'protected_files': {}}
    for group, root, prefix in [('source_sha256', BASE / 'scripts', 'scripts/'),
                                ('artifact_sha256', BASE, ''),
                                ('old_readonly_sha256', OLD, '../dac_visual_calibration_20261009/')]:
        for name, expected in plan[group].items():
            actual = sha(root / name)
            if actual != expected:
                raise RuntimeError(f'Frozen fingerprint mismatch: {group}/{name}')
            result['protected_files'][prefix + name] = actual
    archived = read(BASE / 'analysis/RUNTIME_INTERRUPTION_20261010.json')
    for item in archived['raw_inventory'] + archived['evidence']:
        actual = sha(BASE / item['path'])
        if actual != item['sha256']:
            raise RuntimeError('Interrupted-run evidence changed: ' + item['path'])
        result['protected_files'][item['path']] = actual
    test_folder = BASE / 'measurements/test'
    if test_folder.exists() and any(test_folder.glob('*.csv')):
        raise RuntimeError('Retained-test measurements exist; this recovery is confirmation-only')
    if (BASE / 'protocol/VALIDATION_ACCEPTED.json').exists():
        raise RuntimeError('Unexpected automatic test gate; refuse confirmation-only recovery')
    return result


def child(arguments, label):
    safe_environment()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    stdout = RECOVERY / (label + '_stdout.log')
    stderr = RECOVERY / (label + '_stderr.log')
    if stdout.exists() or stderr.exists():
        raise FileExistsError('Refuse to overwrite analysis child logs: ' + label)
    with stdout.open('x', encoding='utf-8') as out, stderr.open('x', encoding='utf-8') as err:
        completed = subprocess.run([sys.executable, '-X', 'utf8', *map(str, arguments)],
                                   stdout=out, stderr=err, check=False)
    result = dict(label=label, exit_code=completed.returncode,
                  stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
    print(json.dumps(result), flush=True)
    return result


def probe_worker(kind):
    safe_environment()
    if kind in ('torch', 'mixed'):
        from runner import check
        check()
        print('TORCH_RUNTIME_PATHS ' + json.dumps(runtime_paths()), flush=True)
    if kind in ('numpy', 'mixed'):
        import numpy as np
        print('NUMPY_IMPORTED_RUNTIME_PATHS ' + json.dumps(runtime_paths()), flush=True)
        a = np.arange(1024, dtype=np.float64)
        value = float(np.dot(a, a))
        if value != sum(i * i for i in range(1024)):
            raise RuntimeError('NumPy dot witness failed')
        print('NUMPY_DOT_WITNESS_OK ' + str(value), flush=True)
        print('NUMPY_RUNTIME_PATHS ' + json.dumps(runtime_paths()), flush=True)
    print('PROBE_OK ' + kind, flush=True)


def probe():
    safe_environment()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    if (RECOVERY / 'RUNTIME_PROBE.json').exists():
        raise FileExistsError('Probe already archived')
    before = snapshot()
    results = [child([__file__, 'probe-worker', '--kind', kind], 'probe_' + kind)
               for kind in ('torch', 'numpy', 'mixed')]
    after = snapshot()
    if before != after:
        raise RuntimeError('Protected evidence changed during probe')
    report = dict(checked_at=datetime.now().astimezone().isoformat(), results=results,
                  isolated_torch_ok=results[0]['exit_code'] == 0,
                  isolated_numpy_ok=results[1]['exit_code'] == 0,
                  mixed_process_exit_code=results[2]['exit_code'],
                  duplicate_override_used=False, protected_snapshot=before)
    save(RECOVERY / 'RUNTIME_PROBE.json', report, exclusive=True)
    if not report['isolated_torch_ok'] or not report['isolated_numpy_ok']:
        raise RuntimeError('Isolated runtime witness failed')


def evaluate_worker():
    safe_environment()
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch is present in the statistics process')
    from runner import audit, evaluate
    integrity, _ = audit(False)
    save(RECOVERY / 'STRICT_CONFIRMATION_AUDIT.json', integrity, exclusive=True)
    try:
        report = evaluate(False)
    except RuntimeError:
        path = BASE / 'analysis/VALIDATION_REPORT.json'
        if not path.exists():
            raise
        report = read(path)
        if report.get('accepted') is not False or report.get('role') != 'validation':
            raise
        print('NUMERICAL_ACCEPTANCE_FAILED_REPORT_PRESERVED', flush=True)
        exit_code = 2
    else:
        exit_code = 0
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch was loaded into the statistics process')
    print('ANALYSIS_RUNTIME_PATHS ' + json.dumps(runtime_paths()), flush=True)
    print(json.dumps(dict(accepted=report['accepted'], criteria=report['criteria'])), flush=True)
    return exit_code


def analyze():
    safe_environment()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    if (BASE / 'analysis/VALIDATION_REPORT.json').exists():
        raise FileExistsError('Validation report already exists; do not rerun statistics')
    lock = RECOVERY / 'ANALYSIS_STARTED.json'
    before = snapshot()
    save(lock, dict(pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
                    scope='confirmation analysis only; no measurement or test',
                    entry_sha256=sha(__file__), protected_snapshot=before), exclusive=True)
    state_path = RECOVERY / 'RECOVERY_STATE.json'
    state = dict(status='checking_frozen_protocol', measured_test_rows=0)
    save(state_path, state)
    try:
        check_result = child([BASE / 'scripts/runner.py', 'check'], 'isolated_frozen_check')
        if check_result['exit_code']:
            raise RuntimeError('Isolated frozen-protocol/CPU check failed')
        state['status'] = 'auditing_and_analyzing_confirmation'
        save(state_path, state)
        result = child([__file__, 'evaluate-worker'], 'isolated_statistics')
        report_path = BASE / 'analysis/VALIDATION_REPORT.json'
        if result['exit_code'] not in (0, 2) or not report_path.exists():
            raise RuntimeError('Isolated statistics runtime failed')
        report = read(report_path)
        if bool(report['accepted']) != (result['exit_code'] == 0):
            raise RuntimeError('Analysis exit/report disagreement')
        after = snapshot()
        if after != before:
            raise RuntimeError('Protected evidence changed during analysis')
        state.update(status='analysis_complete', acceptance=bool(report['accepted']),
                     ended_at=datetime.now().astimezone().isoformat(),
                     validation_report_sha256=sha(report_path),
                     numerical_result_status='passed' if report['accepted'] else 'failed',
                     independent_review='pending', test_not_started=True)
        save(RECOVERY / 'AFTER_ANALYSIS_SNAPSHOT.json', after, exclusive=True)
        save(state_path, state)
        print(json.dumps(state), flush=True)
    except Exception as error:
        state.update(status='analysis_recovery_failed', error=repr(error),
                     ended_at=datetime.now().astimezone().isoformat())
        save(state_path, state)
        raise


def export_worker():
    safe_environment()
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch must not be loaded for postprocessing')
    snapshot()
    from runner import audit
    report = read(BASE / 'analysis/VALIDATION_REPORT.json')
    if report['plan_sha256'] != EXPECTED_PLAN or report['model_sha256'] != sha(BASE / 'model/CONDITIONAL_MODEL.json'):
        raise RuntimeError('Validation report fingerprint mismatch')
    _, rows = audit(False)
    outputs = [BASE / 'analysis/v2_service_requests.csv',
               BASE / 'figures/fig_visual_service_v2.png', BASE / 'figures/fig_visual_service_v2.pdf']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Do not overwrite postprocessing output')
    # Some tool-launched Windows shells omit USERPROFILE/HOME. Keep plotting
    # configuration inside this experiment and change only this process.
    config = BASE / '.runtime_cache/matplotlib'
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(config))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, profile in zip(axes, PROFILES):
        summary = report['summary'][profile]
        keys = ['mean', 'p95', 'p99']
        x = np.arange(3)
        measured = [summary['measured_ms'][k] for k in keys]
        ax.bar(x - .18, [summary['predicted_ms'][k] for k in keys], .36,
               label='Frozen conditional prediction', color='#0072B2')
        ax.bar(x + .18, measured, .36, label='Measured full service', color='#D55E00')
        for i, value in enumerate(measured):
            ax.text(i + .18, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x, ['Mean', 'P95', 'P99'])
        ax.set_ylabel('Service time (ms)')
        ax.set_title('Confirmation: ' + profile)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    axes[0].legend(fontsize=8)
    fig.suptitle('Target-weighted confirmation only; retained test not measured', fontsize=10)
    fig.tight_layout()
    (BASE / 'figures').mkdir(exist_ok=True)
    fig.savefig(outputs[1], dpi=180)
    fig.savefig(outputs[2])
    plt.close(fig)
    csv_write(outputs[0], rows, exclusive=True)
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch loaded during postprocessing')
    save(RECOVERY / 'EXPORT.json', dict(rows=len(rows), scope='confirmation and reference only',
                                      runtime_paths=runtime_paths(),
                                      output_sha256={str(p.relative_to(BASE)): sha(p) for p in outputs}),
         exclusive=True)
    print('CONFIRMATION_EXPORT_OK', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'probe-worker', 'analyze', 'evaluate-worker', 'export-worker'])
    parser.add_argument('--kind', choices=['torch', 'numpy', 'mixed'])
    options = parser.parse_args()
    if options.action == 'probe':
        probe()
    elif options.action == 'probe-worker':
        if options.kind is None:
            parser.error('--kind is required')
        probe_worker(options.kind)
    elif options.action == 'analyze':
        analyze()
    elif options.action == 'evaluate-worker':
        sys.exit(evaluate_worker())
    else:
        export_worker()
