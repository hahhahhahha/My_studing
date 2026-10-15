"""Pure pre-import closure and consumed-input guards; no model imports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
ROOT = BASE.parent
OLD = ROOT/'results_qla_joint_reward_ablation_20261010_repaired'
SOURCE = OLD/'execution_source'
RAW = ROOT/'Traces_TimeXY_10sec_txt/KAIST'
SCHEDULE_ROOT = HERE/'schedules'
BASELINE_SHA = '742d529b7f02301734a24f70de42e540a4b4ad5080870179762846174cf1387d'
OWN_FILES = ('visual_model_smoke_small.py', 'reward_smoke_small.py', 'reward_input_integrity.py',
             'FROZEN_BASELINE.json', 'README_SMALL_SMOKE_中文.md', 'USER_REQUEST.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def require_map(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise RuntimeError(f'Protected input/source changed: {path}')


def package_digest(folder):
    result = hashlib.sha256()
    for path in sorted(Path(folder).glob('*.py')):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def check_raw(mapping, folder=RAW):
    # Match precisely the original loader's sorted-first30 rule. New earlier
    # files or renamed members must not change consumption silently.
    selected = sorted(Path(folder).glob('*.txt'))[:30]
    actual = {str(path.resolve()): sha(path) for path in selected}
    expected = {str(Path(path).resolve()): value for path, value in mapping.items()}
    if len(selected) != 30 or actual != expected:
        raise RuntimeError('Original consumed KAIST membership/bytes changed')
    return actual


def actual_environment(spec):
    # Validate this process without importing NumPy/Torch or allocating CUDA.
    isolated = Path(spec['isolated_environment']).resolve()
    if Path(sys.executable).resolve() != isolated/'Scripts/python.exe':
        raise RuntimeError('Actual caller/worker must use the declared isolated interpreter')
    if '.'.join(map(str, sys.version_info[:3])) != spec['python']:
        raise RuntimeError('Actual Python version differs from the fixed specification')
    for package in ('numpy', 'torch'):
        if importlib.metadata.version(package) != spec[package]:
            raise RuntimeError(f'Actual {package} distribution version differs from specification')
    numpy_folder = Path(importlib.metadata.distribution('numpy').locate_file('numpy')).resolve()
    if not numpy_folder.is_relative_to(isolated):
        raise RuntimeError('Actual NumPy must be owned by the isolated environment')
    return {'python_executable': str(Path(sys.executable).resolve()),
        'python': spec['python'], 'numpy': spec['numpy'], 'torch': spec['torch'],
        'numpy_distribution_folder': str(numpy_folder)}


def parent_creation_time():
    import psutil
    return psutil.Process(os.getpid()).create_time()


def require_worker_parent(parent, expected_command):
    # Windows venv python.exe redirects through one live launcher. Admit that
    # exact bounded chain, not arbitrary ancestors or an unverified PID.
    source_guard()
    import psutil
    isolated_python = (Path(read(BASE/'protocol/QLA_ENVIRONMENT_SPEC.json')['isolated_environment'])/'Scripts/python.exe').resolve()
    current = psutil.Process(os.getpid())
    direct = current.parent()
    if direct is None or direct.pid != os.getppid():
        raise RuntimeError('Actual direct parent is absent or inconsistent')
    if len(expected_command) < 5 or Path(expected_command[0]).resolve() != isolated_python or expected_command[1:4] != ['-B', '-X', 'utf8'] or expected_command[4:] != sys.argv:
        raise RuntimeError('Exact isolated worker invocation required')
    actual_worker_command = current.cmdline()
    if not actual_worker_command or Path(actual_worker_command[0]).resolve() != isolated_python or actual_worker_command[1:] != expected_command[1:]:
        raise RuntimeError('Actual OS worker command differs from the canonical invocation')
    launcher = None
    logical = direct
    if direct.pid != parent['parent_pid']:
        if os.name != 'nt' or Path(direct.exe()).resolve() != isolated_python:
            raise RuntimeError('Only the declared Windows venv launcher may intervene')
        actual_command = direct.cmdline()
        if not actual_command or Path(actual_command[0]).resolve() != isolated_python or actual_command[1:] != expected_command[1:]:
            raise RuntimeError('Intervening launcher command differs from exact worker command')
        launcher = direct
        logical = direct.parent()
    if logical is None or logical.pid != parent['parent_pid'] or logical.create_time() != parent['parent_creation_time']:
        raise RuntimeError('Recorded logical parent is absent, replaced or not directly above launcher')
    command = logical.cmdline()
    if len(command) < 6 or Path(command[0]).resolve() != isolated_python or command[1:4] != ['-B', '-X', 'utf8'] or Path(command[4]).resolve() != Path(sys.argv[0]).resolve() or command[5] != 'run':
        raise RuntimeError('Logical parent is not the exact entry run invocation')
    return {'logical_parent_pid': logical.pid, 'logical_parent_creation_time': logical.create_time(),
        'direct_parent_pid': direct.pid, 'venv_launcher_pid': launcher.pid if launcher else None,
        'actual_worker_pid': current.pid, 'launched_pid': launcher.pid if launcher else current.pid,
        'worker_command': expected_command, 'actual_OS_worker_command': actual_worker_command,
        'bounded_verified_launcher_chain': True}


def validate_worker_parent_receipt(receipt, launched_pid, parent):
    if (receipt['logical_parent_pid'] != parent['parent_pid']
            or receipt['logical_parent_creation_time'] != parent['parent_creation_time']
            or receipt['launched_pid'] != launched_pid
            or receipt['bounded_verified_launcher_chain'] is not True):
        raise RuntimeError('Actual worker/launcher receipt differs from parent launch')


def source_guard():
    path = HERE/'FROZEN_BASELINE.json'
    if sha(path) != BASELINE_SHA:
        raise RuntimeError('Frozen reviewed baseline changed; no reseal')
    baseline = read(path)
    require_map(baseline['immutable_file_sha256'])
    closure = baseline['execution_source_file_sha256']
    expected_members = {str(Path(p).resolve()) for p in closure}
    if {str(p.resolve()) for p in SOURCE.rglob('*.py')} != expected_members:
        raise RuntimeError('Frozen executable closure membership changed')
    require_map(closure)
    for path, entry in baseline['authorized_resource_replacements'].items():
        if sha(entry['before_path']) != entry['old_sha256'] or sha(path) != entry['current_sha256']:
            raise RuntimeError('Only exact human-authorized resource revision accepted')
    old_binding = read(OLD/'ADDITIVE_ENTRY_BINDING.json')
    replacements = {str(Path(p).resolve()): v for p, v in baseline['authorized_resource_replacements'].items()}
    for p, old_sha in old_binding['files'].items():
        revision = replacements.get(str(Path(p).resolve()))
        if revision:
            if old_sha != revision['old_sha256']:
                raise RuntimeError('Original binding resource provenance mismatch')
        elif sha(p) != old_sha:
            raise RuntimeError('Original entry source/doc changed')
    v2 = read(OLD/'ADDITIVE_V2_BINDING.json')
    require_map(v2['files'])
    if v2['original_binding_sha256'] != sha(OLD/'ADDITIVE_ENTRY_BINDING.json'):
        raise RuntimeError('Original V2 binding changed')
    design = read(OLD/'DESIGN.json')
    lock = design['source_lock']
    if package_digest(SOURCE/'mappo_gp_sf_dsr') != lock['source_sha256'] or package_digest(SOURCE/'repro_risk_averse_edge') != lock['repro_source_sha256']:
        raise RuntimeError('Both frozen package digests required before import')
    helpers = {'repair_worker_sha256': SOURCE/'paper_doc_tools/paired_training_repair_worker.py',
        'repair_helper_sha256': SOURCE/'paper_doc_tools/paired_training_repair_core.py',
        'runner_sha256': ROOT/'paper_doc_tools/run_joint_reward_ablation.py',
        'intervention_core_sha256': ROOT/'paper_doc_tools/joint_reward_ablation_core.py',
        'intervention_worker_sha256': ROOT/'paper_doc_tools/joint_reward_ablation_worker.py',
        'tests_sha256': ROOT/'paper_doc_tools/test_joint_reward_ablation.py'}
    if any(sha(p) != lock[k] for k,p in helpers.items()):
        raise RuntimeError('Frozen executable/helper bytes changed before import')
    check_raw(baseline['consumed_KAIST_first30_file_sha256'])
    if len(design['Full_references']) != 25:
        raise RuntimeError('Incomplete historical Full registry')
    for item in design['Full_references']:
        if sha(item['checkpoint']) != item['checkpoint_sha256'] or sha(item['config']) != item['config_sha256']:
            raise RuntimeError('Historical reference changed')
    spec = read(BASE/'protocol/QLA_ENVIRONMENT_SPEC.json')
    if digest(spec) != baseline['environment_spec_canonical_sha256'] or spec['gpu_occupancy_max_mib_exclusive'] != 1800:
        raise RuntimeError('Exact current environment policy required')
    actual_environment(spec)
    return baseline, design


def own_source_map():
    paths = [HERE/name for name in OWN_FILES] + [BASE/'scripts/qla_environment.py',
        BASE/'scripts/windows_runtime.py', BASE/'protocol/QLA_ENVIRONMENT_SPEC.json']
    return {str(path): sha(path) for path in paths}


def require_review(review_directory=None, *, reviewed_source_map=None, protected_map=None):
    if review_directory is None:
        directory = HERE/'independent_command_rereview'
        visual_review = False
    else:
        directory = Path(review_directory).resolve()
        allowed = (HERE.parent/'20261013_20mhz_receiver_repair/independent_repair_rereview_revision_3').resolve()
        if directory != allowed:
            raise ValueError('Only the separately designated 20 MHz repair review may authorize this visual entry')
        visual_review = True
    path = directory/'REVIEW_RESULT.json'
    report = directory/'REVIEW_RESPONSE.md'
    if path.is_symlink() or report.is_symlink():
        raise RuntimeError('Independent review artifacts must not be redirected through symlinks')
    review = read(path)
    if review.get('status') not in ('PASS', 'WARN') or review.get('blockers') != [] or review.get('review_type') != 'Type-A':
        raise RuntimeError('Complete new-scope source review without blockers required')
    if visual_review:
        # The 20 MHz visual entry is admitted only by the complete review maps,
        # not by a freshly regenerated plan or the smaller own/resource map.
        # Recheck the helper's own pins are represented in the caller's map so
        # a caller cannot supply an unrelated 209-entry dictionary.
        if (type(reviewed_source_map) is not dict or len(reviewed_source_map) != 209
                or type(protected_map) is not dict or len(protected_map) != 210):
            raise RuntimeError('The exact complete 209-source and 210-protected maps are required')
        own = own_source_map()
        if (not set(own).issubset(reviewed_source_map)
                or any(reviewed_source_map[name] != value for name, value in own.items())):
            raise RuntimeError('Complete source map does not include the current own/resource bytes')
        if (review.get('reviewed_sources_sha256') != reviewed_source_map
                or review.get('protected_input_and_source_sha256') != protected_map
                or not report.is_file()):
            raise RuntimeError('Independent review must bind exact complete current source and protected maps')
    elif review.get('reviewed_sources_sha256') != own_source_map() or not report.is_file():
        # Preserve the reward smoke's historical review contract; this change
        # does not recertify or re-route that independent branch.
        raise RuntimeError('Independent review must bind exact current own/resource sources')
    return {'JSON': sha(path), 'MD': sha(report)}


def load_frozen(path, name):
    # No frozen project executable is dynamically imported before full guard.
    source_guard()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    source_guard()
    return module


def schedule_records():
    # No creation/replacement on any runtime path. Exact whole membership and
    # byte hashes catch an orphan counterpart before the old generator runs.
    registry = read(HERE/'SCHEDULE_REGISTRY.json')
    files = registry['files_sha256']
    if {str(p.resolve()) for p in SCHEDULE_ROOT.rglob('*') if p.is_file()} != {str(Path(p).resolve()) for p in files}:
        raise RuntimeError('Schedule membership changed or partial artifact exists')
    require_map(files)
    return registry


def build_schedules_once():
    baseline, _ = source_guard()
    if SCHEDULE_ROOT.exists() or (HERE/'SCHEDULE_REGISTRY.json').exists():
        raise FileExistsError('Preserve partial/existing schedules; no regeneration')
    SCHEDULE_ROOT.mkdir(exist_ok=False)
    import numpy as np
    raw = [np.loadtxt(path, dtype=np.float64) for path in sorted(RAW.glob('*.txt'))[:30]]
    n_trace_slots = min(len(value) for value in raw)
    source_guard()
    module = load_frozen(SOURCE/'mappo_gp_sf_dsr/canonical_protocol.py', 'small_v2_canonical_registration')
    records = []
    for role, seed, steps in (('train', 1, 32), ('validation', 10001, 8)):
        result = module.build_or_load_schedule(root=SCHEDULE_ROOT, dataset='KAIST',
            distribution='poisson_bursty', role=role, seed=seed, n_steps=steps,
            n_trace_slots=n_trace_slots, n_users=30, n_edges=35, max_requests=7,
            max_feedback_delay=8, broadcast_period=2, poisson_lambda=3.5,
            burst_prob=.1, burst_multiplier=2.)
        records.append({'role': role, 'seed': seed, 'steps': steps, 'path': str(result.path),
            'content_sha256': result.content_sha256, 'schedule_id': result.schedule_id})
    files = {str(p): sha(p) for p in sorted(SCHEDULE_ROOT.rglob('*')) if p.is_file()}
    if len(files) != 4:
        raise RuntimeError('Exactly two complete NPZ/manifest pairs required')
    save(HERE/'SCHEDULE_REGISTRY.json', {'schema': 'original-formula-small-schedules-v2',
        'root': str(SCHEDULE_ROOT), 'records': records, 'files_sha256': files,
        'n_trace_slots': n_trace_slots, 'old_shared_schedule_root_written': False,
        'source_and_consumed_raw_baseline_sha256': BASELINE_SHA})
    source_guard()
    return schedule_records()
