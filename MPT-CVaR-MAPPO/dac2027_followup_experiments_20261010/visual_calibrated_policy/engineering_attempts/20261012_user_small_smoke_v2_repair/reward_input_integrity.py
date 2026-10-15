"""Pure pre-import closure and consumed-input guards; no model imports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
import json
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


def require_review():
    path = HERE/'independent_repair_rereview/REVIEW_RESULT.json'
    report = HERE/'independent_repair_rereview/REVIEW_RESPONSE.md'
    review = read(path)
    if review.get('status') not in ('PASS', 'WARN') or review.get('blockers') != [] or review.get('review_type') != 'Type-A':
        raise RuntimeError('Complete new-scope source review without blockers required')
    if review.get('reviewed_sources_sha256') != own_source_map() or not report.is_file():
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
