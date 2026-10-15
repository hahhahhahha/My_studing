"""Isolated environment provisioning and bounded CUDA witness; no training.

All attempts are additive. The base Anaconda environment is never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

from windows_runtime import initialize
initialize()

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
SPEC_PATH = BASE / 'protocol/QLA_ENVIRONMENT_SPEC.json'
# User-authorized occupied-memory policy revision; old spec/source preserved.
EXPECTED_SPEC = '8deaf4b6fbbf27332c795dcb5f08febde21269417f7879348e032643a9fe46f3'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def stamp():
    return datetime.now().astimezone().isoformat()


def guard():
    spec = read(SPEC_PATH)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != EXPECTED_SPEC:
        raise RuntimeError('Declared environment spec changed; no silent rebuild')
    state = read(ROOT / 'dac_visual_calibration_20261010_v2/protocol/RETAINED_TEST_STATE.json')
    if state['status'] not in ('complete_pending_independent_audit', 'complete_audited', 'failed'):
        raise RuntimeError('Service timings have not ended; do not provision/use CUDA')
    return spec


def attempt(action):
    folder = BASE / 'environment/attempts' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + action)
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / 'STARTED.json', dict(action=action, at=stamp(), pid=os.getpid(), spec_sha256=EXPECTED_SPEC))
    return folder


def invoke(command, folder, name):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    save(folder / (name + '_command.json'), command)
    with (folder / (name + '_stdout.log')).open('x', encoding='utf-8') as out, (folder / (name + '_stderr.log')).open('x', encoding='utf-8') as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; evidence: {folder}')


def build():
    spec = guard()
    folder = attempt('build')
    location = Path(spec['isolated_environment'])
    python = location / 'Scripts/python.exe'
    try:
        if '.'.join(map(str, sys.version_info[:3])) != spec['python']:
            raise RuntimeError('Wrong base Python')
        if not location.exists():
            venv.EnvBuilder(system_site_packages=True, with_pip=True).create(location)
        elif not python.exists() or not (location / 'pyvenv.cfg').exists():
            raise RuntimeError('Incomplete environment exists; preserve before repair')
        invoke([str(python), '-X', 'utf8', '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps', '--disable-pip-version-check', 'numpy==2.2.6'], folder, 'install_numpy')
        code = "import sys,importlib.metadata as m,json; print(json.dumps({'python':sys.version,'numpy':m.version('numpy'),'torch':m.version('torch'),'prefix':sys.prefix,'base_prefix':sys.base_prefix})); assert '.'.join(map(str,sys.version_info[:3]))=='3.13.5'; assert m.version('numpy')=='2.2.6'; assert m.version('torch')=='2.11.0+cu128'; assert sys.prefix!=sys.base_prefix"
        invoke([str(python), '-X', 'utf8', '-c', code], folder, 'metadata')
        save(folder / 'RESULT.json', dict(status='built_not_ready', at=stamp(), environment=str(location), spec_sha256=EXPECTED_SPEC,
                                         CUDA_witness=False, two_arm_smoke=False, fresh_agent_follows_doc=False, global_environment_changed=False))
        print('ISOLATED_ENV_BUILT_NOT_READY', folder, flush=True)
    except Exception as exc:
        save(folder / 'FAILURE.json', dict(error=repr(exc), at=stamp(), global_environment_changed=False))
        raise


def witness():
    spec = guard()
    folder = attempt('witness')
    python = Path(spec['isolated_environment']) / 'Scripts/python.exe'
    try:
        import psutil
        import shutil
        save(folder / 'HOST_RESOURCES.json', dict(at=stamp(), cpu_physical=psutil.cpu_count(logical=False), cpu_logical=psutil.cpu_count(),
                                                 memory_available_bytes=psutil.virtual_memory().available, disk_free_bytes=shutil.disk_usage(BASE).free))
        cli = Path(os.environ['SystemRoot']) / 'System32/nvidia-smi.exe'
        invoke([str(cli), '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'], folder, 'gpu_occupancy')
        rows = (folder / 'gpu_occupancy_stdout.log').read_text(encoding='utf-8').strip().splitlines()
        values = [item.strip() for item in rows[0].split(',')]
        if len(values) != 4 or values[0] != '0' or values[1] != spec['gpu'] or int(values[2]) >= spec['gpu_occupancy_max_mib_exclusive']:
            raise RuntimeError('GPU is not verified free; no competing allocation, killing or CPU fallback')
        code = f"import sys; sys.path.insert(0,{str(BASE / 'scripts')!r}); from windows_runtime import initialize; initialize(); import numpy as np, torch, json; assert '.'.join(map(str,sys.version_info[:3]))=='3.13.5'; assert np.__version__=='2.2.6'; assert torch.__version__=='2.11.0+cu128'; torch.set_num_threads(1); torch.set_num_interop_threads(1); assert torch.cuda.is_available(); assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4060 Ti'; torch.manual_seed(123); x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4); value=(x@x.T).sum().item(); assert value==3680.0; print(json.dumps({{'kernel_witness':'pass','value':value,'device':torch.cuda.get_device_name(0),'numpy':np.__version__,'torch':torch.__version__,'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads()}}),flush=True)"
        invoke([str(python), '-X', 'utf8', '-c', code], folder, 'cuda_kernel')
        save(folder / 'RESULT.json', dict(status='kernel_pass_not_formal_ready', at=stamp(), spec_sha256=EXPECTED_SPEC,
                                         two_arm_smoke=False, fresh_agent_follows_doc=False))
        print('CUDA_KERNEL_WITNESS_OK_NOT_FORMAL_READY', folder, flush=True)
    except Exception as exc:
        save(folder / 'FAILURE.json', dict(error=repr(exc), at=stamp(), training_started=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'witness'])
    action = parser.parse_args().action
    {'build': build, 'witness': witness}[action]()
