from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value) + b'\n')


class Journal:
    def __init__(self, path):
        self.stream = Path(path).open('x', encoding='utf-8')
        self.lock = threading.Lock()

    def put(self, kind, **values):
        with self.lock:
            self.stream.write(encoded(dict(kind=kind, epoch_ns=time.time_ns(), **values)).decode() + '\n')
            self.stream.flush()

    def close(self):
        with self.lock:
            self.stream.close()


def request_json(url, *, body=None, headers=None, timeout=10):
    request = Request(url, data=body, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def runtime():
    value = dict(python=platform.python_version(), executable=os.sys.executable,
                 platform=platform.platform(), logical_cpus=os.cpu_count(), pid=os.getpid())
    for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            value[name] = path.read_text().strip()
    return value


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    low, high = math.floor(x), math.ceil(x)
    return values[low] + (values[high] - values[low]) * (x - low)


def tail_mean(values, alpha=.99):
    """Upper (1-alpha) empirical probability mass, with fractional boundary."""
    values = sorted(values, reverse=True)
    if not values:
        return None
    mass = len(values) * (1 - alpha)
    whole = int(math.floor(mass))
    total = math.fsum(values[:whole])
    if whole < len(values):
        total += (mass - whole) * values[whole]
    return total / mass


def source_receipt():
    """Executable system source only: package modules plus the delivered tools.

    Ad-hoc driver scripts (e.g. drive_scripts/) are deliberately excluded so
    that one-off orchestration helpers cannot invalidate a frozen protocol or
    an already-issued entry review.
    """
    root = Path(__file__).resolve().parent.parent
    tracked = [path for path in sorted((root / 'systemlab').rglob('*.py'))
               if '__pycache__' not in path.parts]
    tracked += [path for path in sorted((root / 'tools').rglob('*.py'))
                if '__pycache__' not in path.parts]
    return {str(path.relative_to(root)): sha(path) for path in tracked}


def local_windows_runtime():
    """Restore missing process variables before loading the existing Torch DLLs."""
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').upper() not in ('', '0', 'FALSE'):
        raise RuntimeError('Duplicate OpenMP override is forbidden')
    if os.name == 'nt':
        import ctypes
        import winreg
        buf = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetWindowsDirectoryW(buf, len(buf)):
            raise RuntimeError('Cannot resolve Windows directory')
        os.environ.setdefault('SystemRoot', buf.value)
        os.environ.setdefault('windir', buf.value)
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows\CurrentVersion') as key:
            for variable, field in [('ProgramFiles', 'ProgramFilesDir'), ('ProgramW6432', 'ProgramW6432Dir')]:
                os.environ.setdefault(variable, winreg.QueryValueEx(key, field)[0])
        length = ctypes.c_uint(len(buf))
        if not ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(length)):
            raise RuntimeError('Cannot resolve Windows username')
        os.environ.setdefault('USERNAME',buf.value)
        cache=Path(__file__).resolve().parent.parent/'.runtime_cache'
        cache.mkdir(exist_ok=True)
        os.environ.setdefault('TEMP',str(cache))
        os.environ.setdefault('TMP',str(cache))
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(cache/'torchinductor'))
