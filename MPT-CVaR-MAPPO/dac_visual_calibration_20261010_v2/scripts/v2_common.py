from __future__ import annotations
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OLD = BASE.parent / 'dac_visual_calibration_20261009'
SEED = 20261010
PROFILES = ('CPU-1T', 'CPU-4T')
STAGES = ('decode_ms', 'preprocess_ms', 'infer_ms', 'postprocess_ms')
BOUNDARY = 'decode+RGB+preprocess+FP32 inference+argmax+dict; no disk/network/queue/controller'

def init_runtime():
    if os.name != 'nt':
        return
    import winreg
    buf = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetWindowsDirectoryW(buf, len(buf)):
        raise RuntimeError('Cannot resolve Windows directory')
    os.environ.setdefault('SystemRoot', buf.value)
    os.environ.setdefault('windir', buf.value)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows\CurrentVersion',
                        0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
        for name, field in [('ProgramFiles', 'ProgramFilesDir'), ('ProgramW6432', 'ProgramW6432Dir')]:
            os.environ.setdefault(name, winreg.QueryValueEx(key, field)[0])
    n = ctypes.c_uint(len(buf))
    if not ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(n)):
        raise RuntimeError('Cannot resolve username')
    os.environ.setdefault('USERNAME', buf.value)
    tmp = BASE / '.runtime_cache/tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TEMP', str(tmp))
    os.environ.setdefault('TMP', str(tmp))

init_runtime()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def csv_read(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def csv_write(path, rows, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def stratum(e):
    pixels = int(e['width']) * int(e['height'])
    return next((i for i, b in enumerate([250000, 500000, 1000000, 2000000]) if pixels <= b), 4)

def stem(job):
    return f"{job['role']}_s{job['session']}_b{job['block']}_{job['profile']}"
