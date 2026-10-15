"""Process-local Windows path restoration, not system/driver/network repair."""
import ctypes
import os
from pathlib import Path


def initialize():
    if os.environ.get('KMP_DUPLICATE_LIB_OK', '').upper() not in ('', '0', 'FALSE'):
        raise RuntimeError('Unsafe duplicate OpenMP override is not allowed')
    if os.name != 'nt':
        return
    import winreg
    buf = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetWindowsDirectoryW(buf, len(buf)):
        raise RuntimeError('Cannot resolve Windows path')
    os.environ.setdefault('SystemRoot', buf.value)
    os.environ.setdefault('windir', buf.value)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows\CurrentVersion',
                        0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
        for name, field in [('ProgramFiles', 'ProgramFilesDir'), ('ProgramW6432', 'ProgramW6432Dir')]:
            os.environ.setdefault(name, winreg.QueryValueEx(key, field)[0])
    n = ctypes.c_uint(len(buf))
    if not ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(n)):
        raise RuntimeError('Cannot resolve Windows username')
    os.environ.setdefault('USERNAME', buf.value)
    cache = Path(__file__).resolve().parents[1] / '.runtime_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TEMP', str(cache))
    os.environ.setdefault('TMP', str(cache))
    os.environ.setdefault('PIP_CACHE_DIR', str(cache / 'pip'))
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
