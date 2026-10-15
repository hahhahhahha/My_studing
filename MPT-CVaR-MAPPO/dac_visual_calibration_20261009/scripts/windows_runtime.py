"""Restore only missing Windows runtime paths in this process.

Some launchers strip SystemRoot and ProgramFiles. Winsock and NVML use
these paths even when the Python executable and CUDA DLLs are found.
No registry, persistent environment, package, or driver is modified.
"""
from __future__ import annotations

import os


def initialize_windows_runtime():
    if os.name != "nt":
        return {}
    import ctypes
    import winreg

    added = {}

    def missing(name, value):
        if not os.environ.get(name) and value:
            os.environ[name] = value
            added[name] = value

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    directory = kernel.GetWindowsDirectoryW
    directory.argtypes = (ctypes.c_wchar_p, ctypes.c_uint)
    directory.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32768)
    length = directory(buffer, len(buffer))
    if not length or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "Cannot resolve Windows directory")
    missing("SystemRoot", buffer.value)
    missing("windir", os.environ["SystemRoot"])

    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SOFTWARE\Microsoft\Windows\CurrentVersion", 0, access) as key:
        for name, field in (("ProgramFiles", "ProgramFilesDir"),
                            ("ProgramW6432", "ProgramW6432Dir")):
            if not os.environ.get(name):
                value, _ = winreg.QueryValueEx(key, field)
                missing(name, value)

    # TorchVision imports Dynamo, whose cache naming calls getpass.getuser().
    if not os.environ.get("USERNAME"):
        user = ctypes.WinDLL("advapi32", use_last_error=True).GetUserNameW
        user.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint))
        user.restype = ctypes.c_int
        size = ctypes.c_uint(len(buffer))
        if not user(buffer, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), "Cannot resolve Windows username")
        missing("USERNAME", buffer.value)

    # Keep missing temporary/cache paths inside this experiment's workspace.
    if not os.environ.get("TEMP") or not os.environ.get("TMP"):
        from pathlib import Path
        temporary = Path(__file__).resolve().parents[1] / ".runtime_cache" / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        missing("TEMP", str(temporary))
        missing("TMP", str(temporary))
    return added
