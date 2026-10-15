from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .client import replay
from .common import read, request_json, sha, write_new


def _wait_runtime_pid(pid, timeout_s):
    """Wait for the real service process (not the venv shim) to exit.

    On Windows the venv ``Scripts/python.exe`` launcher spawns the actual
    interpreter as a child, so ``Popen.pid`` is only the shim. The service
    identifies itself in RUNTIME.json and /health; those runtime PIDs are the
    authoritative liveness signal. Returns True if the PID exited in time.
    """
    if pid is None:
        return False  # unknown runtime PID: flag for manual inspection, never assume exit
    if timeout_s <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        synchronize = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            # Process no longer exists (or was never started); treat as exited.
            return True
        try:
            result = kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
            return result == 0  # WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False  # exists but is not ours to signal; treat as alive
        time.sleep(.05)
    return False


def _wait_shim(process, timeout_s):
    """Best-effort wait on the launcher shim handle; never fails the run."""
    try:
        process.wait(timeout=max(0., timeout_s))
    except subprocess.TimeoutExpired:
        pass
    return process.poll()


def local_run(config_path, config, schedule, manifest, method, output, bundle=None, entry_review=None):
    """Start only this attempt's processes; graceful authenticated shutdown."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    cwd = Path(__file__).resolve().parent.parent
    health_urls = []
    owned_services = []
    try:
        for node in config['edges']:
            health_urls.append(node['url'])
        health_urls.append(config['controller_url'])
        for url in health_urls:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.hostname not in ('localhost', '127.0.0.1'):
                raise ValueError('Local launcher requires loopback nodes')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', parsed.port))
        for node in config['edges']:
            launch = ['edge', '--server', node['id'], '--output', str(output / 'edges' / node['id'])]
            start_child(launch, node['id'], processes, streams, cwd, output, config_path)
            owned_services.append((node['url'], processes[-1], output / 'edges' / node['id'] / 'RUNTIME.json'))
        wait_ready(health_urls[:-1], processes, timeout=config.get('startup_timeout_s', 120))
        start_child(['controller', '--output', str(output / 'controller')], 'controller', processes, streams, cwd, output, config_path)
        owned_services.append((config['controller_url'], processes[-1], output / 'controller/RUNTIME.json'))
        wait_ready([config['controller_url']], processes, timeout=10)
        for url, process, runtime_path in owned_services:
            code, health = request_json(url + '/health', timeout=2)
            if code != 200 or process.poll() is not None or health.get('pid') != read(runtime_path)['pid']:
                raise RuntimeError('Ready endpoint is not the recorded runtime of this owned child')
        write_new(output / 'PROCESSES.json', dict(config_sha256=sha(config_path),
            children=[dict(pid=p.pid, command=p.args, runtime_pid=read(path)['pid'])
                      for url, p, path in owned_services], owned_by_launcher_pid=os.getpid()))
        replay(config, schedule, manifest, method, output / 'client', bundle, entry_review)
    except Exception as error:
        write_new(output / 'LAUNCH_FAILURE.json', dict(error=repr(error), automatic_retry=False,
                                                       child_pids=[p.pid for p in processes]))
        raise
    finally:
        # Every shutdown step is recorded per service; nothing is swallowed.
        shutdown_records = _graceful_shutdown(owned_services, config)
        shutdown_timeout_s = config.get('shutdown_timeout_s', config['server_response_timeout_s'] + 5)
        deadline = time.perf_counter() + shutdown_timeout_s
        for record in shutdown_records:
            remaining = max(0., deadline - time.perf_counter())
            record['runtime_exited'] = _wait_runtime_pid(record['runtime_pid'], remaining)
            record['shim_exit_code'] = _wait_shim(record['shim_process'], 5.)
            del record['shim_process']
        alive_runtime = [record['runtime_pid'] for record in shutdown_records
                         if not record['runtime_exited']]
        lingering_shims = [record['shim_pid'] for record in shutdown_records
                           if record['shim_exit_code'] is None and record['runtime_exited']]
        write_new(output / 'PROCESS_END.json', dict(
            known_children_still_alive=alive_runtime,
            lingering_launcher_shims=lingering_shims,
            services=shutdown_records,
            exit_codes={str(record['shim_pid']): record['shim_exit_code'] for record in shutdown_records},
            normal_shutdown_requested=True, forced_termination=False,
            liveness_basis='runtime_pid_from_RUNTIME_json'))
        for stream in streams:
            stream.close()
        if alive_runtime:
            raise RuntimeError(f'Owned runtime processes need manual inspection: {alive_runtime}; logs preserved')


def _graceful_shutdown(owned_services, config):
    """Request authenticated shutdown of each owned service, recording every step."""
    records = []
    for url, process, runtime_path in reversed(owned_services):
        record = dict(url=url, shim_pid=process.pid, shim_process=process,
                      runtime_pid=None, shutdown_requested=False)
        try:
            record['runtime_pid'] = read(runtime_path)['pid']
        except Exception as error:
            record['runtime_pid_error'] = repr(error)
        try:
            code, health = request_json(url + '/health', timeout=2)
            record['health'] = dict(code=code, pid=health.get('pid'))
        except Exception as error:
            record['health'] = dict(error=repr(error))
        if (record['runtime_pid'] is not None and process.poll() is None and
                record['health'].get('code') == 200 and
                record['health'].get('pid') == record['runtime_pid']):
            try:
                code, response = request_json(url + '/shutdown', body=b'',
                    headers={'X-Control-Token': config['control_token']}, timeout=2)
                record['shutdown_requested'] = True
                record['shutdown_response'] = dict(code=code, response=response)
            except Exception as error:
                record['shutdown_response'] = dict(error=repr(error))
        records.append(record)
    return records


def start_child(args, name, processes, streams, cwd, output, config):
    stdout = (output / f'{name}.stdout.log').open('xb')
    stderr = (output / f'{name}.stderr.log').open('xb')
    streams.extend((stdout, stderr))
    command = [sys.executable, '-m', 'systemlab', *args, '--config', str(Path(config).resolve())]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    processes.append(subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, creationflags=flags))


def wait_ready(urls, processes, timeout):
    deadline = time.perf_counter() + timeout
    pending = set(urls)
    while pending:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('An owned service exited before readiness; inspect stderr')
        for url in list(pending):
            try:
                code, value = request_json(url + '/health', timeout=.5)
                if code == 200 and value['ready']:
                    pending.remove(url)
            except Exception:
                pass
        if time.perf_counter() >= deadline:
            raise RuntimeError(f'No healthy service: {pending}')
        time.sleep(.05)
