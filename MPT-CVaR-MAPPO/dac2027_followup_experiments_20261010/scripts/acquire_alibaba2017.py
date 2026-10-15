"""One-shot official archive acquisition AFTER ongoing timing ends.

No model run, data replacement, unsafe extraction or acceptance relaxation.
Partial/failed archives and logs are preserved. No automatic retry loop.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request

from windows_runtime import initialize
initialize()

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
FOLDER = BASE / 'data_acquisition'
STATE = FOLDER / 'DOWNLOAD_STATE.json'
LOCK = FOLDER / 'DOWNLOAD_STARTED.json'
URL = 'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz'
EXPECTED = '02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df'
EXPECTED_BYTES = 243321159  # successful approved HEAD on 2026-10-10
V2_STATE = ROOT / 'dac_visual_calibration_20261010_v2/protocol/RETAINED_TEST_STATE.json'


def stamp():
    return datetime.now().astimezone().isoformat()


def save(path, obj, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    FOLDER.mkdir(exist_ok=True)
    partial = FOLDER / 'alibaba-trace-2017.tar.gz.partial'
    archive = FOLDER / 'alibaba-trace-2017.tar.gz'
    extracted = FOLDER / 'server_usage.csv.verified_extract'
    target = ROOT / 'datasets/external_traces/extracted/alibaba2017/server_usage.csv'
    if any(p.exists() for p in [LOCK, STATE, partial, archive, extracted, target]):
        raise FileExistsError('Acquisition evidence already exists; preserve it, do not relaunch')
    import os
    save(LOCK, dict(pid=os.getpid(), at=stamp(), source_sha256=sha(Path(__file__)), url=URL), exclusive=True)
    state = dict(status='waiting_for_measurement', pid=os.getpid(), url=URL, bytes_downloaded=0,
                 service_times_changed=False, started_at=stamp())
    save(STATE, state)
    try:
        while True:
            v2 = json.loads(V2_STATE.read_text(encoding='utf-8'))
            if v2['status'] in ['complete_pending_independent_audit', 'complete_audited']:
                break
            if v2['status'] == 'failed':
                raise RuntimeError('Service run failed; preserve state and let authorized supervisor diagnose before acquisition')
            time.sleep(30)
        if shutil.disk_usage(FOLDER).free < EXPECTED_BYTES + 2 * 1024**3:
            raise RuntimeError('Insufficient reserve disk space')
        state.update(status='downloading', download_started_at=stamp()); save(STATE, state)
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('xb') as f:
            if response.status != 200 or int(response.headers.get('Content-Length', -1)) != EXPECTED_BYTES:
                raise RuntimeError('Official endpoint/length differs from approved probe; verify rather than substitute')
            state['headers'] = dict(response.headers.items())
            state['final_url'] = response.url
            last = time.monotonic()
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                f.write(block); state['bytes_downloaded'] += len(block)
                if state['bytes_downloaded'] > EXPECTED_BYTES:
                    raise RuntimeError('Download exceeded declared archive size')
                if time.monotonic() - last >= 10:
                    save(STATE, state); print('DOWNLOADED_BYTES', state['bytes_downloaded'], flush=True)
                    last = time.monotonic()
        state.update(status='verifying_archive', archive_sha256=sha(partial)); save(STATE, state)
        if partial.stat().st_size != EXPECTED_BYTES or state['archive_sha256'] != EXPECTED:
            raise RuntimeError('Archive does not match official trace_201708.tgz checksum; do not treat rename as verification')
        partial.rename(archive)
        state['status'] = 'extracting_server_usage_only'; save(STATE, state)
        with tarfile.open(archive, 'r:gz') as tar:
            candidates = [m for m in tar.getmembers() if Path(m.name.replace('\\', '/')).name == 'server_usage.csv']
            if len(candidates) != 1 or not candidates[0].isfile():
                raise RuntimeError('Expected one regular server_usage.csv; no arbitrary extraction')
            member = candidates[0]
            if shutil.disk_usage(FOLDER).free < member.size + 1024**3:
                raise RuntimeError('Insufficient space for verified member')
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError('Missing member stream')
            with stream, extracted.open('xb') as f:
                shutil.copyfileobj(stream, f, 1024 * 1024)
        if extracted.stat().st_size != member.size:
            raise RuntimeError('Extracted member length mismatch')
        if target.exists():
            raise FileExistsError('Target appeared during acquisition; preserve both')
        target.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(target)
        state.update(status='verified_download_complete_pending_schema_audit', ended_at=stamp(),
                     official_checksum_name='trace_201708.tgz', member_name=member.name,
                     server_usage_path=str(target), server_usage_bytes=target.stat().st_size,
                     server_usage_sha256=sha(target), model_tests_started=False)
        save(STATE, state); print(json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), ended_at=stamp())
        save(STATE, state)
        raise


if __name__ == '__main__':
    main()
