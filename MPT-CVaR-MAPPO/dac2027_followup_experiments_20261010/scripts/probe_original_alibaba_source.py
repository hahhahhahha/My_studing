"""Bounded read-only provenance probes; never relax the archive checksum."""
from datetime import datetime
import json
from pathlib import Path
import urllib.error
import urllib.request
from windows_runtime import initialize
initialize()

BASE = Path(__file__).resolve().parents[1]
URLS = [
    'https://clusterdata.oss-cn-beijing.aliyuncs.com/trace_201708.tgz',
    'https://clusterdata-us.oss-us-west-1.aliyuncs.com/trace_201708.tgz',
    'https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/trace_201708.tgz',
]

if __name__ == '__main__':
    results = []
    for url in URLS:
        try:
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request, timeout=15) as response:
                result = dict(url=url, status=response.status, final_url=response.url, headers=dict(response.headers))
        except Exception as exc:
            result = dict(url=url, error=repr(exc), status=getattr(exc, 'code', None))
        results.append(result)
    report = dict(at=datetime.now().astimezone().isoformat(), method='HEAD only', source_gate_changed=False,
                  original_worker_restarted=False, archive_extracted=False, results=results)
    path = BASE / 'data_acquisition' / ('original_source_probe_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print('PROBE_EVIDENCE', path, flush=True)
