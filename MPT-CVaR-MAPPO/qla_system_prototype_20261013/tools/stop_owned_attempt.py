"""Gracefully close only an attempt's runtime-PID-verified services."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import local_windows_runtime,read,request_json,sha,write_new

p=argparse.ArgumentParser()
p.add_argument('--attempt',required=True)
p.add_argument('--config',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
local_windows_runtime()
root=Path(a.attempt).resolve()
receipt=read(root/'PROCESSES.json')
if sha(a.config)!=receipt['config_sha256']:
    raise ValueError('Attempt/config binding mismatch')
cfg=read(a.config)
services=[(cfg['controller_url'],root/'controller/RUNTIME.json')]+[
    (e['url'],root/'edges'/e['id']/'RUNTIME.json') for e in cfg['edges']]
results=[]
for url,path in services:
    record=read(path)
    try:
        code,health=request_json(url+'/health',timeout=2)
        if code!=200 or health.get('pid')!=record['pid']:
            raise RuntimeError('Health does not identify the exact recorded runtime process; no closure')
        code,response=request_json(url+'/shutdown',body=b'',
            headers={'X-Control-Token':cfg['control_token']},timeout=2)
        results.append(dict(url=url,runtime_pid=record['pid'],http_status=code,response=response))
    except Exception as error:
        results.append(dict(url=url,runtime_pid=record['pid'],error=repr(error)))
write_new(a.output,dict(attempt=str(root),services=results,forced_termination=False,
    only_recorded_runtime_pids=True,normal_shutdown_only=True))
print(results)
