"""Synthetic, service-free adversarial scheduling check of the unchanged replay.

No performance evidence or entry approval is produced. A trace hook simulates
the sender thread being descheduled immediately after recording its response
time and before acquiring outcome_lock, until after cutoff finalization.
"""
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time

OUT=Path(__file__).resolve().parent
LAB=OUT.parents[1]
sys.path.insert(0,str(LAB))
from systemlab.common import source_receipt,write_new
from systemlab.common import local_windows_runtime
local_windows_runtime()
from unittest.mock import patch
from systemlab.client import replay
from systemlab.analysis import analyze

ROOT=OUT/'synthetic_cutoff_race_attempt_02'
ROOT.mkdir(exist_ok=False)
source_before=source_receipt()
# This local synthetic asset is never decoded and is not a real image run.
asset=ROOT/'NOT_A_JPEG_fixture.bin'
asset.write_bytes(b'SYNTHETIC_FIXTURE_ONLY')
manifest=ROOT/'manifest.json'
write_new(manifest,dict(schema='systemlab-images-v1',role='pilot',image_root=str(ROOT),
    images=[dict(image_id=asset.name,image_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),payload_bytes=asset.stat().st_size)]))
schedule=ROOT/'schedule.json'
write_new(schedule,dict(schema='systemlab-schedule-v1',role='pilot',users=1,seed=1,duration_s=.05,
    image_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    tasks=[dict(task_id='fixture_task',user_id=0,image_id=asset.name,arrival_s=.001,assignment_uniform=.5)]))
cfg=dict(users=1,seed=1,edges=[dict(id='fixture_edge',url='fixture://no-network')],
    feature_scales={},score_temperature_s=.05,controller_url='fixture://no-controller',
    feedback=dict(udp_port=0,injected_loss=0),feedback_start_timeout_s=1,max_inflight=1,
    fixed_drain_s=.05,decision_period_s=.01,client_timeout_s=1)

class FixtureRouter:
    receipt={'method':'SYNTHETIC_ONLY'}
    def __init__(self,*args): pass
    def portfolio(self,*args): return [1.],dict(actor_observation=[],expected_costs_s=[0],feedback_age_s=[0])
    def verify_frozen(self): return {'actor_executed':False,'synthetic_fixture':True}

class FixtureCache:
    def __init__(self,*args): pass
    def observation(self,*args): return {'fixture_edge':dict(source_epoch_ns=0,version=1)}

class FixtureSocket:
    def __init__(self,*args): pass
    def bind(self,*args): pass
    def settimeout(self,*args): pass
    def recvfrom(self,*args): time.sleep(.005); raise socket.timeout()
    def close(self): pass

def fake_request(url,**kwargs):
    if url.endswith('/health'): return 200,dict(ready=True,session='SYNTHETIC')
    return 200,dict(status='completed',task_id='fixture_task',server_id='fixture_edge',
        queue_s=0.,server_service_s=.001,preprocess_s=0.,compute_s=.001)

trace_events=[]
def trace(frame,event,arg):
    if (event=='line' and frame.f_code.co_name=='send_task' and
            Path(frame.f_code.co_filename).resolve()==LAB/'systemlab/client.py' and frame.f_lineno==120):
        trace_events.append(dict(event='pause_before_outcome_lock',response_end=frame.f_locals['end'],cutoff=frame.f_locals['cutoff']))
        end=time.monotonic()+2
        while not (ROOT/'client/CUTOFF.json').exists() and time.monotonic()<end: time.sleep(.005)
        trace_events.append(dict(event='resume_after_cutoff_file',cutoff_written=(ROOT/'client/CUTOFF.json').exists()))
    return trace

threading.settrace(trace)
try:
    with patch('systemlab.client.Router',FixtureRouter),patch('systemlab.client.FeedbackCache',FixtureCache),patch('systemlab.client.socket.socket',FixtureSocket),patch('systemlab.client.request_json',fake_request):
        replay(cfg,schedule,manifest,'score_softmax',ROOT/'client')
finally:
    threading.settrace(None)
outcomes=[json.loads(s) for s in (ROOT/'client/tasks.jsonl').read_text().splitlines()]
try:
    analyze(ROOT/'client',ROOT/'analysis_should_not_exist.json')
    analysis_rejection=None
except Exception as exc:
    analysis_rejection=repr(exc)
write_new(ROOT/'RESULT.json',dict(schema='independent-synthetic-adversarial-cutoff-check-v1',
    is_real_execution=False,performance_evidence=False,source_before=source_before,
    source_unchanged=source_before==source_receipt(),trace_events=trace_events,
    task_log_rows=len(outcomes),unique_task_ids=len({x['task_id'] for x in outcomes}),
    statuses=[x['status'] for x in outcomes],complete=json.loads((ROOT/'client/COMPLETE.json').read_text()),
    analyze_rejected=analysis_rejection))
print(json.dumps(dict(task_log_rows=len(outcomes),statuses=[x['status'] for x in outcomes],analysis_rejection=analysis_rejection)))
