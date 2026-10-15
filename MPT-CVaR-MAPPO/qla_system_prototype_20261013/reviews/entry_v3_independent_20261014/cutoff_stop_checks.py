"""Controlled no-network correctness fixtures; never performance/entry approval."""
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

OUT=Path(__file__).resolve().parent
LAB=OUT.parents[1]
sys.path.insert(0,str(LAB))
spec=importlib.util.spec_from_file_location('reviewed_cutoff_tests',LAB/'tests/test_cutoff_race.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
from systemlab.client import replay
from systemlab.common import write_new,read
from systemlab.analysis import analyze

results=[]
for kind in ('before_cutoff_paused','normal','late','server_timeout','transport_timeout','early_exception','keyboard_interrupt'):
    root=OUT/'controlled_cutoff_attempt_01'/kind
    root.mkdir(parents=True,exist_ok=False)
    cfg,schedule,manifest,request=f._setup(root,response_delay_s=.2 if kind=='late' else 0.)
    observed=dict(pause_reached=False,resumed_after_cutoff_file=False)
    gate=threading.Event()
    if kind in ('early_exception','keyboard_interrupt'):
        s=read(schedule); s['duration_s']=.5
        s['tasks'].append(dict(s['tasks'][0],task_id='unsent_fixture',arrival_s=.03))
        alt=root/'two_task_schedule.json'; write_new(alt,s); schedule=alt
    class Router(f.FixtureRouter):
        calls=0
        def portfolio(self,*args):
            self.calls+=1
            if self.calls==2:
                if kind=='early_exception': raise ValueError('INJECTED controlled early main-thread error')
                if kind=='keyboard_interrupt': raise KeyboardInterrupt('INJECTED user-stop-equivalent KeyboardInterrupt')
            return super().portfolio(*args)
    def fake_request(url,**kwargs):
        if url.endswith('/health'): return 200,dict(ready=True,session='SYNTHETIC')
        if kind in ('early_exception','keyboard_interrupt'): time.sleep(.12)
        if kind=='transport_timeout': raise TimeoutError('INJECTED client transport timeout')
        if kind=='server_timeout':
            return 504,dict(status='server_response_timeout',task_id='fixture_task',server_id='fixture_edge')
        return request(url,**kwargs)
    def trace(frame,event,arg):
        if event=='line' and frame.f_code.co_name=='send_task' and 'end' in frame.f_locals and not observed['pause_reached']:
            observed['pause_reached']=True
            observed['response_end_perf']=frame.f_locals['end']
            observed['cutoff_perf']=frame.f_locals['cutoff']
            limit=time.monotonic()+2
            while not (root/'client/CUTOFF.json').exists() and time.monotonic()<limit: time.sleep(.002)
            observed['resumed_after_cutoff_file']=(root/'client/CUTOFF.json').exists()
        return trace
    caught=None; t0=time.perf_counter()
    try:
        if kind=='before_cutoff_paused': threading.settrace(trace)
        with patch('systemlab.client.Router',Router),patch('systemlab.client.FeedbackCache',f.FixtureCache),\
             patch('systemlab.client.socket.socket',f.FixtureSocket),patch('systemlab.client.request_json',fake_request):
            replay(cfg,schedule,manifest,'score_softmax',root/'client')
    except BaseException as error:
        caught=dict(type=type(error).__name__,message=str(error))
    finally:
        threading.settrace(None)
    elapsed=time.perf_counter()-t0
    tasks=[json.loads(x) for x in (root/'client/tasks.jsonl').read_text().splitlines()]
    late=[json.loads(x) for x in (root/'client/late_responses.jsonl').read_text().splitlines()]
    analysis_error=None
    try: analysis=analyze(root/'client',root/'analysis.json')
    except Exception as error: analysis={}; analysis_error=repr(error)
    value=dict(case=kind,synthetic_fixture=True,performance_evidence=False,caught=caught,
        elapsed_wall_s=elapsed,registered_horizon_s=read(schedule)['duration_s'],fixed_drain_s=cfg['fixed_drain_s'],
        outcome_rows=len(tasks),unique_task_ids=len({x['task_id'] for x in tasks}),
        statuses=[x['status'] for x in tasks],late_rows=len(late),complete=read(root/'client/COMPLETE.json'),
        cutoff=read(root/'client/CUTOFF.json'),analysis_infrastructure_valid=analysis.get('infrastructure_valid'),
        analysis_error=analysis_error,trace_observation=observed)
    write_new(root/'RESULT.json',value); results.append(value)
write_new(OUT/'CUTOFF_STOP_RESULTS.json',results)
print(json.dumps(results,indent=2))
