"""Review-only read/compute audit. No source, prior output, or scientific setting writes."""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
LAB = OUT.parents[1]
REPO = LAB.parent
sys.path.insert(0, str(LAB))
from systemlab.common import local_windows_runtime, source_receipt

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def new(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def quant(values, p):
    a=sorted(values)
    if not a: return None
    x=(len(a)-1)*p; j=int(x)
    return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(x-j)

def tail(values):
    a=sorted(values,reverse=True); mass=len(a)/100
    if not a: return None
    whole=int(mass)
    return (math.fsum(a[:whole])+(mass-whole)*a[whole])/mass

started=time.time_ns()
source0=source_receipt()
run_names=['M1_rr_s1_attempt_01','M1_stress_s1_attempt_01','M2c_qla_s101_attempt_01',
           'M2c_score_softmax_s101_attempt_01','M2c_stale_shortest_s101_attempt_01']
run_dirs=[LAB/'runs'/n for n in run_names]
run_dirs += [LAB/'engineering_repairs/20261013_three_logic_fixes/functional_4edge_attempt_02']
inventory={}
roots=run_dirs+[LAB/'interference_calibration', LAB/'interference_calibration_affinity_attempt_01_broken',
               LAB/'systemlab',LAB/'tools',LAB/'tests',LAB/'configs',LAB/'docs',LAB/'drive_scripts',LAB/'protocol',LAB/'data']
paths=set(p for r in roots for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update([LAB/'README.md',LAB/'requirements.txt'])
for p in sorted(paths):
    raw=p.read_bytes()
    record={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if p.suffix=='.json':
        obj=json.loads(raw.decode('utf-8-sig')); record['json_type']=type(obj).__name__
    elif p.suffix=='.jsonl':
        row=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
        record['rows']=len(row); record['kinds']=dict(collections.Counter(x.get('kind') for x in row))
    elif p.suffix in ('.py','.md','.log','.txt'):
        record['text_lines']=len(raw.decode('utf-8-sig').splitlines())
        if p.suffix=='.log' and raw: record['text']=raw.decode('utf-8-sig')
    inventory[str(p.relative_to(LAB))]=record
new('READ_INVENTORY.json',inventory)

bindings=[]
def visit(obj, origin):
    if isinstance(obj,dict):
        if isinstance(obj.get('path'),str) and isinstance(obj.get('sha256'),str):
            p=Path(obj['path']); actual=digest(p) if p.is_file() else None
            bindings.append({'origin':origin,'path':str(p),'expected':obj['sha256'],'actual':actual,'pass':actual==obj['sha256']})
        for k,v in obj.items():
            if k.endswith('_sha256') and isinstance(v,dict):
                for path, expected in v.items():
                    p=Path(path)
                    if p.is_absolute() and isinstance(expected,str):
                        actual=digest(p) if p.is_file() else None
                        bindings.append({'origin':origin+'/'+k,'path':str(p),'expected':expected,'actual':actual,'pass':actual==expected})
            visit(v,origin+'/'+k)
    elif isinstance(obj,list):
        for j,x in enumerate(obj): visit(x,origin+f'/{j}')
for rel in ['protocol/frozen_v2/FROZEN_SYSTEM_PROTOCOL.json','protocol/E1_test_registry_v2.json','protocol/formal_actor_v2.json','protocol/calibration_scales_v1.json']:
    visit(load(LAB/rel),rel)
actor=load(LAB/'protocol/formal_actor_v2.json')
historical={str(p.relative_to(actor['source_root'])):digest(p) for p in sorted(Path(actor['source_root']).rglob('*.py'))}
bindings += [{'origin':'actor checkpoint','path':actor['checkpoint'],'expected':actor['checkpoint_sha256'],
              'actual':digest(actor['checkpoint']),'pass':digest(actor['checkpoint'])==actor['checkpoint_sha256']}]
cfg=load(LAB/'protocol/frozen_v2/FORMAL_CONFIG.json')
bindings += [{'origin':'vision weight','path':cfg['weight_path'],'expected':cfg['weight_sha256'],
              'actual':digest(cfg['weight_path']),'pass':digest(cfg['weight_path'])==cfg['weight_sha256']}]
new('BINDINGS.json',{'records':bindings,'failed':[x for x in bindings if not x['pass']],
                     'historical_source_sha256':historical,'historical_source_matches':historical==actor['source_sha256'],
                     'current_source_sha256':source0,'frozen_source_matches':source0==load(cfg['frozen_protocol_path'])['source_sha256']})

split_path=REPO/'dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json'
split=load(split_path)
image_summary={}; image_ids={}; actual_images={}
for role in ('calibration','validation','test'):
    manifest=load(LAB/f'data/{role}_images.json')
    original={x['image_id']:x for x in split[role]}
    errors=[]; image_ids[role]=set()
    for item in manifest['images']:
        p=Path(manifest['image_root'])/item['image_id']; raw=p.read_bytes()
        actual=hashlib.sha256(raw).hexdigest()
        if (actual!=item['image_sha256'] or len(raw)!=item['payload_bytes'] or
            item!= {k:original[item['image_id']][k] for k in item}): errors.append(item['image_id'])
        image_ids[role].add(item['image_id']); actual_images[str(p)]={'sha256':actual,'bytes':len(raw)}
    image_summary[role]={'count':len(manifest['images']),'source_split_sha_matches':digest(split_path)==manifest['source_split_sha256'],
                         'mismatches':errors,'manifest_sha256':digest(LAB/f'data/{role}_images.json')}
new('IMAGES.json',{'roles':image_summary,'disjoint':all(not (image_ids[a]&image_ids[b]) for a,b in [('calibration','validation'),('calibration','test'),('validation','test')]),'actual_files':actual_images,'source_split_actual_sha256':digest(split_path)})

schedules={}
for p in (LAB/'data').rglob('*.json'):
    obj=load(p)
    if obj.get('schema')=='systemlab-schedule-v1': schedules[digest(p)]=(p,obj)
schedule_summary={}
for fingerprint,(p,s) in schedules.items():
    if s['provenance'].get('source')!='synthetic_poisson': continue
    rng=random.Random(s['seed']); t=0.; times=[]
    while True:
        t+=rng.expovariate(s['provenance']['aggregate_rate_per_s'])
        if t>=s['duration_s']: break
        times.append(t)
    schedule_summary[str(p.relative_to(LAB))]={'sha256':fingerprint,'tasks':len(s['tasks']), 'role':s['role'],
        'seed':s['seed'],'rate':s['provenance']['aggregate_rate_per_s'],'horizon':s['duration_s'],
        'poisson_arrivals_reproduced':times==[x['arrival_s'] for x in s['tasks']],
        'unique_ids':len(set(x['task_id'] for x in s['tasks']))==len(s['tasks'])}
new('SCHEDULES.json',schedule_summary)

summaries={}; qla_observations=[]; qla_portfolios=[]
for r in run_dirs:
    inputs=load(r/'client/INPUTS.json'); cutoff=load(r/'client/CUTOFF.json'); complete=load(r/'client/COMPLETE.json')
    tasks=lines(r/'client/tasks.jsonl'); decisions=lines(r/'client/decisions.jsonl'); feeds=lines(r/'controller/feedback_send.jsonl')
    recvs=lines(r/'client/feedback_receive.jsonl'); late=lines(r/'client/late_responses.jsonl')
    c=inputs['config']; states={(x['snapshot']['server_id'],x['snapshot']['source_epoch_ns']):x['snapshot'] for x in feeds if x['kind']=='monitor_snapshot'}
    sent={(x['user_id'],x['version']):x for x in feeds if x['kind']=='feedback_sent'}
    received={(x['user_id'],x['version']):x for x in recvs if x['kind']=='feedback_received' and not x['injected_drop']}
    errs=collections.Counter(); max_feature_error=0.; max_portfolio_error=0.; edge_summary={}; all_execution={}
    for e in c['edges']:
        rows=lines(r/'edges'/e['id']/'edge.jsonl'); admitted=[x for x in rows if x['kind']=='admitted']; executed=[x for x in rows if x['kind']=='execution_result']
        rejected=[x for x in rows if x['kind']=='rejected']; end=next(x['snapshot'] for x in rows if x['kind']=='stopped')
        all_execution.update({x['task_id']:x for x in executed})
        durations=[x['server_service_s'] for x in executed if x['status']=='completed']
        edge_summary[e['id']]={'admitted':len(admitted),'executed':len(executed),'rejected':len(rejected),
            'fifo_order_pass':[x['task_id'] for x in admitted]==[x['task_id'] for x in executed],
            'worker_nonoverlap':all(a['compute_end_epoch_ns']<=b['compute_start_epoch_ns'] for a,b in zip(executed,executed[1:])),
            'mean_service_s':statistics.mean(durations) if durations else None,'final_snapshot':end,
            'runtime_source_matches_current':load(r/'edges'/e['id']/'RUNTIME.json')['source_sha256']==source0}
    schedule=schedules.get(inputs['schedule_sha256'],(None,{}))[1]
    stasks={x['task_id']:x for x in schedule.get('tasks',[])}
    if not stasks and r.name!='functional_4edge_attempt_02': errs['schedule_not_found']+=1
    manifest=load(LAB/('verification/images.json' if r.name=='functional_4edge_attempt_02' else f"data/{inputs['role']}_images.json"))
    assets={x['image_id']:x for x in manifest['images']}
    for task in tasks:
        if task['status']=='completed':
            if abs(task['e2e_from_release_s']-(task['e2e_from_send_s']+task['release_to_send_s']))>1e-8: errs['e2e_arithmetic']+=1
            ev=all_execution.get(task['task_id'])
            if ev is None or any(task['server_result'].get(k)!=ev.get(k) for k in task['server_result']): errs['server_event_response_mismatch']+=1
        if stasks and (task['task_id'] not in stasks or task.get('planned_release_s')!=stasks[task['task_id']]['arrival_s']): errs['task_schedule']+=1
    for x in feeds:
        if x['kind']=='monitor_snapshot':
            s=x['snapshot']
            if s['admitted']!=s['completed']+s['inference_errors']+s['waiting_count']+int(s['busy']) or s['received']!=s['admitted']+s['rejected']: errs['monitor_mass']+=1
    for d in decisions:
        for edge,ns in d['source_timestamps'].items():
            key=(d['user_id'],d['feedback_versions'][edge]); sf=sent.get(key); rf=received.get(key)
            if (sf is None or sf['server_sources'].get(edge)!=ns or rf is None or rf['received_epoch_ns']>d['epoch_ns']): errs['undelivered_feature_source']+=1
        if not stasks: continue
        ss=[states[(e['id'],d['source_timestamps'][e['id']])] for e in c['edges']]
        scale=c['feature_scales']; n=len(d['task_ids']); m=len(c['edges'])
        costs=[s['predicted_wait_s']+s['service_estimate_s']+t for s,t in zip(ss,scale['transfer_seconds_per_server'])]
        prices=[n/scale['request_scale']*min(1,max(0,s['predicted_wait_s'])/scale['queue_reference_s'])+.5*v/scale['latency_scale_s'] for s,v in zip(ss,costs)]
        span=max(max(prices)-min(prices),1e-8)
        obs=[0.,0.]+[0.]*m+[n/scale['request_scale'],sum(assets[stasks[i]['image_id']]['payload_bytes'] for i in d['task_ids'])/(scale['request_scale']*scale['mean_payload_bytes']),n/scale['request_scale']]+[1/(1+v/scale['latency_scale_s']) for v in costs]+[sum(d['feedback_age_s'])/(m*scale['age_scale_s'])]+[(v-min(prices))/span for v in prices]+[min(10,math.log1p(sum(max(0,v) for v in prices)/m))]
        max_feature_error=max(max_feature_error,max(abs(a-b) for a,b in zip(obs,d['actor_observation'])))
        if inputs['method']=='score_softmax':
            logits=[-v/c['score_temperature_s'] for v in costs]; weights=[math.exp(v-max(logits)) for v in logits]; weights=[v/sum(weights) for v in weights]
            max_portfolio_error=max(max_portfolio_error,max(abs(a-b) for a,b in zip(weights,d['portfolio'])))
        elif inputs['method']=='qla': qla_observations.append(d['actor_observation']); qla_portfolios.append(d['portfolio'])
        elif inputs['method']=='stale_shortest':
            target=min(range(m),key=lambda j:(costs[j],j))
            if d['portfolio']!=[float(i==target) for i in range(m)]: errs['shortest_portfolio']+=1
        for tid in d['task_ids']:
            u=stasks[tid]['assignment_uniform']; accum=0.; selected=m-1
            for i,w in enumerate(d['portfolio']):
                accum+=w
                if u<accum: selected=i; break
            if c['edges'][selected]['id']!=d['assigned_task_servers'][tid]: errs['common_uniform_assignment']+=1
    lat=[t['e2e_from_release_s'] for t in tasks if t['status']=='completed']
    metrics={'mean_latency_s':statistics.mean(lat) if lat else None,'p95_latency_s':quant(lat,.95), 'p99_latency_s':quant(lat,.99),'cvar99_s':tail(lat)}
    recorded=load(r/'analysis.json') if (r/'analysis.json').exists() else {}
    sums={'status_counts':dict(collections.Counter(x['status'] for x in tasks)), 'offered':len(tasks),'unique_outcomes':len(set(x['task_id'] for x in tasks))==len(tasks),
        'expected_outcomes':cutoff['scheduled_tasks'],'completed_metrics':metrics,'tail_mass':len(lat)/100,
        'maximum_metric_recompute_error':max((abs(v-recorded[k]) for k,v in metrics.items() if v is not None and k in recorded),default=None),
        'edges':edge_summary,'errors':dict(errs),'max_feature_error':max_feature_error,'max_baseline_portfolio_error':max_portfolio_error,
        'complete':complete,'process_end':load(r/'PROCESS_END.json'),'initial_source_matches_current':inputs['source_sha256']==source0,
        'schedule_sha256':inputs['schedule_sha256'],'late_responses':len(late),'decision_count':len(decisions),
        'max_release_to_send_s':max((t.get('release_to_send_s',0) for t in tasks),default=0),
        'p99_release_to_send_s':quant([t['release_to_send_s'] for t in tasks if 'release_to_send_s' in t],.99),
        'feedback_event_kinds':dict(collections.Counter(x['kind'] for x in feeds)),
        'receiver_event_kinds':dict(collections.Counter(x['kind'] for x in recvs))}
    summaries[r.name]=sums
new('RAW_RUN_RECOMPUTATION.json',summaries)

interference={}
for folder in ('interference_calibration','interference_calibration_affinity_attempt_01_broken'):
    root=LAB/folder; plan=load(root/'PLAN.json'); result=[]
    for unit in plan['plan']:
        for p in sorted((root/f"run_{unit['run']:02}").glob('edge*.json')):
            row=load(p)
            if 'error' in row: result.append({'run':unit['run'],'kind':unit['kind'],'file':str(p),'error':row['error']}); continue
            comp=[x['compute_s'] for x in row['rows']]
            result.append({'run':unit['run'],'kind':unit['kind'],'node':row['node_id'],'threads':row['torch_threads'],
                'warmup':row['warmup'],'n':row['n'],'affinity_mask':row['affinity_mask'],
                'same_image_order':[x['image_id'] for x in row['rows']]==plan['image_ids'],
                'compute_median_s':statistics.median(comp),'compute_p95_s':quant(comp,.95),
                'cpu_s':row['cpu_s'],'wall_s':row['wall_s'],'cpu_wall_ratio':row['cpu_wall_ratio'],
                'cpu_ratio_recompute_error':abs(row['cpu_s']/row['wall_s']-row['cpu_wall_ratio']),
                'row_fields':sorted(row['rows'][0]),'top_level_fields':sorted(row)})
    interference[folder]={'plan':plan,'results':result}
new('INTERFERENCE_RECOMPUTATION.json',interference)

local_windows_runtime()
import numpy,torch,torchvision,PIL
from systemlab.policy import Router
router=Router('qla',cfg,LAB/'protocol/formal_actor_v2.json')
with torch.inference_mode():
    x=torch.tensor(qla_observations,dtype=torch.float32); out=router.actor.portfolio(router.actor.distribution(x).mean,x)
    actual=numpy.asarray(out.tolist()); original=numpy.asarray(qla_portfolios)
    qla_delta=float(numpy.max(numpy.abs(actual-original)))
deps={}
for name,mod in [('numpy',numpy),('torch',torch),('torchvision',torchvision),('Pillow',PIL)]:
    dist=importlib.metadata.distribution(name); path=Path(mod.__file__)
    deps[name]={'version':mod.__version__,'path':str(path),'module_sha256':digest(path),
                'distribution_root':str(dist.locate_file('')),'metadata_path':str(dist._path)}
    for file in ('METADATA','RECORD'):
        p=Path(dist._path)/file
        if p.exists(): deps[name][file+'_sha256']=digest(p)
dep_out={'packages':deps,'sys_executable':sys.executable,'base_prefix':sys.base_prefix,'prefix':sys.prefix,'sys_path':sys.path,
         'pyvenv':(Path(sys.prefix)/'pyvenv.cfg').read_text(),'qla_recomputed_decisions':len(qla_observations),
         'qla_max_portfolio_error_batched_fp32':qla_delta,'actor_verification':router.verify_frozen(),
         'loaded_historical_modules':{k:v.__file__ for k,v in sys.modules.items() if (k.startswith('mappo_gp_sf_dsr') or k.startswith('repro_risk_averse_edge')) and hasattr(v,'__file__')}}
new('DEPENDENCIES_AND_ACTOR.json',dep_out)

net=subprocess.run(['C:/Windows/System32/netstat.exe','-ano'],capture_output=True,text=True,check=True)
ports={}
for kind, nums in [('TCP',[8100,8101,8102,8103,8104,8400,8401,8402,8403,8404]),('UDP',[9100,9400])]:
    for port in nums:
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM if kind=='TCP' else socket.SOCK_DGRAM)
        try: sock.bind(('127.0.0.1',port)); ports[f'{kind}/{port}']='bind_available'
        except OSError as e: ports[f'{kind}/{port}']=repr(e)
        finally: sock.close()
new('PRE_EXECUTION_RECEIPT.json',{'started_epoch_ns':started,'ended_epoch_ns':time.time_ns(),'ports':ports,'netstat_stdout':net.stdout,
    'source_sha256':source_receipt(),'source_unchanged_during_audit':source0==source_receipt(),'protocol_sha256':digest(cfg['frozen_protocol_path']),
    'test_registry_sha256':digest(LAB/'protocol/E1_test_registry_v2.json'),'files_read':len(inventory),'image_bytes_read':len(actual_images),
    'no_services_started':True,'no_training':True,'no_test_execution':True})
print(json.dumps({'bindings_failed':len([x for x in bindings if not x['pass']]),'historical_source_matches':historical==actor['source_sha256'],
 'runs':{k:{a:v[a] for a in ('status_counts','completed_metrics','errors','max_feature_error','maximum_metric_recompute_error')} for k,v in summaries.items()},
 'qla_forward_max_error':qla_delta,'ports':ports},indent=2))
