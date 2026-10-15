"""Read-only independent evidence checks. Outputs only below this review directory."""
import ast
import collections
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

OUT = Path(__file__).resolve().parent
LAB = OUT.parents[1]
REPO = LAB.parent
sys.path.insert(0, str(LAB))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('TEMP', str(OUT / 'tmp'))
os.environ.setdefault('TMP', str(OUT / 'tmp'))
(OUT / 'tmp').mkdir(exist_ok=True)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_bytes())

def save(p, value):
    p = OUT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))

def rows(p):
    return [json.loads(x) for x in Path(p).read_bytes().splitlines() if x.strip()]

def q(x, p):
    x = sorted(x)
    if not x: return None
    a = (len(x)-1)*p
    return x[math.floor(a)] + (x[math.ceil(a)]-x[math.floor(a)])*(a-math.floor(a))

def cv(x):
    x = sorted(x, reverse=True)
    if not x: return None
    mass = .01*len(x)
    return sum(v*max(0., min(1., mass-i)) for i,v in enumerate(x))/mass

def inventory():
    folders = ['systemlab','tools','tests','configs','docs','drive_scripts','protocol/frozen_v3',
        'legacy_source','data/E1_common_v3','interference_calibration',
        'interference_calibration_affinity_attempt_01_broken',
        'engineering_repairs/20261013_three_logic_fixes/BEFORE',
        'engineering_repairs/20261013_three_logic_fixes/functional_4edge_attempt_02']
    folders += ['runs/'+x for x in ('M1_rr_s1_attempt_01','M1_stress_s1_attempt_01',
        'M2d_qla_s101_attempt_01','M2d_score_softmax_s101_attempt_01','M2d_stale_shortest_s101_attempt_01')]
    files = ['README.md','requirements.txt','Dockerfile','compose.yaml',
        'protocol/formal_actor_v3.json','protocol/E1_grid_v3.json','protocol/E1_test_registry_v3.json',
        'protocol/validation_registration_v1.json','protocol/calibration_scales_v1.json',
        'protocol/validation_config_v1.json','protocol/validation_actor_v1.json',
        'data/calibration_images.json','data/validation_images.json','data/test_images.json',
        'verification/images.json','verification/schedule_4edge.json','verification/actor_fixture_bundle.json',
        'reviews/entry_v2_independent_20261014/REVIEW_RESPONSE.md',
        'engineering_repairs/20261013_three_logic_fixes/REPAIR_REPORT_中文.md',
        'engineering_repairs/20261013_three_logic_fixes/RESULT.json',
        'engineering_repairs/20261013_three_logic_fixes/UNIT_RESULT_attempt_02.json']
    paths = set(LAB/f for f in files)
    for folder in folders:
        paths.update(p for p in (LAB/folder).rglob('*') if p.is_file())
    paths.update((LAB/'data').glob('*.json'))
    records=[]
    for p in sorted(paths):
        b=p.read_bytes()
        v=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        if p.suffix=='.json':
            a=json.loads(b); v['json_type']=type(a).__name__
        elif p.suffix=='.jsonl':
            a=[json.loads(x) for x in b.splitlines() if x.strip()]
            v.update(rows=len(a),kinds=dict(collections.Counter(x.get('kind','') for x in a)))
        elif p.suffix in ('.py','.md','.txt','.yaml','.log') or p.name=='Dockerfile':
            v['text_lines']=len(b.decode('utf-8-sig').splitlines())
            if p.suffix=='.py': ast.parse(b.decode('utf-8-sig'))
        records.append(v)
    save('READ_INVENTORY.json', records)
    return records

def bindings():
    index=read(OUT/'REVIEW_REQUEST_INDEX.json')
    source={str(p.relative_to(LAB)):digest(p) for folder in ('systemlab','tools')
        for p in sorted((LAB/folder).rglob('*.py'))}
    checks=[]
    def check(p,h):
        actual=digest(p)
        checks.append(dict(path=str(Path(p).resolve()),expected=h,actual=actual,match=actual==h))
    actor=read(LAB/'protocol/formal_actor_v3.json')
    cfg=read(LAB/'protocol/frozen_v3/FORMAL_CONFIG.json')
    for name,h in index['artifacts'].items():
        if name=='actor checkpoint': p=actor['checkpoint']
        elif name=='vision weight': p=cfg['weight_path']
        elif name.startswith(('data/','runs/','legacy_source/','tests/')): p=LAB/name
        else: p=LAB/'protocol'/name
        check(p,h)
    for name,h in source.items(): check(LAB/name,index['source_sha256'][name])
    protocol=read(cfg['frozen_protocol_path'])
    for p,h in protocol['validation_artifacts_sha256'].items(): check(p,h)
    for p,h in cfg['feature_scales']['calibration_edge_logs'].items(): check(p,h)
    registry=read(LAB/'protocol/E1_test_registry_v3.json')
    for key in ('config','grid','protocol','validation_registration','feature_scales','test_manifest'):
        check(registry[key]['path'],registry[key]['sha256'])
    for b in registry['common_schedules']+registry['generator_sources']+list(registry['actor_bundles'].values()):
        check(b['path'],b['sha256'])
    hist=[]
    for p in sorted(Path(actor['source_root']).rglob('*.py')):
        b=p.read_bytes(); ast.parse(b.decode('utf-8-sig'))
        hist.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest()))
    actual_hist={str(Path(x['path']).relative_to(actor['source_root'])):x['sha256'] for x in hist}
    save('BINDINGS.json',dict(checks=checks,source_sha256=source,
        source_table_exact=source==index['source_sha256'],actor_source=hist,
        actor_source_exact=actual_hist==actor['source_sha256'],
        protocol_source_exact=source==protocol['source_sha256'],registry_source_exact=source==registry['source_sha256']))
    return source

def images_schedules():
    splitp=REPO/'dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json'
    split=read(splitp); image_records=[]; roles={}; schedule_records=[]
    for role in ('calibration','validation','test'):
        m=read(LAB/f'data/{role}_images.json')
        expected={x['image_id']:x for x in split[role]}
        roles[role]=m['images']
        for x in m['images']:
            p=Path(m['image_root'])/x['image_id']; b=p.read_bytes()
            actual=hashlib.sha256(b).hexdigest()
            image_records.append(dict(path=str(p),role=role,sha256=actual,bytes=len(b),
                matches=actual==x['image_sha256'] and len(b)==x['payload_bytes'] and
                    all(x[k]==expected[x['image_id']][k] for k in x)))
        assert m['source_split_sha256']==digest(splitp)
    overlap={a+'|'+b:len({x['image_sha256'] for x in roles[a]} & {x['image_sha256'] for x in roles[b]})
        for a,b in [('calibration','validation'),('calibration','test'),('validation','test')]}
    for p in sorted((LAB/'data').rglob('*.json')):
        if 'E1_common_v3' not in str(p) and p.parent!=LAB/'data': continue
        s=read(p)
        if s.get('schema')!='systemlab-schedule-v1': continue
        m=read(LAB/f"data/{s['role']}_images.json")
        ok=s['image_manifest_sha256']==digest(LAB/f"data/{s['role']}_images.json")
        rng=random.Random(s['seed']); ts=[]; t=0.
        rate=s['provenance'].get('aggregate_rate_per_s')
        if rate is not None:
            while True:
                t+=rng.expovariate(rate)
                if t>=s['duration_s']: break
                ts.append(t)
            r=random.Random(s['seed']+10000)
            rebuilt=[dict(task_id=f't{i:08d}',arrival_s=float(t),user_id=r.randrange(s['users']),
                image_id=r.choice(m['images'])['image_id'],assignment_uniform=r.random()) for i,t in enumerate(ts)]
            ok=ok and rebuilt==s['tasks']
        schedule_records.append(dict(path=str(p),sha256=digest(p),role=s['role'],tasks=len(s['tasks']),
            seed=s['seed'],duration_s=s['duration_s'],rate=rate,full_reconstruction_match=ok))
    save('IMAGES_SCHEDULES.json',dict(split_path=str(splitp),split_sha256=digest(splitp),
        images=image_records,role_content_overlap=overlap,schedules=schedule_records))

def legacy():
    before=LAB/'engineering_repairs/20261013_three_logic_fixes/BEFORE'
    candidates=list(before.rglob('*.py'))+list((LAB/'legacy_source').rglob('*.py'))
    for folder in ('systemlab','tools','drive_scripts','tests'): candidates+=list((LAB/folder).rglob('*.py'))
    by_hash=collections.defaultdict(list)
    for p in candidates: by_hash[digest(p)].append(str(p))
    result={}
    for name in ('M1_rr_s1_attempt_01','M1_stress_s1_attempt_01'):
        rec=read(LAB/'runs'/name/'client/INPUTS.json')['source_sha256']
        result[name]={p:dict(sha256=h,resolved_paths=by_hash[h]) for p,h in rec.items()}
    old=(LAB/'legacy_source/systemlab/common.py').read_text(encoding='utf-8')
    new=(LAB/'systemlab/common.py').read_text(encoding='utf-8')
    def omit(text):
        tree=ast.parse(text); f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='source_receipt')
        return '\n'.join(text.splitlines()[:f.lineno-1]+text.splitlines()[f.end_lineno:])
    diffs={}
    for p in before.rglob('*.py'):
        cur=LAB/p.relative_to(before)
        if cur.exists() and digest(p)!=digest(cur):
            diffs[str(p.relative_to(before))]=list(difflib.unified_diff(p.read_text(encoding='utf-8').splitlines(),
                cur.read_text(encoding='utf-8').splitlines(),lineterm=''))
    save('LEGACY_SOURCE_RESOLUTION.json',dict(sources=result,common_sha256=digest(LAB/'legacy_source/systemlab/common.py'),
        all_non_source_receipt_text_identical=omit(old)==omit(new),
        common_diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),lineterm='')),
        before_current_source_diffs=diffs))

def run_checks(root):
    inp=read(root/'client/INPUTS.json'); cutoff=read(root/'client/CUTOFF.json')
    complete=read(root/'client/COMPLETE.json'); task=rows(root/'client/tasks.jsonl')
    decisions=rows(root/'client/decisions.jsonl'); controller=rows(root/'controller/feedback_send.jsonl')
    received=rows(root/'client/feedback_receive.jsonl'); cfg=inp['config']
    byid={x['task_id']:x for x in task}; violations=[]
    def ck(ok,msg):
        if not ok: violations.append(msg)
    ck(len(task)==len(byid)==cutoff['scheduled_tasks']==complete['actual_outcomes'],'outcome uniqueness/count')
    lat=[x['e2e_from_release_s'] for x in task if x['status']=='completed']
    metrics=dict(mean_latency_s=sum(lat)/len(lat) if lat else None,p95_latency_s=q(lat,.95),
        p99_latency_s=q(lat,.99),cvar99_s=cv(lat),cvar_tail_effective_tasks=.01*len(lat))
    analysis=read(root/'analysis.json') if (root/'analysis.json').exists() else {}
    delta={k:abs(v-analysis[k]) for k,v in metrics.items() if v is not None and k in analysis}
    edges={}; exec_byid={}; snaps={}
    for e in cfg['edges']:
        eroot=root/'edges'/e['id']; rr=read(eroot/'RUNTIME.json'); ev=rows(eroot/'edge.jsonl')
        admitted=[x for x in ev if x['kind']=='admitted']; executed=[x for x in ev if x['kind']=='execution_result']
        ck([x['task_id'] for x in admitted]==[x['task_id'] for x in executed],e['id']+' FIFO')
        ex=[x for x in executed if x['status']=='completed']
        ck(all(a['compute_end_epoch_ns']<=b['compute_start_epoch_ns'] for a,b in zip(ex,ex[1:])),e['id']+' overlap')
        for x in executed: exec_byid[x['task_id']]=x
        edges[e['id']]=dict(admitted=len(admitted),completed=len(ex),
            rejected=sum(x['kind']=='rejected' for x in ev),runtime=rr,
            mean_service_s=statistics.mean(x['server_service_s'] for x in ex),
            stopped=[x for x in ev if x['kind']=='stopped'])
    for x in task:
        if x['status']=='completed':
            ck(x['task_id'] in exec_byid,x['task_id']+' execution absent')
            ex=exec_byid.get(x['task_id'],{})
            ck(all(v==ex[k] for k,v in x['server_result'].items()),x['task_id']+' response != execution')
            ck(x['client_response_elapsed_s']<=cutoff['protocol_horizon_s']+cutoff['fixed_drain_s'],x['task_id']+' after cutoff completed')
    for x in controller:
        if x['kind']=='monitor_snapshot':
            s=x['snapshot']; snaps[(s['server_id'],s['source_epoch_ns'])]=s
            ck(s['admitted']==s['completed']+s['inference_errors']+s['waiting_count']+int(s['busy']),'snapshot mass')
            ck(s['received']==s['admitted']+s['rejected'],'snapshot arrival mass')
    sent={(x['user_id'],x['version']):x for x in controller if x['kind']=='feedback_sent'}
    rec={(x['user_id'],x['version']):x for x in received if x['kind']=='feedback_received' and not x['injected_drop']}
    features_max=0.; dest_errors=0
    for d in decisions:
        states=[]
        for sid,stamp in d['source_timestamps'].items():
            k=(d['user_id'],d['feedback_versions'][sid]); a=rec.get(k); b=sent.get(k)
            ck(a is not None and b is not None and b['server_sources'].get(sid)==stamp and a['received_epoch_ns']<=d['epoch_ns'],'delivered cache provenance')
        states=[snaps[(e['id'],d['source_timestamps'][e['id']])] for e in cfg['edges']]
        sc=cfg['feature_scales']; n=len(d['task_ids'])
        # independent reconstruction from recorded sources; age is already logged and cross-checked below
        costs=[max(0,s['predicted_wait_s'])+s['service_estimate_s']+t for s,t in zip(states,sc['transfer_seconds_per_server'])]
        prices=[n/sc['request_scale']*min(1,max(0,s['predicted_wait_s'])/sc['queue_reference_s'])+.5*c/sc['latency_scale_s'] for s,c in zip(states,costs)]
        norm=[(p-min(prices))/max(max(prices)-min(prices),1e-8) for p in prices]
        payload=sum(byid[t].get('payload_bytes',0) for t in d['task_ids'])
        if all('payload_bytes' in byid[t] for t in d['task_ids']):
            obs=[0.,0.]+[0.]*len(states)+[n/sc['request_scale'],payload/(sc['request_scale']*sc['mean_payload_bytes']),n/sc['request_scale']]+[1/(1+c/sc['latency_scale_s']) for c in costs]+[sum(d['feedback_age_s'])/(len(states)*sc['age_scale_s'])]+norm+[min(10.,math.log1p(sum(max(0,p) for p in prices)/len(states)))]
            features_max=max(features_max,max(abs(a-b) for a,b in zip(obs,d['actor_observation'])))
        ck(max(abs(a-b) for a,b in zip(costs,d['expected_costs_s']))<1e-12,'cost mismatch')
        if inp['method']=='stale_shortest':
            expected=[float(j==min(range(len(costs)),key=lambda j:(costs[j],j))) for j in range(len(costs))]
            ck(expected==d['portfolio'],'stale action')
        elif inp['method']=='score_softmax':
            log=[-c/cfg['score_temperature_s'] for c in costs]; w=[math.exp(z-max(log)) for z in log]
            ck(max(abs(a/sum(w)-b) for a,b in zip(w,d['portfolio']))<1e-12,'score action')
        for tid,sid in d['assigned_task_servers'].items():
            t=byid[tid]
            if 'assignment_uniform' in t:
                u=t['assignment_uniform']; acc=0.; chosen=len(cfg['edges'])-1
                for j,w in enumerate(d['portfolio']):
                    acc+=w
                    if u<acc: chosen=j; break
                if cfg['edges'][chosen]['id']!=sid or t['server_id']!=sid: dest_errors+=1
    h0=cutoff['begin_epoch_ns']; h1=h0+int(cutoff['protocol_horizon_s']*1e9)
    offered_sent=[x for x in sent.values() if h0<=x['send_epoch_ns']<h1]
    mon_errors=[x for x in controller if x['kind']=='monitor_error' and h0<=x['epoch_ns']<=cutoff['cutoff_epoch_ns']]
    ck(features_max<1e-12,'features mismatch'); ck(dest_errors==0,'assignment mismatch')
    return dict(run=str(root),offered=len(task),status_counts=dict(collections.Counter(x['status'] for x in task)),
        metrics=metrics,analysis_absolute_deltas=delta,violations=violations,edges=edges,
        source_receipt=inp['source_sha256'],complete=complete,cutoff=cutoff,
        decisions=len(decisions),features_max_error=features_max,destination_errors=dest_errors,
        late_responses=len(rows(root/'client/late_responses.jsonl')),
        feedback_received=len(received),feedback_application_bytes_per_s=sum(x['payload_bytes'] for x in offered_sent)/cutoff['protocol_horizon_s'],
        monitor_errors_during_measurement=mon_errors,process_end=read(root/'PROCESS_END.json'),
        p99_release_to_send_s=q([x['release_to_send_s'] for x in task if 'release_to_send_s' in x],.99))

def runs():
    results=[run_checks(LAB/'runs'/name) for name in ('M1_rr_s1_attempt_01','M1_stress_s1_attempt_01',
        'M2d_qla_s101_attempt_01','M2d_score_softmax_s101_attempt_01','M2d_stale_shortest_s101_attempt_01')]
    save('RAW_RUN_RECOMPUTATION.json',results)
    for x in results:
        print(Path(x['run']).name,x['status_counts'],x['metrics'],'violations',len(x['violations']),flush=True)
    # Calibration numbers independently rebuilt from raw client/edge events.
    rr=results[0]; rt=rows(LAB/'runs/M1_rr_s1_attempt_01/client/tasks.jsonl')
    scale=read(LAB/'protocol/calibration_scales_v1.json')
    transfer=[]
    for sid in rr['edges']:
        transfer.append(statistics.mean(max(0,x['e2e_from_send_s']-x['server_result']['queue_s']-x['server_result']['server_service_s']) for x in rt if x['status']=='completed' and x['server_id']==sid))
    service={sid:e['mean_service_s'] for sid,e in rr['edges'].items()}
    save('CALIBRATION_RECOMPUTATION.json',dict(service_s=service,transfer_s=transfer,
        max_service_error=max(abs(v-scale['calibration_service_s_by_edge'][k]) for k,v in service.items()),
        max_transfer_error=max(abs(a-b) for a,b in zip(transfer,scale['transfer_seconds_per_server'])),
        capacity=sum(1/v for v in service.values())))

def interference():
    allresult={}
    for name in ('interference_calibration','interference_calibration_affinity_attempt_01_broken'):
        root=LAB/name; plan=read(root/'PLAN.json'); rs=[]
        for cell in plan['plan']:
            for p in sorted((root/f"run_{cell['run']:02d}").glob('edge*.json')):
                v=read(p)
                if 'error' in v: rs.append(dict(path=str(p),error=v['error'],kind=cell['kind'])); continue
                rs.append(dict(path=str(p),kind=cell['kind'],node=v['node_id'],n=v['n'],
                    image_sequence_match=[x['image_id'] for x in v['rows']]==plan['image_ids'],
                    warmup=v['warmup'],torch_threads=v['torch_threads'],affinity=v['affinity_mask'],
                    compute_median_s=statistics.median(x['compute_s'] for x in v['rows']),
                    compute_p95_s=q([x['compute_s'] for x in v['rows']],.95),cpu_s=v['cpu_s'],wall_s=v['wall_s'],
                    cpu_wall_ratio=v['cpu_wall_ratio'],ratio_match=abs(v['cpu_s']/v['wall_s']-v['cpu_wall_ratio'])<1e-14,
                    has_measurement_epoch=any('epoch' in k for k in v)))
        agg={}
        for node in ('edge0','edge1','edge2','edge3'):
            agg[node]={}
            for kind in ('solo','concurrent','concurrent_affinity'):
                vals=[x for x in rs if x.get('node')==node and x['kind']==kind]
                agg[node][kind]=dict(n=len(vals),mean_median_s=statistics.mean(x['compute_median_s'] for x in vals) if vals else None,
                    mean_cpu_s=statistics.mean(x['cpu_s'] for x in vals) if vals else None,
                    mean_wall_s=statistics.mean(x['wall_s'] for x in vals) if vals else None)
        allresult[name]=dict(runs=rs,aggregate=agg)
    save('INTERFERENCE_RECOMPUTATION.json',allresult)

def actor_forward():
    from systemlab.common import local_windows_runtime
    local_windows_runtime()
    from systemlab.policy import Router
    cfg=read(LAB/'protocol/frozen_v3/FORMAL_CONFIG.json')
    router=Router('qla',cfg,LAB/'protocol/formal_actor_v3.json')
    t=router.torch
    d=rows(LAB/'runs/M2d_qla_s101_attempt_01/client/decisions.jsonl')
    maxerr=0.
    with t.inference_mode():
        for start in range(0,len(d),256):
            block=d[start:start+256]
            x=t.tensor([v['actor_observation'] for v in block],dtype=t.float32)
            w=router.actor.portfolio(router.actor.distribution(x).mean,x).tolist()
            maxerr=max(maxerr,max(abs(a-b) for row,v in zip(w,block) for a,b in zip(row,v['portfolio'])))
    modules={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if getattr(m,'__file__',None) and n.startswith(('mappo_gp_sf_dsr','repro_risk_averse_edge'))}
    deps={}
    for name in ('numpy','torch','torchvision','Pillow','pytest'):
        dist=importlib.metadata.distribution(name)
        meta=Path(dist._path)
        deps[name]=dict(version=dist.version,path=str(meta),metadata_sha256={p.name:digest(p) for p in meta.iterdir() if p.name in ('METADATA','RECORD','WHEEL')})
    payload=t.load(router.bundle['checkpoint'],map_location='cpu',weights_only=True)
    save('ACTOR_DEPENDENCY_CHECK.json',dict(runtime_executable=sys.executable,checkpoint_sha256=digest(router.bundle['checkpoint']),
        payload_scalar_metadata={k:v for k,v in payload.items() if isinstance(v,(str,int,float,bool))},
        actor_state_shapes={k:list(v.shape) for k,v in payload['actor'].items()},
        decisions_recomputed=len(d),portfolio_max_error=maxerr,frozen=router.verify_frozen(),
        imported_historical_modules=modules,imported_module_sha256={n:digest(p) for n,p in modules.items()},dependencies=deps,
        no_optimizer=True,scope='read-only offline inference correctness, not performance'))

if __name__=='__main__':
    modes={'inventory':inventory,'bindings':bindings,'images':images_schedules,'legacy':legacy,
        'runs':runs,'interference':interference,'actor':actor_forward}
    try:
        modes[sys.argv[1]]()
    except BaseException as e:
        import traceback
        save('DIAGNOSTIC_FAILURE_'+sys.argv[1]+'.json',dict(error=repr(e),traceback=traceback.format_exc(),time_ns=time.time_ns()))
        raise
