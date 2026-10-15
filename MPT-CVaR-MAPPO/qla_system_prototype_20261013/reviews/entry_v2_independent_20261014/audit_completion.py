"""Read-only completion checks for the independent review; writes only here."""
import collections
import ctypes
import hashlib
import json
import math
from pathlib import Path
import random
import socket
import statistics
import sys
import time
OUT=Path(__file__).resolve().parent; LAB=OUT.parents[1]
sys.path.insert(0,str(LAB))
from systemlab.common import source_receipt,local_windows_runtime
local_windows_runtime()
from systemlab.launcher import _wait_runtime_pid
from systemlab.admission import check_protocol,validate_test_admission

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def save(p,x):
    with (OUT/p).open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')

inventory=load(OUT/'READ_INVENTORY.json')
changed=[p for p,r in inventory.items() if sha(LAB/p)!=r['sha256']]
images=load(OUT/'IMAGES.json')
image_changed=[p for p,r in images['actual_files'].items() if sha(p)!=r['sha256']]
hashsets={role:{x['image_sha256'] for x in load(LAB/f'data/{role}_images.json')['images']} for role in ('calibration','validation','test')}
overlap={a+'_'+b:len(hashsets[a]&hashsets[b]) for a,b in [('calibration','validation'),('calibration','test'),('validation','test')]}
binding=load(OUT/'BINDINGS.json')
changed_bindings=[r['path'] for r in binding['records'] if sha(r['path'])!=r['actual']]
actor=load(LAB/'protocol/formal_actor_v2.json')
historical_changed=[p for p,h in binding['historical_source_sha256'].items() if sha(Path(actor['source_root'])/p)!=h]
registry=load(LAB/'protocol/E1_test_registry_v2.json');cfg=load(registry['config']['path']);grid=load(registry['grid']['path'])
check_protocol(cfg)
schedules={}; schedule_regen={}
for rec in registry['common_schedules']:
    sch=load(rec['path']);pool=load(registry['test_manifest']['path'])['images']; rng=random.Random(sch['seed']+10000)
    expected=[dict(task_id=f't{i:08d}',arrival_s=t['arrival_s'],user_id=rng.randrange(sch['users']),image_id=rng.choice(pool)['image_id'],assignment_uniform=rng.random()) for i,t in enumerate(sch['tasks'])]
    schedule_regen[rec['common_schedule_key']]=expected==sch['tasks']; schedules[rec['common_schedule_key']]=rec['path']
admission=[]
for cell in grid['cells']:
    try:
        validate_test_admission(cfg,schedules[cell['common_schedule_key']],registry['test_manifest']['path'],cell['method'],None,LAB/'protocol/E1_test_registry_v2.json',cell['id'],LAB/'protocol/formal_actor_v2.json')
        outcome='UNEXPECTED_ACCEPT'
    except Exception as e: outcome=repr(e)
    admission.append(dict(cell_id=cell['id'],without_review=outcome))

class LPI(ctypes.Structure):
    _fields_=[('mask',ctypes.c_size_t),('relationship',ctypes.c_uint32),('info',ctypes.c_uint64*2)]
api=ctypes.WinDLL('kernel32',use_last_error=True)
api.GetLogicalProcessorInformation.argtypes=[ctypes.POINTER(LPI),ctypes.POINTER(ctypes.c_uint32)]
api.GetLogicalProcessorInformation.restype=ctypes.c_int
n=ctypes.c_uint32(0);api.GetLogicalProcessorInformation(None,ctypes.byref(n))
arr=(LPI*(n.value//ctypes.sizeof(LPI)))()
ok=bool(api.GetLogicalProcessorInformation(arr,ctypes.byref(n)))
topology={'api':'GetLogicalProcessorInformation','current_read_only_measurement_not_historical_proof':True,'success':ok,
          'physical_core_masks':[int(r.mask) for r in arr if r.relationship==0] if ok else [],'epoch_ns':time.time_ns()}

run=OUT/'functional_4edge_attempt_01'
ends=load(run/'PROCESS_END.json'); tasks=rows(run/'client/tasks.jsonl'); events={}
for edge in cfg['edges']:
    ed=rows(run/'edges'/edge['id']/'edge.jsonl'); events[edge['id']]={'kinds':dict(collections.Counter(x['kind'] for x in ed)),
        'stopped':next(x for x in ed if x['kind']=='stopped'),'runtime':load(run/'edges'/edge['id']/'RUNTIME.json')}
functional={'complete':load(run/'client/COMPLETE.json'),'cutoff':load(run/'client/CUTOFF.json'),
    'status_counts':dict(collections.Counter(x['status'] for x in tasks)),'unique_outcomes':len(tasks)==len({x['task_id'] for x in tasks}),
    'process_end':ends,'runtime_exited_fresh':{str(x['runtime_pid']):_wait_runtime_pid(x['runtime_pid'],.1) for x in ends['services']},
    'edges':events,'feedback_received':len(rows(run/'client/feedback_receive.jsonl')),
    'source_matches':load(run/'client/INPUTS.json')['source_sha256']==source_receipt(),
    'stderr_nonempty':[str(p) for p in run.glob('*.stderr.log') if p.stat().st_size],
    'receipt_sha256':{str(p.relative_to(OUT)):sha(p) for p in run.rglob('*') if p.is_file()}}
save('FUNCTIONAL_VERIFICATION.json',functional)
diagnostics={}
raw=load(OUT/'RAW_RUN_RECOMPUTATION.json')
for name in raw:
    root=LAB/'runs'/name if not name.startswith('functional') else LAB/'engineering_repairs/20261013_three_logic_fixes'/name
    cut=load(root/'client/CUTOFF.json'); feeds=rows(root/'controller/feedback_send.jsonl')
    errs=[dict(x,relative_to_begin_s=(x['epoch_ns']-cut['begin_epoch_ns'])/1e9) for x in feeds if x['kind']=='monitor_error']
    begin=cut['begin_epoch_ns']; horizon_end=begin+int(cut['protocol_horizon_s']*1e9)
    monitor=[x for x in feeds if x['kind']=='monitor_snapshot' and begin<=x['epoch_ns']<horizon_end]
    sent=[x for x in feeds if x['kind']=='feedback_sent' and begin<=x['send_epoch_ns']<horizon_end]
    diagnostics[name]={'monitor_errors':errs,'monitor_response_serialized_bytes':sum(len(json.dumps(x['snapshot'],sort_keys=True,separators=(',',':')).encode()) for x in monitor),
       'application_feedback_bytes':sum(x['payload_bytes'] for x in sent),'application_rate':sum(x['payload_bytes'] for x in sent)/cut['protocol_horizon_s'],
       'feedback_messages':len(sent),'max_observed_waiting':max(x['snapshot']['waiting_count'] for x in feeds if x['kind']=='monitor_snapshot')}
save('FEEDBACK_DIAGNOSTICS.json',diagnostics)

deps=load(OUT/'DEPENDENCIES_AND_ACTOR.json');binary_paths=[]
for mod in ('torch','torchvision','numpy','Pillow'):
    root=Path(deps['packages'][mod]['path']).parent
    binary_paths.extend(root.glob('*.pyd'))
    if mod=='torch': binary_paths.extend(p for name in ('c10.dll','torch_cpu.dll','fbgemm.dll','libiomp5md.dll') for p in (root/'lib').glob(name))
    if mod=='numpy': binary_paths.extend((root/'_core').glob('*.pyd'))
binary_hashes={str(p):sha(p) for p in binary_paths}
save('DEPENDENCY_BINARY_SAMPLE.json',{'scope':'Current selected native execution components; not a complete hermetic transitive dependency lock', 'actual_sha256':binary_hashes})
save('POST_EXECUTION_RECEIPT.json',{'epoch_ns':time.time_ns(),'source_sha256':source_receipt(),'source_matches_initial':source_receipt()==binding['current_source_sha256'],
    'read_inventory_files':len(inventory),'changed_input_files':changed,'image_count':len(images['actual_files']), 'changed_image_files':image_changed,
    'role_sha_overlap_counts':overlap,'changed_bound_files':changed_bindings,'changed_historical_source_files':historical_changed,
    'schedule_assignments_regenerated':schedule_regen,'actual_cells_without_review_rejected':admission,'topology':topology,
    'no_E1_started_by_this_review':True,'no_training':True,'all_owned_functional_runtime_pids_exited':all(functional['runtime_exited_fresh'].values())})
print(json.dumps({'changed_input_files':changed,'images_changed':image_changed,'role_overlap':overlap,'topology':topology,
 'functional_counts':functional['status_counts'],'functional_runtime_exited':functional['runtime_exited_fresh'],'inventory_files':len(inventory),
 'all_schedule_assignments_reproduced':all(schedule_regen.values()),'real_test_cells_checked_without_execution':len(admission)},indent=2))
