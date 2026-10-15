"""Independent, stdlib-only record audit. Does not import experiment modules.

Only existing calibration/confirmation rows are read. Retained test images and
labels are never read; split metadata is used solely to verify separation.
Outputs are restricted to this new reviewer directory.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from fractions import Fraction

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]
OLD = BASE.parent / 'dac_visual_calibration_20261009'
PROFILES = ('CPU-1T', 'CPU-4T')
STAGES = ('decode_ms', 'preprocess_ms', 'infer_ms', 'postprocess_ms')
BOUNDARY = 'decode+RGB+preprocess+FP32 inference+argmax+dict; no disk/network/queue/controller'
checks = Counter()
issues = []
hashes = {}

def verify(condition, check, evidence):
    checks[check] += 1
    if not condition:
        issues.append(dict(check=check, evidence=str(evidence)))

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    with p.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256').hexdigest()
    hashes[str(p)] = h
    return h

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def size(e):
    pixels = int(e['width']) * int(e['height'])
    return sum(pixels > b for b in (250000, 500000, 1000000, 2000000))

def cell(e):
    return f"{size(e)}|{e['class_id']}"

def stem(j):
    return f"{j['role']}_s{j['session']}_b{j['block']}_{j['profile']}"

plan = read(BASE/'protocol/PLAN.json')
splits = read(BASE/'protocol/SPLITS.json')
old_splits = read(OLD/'protocol/IMAGE_SPLITS.json')
pop = read(BASE/'protocol/POPULATION_PROFILE.json')
model = read(BASE/'model/CONDITIONAL_MODEL.json')
mapping = read(BASE/'model/VALIDATION_NEIGHBORS.json')
env = read(BASE/'protocol/ENVIRONMENT_SPEC.json')
topology = read(BASE/'protocol/CPU_TOPOLOGY.json')
archive = read(BASE/'analysis/RUNTIME_INTERRUPTION_20261010.json')
verify(read(BASE/'protocol/ENTRY_REVIEW.json')['plan_sha256'] == sha(BASE/'protocol/PLAN.json'), 'entry_plan_hash', 'protocol/ENTRY_REVIEW.json:3')
for key, root in [('source_sha256',BASE/'scripts'), ('artifact_sha256',BASE), ('old_readonly_sha256',OLD)]:
    for name, expected in plan[key].items():
        verify(sha(root/name) == expected, key, root/name)
for e in archive['raw_inventory'] + archive['evidence']:
    path = BASE/e['path']
    verify(sha(path) == e['sha256'] and path.stat().st_size == e['bytes'], 'interruption_archive_hash', path)
verify(sha(OLD/'models/resnet18-f37072fd.pth') == plan['weights_sha256'], 'weights_hash', 'models/resnet18-f37072fd.pth')

# Reconstruct eligible inventory using direct SHA + exhaustive Hamming checks,
# independently of the runner's segmented near-duplicate index.
manifest = rows(OLD/'protocol/IMAGE_MANIFEST.csv')
manifest_by_id = {e['image_id']:e for e in manifest}
used = [e for group in old_splits.values() for e in group]
used_ids = {e['image_id'] for e in used}
used_sha = {e['image_sha256'] for e in used}
prior_dhash = [int(e['dhash64'],16) for e in used]
candidates = sorted((e for e in manifest if e['official_split']=='train' and e['image_id'] not in used_ids), key=lambda e:e['image_id'])
random.Random(plan['seed']).shuffle(candidates)
eligible = []
for e in candidates:
    dh = int(e['dhash64'],16)
    if e['image_sha256'] in used_sha or any((dh ^ prev).bit_count() <= 2 for prev in prior_dhash):
        continue
    eligible.append(e)
    used_sha.add(e['image_sha256'])
    prior_dhash.append(dh)
eligible_ids = {e['image_id'] for e in eligible}
population_counts = Counter(cell(e) for e in eligible)
sample_counts = Counter(cell(e) for e in splits['validation'])
verify(len(eligible)==pop['eligible_images']==plan['validation_population']['eligible_images'], 'population_count', 'protocol/POPULATION_PROFILE.json:2')
verify(len(candidates)-len(eligible)==plan['validation_population']['duplicates_excluded'], 'population_exclusions', 'protocol/PLAN.json')
verify(dict(population_counts)==pop['size_class_counts'], 'population_cells', 'protocol/POPULATION_PROFILE.json:3')
verify(dict(sample_counts)==pop['selected_size_class_counts'], 'sample_cells', 'protocol/POPULATION_PROFILE.json:54')
verify(set(population_counts)==set(sample_counts), 'all_nonempty_cells_covered', 'protocol/POPULATION_PROFILE.json')
verify(splits['test']==old_splits['test'], 'retained_test_metadata_unchanged', 'protocol/SPLITS.json')
weights = {}
for e in splits['validation']:
    w = Fraction(population_counts[cell(e)],len(eligible)*sample_counts[cell(e)])
    weights[e['image_id']] = w
    verify(e['image_id'] in eligible_ids, 'validation_from_eligible_population', e['image_id'])
    verify(abs(float(w)-e['target_weight']) < 1e-18, 'target_weights', e['image_id'])
verify(sum(weights.values())==1 and len(weights)==2000, 'weights_sum_and_count', 'protocol/SPLITS.json')
verify(len({e['image_id'] for e in splits['reference']})==200 and all(e['image_id'] in {x['image_id'] for x in old_splits['calibration']} for e in splits['reference']), 'reference_calibration_subset', 'protocol/SPLITS.json')
data_to_hash = {e['image_id']:e for e in old_splits['calibration']+splits['validation']}
for image_id,e in data_to_hash.items():
    path = OLD/'data/extracted'/image_id
    verify(sha(path)==e['image_sha256'] and path.stat().st_size==int(e['payload_bytes']), 'image_hash_bytes', image_id)
    for field in ('class_id','official_split','width','height','payload_bytes','image_sha256','dhash64'):
        verify(str(e[field])==str(manifest_by_id[image_id][field]), 'manifest_identity', image_id+':'+field)

def inspect_block(root, job, entries, require_job=True):
    path = root/'measurements'/job['role']/(stem(job)+'.csv')
    meta = read(path.with_suffix('.json'))
    data = rows(path)
    threads = 1 if job['profile']=='CPU-1T' else 4
    affinity = topology['logical_cpu_per_physical_core'][:threads]
    verify(meta['job']==job, 'metadata_job', path)
    if require_job:
        jp = root/'protocol/jobs'/(stem(job)+'.json')
        verify(read(jp)==job and meta['job_sha256']==sha(jp), 'job_hash_and_identity', jp)
    for k,v in [('versions',env['versions']),('weights_sha256',plan['weights_sha256']),('warmup_requests',200),('errors',0),('timing_boundary',BOUNDARY),('affinity',affinity)]:
        verify(meta[k]==v, 'metadata_'+k, path.with_suffix('.json'))
    verify(len(data)==meta['requests']==len(job['image_ids']) and [r['image_id'] for r in data]==job['image_ids'], 'block_row_count_order', path)
    start = datetime.fromisoformat(meta['sensor_before']['wall_clock'])
    end = datetime.fromisoformat(meta['sensor_after']['wall_clock'])
    verify(start < end, 'interval_positive', path)
    previous = start
    for n,r in enumerate(data,2):
        e = entries[r['image_id']]
        evidence = str(path)+':'+str(n)
        expected = dict(session_id=str(job['session']),block_id=str(job['block']),run_order=str(n-1),profile_id=job['profile'],image_split=job['role'],device='cpu',cpu_threads=str(threads),model_id='ResNet18_IMAGENET1K_V1',weights_sha256=plan['weights_sha256'],error_status='ok')
        expected.update({k:str(e[k]) for k in ('image_sha256','class_id','width','height','payload_bytes')})
        verify(all(r[k]==v for k,v in expected.items()) and json.loads(r['cpu_affinity'])==affinity and 0<=int(r['predicted_class'])<1000, 'row_identity', evidence)
        values = [float(r[k]) for k in STAGES]
        total = float(r['service_ms'])
        verify(all(math.isfinite(v) and v>=0 for v in values) and math.isfinite(total) and total>0 and math.isclose(math.fsum(values),total,rel_tol=1e-10,abs_tol=1e-8), 'finite_stage_sum', evidence)
        stamp = datetime.fromisoformat(r['measurement_timestamp'])
        verify(start<=stamp<=end and previous<=stamp, 'timestamp_order', evidence)
        # Timer and wall clock are distinct; allow 2 ms timestamp scheduling/resolution tolerance.
        if n>2:
            verify(stamp+timedelta(milliseconds=2)>=previous+timedelta(milliseconds=float(data[n-3]['service_ms'])), 'row_interval_serial', evidence)
        verify(stamp+timedelta(milliseconds=total)<=end+timedelta(milliseconds=2), 'row_end_within_block', evidence)
        previous=stamp
        r['_evidence']=evidence
    sha(path);sha(path.with_suffix('.json'))
    return data, (start,end,stem(job)), meta

allrows=[];intervals=[];jobs=[];session_times=[];idle=[]
for item in plan['sessions']:
    if item['kind']!='confirmation':
        continue
    s=item['session']
    # Independently reproduce deterministic schedule from documented random seed.
    rng=random.Random(f"{plan['seed']}:jobs:{s}:False")
    chunks={}
    for role,chunk in [('validation',200),('reference',20)]:
        ids=[e['image_id'] for e in splits[role]]
        rng.shuffle(ids)
        chunks[role]=[ids[k:k+chunk] for k in range(0,len(ids),chunk)]
    expected=[]
    for b in range(10):
        roles=['validation','reference'];rng.shuffle(roles)
        for role in roles:
            profiles=list(PROFILES) if (s+b)%2 else list(reversed(PROFILES))
            for profile in profiles:
                expected.append(dict(role=role,session=s,block=b+1,profile=profile,image_ids=chunks[role][b]))
    verify(item['jobs']==expected, 'randomized_counterbalanced_schedule', f'protocol/PLAN.json session {s}')
    session_intervals=[]
    for j in item['jobs']:
        data,interval,meta=inspect_block(BASE,j,{e['image_id']:e for e in splits[j['role']]})
        allrows.extend(data);intervals.append(interval);session_intervals.append(interval);jobs.append(j)
        log=BASE/'logs'/(stem(j)+'.log')
        verify(log.read_text(encoding='utf-8').strip()==f"MEASURED {stem(j)} {len(data)}", 'block_completion_log', log)
        sha(log)
    session_times.append(dict(session=s,start=min(i[0] for i in session_intervals).isoformat(),end=max(i[1] for i in session_intervals).isoformat()))
    idle_rows=read(BASE/f'protocol/idle_confirmation_s{s}.json')
    gate=idle_rows[-1]
    verify(gate['mean']<=plan['machine_idle_percent'] and gate['selected_max']<=plan['selected_core_idle_percent'] and gate['available_memory_bytes']>=plan['minimum_memory_bytes'], 'idle_gate_thresholds', f'protocol/idle_confirmation_s{s}.json:3')
    verify(datetime.fromisoformat(gate['at'])<session_intervals[0][0], 'idle_before_session', f'protocol/idle_confirmation_s{s}.json:3')
    idle.append(gate)
verify([i[2] for i in sorted(intervals)]==[stem(j) for j in jobs], 'actual_execution_order', 'all confirmation intervals')
verify(all(a[1]<=b[0] for a,b in zip(sorted(intervals),sorted(intervals)[1:])), 'blocks_serial', 'all confirmation intervals')
gaps=[(datetime.fromisoformat(b['at'])-datetime.fromisoformat(a['at'])).total_seconds() for a,b in zip(idle,idle[1:])]
verify(all(g>=plan['spacing_seconds'] for g in gaps), 'session_spacing', gaps)
for role in ('reference','validation'):
    expected_files={stem(j)+ext for j in jobs if j['role']==role for ext in ('.csv','.json')}
    actual_files={p.name for p in (BASE/'measurements'/role).iterdir() if p.is_file()}
    verify(actual_files==expected_files, 'exact_measurement_file_set', role)
verify({p.name for p in (BASE/'protocol/jobs').iterdir() if p.is_file()}=={stem(j)+'.json' for j in jobs}, 'exact_job_set_no_test', 'protocol/jobs')
verify(not list((BASE/'measurements/test').iterdir()), 'no_retained_test_records', 'measurements/test')
verify(len(allrows)==13200, 'confirmation_row_count', len(allrows))
identities=[(r['image_split'],r['session_id'],r['profile_id'],r['image_id']) for r in allrows]
verify(len(set(identities))==len(identities), 'no_duplicate_row_identity', 'all confirmation rows')
predictions=defaultdict(set)
for r in allrows:predictions[(r['image_split'],r['image_id'])].add(r['predicted_class'])
verify(all(len(v)==1 for v in predictions.values()), 'prediction_agreement_across_profiles_sessions', 'all confirmation rows')
for role in ('validation','reference'):
    for s in range(1,4):
        for p in PROFILES:
            verify({r['image_id'] for r in allrows if r['image_split']==role and int(r['session_id'])==s and r['profile_id']==p}=={e['image_id'] for e in splits[role]}, 'full_session_profile_pairing', (role,s,p))
stdout=(BASE/'logs/runner_stdout.log').read_text(encoding='utf-8')
complete_lines=[x for x in stdout.splitlines() if x.startswith('BLOCK_COMPLETE ')]
running_count=0;expected_lines=[]
for j in jobs:
    running_count+=len(j['image_ids']);expected_lines.append(f'BLOCK_COMPLETE {stem(j)} {running_count}')
verify(complete_lines==expected_lines, 'runner_completion_log', 'logs/runner_stdout.log')

# Reconstruct frozen model directly from every original calibration row.
cal_entries={e['image_id']:e for e in old_splits['calibration']}
cal_rows=[]
for path in sorted((OLD/'measurements/calibration').glob('*.json')):
    meta=read(path)
    data,_,_=inspect_block(OLD,meta['job'],cal_entries,require_job=False)
    cal_rows.extend(data)
verify(len(cal_rows)==20000, 'calibration_row_count', len(cal_rows))
cal_by_key={}
for r in cal_rows:
    key=(r['image_id'],int(r['session_id']),r['profile_id'])
    verify(key not in cal_by_key, 'calibration_unique_keys', r['_evidence'])
    cal_by_key[key]=r
verify(set(cal_by_key)=={(i,s,p) for i in cal_entries for s in range(1,6) for p in PROFILES}, 'calibration_full_pairing', 'all calibration CSV')
cal_predictions=defaultdict(set)
for r in cal_rows:cal_predictions[r['image_id']].add(r['predicted_class'])
verify(all(len(v)==1 for v in cal_predictions.values()), 'calibration_prediction_agreement', 'all calibration CSV')
for path in sorted((OLD/'measurements/calibration').glob('*.json')):
    meta=read(path);jp=OLD/'protocol'/('job_'+path.name)
    verify(jp.exists() and read(jp)==meta['job'] and sha(jp)==meta['job_sha256'], 'calibration_job_hash_identity', path)
verify(model['training_entries']==old_splits['calibration'] and model['k']==20 and model['support_distance']==2 and model['strata_upper_bounds']==[250000,500000,1000000,2000000], 'model_definition_entries', 'model/CONDITIONAL_MODEL.json:2')
model_keys=set()
for item in model['paired_rows']:
    key=(item['image_id'],item['session']);model_keys.add(key)
    for p in PROFILES:
        r=cal_by_key[(*key,p)]
        verify(item['profiles'][p]=={k:float(r[k]) for k in STAGES+('service_ms',)}, 'model_original_stage_agreement', r['_evidence'])
verify(len(model['paired_rows'])==10000 and len(model_keys)==10000, 'model_all_calibration_pairs', 'model/CONDITIONAL_MODEL.json:22013')
rebuilt_neighbors={};unsupported=[]
for e in splits['validation']:
    x=(math.log2(int(e['width'])*int(e['height'])),math.log2(int(e['payload_bytes'])))
    distances=[]
    for c in cal_entries.values():
        if size(c)!=size(e):continue
        y=(math.log2(int(c['width'])*int(c['height'])),math.log2(int(c['payload_bytes'])))
        distances.append((math.hypot(x[0]-y[0],x[1]-y[1]),c['image_id']))
    distances.sort();chosen=distances[:20]
    rebuilt_neighbors[e['image_id']]=[i for d,i in chosen]
    if not chosen or chosen[0][0]>2:unsupported.append(e['image_id'])
    frozen=mapping[e['image_id']]
    verify(frozen['neighbor_ids']==rebuilt_neighbors[e['image_id']] and frozen['nearest_distance']==chosen[0][0] and frozen['supported']==(chosen[0][0]<=2), 'recomputed_neighbor_agreement', e['image_id'])
verify(set(mapping)==set(weights), 'neighbor_exact_input_set', 'model/VALIDATION_NEIGHBORS.json')

def statistics(pairs):
    # Exact rational probability masses; left empirical CDF inverse, no interpolation.
    ordered=sorted(pairs,key=lambda t:t[0]);total=sum(w for v,w in ordered)
    mean=math.fsum(v*float(w/total) for v,w in ordered)
    result={'mean':mean};witnesses={}
    for name,q in [('p95',Fraction(95,100)),('p99',Fraction(99,100))]:
        cumulative=Fraction(0)
        for v,w in ordered:
            previous=cumulative;cumulative+=w
            if cumulative>=q*total:
                result[name]=v;witnesses[name]=dict(value_ms=v,cdf_before=float(previous/total),cdf_at=float(cumulative/total));break
    return result,witnesses

# Aggregate neighbor reuse before sorting, avoiding dependence on evaluator loop order.
cal_mass=defaultdict(Fraction)
for image_id,ns in rebuilt_neighbors.items():
    for neighbor_id in ns:cal_mass[neighbor_id]+=weights[image_id]/len(ns)/5
metrics=[];summary={};quantile_witness={};reference=[]
for p in PROFILES:
    predicted=[(float(cal_by_key[(i,s,p)]['service_ms']),w) for i,w in cal_mass.items() for s in range(1,6)]
    observed=[(float(r['service_ms']),weights[r['image_id']]/3) for r in allrows if r['image_split']=='validation' and r['profile_id']==p]
    pred,pw=statistics(predicted);obs,ow=statistics(observed)
    summary[p]=dict(predicted_ms=pred,measured_ms=obs)
    quantile_witness[p]=dict(predicted=pw,measured=ow)
    for metric in ('mean','p95','p99'):
        err=abs(pred[metric]-obs[metric])/obs[metric]
        metrics.append(dict(profile=p,metric=metric,predicted_ms=pred[metric],measured_ms=obs[metric],relative_error=err,tolerance=plan['thresholds'][metric],passed=err<=plan['thresholds'][metric]))
    for s in range(1,4):
        data=[(float(r['service_ms']),Fraction(1)) for r in allrows if r['image_split']=='reference' and r['profile_id']==p and int(r['session_id'])==s]
        stats,_=statistics(data);reference.append(dict(profile=p,session=s,**stats))
compare=[]
if (BASE/'analysis/VALIDATION_REPORT.json').exists():
    report=read(BASE/'analysis/VALIDATION_REPORT.json')
    for x in metrics:
        other=next(y for y in report['criteria'] if (y['profile'],y['metric'])==(x['profile'],x['metric']))
        delta={k:other[k]-x[k] for k in ('predicted_ms','measured_ms','relative_error','tolerance')}
        compare.append(dict(profile=x['profile'],metric=x['metric'],deltas=delta,passed_agrees=other['passed']==x['passed']))
        verify(all(abs(v)<1e-10 for v in delta.values()) and other['passed']==x['passed'], 'new_report_metric_agreement', 'analysis/VALIDATION_REPORT.json')
    verify(report['accepted']==(all(x['passed'] for x in metrics) and not unsupported), 'new_report_acceptance_agreement', 'analysis/VALIDATION_REPORT.json')
    verify(report['plan_sha256']==sha(BASE/'protocol/PLAN.json') and report['model_sha256']==sha(BASE/'model/CONDITIONAL_MODEL.json'), 'new_report_frozen_hashes', 'analysis/VALIDATION_REPORT.json')

# Read and independently verify recovery primary artifacts, not only reported PASS.
recovery=BASE/'analysis/recovery_20261010'
probe=read(recovery/'RUNTIME_PROBE.json')
started=read(recovery/'ANALYSIS_STARTED.json')
after=read(recovery/'AFTER_ANALYSIS_SNAPSHOT.json')
state=read(recovery/'RECOVERY_STATE.json')
strict=read(recovery/'STRICT_CONFIRMATION_AUDIT.json')
verify(probe['protected_snapshot']==started['protected_snapshot']==after, 'recovery_before_after_snapshots', recovery)
for name, expected in after['protected_files'].items():
    verify(sha(BASE/name)==expected, 'recovery_snapshot_hash', name)
analysis_source=recovery/'analyze_confirmation_isolated_analysis_run_source.py'
verify(started['entry_sha256']==sha(analysis_source), 'recovery_analysis_run_source_hash', 'analysis/recovery_20261010/ANALYSIS_STARTED.json:5')
current_source=BASE/'scripts/analyze_confirmation_isolated.py'
sha(current_source)
old_text=analysis_source.read_text(encoding='utf-8')
new_text=current_source.read_text(encoding='utf-8')
verify(old_text.split('def export_worker():')[0]==new_text.split('def export_worker():')[0], 'unchanged_probe_analysis_semantics', current_source)
for item in probe['results']:
    for stream in ('stdout','stderr'):
        verify(sha(recovery/(item['label']+'_'+stream+'.log'))==item[stream+'_sha256'], 'probe_log_hash', item['label']+'_'+stream)
verify(strict['rows']==len(allrows) and strict['blocks']==len(jobs), 'recovery_strict_counts', 'analysis/recovery_20261010/STRICT_CONFIRMATION_AUDIT.json:2')
verify(len(strict['measurement_sha256'])==240, 'recovery_strict_hash_count', 'analysis/recovery_20261010/STRICT_CONFIRMATION_AUDIT.json')
for name,expected in strict['measurement_sha256'].items():
    verify(sha(BASE/name)==expected, 'recovery_strict_measurement_hash', name)
verify(state['validation_report_sha256']==sha(BASE/'analysis/VALIDATION_REPORT.json'), 'recovery_report_hash', 'analysis/recovery_20261010/RECOVERY_STATE.json:6')
verify(not (BASE/'protocol/VALIDATION_ACCEPTED.json').exists() and not (BASE/'analysis/TEST_REPORT.json').exists(), 'no_retained_test_gate_report', 'protocol/VALIDATION_ACCEPTED.json and analysis/TEST_REPORT.json')
for label in ('isolated_statistics','isolated_frozen_check'):
    verify((recovery/(label+'_stderr.log')).read_text(encoding='utf-8')=='', 'isolated_child_stderr_empty', label)
stats_log=(recovery/'isolated_statistics_stdout.log').read_text(encoding='utf-8').splitlines()
emitted=json.loads(stats_log[-1])
verify(emitted['criteria']==report['criteria'] and emitted['accepted']==report['accepted'], 'statistics_called_output_agreement', 'analysis/recovery_20261010/isolated_statistics_stdout.log:2')
for a,b in zip(rows(BASE/'analysis/validation_errors.csv'),metrics):
    verify(a['profile']==b['profile'] and a['metric']==b['metric'] and all(abs(float(a[k])-b[k])<1e-10 for k in ('predicted_ms','measured_ms','relative_error','tolerance')) and a['passed']==str(b['passed']), 'errors_csv_agreement', 'analysis/validation_errors.csv')
old_report=read(OLD/'calibration/VALIDATION_REPORT.json')
export=read(recovery/'EXPORT.json')
consolidated=rows(BASE/'analysis/v2_service_requests.csv')
original=[{k:v for k,v in r.items() if k!='_evidence'} for r in allrows]
verify(consolidated==original and export['rows']==len(original), 'exact_consolidated_source_rows', 'analysis/v2_service_requests.csv:2')
for name,expected in export['output_sha256'].items():
    verify(sha(BASE/name)==expected, 'export_output_hash', name)
pdf=(BASE/'figures/fig_visual_service_v2.pdf').read_bytes()
streams=[]
for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',pdf,re.S):
    data=match.group(1)
    try:data=zlib.decompress(data)
    except zlib.error:pass
    streams.append(data)
pdf_contents=b'\n'.join(streams)
pdf_labels=[f"{x['measured_ms']:.2f}" for x in metrics]
for label in pdf_labels:
    verify(label.encode() in pdf_contents, 'pdf_measured_value_labels', label)
model_size_counts=Counter(size(e) for e in cal_entries.values())
additional=dict(recovery_entry_sha256=started['entry_sha256'],current_recovery_entry_sha256=hashes[str(current_source)],export_scope=export['scope'],pdf_value_labels=pdf_labels,recovery_snapshots_protected_file_count=len(after['protected_files']),original_v1_acceptance=old_report.get('accepted'),minimal_mixed_probe_exit_code=probe['mixed_process_exit_code'],original_interruption_reproduced_by_probe=False,nearest_distance_max=max(x['nearest_distance'] for x in mapping.values()),calibration_size_counts=dict(model_size_counts),validation_size_counts=dict(Counter(size(e) for e in splits['validation'])),observed_runtime_versions=env['versions'],bootstrap_numeric_endpoints_independently_recomputed=False,bootstrap_review='Source inspected: 500 crossed image-within-cell and entire-session resamples; measured-distribution CI only, not acceptance error CI')

result=dict(schema='independent_stdlib_confirmation_audit_v1',reviewer='gpt-6-astra xhigh; Type-A second GPT-family reviewer, not Type-B cross-family',checks=dict(checks),issues=issues,strict_integrity_checks_passed=not issues,criteria=metrics,summary=summary,accepted=all(x['passed'] for x in metrics) and not unsupported and not issues,quantile_convention='left empirical CDF inverse with exact rational population weights; no interpolation',quantile_witness=quantile_witness,unsupported_inputs=unsupported,session_intervals=session_times,idle_gate_start_gaps_seconds=gaps,reference_drift_diagnostic_only=reference,calibration_rows=len(cal_rows),confirmation_rows=len(allrows),target_population_images=len(eligible),nonempty_size_class_cells=len(sample_counts),singleton_size_class_cells=[k for k,v in sample_counts.items() if v==1],new_report_comparison=compare,additional=additional,protected_file_hashes=hashes,notes=['No runner import or run/prepare/check/repair/seal execution','No PyTorch/NumPy import, measurements, model inference, package install or unsafe OpenMP workaround','No retained-test image/label or measurement access; only frozen split metadata and empty-directory check','Checks do not independently authenticate historical OS execution; file/code/timestamps provide convergent record evidence'])
(OUT/'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
with (OUT/'INDEPENDENT_METRICS.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print(json.dumps({k:result[k] for k in ('checks','issues','criteria','strict_integrity_checks_passed','accepted','session_intervals','idle_gate_start_gaps_seconds','target_population_images','nonempty_size_class_cells','singleton_size_class_cells','new_report_comparison')},ensure_ascii=False,indent=2))
