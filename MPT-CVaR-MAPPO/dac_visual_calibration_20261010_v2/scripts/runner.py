"""Isolated, no-restart V2 protocol, conditional service model and strict audit."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import importlib.util
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from v2_common import BASE, OLD, SEED, PROFILES, STAGES, BOUNDARY, read, save, sha, csv_read, csv_write, stratum, stem

SOURCE_NAMES = ('v2_common.py', 'measure_block.py', 'runner.py')

def proportional(counts, total, minimum=0):
    quotas = {k: min(v, minimum) for k, v in counts.items()}
    while sum(quotas.values()) < total:
        candidates = [k for k in counts if quotas[k] < counts[k]]
        if not candidates:
            raise RuntimeError('Insufficient eligible images')
        k = max(candidates, key=lambda k: (total * counts[k] / sum(counts.values()) - quotas[k], -int(k)))
        quotas[k] += 1
    if sum(quotas.values()) != total:
        raise RuntimeError('Minimum quotas exceed requested size')
    return quotas

class NearIndex:
    def __init__(self):
        self.parts = defaultdict(list)
        self.hashes = set()
    def near(self, e):
        if e['image_sha256'] in self.hashes:
            return True
        value = int(e['dhash64'], 16)
        for offset, width in [(0, 22), (22, 21), (43, 21)]:
            for old in self.parts[(offset, (value >> offset) & ((1 << width)-1))]:
                if (value ^ old).bit_count() <= 2:
                    return True
        return False
    def add(self, e):
        value = int(e['dhash64'], 16)
        self.hashes.add(e['image_sha256'])
        for offset, width in [(0, 22), (22, 21), (43, 21)]:
            self.parts[(offset, (value >> offset) & ((1 << width)-1))].append(value)

def eligible_inventory():
    old_splits=read(OLD/'protocol/IMAGE_SPLITS.json')
    manifest=csv_read(OLD/'protocol/IMAGE_MANIFEST.csv')
    for e in manifest:
        for field in ('width','height','payload_bytes'):
            e[field]=int(e[field])
    used=[e for role in old_splits.values() for e in role]
    index=NearIndex()
    for e in used:index.add(e)
    used_ids={e['image_id'] for e in used}
    candidates=sorted([e for e in manifest if e['official_split']=='train' and e['image_id'] not in used_ids],key=lambda e:e['image_id'])
    random.Random(SEED).shuffle(candidates)
    eligible=[]
    for e in candidates:
        if not index.near(e):
            eligible.append(e);index.add(e)
    return eligible,len(candidates)-len(eligible)

def population_weights(eligible,validation):
    population=Counter((stratum(e),e['class_id']) for e in eligible)
    sampled=Counter((stratum(e),e['class_id']) for e in validation)
    if set(population)!=set(sampled):
        raise RuntimeError('Nonempty population cells lack confirmation support')
    for e in validation:
        key=(stratum(e),e['class_id'])
        e['target_weight']=population[key]/len(eligible)/sampled[key]
    return dict(eligible_images=len(eligible),size_class_counts={f'{s}|{c}':n for (s,c),n in sorted(population.items())},
                selected_size_class_counts={f'{s}|{c}':n for (s,c),n in sorted(sampled.items())},
                estimand='Uniform deduplicated eligible official train inventory, inverse size-class inclusion probability weights')

def choose(entries, total, minimum):
    by_size = defaultdict(list)
    for e in entries:
        by_size[stratum(e)].append(e)
    quotas = proportional({k: len(v) for k, v in by_size.items()}, total, minimum)
    selected = []
    for size in sorted(by_size):
        by_class = defaultdict(list)
        for e in by_size[size]:
            by_class[e['class_id']].append(e)
        # Class identifiers are strings: allocation tie-break is deterministic lexical order.
        order = {c: n for n, c in enumerate(sorted(by_class))}
        class_quota = proportional({order[c]: len(v) for c, v in by_class.items()}, quotas[size])
        for c in sorted(by_class):
            pool = sorted(by_class[c], key=lambda e: e['image_id'])
            random.Random(f'{SEED}:{total}:{size}:{c}').shuffle(pool)
            selected.extend(pool[:class_quota[order[c]]])
    return sorted(selected, key=lambda e: e['image_id']), quotas

def old_calibration():
    sys.path.insert(0, str(OLD / 'scripts'))
    import measurement_audit
    report, rows = measurement_audit.audit_role('calibration')
    if len(rows) != 20000 or report['paired_prediction_disagreements']:
        raise RuntimeError('V1 calibration is not eligible')
    return report, rows

def neighbors(cal_entries, entries):
    out = {}
    for e in entries:
        x = (math.log2(int(e['width'])*int(e['height'])), math.log2(int(e['payload_bytes'])))
        distances = []
        for c in cal_entries:
            if stratum(c) != stratum(e):
                continue
            y = (math.log2(int(c['width'])*int(c['height'])), math.log2(int(c['payload_bytes'])))
            d = math.hypot(x[0]-y[0], x[1]-y[1])
            distances.append((d, c['image_id']))
        if not distances:
            raise RuntimeError('Unsupported size stratum')
        distances.sort()
        top = distances[:20]
        out[e['image_id']] = dict(neighbor_ids=[image_id for _, image_id in top],
                                 nearest_distance=top[0][0], supported=top[0][0] <= 2.0)
    return out

def jobs_for_session(splits, session, test=False):
    rng = random.Random(f'{SEED}:jobs:{session}:{test}')
    roles = ['test'] if test else ['validation', 'reference']
    chunks = {}
    for role in roles:
        ids = [e['image_id'] for e in splits[role]]
        rng.shuffle(ids)
        size = 200 if role != 'reference' else 20
        chunks[role] = [ids[i:i+size] for i in range(0, len(ids), size)]
    jobs = []
    for b in range(len(chunks[roles[0]])):
        role_order = roles.copy()
        rng.shuffle(role_order)
        for role in role_order:
            profiles = list(PROFILES)
            if (session+b) % 2 == 0:
                profiles.reverse()
            for profile in profiles:
                jobs.append(dict(role=role, session=session, block=b+1, profile=profile,
                                 image_ids=chunks[role][b]))
    return jobs

def resources():
    import psutil
    detector_path = Path('C:/Users/CKX/.codex/skills/get-available-resources/scripts/detect_resources.py')
    spec = importlib.util.spec_from_file_location('resource_skill', detector_path)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    disk = shutil.disk_usage(BASE)
    return dict(cpu=detector.get_cpu_info(), memory=detector.get_memory_info(),
                disk=dict(total_gb=disk.total/2**30, available_gb=disk.free/2**30),
                gpu=detector.detect_nvidia_gpus(),
                detector_note='Full detector failed: missing disk.available_gb; skill probes + shutil fallback. Process-only Windows paths restored before NVML.',
                strategy='CPU timing only, sequential independent blocks; no GPU allocation or installs')

def prepare():
    if (BASE / 'protocol/PLAN.json').exists():
        raise FileExistsError('V2 already frozen; prepare is not a restart')
    for folder in ('protocol','logs','analysis','model','figures','measurements/reference','measurements/validation','measurements/test'):
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    save(BASE / 'protocol/RESOURCES.json', resources(), exclusive=True)
    for name in ('ENVIRONMENT_SPEC.json','CPU_TOPOLOGY.json'):
        save(BASE / 'protocol' / name, read(OLD / 'protocol' / name), exclusive=True)
    report, rows = old_calibration()
    save(BASE / 'protocol/V1_CALIBRATION_INTEGRITY.json', report, exclusive=True)
    old_splits = read(OLD / 'protocol/IMAGE_SPLITS.json')
    manifest = csv_read(OLD / 'protocol/IMAGE_MANIFEST.csv')
    for e in manifest:
        for field in ('width','height','payload_bytes'):
            e[field] = int(e[field])
    used = [e for role in old_splits.values() for e in role]
    index = NearIndex()
    for e in used:
        index.add(e)
    used_ids = {e['image_id'] for e in used}
    candidates = sorted([e for e in manifest if e['official_split'] == 'train' and e['image_id'] not in used_ids], key=lambda e: e['image_id'])
    random.Random(SEED).shuffle(candidates)
    eligible = []
    for e in candidates:
        if not index.near(e):
            eligible.append(e)
            index.add(e)
    validation, quotas = choose(eligible, 2000, 25)
    reference, _ = choose(old_splits['calibration'], 200, 2)
    counts = Counter(stratum(e) for e in eligible)
    population_profile=population_weights(eligible,validation)
    save(BASE/'protocol/POPULATION_PROFILE.json',population_profile,exclusive=True)
    splits = dict(validation=validation, reference=reference, test=old_splits['test'])
    save(BASE / 'protocol/SPLITS.json', splits, exclusive=True)
    paired = defaultdict(dict)
    for r in rows:
        key = (r['image_id'], int(r['session_id']))
        paired[key][r['profile_id']] = {k: float(r[k]) for k in (*STAGES, 'service_ms')}
    model = dict(kind='pixel-stratum log2(pixel, JPEGbytes) kNN empirical paired mixture', k=20,
                 strata_upper_bounds=[250000,500000,1000000,2000000], support_distance=2.0,
                 training_entries=old_splits['calibration'],
                 paired_rows=[dict(image_id=i, session=s, profiles=p) for (i,s),p in sorted(paired.items())],
                 fitted_from='V1 calibration only; V1 validation used for design diagnosis, not timing fit')
    save(BASE / 'model/CONDITIONAL_MODEL.json', model, exclusive=True)
    # Only confirmation covariates are mapped before timing; retained test labels remain inaccessible.
    save(BASE / 'model/VALIDATION_NEIGHBORS.json', neighbors(old_splits['calibration'], validation), exclusive=True)
    sessions = [dict(kind='confirmation', session=s, jobs=jobs_for_session(splits,s)) for s in range(1,4)]
    sessions += [dict(kind='retained_test', session=s, jobs=jobs_for_session(splits,s,True)) for s in range(1,6)]
    protected = ['protocol/IMAGE_SPLITS.json','protocol/IMAGE_MANIFEST.csv','protocol/IMAGE_FREEZE.json',
                 'protocol/ENVIRONMENT_SPEC.json','protocol/CPU_TOPOLOGY.json','protocol/FORMAL_EXECUTION_PLAN.json',
                 'protocol/SERVICE_DISTRIBUTION_VALIDATION_RULE.json','calibration/VALIDATION_REPORT.json',
                 'protocol/CPU_WITNESS.json','protocol/PILOT_GATE.json',
                 'protocol/FORMAL_EXECUTION_STATE.json','STATUS.json',
                 'logs/formal_runner_stdout.log','logs/formal_runner_stderr.log',
                 'models/resnet18-f37072fd.pth']
    protected += [f'scripts/{x}' for x in ['common.py','windows_runtime.py','pipeline.py','measure_block.py','measurement_audit.py','service_summary.py','formal_runner.py']]
    old_hashes = {p: sha(OLD / p) for p in protected}
    old_hashes.update(report['measurement_sha256'])
    for p in (OLD / 'measurements').rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json'):
            old_hashes[str(p.relative_to(OLD))] = sha(p)
    plan = dict(seed=SEED, frozen_at=datetime.now().astimezone().isoformat(), sessions=sessions,
                spacing_seconds=900, maximum_idle_wait_seconds=3600, machine_idle_percent=35,
                selected_core_idle_percent=60, minimum_memory_bytes=2*2**30,
                confirmation_rows=13200, retained_test_rows=10000, warmup=200,
                thresholds=dict(mean=.10,p95=.15,p99=.20), bootstrap_replicates=500,
                validation_population=dict(eligible_images=len(eligible), size_counts=dict(counts),
                                           selected_quotas=quotas, duplicates_excluded=len(candidates)-len(eligible)),
                weights_sha256=sha(OLD / 'models/resnet18-f37072fd.pth'), old_readonly_sha256=old_hashes,
                source_sha256={n: sha(BASE / 'scripts' / n) for n in SOURCE_NAMES},
                artifact_sha256={p: sha(BASE / p) for p in ['README_中文.md','protocol/SPLITS.json',
                'protocol/ENVIRONMENT_SPEC.json','protocol/CPU_TOPOLOGY.json','model/CONDITIONAL_MODEL.json',
                'model/VALIDATION_NEIGHBORS.json','protocol/POPULATION_PROFILE.json']})
    save(BASE / 'protocol/PLAN.json', plan, exclusive=True)
    print('V2_PREPARED', len(eligible), dict(counts), quotas, flush=True)

def seal_preflight():
    """Only before RUN_STARTED: preserve pre-review plan, strengthen immutable guards."""
    if (BASE/'protocol/RUN_STARTED.json').exists():
        raise RuntimeError('No preflight revision is allowed after measurement launch')
    path=BASE/'protocol/PLAN.json';plan=read(path)
    save(BASE/'protocol/PLAN_PRE_REVIEW_1.json',plan,exclusive=True)
    for p in (OLD/'measurements').rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json'):
            plan['old_readonly_sha256'][str(p.relative_to(OLD))]=sha(p)
    for name in ['protocol/CPU_WITNESS.json','protocol/PILOT_GATE.json',
                 'protocol/FORMAL_EXECUTION_STATE.json','STATUS.json',
                 'logs/formal_runner_stdout.log','logs/formal_runner_stderr.log']:
        plan['old_readonly_sha256'][name]=sha(OLD/name)
    for name in SOURCE_NAMES:
        plan['source_sha256'][name]=sha(BASE/'scripts'/name)
    for name in plan['artifact_sha256']:
        plan['artifact_sha256'][name]=sha(BASE/name)
    save(path,plan)
    save(BASE/'protocol/PRE_LAUNCH_REPAIR.json',dict(at=datetime.now().astimezone().isoformat(),
         reason='Independent preflight found V1 validation raw files were not traversed by check; bind all V1 raw files and failure runtime evidence.',
         before_sha256=sha(BASE/'protocol/PLAN_PRE_REVIEW_1.json'),after_sha256=sha(path),
         measurements_started=False,split_or_model_or_threshold_changed=False),exclusive=True)
    print('V2_PREFLIGHT_RESEALED',sha(path),flush=True)

def repair_poststratification():
    if (BASE/'protocol/RUN_STARTED.json').exists():
        raise RuntimeError('No design revision after measurement launch')
    path=BASE/'protocol/PLAN.json';plan=read(path)
    splits=read(BASE/'protocol/SPLITS.json')
    save(BASE/'protocol/PLAN_PRE_REVIEW_2.json',plan,exclusive=True)
    save(BASE/'protocol/SPLITS_PRE_REVIEW_2.json',splits,exclusive=True)
    eligible,excluded=eligible_inventory()
    if len(eligible)!=plan['validation_population']['eligible_images'] or excluded!=plan['validation_population']['duplicates_excluded']:
        raise RuntimeError('Eligible population unexpectedly changed')
    profile=population_weights(eligible,splits['validation'])
    save(BASE/'protocol/POPULATION_PROFILE.json',profile,exclusive=True)
    save(BASE/'protocol/SPLITS.json',splits)
    plan['artifact_sha256']['protocol/POPULATION_PROFILE.json']=sha(BASE/'protocol/POPULATION_PROFILE.json')
    for name in plan['artifact_sha256']:plan['artifact_sha256'][name]=sha(BASE/name)
    for name in SOURCE_NAMES:plan['source_sha256'][name]=sha(BASE/'scripts'/name)
    save(path,plan)
    save(BASE/'protocol/PRE_LAUNCH_WEIGHT_REPAIR.json',dict(at=datetime.now().astimezone().isoformat(),
         reason='Independent audit required exact size-class inverse inclusion weights and matching crossed bootstrap strata before timing.',
         before_sha256=sha(BASE/'protocol/PLAN_PRE_REVIEW_2.json'),after_sha256=sha(path),
         measurements_started=False,selected_image_ids_changed=False,model_or_threshold_changed=False),exclusive=True)
    print('V2_POSTSTRATIFICATION_REPAIRED',sha(path),flush=True)

def check():
    plan = read(BASE / 'protocol/PLAN.json')
    for name, expected in plan['source_sha256'].items():
        if sha(BASE / 'scripts' / name) != expected:
            raise RuntimeError(f'V2 source changed: {name}')
    for name, expected in plan['artifact_sha256'].items():
        if sha(BASE / name) != expected:
            raise RuntimeError(f'V2 artifact changed: {name}')
    for name, expected in plan['old_readonly_sha256'].items():
        if sha(OLD / name) != expected:
            raise RuntimeError(f'V1 evidence changed: {name}')
    splits = read(BASE / 'protocol/SPLITS.json')
    if [len(splits[r]) for r in ('validation','reference','test')] != [2000,200,1000]:
        raise RuntimeError('Invalid role counts')
    if splits['test'] != read(OLD / 'protocol/IMAGE_SPLITS.json')['test']:
        raise RuntimeError('Retained test changed')
    if not math.isclose(sum(e['target_weight'] for e in splits['validation']), 1, abs_tol=1e-12):
        raise RuntimeError('Invalid target weights')
    population=read(BASE/'protocol/POPULATION_PROFILE.json')
    for e in splits['validation']:
        key=f"{stratum(e)}|{e['class_id']}"
        expected=population['size_class_counts'][key]/population['eligible_images']/population['selected_size_class_counts'][key]
        if e['target_weight']!=expected:
            raise RuntimeError('Target weight changed')
    for item in plan['sessions']:
        if item['jobs'] != jobs_for_session(splits,item['session'],item['kind']=='retained_test'):
            raise RuntimeError('Job schedule changed')
    mappings = read(BASE / 'model/VALIDATION_NEIGHBORS.json')
    if any(not v['supported'] for v in mappings.values()):
        raise RuntimeError('Confirmation inputs lack model support; do not silently remove them')
    import torch
    import importlib.metadata
    import platform
    versions = read(BASE / 'protocol/ENVIRONMENT_SPEC.json')['versions']
    for package, expected in versions.items():
        actual = platform.python_version() if package == 'python' else importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError('Environment changed')
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    x = torch.randn(64,64)
    y = x @ torch.eye(64)
    witness = __import__('hashlib').sha256(y.numpy().tobytes()).hexdigest()
    if not torch.isfinite(y).all() or not torch.equal(x,y) or witness != read(OLD / 'protocol/CPU_WITNESS.json')['tensor_sha256']:
        raise RuntimeError('CPU kernel witness mismatch')
    print('V2_FROZEN_PROTOCOL_AND_CPU_WITNESS_OK', witness, flush=True)

def audit(test=False):
    plan = read(BASE / 'protocol/PLAN.json')
    splits = read(BASE / 'protocol/SPLITS.json')
    expected_jobs = [j for s in plan['sessions'] if (s['kind']=='retained_test')==test for j in s['jobs']]
    entries = {r: {e['image_id']:e for e in es} for r,es in splits.items()}
    versions = read(BASE / 'protocol/ENVIRONMENT_SPEC.json')['versions']
    cpus = read(BASE / 'protocol/CPU_TOPOLOGY.json')['logical_cpu_per_physical_core']
    rows, intervals, hashes, pairs = [], [], {}, defaultdict(dict)
    for job in expected_jobs:
        path = BASE / 'measurements' / job['role'] / (stem(job)+'.csv')
        meta = read(path.with_suffix('.json'))
        job_path = BASE / 'protocol/jobs' / (stem(job)+'.json')
        threads = 1 if job['profile']=='CPU-1T' else 4
        if (read(job_path)!=job or meta['job']!=job or meta['job_sha256']!=sha(job_path)
            or meta['versions']!=versions or meta['weights_sha256']!=plan['weights_sha256']
            or meta['warmup_requests']!=200 or meta['errors']!=0 or meta['timing_boundary']!=BOUNDARY
            or meta['affinity']!=cpus[:threads]):
            raise RuntimeError('Metadata mismatch '+stem(job))
        block = csv_read(path)
        if len(block)!=meta['requests'] or [r['image_id'] for r in block]!=job['image_ids']:
            raise RuntimeError('Missing/reordered requests')
        start,end = [datetime.fromisoformat(meta[k]['wall_clock']) for k in ('sensor_before','sensor_after')]
        last = start
        for n,r in enumerate(block,1):
            e=entries[job['role']][r['image_id']]
            if (int(r['session_id'])!=job['session'] or int(r['block_id'])!=job['block'] or int(r['run_order'])!=n
                or r['profile_id']!=job['profile'] or r['image_split']!=job['role'] or r['device']!='cpu'
                or int(r['cpu_threads'])!=threads or __import__('json').loads(r['cpu_affinity'])!=cpus[:threads]
                or r['weights_sha256']!=plan['weights_sha256'] or r['image_sha256']!=e['image_sha256']
                or r['model_id']!='ResNet18_IMAGENET1K_V1' or r['class_id']!=e['class_id']
                or any(int(r[k])!=int(e[k]) for k in ('width','height','payload_bytes')) or r['error_status']!='ok'
                or not 0<=int(r['predicted_class'])<1000):
                raise RuntimeError('Identity mismatch '+stem(job))
            stage=[float(r[k]) for k in STAGES]
            total=float(r['service_ms'])
            if not all(math.isfinite(v) and v>=0 for v in stage) or not math.isfinite(total) or total<=0 or not math.isclose(sum(stage),total,abs_tol=1e-8,rel_tol=1e-10):
                raise RuntimeError('Invalid timing')
            stamp=datetime.fromisoformat(r['measurement_timestamp'])
            if not start<=stamp<=end or stamp<last:
                raise RuntimeError('Invalid timestamp')
            last=stamp
            pairs[(job['role'],job['session'],r['image_id'])][r['profile_id']]=r['predicted_class']
        rows.extend(block)
        intervals.append((start,end,stem(job)))
        for p in (path,path.with_suffix('.json')):
            hashes[str(p.relative_to(BASE))]=sha(p)
    for role in (['test'] if test else ['validation','reference']):
        actual={p.name for p in (BASE/'measurements'/role).iterdir() if p.suffix in ('.csv','.json')}
        expected={stem(j)+suffix for j in expected_jobs if j['role']==role for suffix in ('.csv','.json')}
        if actual!=expected:
            raise RuntimeError('Extra/missing measurement files')
    intervals.sort()
    if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
        raise RuntimeError('Concurrent measurements')
    if any(set(p)!=set(PROFILES) or len(set(p.values()))!=1 for p in pairs.values()):
        raise RuntimeError('Missing paired profiles or prediction disagreement')
    if len(rows)!=(10000 if test else 13200):
        raise RuntimeError('Incomplete row count')
    return dict(passed=True, rows=len(rows), blocks=len(expected_jobs), measurement_sha256=hashes,
                first_start=intervals[0][0].isoformat(), last_end=intervals[-1][1].isoformat()), rows

def weighted_stats(values, weights):
    import numpy as np
    a=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
    order=np.argsort(a,kind='stable'); a=a[order]; w=w[order]
    if not len(a) or not np.isfinite(a).all() or (w<0).any() or w.sum()<=0:
        raise RuntimeError('Invalid weighted observations')
    w=w/w.sum(); c=np.cumsum(w)
    return dict(mean=float(np.dot(a,w)), p95=float(a[min(int(np.searchsorted(c,.95)),len(a)-1)]),
                p99=float(a[min(int(np.searchsorted(c,.99)),len(a)-1)]))

def evaluate(test=False):
    integrity,rows=audit(test)
    plan=read(BASE/'protocol/PLAN.json'); splits=read(BASE/'protocol/SPLITS.json')
    role='test' if test else 'validation'
    mapping=neighbors(read(BASE/'model/CONDITIONAL_MODEL.json')['training_entries'],splits[role]) if test else read(BASE/'model/VALIDATION_NEIGHBORS.json')
    unsupported=[i for i,v in mapping.items() if not v['supported']]
    model=read(BASE/'model/CONDITIONAL_MODEL.json')
    pairs=defaultdict(list)
    for r in model['paired_rows']:
        pairs[r['image_id']].append(r)
    input_weights={e['image_id']:(1/len(splits[role]) if test else e['target_weight']) for e in splits[role]}
    sessions=5 if test else 3
    results=[]; summary={}; uncertainty={}
    for profile in PROFILES:
        predicted=[]; pw=[]
        for e in splits[role]:
            near=mapping[e['image_id']]['neighbor_ids']
            for image_id in near:
                for r in pairs[image_id]:
                    predicted.append(r['profiles'][profile]['service_ms'])
                    pw.append(input_weights[e['image_id']]/len(near)/len(pairs[image_id]))
        measured=[r for r in rows if r['image_split']==role and r['profile_id']==profile]
        actual=[float(r['service_ms']) for r in measured]
        aw=[input_weights[r['image_id']]/sessions for r in measured]
        pred=weighted_stats(predicted,pw); obs=weighted_stats(actual,aw)
        summary[profile]=dict(predicted_ms=pred, measured_ms=obs)
        for metric in ('mean','p95','p99'):
            error=abs(pred[metric]-obs[metric])/obs[metric]
            results.append(dict(profile=profile,metric=metric,predicted_ms=pred[metric],measured_ms=obs[metric],
                                relative_error=error,tolerance=plan['thresholds'][metric],passed=error<=plan['thresholds'][metric]))
        # Crossed resampling of independent images (within design strata) and entire sessions.
        import numpy as np
        values={(r['image_id'],int(r['session_id'])):float(r['service_ms']) for r in measured}
        by_size=defaultdict(list)
        for e in splits[role]:
            by_size[(stratum(e),e['class_id'])].append(e['image_id'])
        rng=np.random.default_rng(SEED+int(test)); boots=defaultdict(list)
        for _ in range(plan['bootstrap_replicates']):
            selected_sessions=rng.integers(1,sessions+1,size=sessions)
            bv=[];bw=[]
            for group in by_size.values():
                for image_id in rng.choice(group,size=len(group),replace=True):
                    for s in selected_sessions:
                        bv.append(values[(image_id,int(s))]);bw.append(input_weights[image_id]/sessions)
            bs=weighted_stats(bv,bw)
            for k,v in bs.items():
                boots[k].append(v)
        uncertainty[profile]={k:[float(v) for v in np.quantile(vals,[.025,.975])] for k,vals in boots.items()}
    accepted=all(r['passed'] for r in results) and not unsupported
    result=dict(accepted=accepted,role=role,criteria=results,summary=summary,
                measured_crossed_image_session_bootstrap_95ci_ms=uncertainty,
                bootstrap_scope='Measured-distribution CI only; not model-selection adjusted or CI of prediction error',
                unsupported_inputs=unsupported, integrity=integrity,plan_sha256=sha(BASE/'protocol/PLAN.json'),
                model_sha256=sha(BASE/'model/CONDITIONAL_MODEL.json'), reference_used_to_refit=False,
                evidence_scope='Single-machine repeated CPU-profile service measurement; not queue validation or QLA benefit')
    save(BASE/f'analysis/{role.upper()}_REPORT.json',result,exclusive=True)
    csv_write(BASE/f'analysis/{role}_errors.csv',results,exclusive=True)
    if not test:
        reference=[]
        for s in range(1,4):
            for profile in PROFILES:
                group=[float(r['service_ms']) for r in rows if r['image_split']=='reference' and int(r['session_id'])==s and r['profile_id']==profile]
                reference.append(dict(session=s,profile=profile,**weighted_stats(group,[1]*len(group))))
        csv_write(BASE/'analysis/reference_drift.csv',reference,exclusive=True)
    lines=['# V2 '+('保留测试' if test else '确认验证')+'结果','',f'整体验收：{accepted}。原 V1 失败结论未更改。','',
           '| CPU | 指标 | 预测 ms | 实测 ms | 相对误差 | 容差 | 通过 |','|---|---|---:|---:|---:|---:|---|']
    lines += [f"| {r['profile']} | {r['metric']} | {r['predicted_ms']:.4f} | {r['measured_ms']:.4f} | {r['relative_error']:.2%} | {r['tolerance']:.0%} | {r['passed']} |" for r in results]
    lines += ['', '这是固定预训练视觉模型的服务分布校准；未训练视觉模型，未接入卸载模拟器，未证明 QLA 收益。',
              '验证统计按预登记目标库存分层加权；参考块不计入验证统计，也不用于漂移归一化。',
              '图片与会话的交叉重采样区间见 JSON；它是实测分布区间，不是预测误差区间。',
              f'缺乏局部支持的输入数：{len(unsupported)}。']
    path=BASE/f'analysis/V2_{"保留测试" if test else "确认验证"}结果_中文.md'
    with path.open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    if not accepted:
        raise RuntimeError(role+' service-distribution validation failed; evidence retained, no retries')
    return result

def idle_gate(item):
    import psutil
    plan=read(BASE/'protocol/PLAN.json'); observations=[]; deadline=time.monotonic()+plan['maximum_idle_wait_seconds']
    cpus=read(BASE/'protocol/CPU_TOPOLOGY.json')['logical_cpu_per_physical_core'][:4]
    while True:
        per=psutil.cpu_percent(interval=5,percpu=True)
        row=dict(at=datetime.now().astimezone().isoformat(),mean=sum(per)/len(per),selected_max=max(per[i] for i in cpus),
                 available_memory_bytes=psutil.virtual_memory().available)
        observations.append(row)
        if row['mean']<=plan['machine_idle_percent'] and row['selected_max']<=plan['selected_core_idle_percent'] and row['available_memory_bytes']>=plan['minimum_memory_bytes']:
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Idle gate timed out; not dropping sessions')
        time.sleep(25)
    save(BASE/f"protocol/idle_{item['kind']}_s{item['session']}.json",observations,exclusive=True)

def run():
    lock=BASE/'protocol/RUN_STARTED.json'
    save(lock,dict(pid=__import__('os').getpid(),started_at=datetime.now().astimezone().isoformat()),exclusive=True)
    state=dict(status='preflight',pid=__import__('os').getpid(),completed_blocks=0,completed_rows=0)
    state_path=BASE/'protocol/STATE.json'
    try:
        check()
        review=read(BASE/'protocol/ENTRY_REVIEW.json')
        if not review.get('passed') or review['plan_sha256']!=sha(BASE/'protocol/PLAN.json'):
            raise RuntimeError('Missing independent entry review')
        plan=read(BASE/'protocol/PLAN.json'); last_start=None
        for item in plan['sessions']:
            if item['kind']=='retained_test' and item['session']==1:
                state['status']='auditing_confirmation';save(state_path,state)
                evaluate(False)
                save(BASE/'protocol/VALIDATION_ACCEPTED.json',dict(plan_sha256=sha(BASE/'protocol/PLAN.json'),
                     report_sha256=sha(BASE/'analysis/VALIDATION_REPORT.json')),exclusive=True)
            check()
            state.update(status='waiting_for_session',kind=item['kind'],session=item['session']);save(state_path,state)
            if last_start is not None:
                while time.monotonic()-last_start<plan['spacing_seconds']:
                    time.sleep(min(30,plan['spacing_seconds']-(time.monotonic()-last_start)))
            idle_gate(item)
            last_start=time.monotonic()
            state.update(status='measuring',session_started_at=datetime.now().astimezone().isoformat());save(state_path,state)
            for job in item['jobs']:
                job_path=BASE/'protocol/jobs'/(stem(job)+'.json')
                save(job_path,job,exclusive=True)
                for suffix in ('.csv','.json'):
                    if (BASE/'measurements'/job['role']/(stem(job)+suffix)).exists():
                        raise FileExistsError('Refuse measurement overwrite')
                log=BASE/'logs'/(stem(job)+'.log')
                with log.open('x',encoding='utf-8') as f:
                    completed=subprocess.run([sys.executable,'-X','utf8',str(BASE/'scripts/measure_block.py'),'--job',str(job_path)],
                                             stdout=f,stderr=subprocess.STDOUT,check=False)
                if completed.returncode:
                    raise RuntimeError(f'Block failed: {stem(job)}, exit={completed.returncode}')
                state['completed_blocks']+=1;state['completed_rows']+=len(job['image_ids'])
                state['last_block']=stem(job);save(state_path,state)
                print('BLOCK_COMPLETE',stem(job),state['completed_rows'],flush=True)
        state['status']='auditing_retained_test';save(state_path,state)
        evaluate(True)
        check()
        state.update(status='complete',ended_at=datetime.now().astimezone().isoformat());save(state_path,state)
        print('V2_COMPLETE',flush=True)
    except Exception as error:
        state.update(status='failed',error=repr(error),ended_at=datetime.now().astimezone().isoformat());save(state_path,state)
        raise

def unit_tests():
    assert proportional({0:100,1:10},40,5)=={0:35,1:5}
    assert stratum(dict(width=500,height=500))==0
    assert stratum(dict(width=1000,height=2001))==4
    q=weighted_stats([1,2,3],[.5,.4,.1])
    assert math.isclose(q['mean'],1.6) and q['p95']==3. and q['p99']==3.
    index=NearIndex();e=dict(image_sha256='a',dhash64='0000000000000000');index.add(e)
    assert index.near(dict(image_sha256='b',dhash64='0000000000000003'))
    assert not index.near(dict(image_sha256='c',dhash64='0000000000000007'))
    print('V2_UNIT_TESTS_OK',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','seal-preflight','repair-poststratification','check','run','unit-tests','audit-confirmation','audit-test'])
    action=p.parse_args().action
    if action=='prepare':prepare()
    elif action=='seal-preflight':seal_preflight()
    elif action=='repair-poststratification':repair_poststratification()
    elif action=='check':check()
    elif action=='run':run()
    elif action=='unit-tests':unit_tests()
    else: print(audit(action=='audit-test')[0])
