"""Independent read-only reconstruction. Imports no experiment module, no torch.
Only prints JSON. Capture stdout using apply_patch; does not write evidence.
"""
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

B = Path('X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2')
V = B.parent / 'dac_visual_calibration_20261009'
PROFILES = ['CPU-1T', 'CPU-4T']
STAGES = ['decode_ms', 'preprocess_ms', 'infer_ms', 'postprocess_ms']
BOUNDARY = 'decode+RGB+preprocess+FP32 inference+argmax+dict; no disk/network/queue/controller'
result = {'audit_type': 'Type-A independent agent; not cross-family certification', 'failures': []}

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def ck(ok, what):
    if not ok:
        result['failures'].append(what)

def size(e):
    px = int(e['width']) * int(e['height'])
    return sum(px > b for b in [250000, 500000, 1000000, 2000000])

def name(j):
    return f"{j['role']}_s{j['session']}_b{j['block']}_{j['profile']}"

plan = read(B / 'protocol/PLAN.json')
freeze = read(B / 'protocol/RETAINED_TEST_CONTINUATION.json')
entry = read(B / 'protocol/RETAINED_TEST_ENTRY_REVIEW.json')
lock = read(B / 'protocol/RETAINED_TEST_RUN_STARTED.json')
splits = read(B / 'protocol/SPLITS.json')
old_splits = read(V / 'protocol/IMAGE_SPLITS.json')
versions = read(B / 'protocol/ENVIRONMENT_SPEC.json')['versions']
cpus = read(B / 'protocol/CPU_TOPOLOGY.json')['logical_cpu_per_physical_core']
model = read(B / 'model/CONDITIONAL_MODEL.json')
hash_counts = {}
for label, table, root in [('v2_sources', plan['source_sha256'], B / 'scripts'),
                           ('v2_artifacts', plan['artifact_sha256'], B),
                           ('protected_v1', plan['old_readonly_sha256'], V),
                           ('protected_confirmation', freeze['protected_sha256'], B)]:
    for f, h in table.items():
        ck((root / f).is_file() and sha(root / f) == h, label + ':' + f)
    hash_counts[label] = len(table)
bindings = [('plan', B / 'protocol/PLAN.json', freeze['original_plan_sha256']),
            ('validation', B / 'analysis/VALIDATION_REPORT.json', freeze['validation_report_sha256']),
            ('historical_review', B / 'analysis/recovery_20261010/reviewer/AUDIT_REPORT.json', freeze['confirmation_review_sha256']),
            ('continuation_source', B / 'scripts/run_retained_test_isolated.py', freeze['entry_source_sha256']),
            ('readme', B / 'README_RETAINED_TEST_中文.md', freeze['readme_sha256']),
            ('lock_freeze', B / 'protocol/RETAINED_TEST_CONTINUATION.json', lock['continuation_sha256']),
            ('lock_entry', B / 'protocol/RETAINED_TEST_ENTRY_REVIEW.json', lock['entry_review_sha256'])]
for label, p, h in bindings:
    ck(sha(p) == h, 'binding:' + label)
ck(entry['passed'] is True and entry['continuation_sha256'] == sha(B / 'protocol/RETAINED_TEST_CONTINUATION.json') and entry['entry_source_sha256'] == freeze['entry_source_sha256'], 'entry binding')
ck(read(V / 'protocol/IMAGE_FREEZE.json')['split_sha256'] == sha(V / 'protocol/IMAGE_SPLITS.json'), 'V1 image freeze')
ck(splits['test'] == old_splits['test'], 'original retained test pool')
ck(model['training_entries'] == old_splits['calibration'], 'training entry identity')
ck(plan['thresholds'] == {'mean': .10, 'p95': .15, 'p99': .20}, 'thresholds unchanged')
ck(freeze['sessions'] == [s for s in plan['sessions'] if s['kind'] == 'retained_test'], 'retained sessions')
manifest = {e['image_id']: e for e in rows(V / 'protocol/IMAGE_MANIFEST.csv')}
images = {e['image_id']: e for es in [old_splits['calibration'], *splits.values()] for e in es}
for i, e in images.items():
    p = V / 'data/extracted' / i
    ck(p.is_file() and sha(p) == e['image_sha256'], 'JPEG:' + i)
    m = manifest[i]
    ck(all(str(e[k]) == str(m[k]) for k in ['image_sha256', 'class_id', 'width', 'height', 'payload_bytes']), 'manifest:' + i)
    ck(p.stat().st_size == int(e['payload_bytes']), 'JPEG byte length:' + i)
cal_ids = set(e['image_id'] for e in old_splits['calibration'])
test_ids = set(e['image_id'] for e in splits['test'])
val_ids = set(e['image_id'] for e in splits['validation'])
ck(len(test_ids) == 1000 and not test_ids.intersection(cal_ids | val_ids), 'test leakage')
result['fingerprint_checks'] = {**hash_counts, 'JPEG_files': len(images), 'plan_sha256': sha(B / 'protocol/PLAN.json'), 'model_sha256': sha(B / 'model/CONDITIONAL_MODEL.json')}

all_rows = []
role_counts = Counter()
intervals = []
pair_preds = defaultdict(dict)
raw_hashes = {}
max_sum_error = 0.0
for item in plan['sessions']:
    # Independently replay deterministic job randomization from frozen split lists.
    rng = random.Random(f"20261010:jobs:{item['session']}:{item['kind'] == 'retained_test'}")
    roles = ['test'] if item['kind'] == 'retained_test' else ['validation', 'reference']
    chunks = {}
    for role in roles:
        ids = [e['image_id'] for e in splits[role]]
        rng.shuffle(ids)
        blocksize = 20 if role == 'reference' else 200
        chunks[role] = [ids[k:k+blocksize] for k in range(0, len(ids), blocksize)]
    replay = []
    for b in range(len(chunks[roles[0]])):
        rr = roles.copy(); rng.shuffle(rr)
        pp = PROFILES[::-1] if (item['session'] + b) % 2 == 0 else PROFILES
        for role in rr:
            for profile in pp:
                replay.append(dict(role=role, session=item['session'], block=b+1, profile=profile, image_ids=chunks[role][b]))
    ck(replay == item['jobs'], 'randomized schedule:' + str(item['session']) + item['kind'])
    for j in item['jobs']:
        n = name(j); path = B / 'measurements' / j['role'] / (n + '.csv')
        meta = read(path.with_suffix('.json')); data = rows(path)
        jp = B / 'protocol/jobs' / (n + '.json')
        t = 1 if j['profile'] == 'CPU-1T' else 4
        ck(read(jp) == j == meta['job'] and sha(jp) == meta['job_sha256'], 'job:' + n)
        ck(meta['versions'] == versions and meta['weights_sha256'] == plan['weights_sha256'] and meta['affinity'] == cpus[:t] and meta['warmup_requests'] == 200 and meta['errors'] == 0 and meta['requests'] == len(data) and meta['timing_boundary'] == BOUNDARY, 'metadata:' + n)
        ck([r['image_id'] for r in data] == j['image_ids'], 'row grid:' + n)
        start = datetime.fromisoformat(meta['sensor_before']['wall_clock'])
        end = datetime.fromisoformat(meta['sensor_after']['wall_clock'])
        intervals.append((start, end, n)); last = start
        emap = {e['image_id']: e for e in splits[j['role']]}
        for order, r in enumerate(data, 1):
            e = emap[r['image_id']]
            ck(int(r['session_id']) == j['session'] and int(r['block_id']) == j['block'] and int(r['run_order']) == order and r['profile_id'] == j['profile'] and r['image_split'] == j['role'], 'row identity:' + n + ':' + str(order))
            ck(r['device'] == 'cpu' and int(r['cpu_threads']) == t and json.loads(r['cpu_affinity']) == cpus[:t] and r['weights_sha256'] == plan['weights_sha256'] and r['model_id'] == 'ResNet18_IMAGENET1K_V1' and r['error_status'] == 'ok' and 0 <= int(r['predicted_class']) < 1000, 'row runtime:' + n + ':' + str(order))
            ck(all(str(r[k]) == str(e[k]) for k in ['image_sha256', 'class_id', 'width', 'height', 'payload_bytes']), 'row input:' + n + ':' + str(order))
            stages = [float(r[k]) for k in STAGES]; service = float(r['service_ms'])
            residual = abs(sum(stages) - service); max_sum_error = max(max_sum_error, residual)
            ck(all(math.isfinite(v) and v >= 0 for v in stages) and math.isfinite(service) and service > 0 and residual < 1e-8, 'row timing:' + n + ':' + str(order))
            stamp = datetime.fromisoformat(r['measurement_timestamp'])
            ck(last <= stamp <= end and stamp.date().isoformat() == '2026-10-10', 'row timestamp:' + n + ':' + str(order))
            last = stamp
            pair_preds[(j['role'], j['session'], r['image_id'])][r['profile_id']] = r['predicted_class']
        ck((B / 'logs' / (n + '.log')).read_text(encoding='utf-8').strip() == f'MEASURED {n} {len(data)}', 'job invocation log:' + n)
        all_rows.extend(data); role_counts[j['role']] += len(data)
        for p in [path, path.with_suffix('.json')]:
            raw_hashes[str(p.relative_to(B))] = sha(p)
for role in splits:
    actual = {p.name for p in (B / 'measurements' / role).iterdir() if p.suffix in ['.csv', '.json']}
    expected = {name(j) + ext for it in plan['sessions'] for j in it['jobs'] if j['role'] == role for ext in ['.csv', '.json']}
    ck(actual == expected, 'files grid:' + role)
ck(dict(role_counts) == dict(reference=1200, validation=12000, test=10000), 'all new row counts')
intervals.sort()
ck(all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:])), 'serial nonoverlap')
ck(all(set(p) == set(PROFILES) and len(set(p.values())) == 1 for p in pair_preds.values()), 'paired prediction identity')
export = rows(B / 'analysis/v2_complete_service_requests.csv')
ck(export == all_rows, 'complete CSV row-for-row, field-for-field, order')
for f, h in read(B / 'analysis/retained_test_continuation/COMPLETE_EXPORT.json')['artifacts'].items():
    ck(sha(B / f) == h, 'export hash:' + f)
for f in ['analysis/TEST_REPORT.json', 'analysis/VALIDATION_REPORT.json', 'analysis/retained_test_continuation/STRICT_TEST_AUDIT.json']:
    report = read(B / f); integrity = report.get('integrity', report)
    for p, h in integrity['measurement_sha256'].items():
        ck(raw_hashes.get(p) == h, 'reported raw hash:' + f + ':' + p)
result['records'] = {'counts': dict(role_counts), 'blocks': len(intervals), 'paired_inputs_sessions': len(pair_preds), 'max_stage_sum_error_ms': max_sum_error, 'complete_csv_exact_match': export == all_rows, 'versions': versions, 'affinities': {'CPU-1T': cpus[:1], 'CPU-4T': cpus[:4]}, 'warmup_per_block': 200, 'first_new_block_start': intervals[0][0].isoformat(), 'last_new_block_end': intervals[-1][1].isoformat()}

session_details = []; previous = None
for s in range(1,6):
    ev = read(B / f'protocol/retained_test_session_{s}.json')
    start = datetime.fromisoformat(ev['started_at'])
    ck(ev['session'] == s and ev['continuation_sha256'] == lock['continuation_sha256'] and start.date().isoformat() == '2026-10-10', 'session identity:' + str(s))
    gap = (start - previous).total_seconds() if previous else None
    ck(gap is None or gap >= 900, 'session spacing:' + str(s))
    idle = read(B / f'protocol/idle_retained_test_s{s}.json')
    def idle_ok(r):
        return r['mean'] <= 35 and r['selected_max'] <= 60 and r['available_memory_bytes'] >= 2147483648
    ck(len(idle) > 0 and idle_ok(idle[-1]) and all(not idle_ok(r) for r in idle[:-1]), 'idle stop rule:' + str(s))
    ck(datetime.fromisoformat(idle[-1]['at']) <= start and (datetime.fromisoformat(idle[-1]['at']) - datetime.fromisoformat(idle[0]['at'])).total_seconds() <= 3600, 'idle time:' + str(s))
    actual_starts = [a for a,b,n in intervals if n.startswith(f'test_s{s}_')]
    actual_ends = [b for a,b,n in intervals if n.startswith(f'test_s{s}_')]
    ck(min(actual_starts) >= start, 'block precedes session:' + str(s))
    session_details.append(dict(session=s, start=start.isoformat(), spacing_seconds=gap, first_block_start=min(actual_starts).isoformat(), last_block_end=max(actual_ends).isoformat(), idle=idle))
    previous = start
result['test_sessions'] = session_details

# Reconstruct frozen empirical pairs from original CSVs, not model timing values.
cal = []
cal_pair = defaultdict(dict)
cal_pred = defaultdict(dict)
for p in sorted((V / 'measurements/calibration').glob('*.csv')):
    meta = read(p.with_suffix('.json')); rr = rows(p)
    ck(len(rr) == meta['requests'] == 200 and meta['errors'] == 0 and meta['warmup_requests'] == 200 and meta['versions'] == versions, 'cal metadata:' + p.name)
    ck([r['image_id'] for r in rr] == meta['job']['image_ids'], 'cal job order:' + p.name)
    for r in rr:
        ck(r['image_id'] in cal_ids and r['error_status'] == 'ok', 'cal role:' + p.name)
        values = {k: float(r[k]) for k in STAGES + ['service_ms']}
        ck(all(math.isfinite(v) and v >= 0 for v in values.values()) and abs(sum(values[k] for k in STAGES)-values['service_ms']) < 1e-8, 'cal timing:' + p.name)
        key = (r['image_id'], int(r['session_id']))
        ck(r['profile_id'] not in cal_pair[key], 'duplicate cal pair')
        cal_pair[key][r['profile_id']] = values
        cal_pred[key][r['profile_id']] = r['predicted_class']
    cal.extend(rr)
ck(len(cal) == 20000 and len(cal_pair) == 10000 and all(set(p) == set(PROFILES) for p in cal_pair.values()), 'cal complete grid')
ck(all(len(set(p.values())) == 1 for p in cal_pred.values()), 'cal paired prediction')
reconstructed = [dict(image_id=i, session=s, profiles=p) for (i,s),p in sorted(cal_pair.items())]
ck(reconstructed == model['paired_rows'], 'model paired rows reconstructed')
ck(model['k'] == 20 and model['strata_upper_bounds'] == [250000,500000,1000000,2000000] and model['support_distance'] == 2.0, 'model fixed definition')
result['model_rebuild'] = {'calibration_rows': len(cal), 'paired_rows': len(cal_pair), 'paired_stage_values_exact': reconstructed == model['paired_rows'], 'test_label_in_fit': False}

# Independent all-input kNN construction, uses only frozen input covariates.
maps = {}
for role in ['test', 'validation']:
    mm = {}
    for e in splits[role]:
        x = (math.log2(int(e['width'])*int(e['height'])), math.log2(int(e['payload_bytes'])))
        dist = []
        for c in old_splits['calibration']:
            if size(c) == size(e):
                d = math.hypot(x[0]-math.log2(int(c['width'])*int(c['height'])), x[1]-math.log2(int(c['payload_bytes'])))
                dist.append((d,c['image_id']))
        dist.sort(); selected = dist[:20]
        mm[e['image_id']] = {'neighbor_ids': [i for d,i in selected], 'nearest_distance': selected[0][0], 'supported': selected[0][0] <= 2}
    maps[role] = mm
ck(maps['validation'] == read(B / 'model/VALIDATION_NEIGHBORS.json'), 'all confirmation neighbors')
result['neighbors'] = {r: {'inputs': len(m), 'unsupported': sum(not v['supported'] for v in m.values()), 'neighbor_count_histogram': dict(Counter(len(v['neighbor_ids']) for v in m.values())), 'max_nearest_distance': max(v['nearest_distance'] for v in m.values()), 'canonical_sha256': hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()} for r,m in maps.items()}

import numpy as np
def stats(vals, weights):
    # Reproduce the frozen floating weighted-CDF estimator, not silently a replacement.
    ordered = sorted(zip(vals, weights), key=lambda z: z[0])
    a = np.array([v for v,w in ordered], dtype=float)
    w = np.array([w for v,w in ordered], dtype=float)
    w /= np.sum(w)
    cumulative = np.cumsum(w)
    return dict(mean=float(np.dot(a,w)), p95=float(a[np.searchsorted(cumulative,.95)]), p99=float(a[np.searchsorted(cumulative,.99)]))

def exact_uniform(vals):
    a = sorted(vals); n = len(a)
    return dict(mean=math.fsum(a)/n, p95=a[(95*n+99)//100-1], p99=a[(99*n+99)//100-1])

result['statistics'] = {}
for role, sessions in [('test',5), ('validation',3)]:
    recorded = read(B / ('analysis/' + role.upper() + '_REPORT.json'))
    results = {}; cells = defaultdict(list)
    input_weights = {e['image_id']: (1/1000 if role == 'test' else e['target_weight']) for e in splits[role]}
    for e in splits[role]: cells[(size(e),e['class_id'])].append(e['image_id'])
    for profile in PROFILES:
        pred = []; pw = []
        for e in splits[role]:
            near = maps[role][e['image_id']]['neighbor_ids']
            for i in near:
                for s in range(1,6):
                    pred.append(cal_pair[i,s][profile]['service_ms'])
                    pw.append(input_weights[e['image_id']]/len(near)/5)
        mr = [r for r in all_rows if r['image_split'] == role and r['profile_id'] == profile]
        obs = [float(r['service_ms']) for r in mr]
        ow = [input_weights[r['image_id']]/sessions for r in mr]
        ps = stats(pred,pw); os = stats(obs,ow)
        gates = {k: {'relative_error': abs(ps[k]-os[k])/os[k], 'tolerance': plan['thresholds'][k], 'passed': abs(ps[k]-os[k])/os[k] <= plan['thresholds'][k]} for k in ps}
        for k in ps:
            ck(math.isclose(ps[k],recorded['summary'][profile]['predicted_ms'][k],rel_tol=1e-13,abs_tol=1e-12) and math.isclose(os[k],recorded['summary'][profile]['measured_ms'][k],rel_tol=1e-13,abs_tol=1e-12), 'point metric:'+role+profile+k)
            original_gate = next(x for x in recorded['criteria'] if x['profile'] == profile and x['metric'] == k)
            ck(original_gate['passed'] == gates[k]['passed'] and original_gate['tolerance'] == gates[k]['tolerance'] and math.isclose(original_gate['relative_error'],gates[k]['relative_error'],abs_tol=1e-12), 'gate:'+role+profile+k)
        # Independent bootstrap endpoints. Generator and draw order honor preregistration.
        lookup = {(r['image_id'],int(r['session_id'])):float(r['service_ms']) for r in mr}
        rng = np.random.default_rng(20261010 + int(role == 'test'))
        boot = defaultdict(list)
        for _ in range(500):
            ss = rng.integers(1,sessions+1,size=sessions)
            bv=[]; bw=[]
            for ids in cells.values():
                picked=rng.choice(ids,size=len(ids),replace=True)
                for i in picked:
                    bv.extend(lookup[i,int(s)] for s in ss)
                    bw.extend([input_weights[i]/sessions]*sessions)
            st = stats(bv,bw)
            for k,v in st.items(): boot[k].append(v)
        ci = {k: [float(v) for v in np.quantile(vals,[.025,.975])] for k,vals in boot.items()}
        claimed_ci = recorded['measured_crossed_image_session_bootstrap_95ci_ms'][profile]
        ck(all(np.allclose(ci[k],claimed_ci[k],rtol=1e-13,atol=1e-12) for k in ci), 'bootstrap:'+role+profile)
        results[profile] = dict(measured_rows=len(obs), predicted_mixture_components=len(pred), predicted_ms=ps, measured_ms=os, gates=gates, measured_distribution_95ci_ms=ci, CI_independently_recomputed=True, observed_max_ms=max(obs))
        if role == 'test':
            # Some sparse strata have fewer than twenty neighbors: prediction components
            # need not be uniform. Exact weighted CDF uses rational component weights.
            results[profile]['exact_uniform_measured_ms'] = exact_uniform(obs)
            from fractions import Fraction
            frequencies = defaultdict(Fraction)
            for e in splits[role]:
                near = maps[role][e['image_id']]['neighbor_ids']
                for i in near:
                    for s in range(1,6): frequencies[cal_pair[i,s][profile]['service_ms']] += Fraction(1,1000*len(near)*5)
            exact={}; cumulative=Fraction(0)
            for v,w in sorted(frequencies.items()):
                cumulative += w
                for k,q in [('p95',Fraction(95,100)),('p99',Fraction(99,100))]:
                    if k not in exact and cumulative >= q: exact[k]=v
            results[profile]['exact_rational_predicted_quantiles_ms'] = exact
            results[profile]['exact_empirical_alternative_gates'] = {k: abs(exact[k]-exact_uniform(obs)[k])/exact_uniform(obs)[k] <= plan['thresholds'][k] for k in ['p95','p99']}
    result['statistics'][role] = results
    result['statistics'][role + '_size_class_cells'] = {'cells':len(cells),'minimum_images': min(len(v) for v in cells.values()),'singleton_cells':sum(len(v)==1 for v in cells.values())}
result['bootstrap_scope'] = '500 observed-distribution crossed image-within-size-class / whole-session draws, independently recomputed; not prediction/model error CI; 5 test and 3 confirmation sessions, same machine/day.'
result['originals_unchanged_after_recompute'] = True
for label, table, root in [('source',plan['source_sha256'],B/'scripts'),('artifact',plan['artifact_sha256'],B),('V1',plan['old_readonly_sha256'],V),('confirmation',freeze['protected_sha256'],B),('new_raw',raw_hashes,B)]:
    for p,h in table.items():
        ok = sha(root/p)==h
        ck(ok,'post-audit hash:'+label+':'+p)
        result['originals_unchanged_after_recompute'] &= ok
result['passed_computational_checks'] = not result['failures']
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
