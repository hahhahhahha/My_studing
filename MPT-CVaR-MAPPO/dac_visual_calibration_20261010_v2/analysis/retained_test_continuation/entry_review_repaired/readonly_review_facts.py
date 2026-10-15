"""Reviewer-owned read-only checks. Stdlib only; never imports experiment code."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import ast
import difflib
import hashlib
import json
import math

BASE = Path('X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2')
OLD = BASE.parent / 'dac_visual_calibration_20261009'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def binding(root, values):
    mismatches = []
    actual = {}
    for name, expected in values.items():
        path = root / name
        observed = sha(path) if path.is_file() else None
        actual[name] = observed
        if observed != expected:
            mismatches.append(dict(path=name, expected=expected, actual=observed))
    return dict(count=len(values), mismatches=mismatches,
                actual_map_canonical_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest())

plan = read(BASE / 'protocol/PLAN.json')
freeze = read(BASE / 'protocol/RETAINED_TEST_CONTINUATION.json')
before = read(BASE / 'protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json')
historical = read(BASE / 'protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json')
initial = read(BASE / 'analysis/retained_test_continuation/entry_review/ENTRY_REVIEW_20261010_INITIAL_FAIL.json')
splits = read(BASE / 'protocol/SPLITS.json')
old_splits = read(OLD / 'protocol/IMAGE_SPLITS.json')
validation = read(BASE / 'analysis/VALIDATION_REPORT.json')
confirmation_review = read(BASE / 'analysis/recovery_20261010/reviewer/AUDIT_REPORT.json')
entry_path = BASE / 'scripts/run_retained_test_isolated.py'
archive_path = BASE / 'analysis/retained_test_continuation/run_retained_test_isolated_pre_review.py'
archived_readme = BASE / 'analysis/retained_test_continuation/README_RETAINED_TEST_PRE_REVIEW_中文.md'
source = entry_path.read_text(encoding='utf-8')
archived_source = archive_path.read_text(encoding='utf-8')
tree = ast.parse(source)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
def calls(name):
    return [dict(call=ast.unparse(node.func), line=node.lineno) for node in ast.walk(functions[name]) if isinstance(node, ast.Call)]

test_sessions = [item for item in plan['sessions'] if item['kind'] == 'retained_test']
test_ids = {e['image_id'] for e in splits['test']}
session_facts = []
for session in test_sessions:
    jobs = session['jobs']
    pairs = {}
    for job in jobs:
        pairs.setdefault(job['block'], {})[job['profile']] = job['image_ids']
    profiles = {}
    for profile in ['CPU-1T', 'CPU-4T']:
        ids = [image_id for j in jobs if j['profile'] == profile for image_id in j['image_ids']]
        profiles[profile] = dict(rows=len(ids), unique_images=len(set(ids)), exact_test_set=set(ids) == test_ids,
                                 multiplicities=sorted(set(Counter(ids).values())))
    session_facts.append(dict(session=session['session'], jobs=len(jobs), rows=sum(len(j['image_ids']) for j in jobs),
                              sizes=sorted(set(len(j['image_ids']) for j in jobs)), block_ids=sorted(pairs),
                              every_pair_exactly_same_images=all(set(p)=={'CPU-1T','CPU-4T'} and p['CPU-1T']==p['CPU-4T'] for p in pairs.values()),
                              roles_valid=all(j['role']=='test' and j['session']==session['session'] for j in jobs), profiles=profiles))

evidence_paths = ['protocol/RETAINED_TEST_RUN_STARTED.json', 'protocol/RETAINED_TEST_STATE.json',
    'analysis/TEST_REPORT.json', 'analysis/test_errors.csv', 'analysis/V2_保留测试结果_中文.md',
    'logs/retained_test_stdout.log', 'logs/retained_test_stderr.log',
    'analysis/v2_complete_service_requests.csv', 'figures/fig_visual_service_v2_complete.png',
    'figures/fig_visual_service_v2_complete.pdf']
existing_evidence = [p for p in evidence_paths if (BASE / p).exists()]
for folder, pattern in [('measurements/test','*'), ('protocol/jobs','test_*.json'), ('logs','test_*.log'),
                        ('protocol','idle_retained_test_*.json'), ('protocol','retained_test_session_*.json')]:
    existing_evidence.extend(p.relative_to(BASE).as_posix() for p in (BASE / folder).glob(pattern))
numeric_checks=[]
for c in validation['criteria']:
    error=abs(c['predicted_ms']-c['measured_ms'])/c['measured_ms']
    numeric_checks.append(dict(profile=c['profile'], metric=c['metric'], error=error, tolerance=c['tolerance'],
        numeric_recalculation_matches=math.isclose(error,c['relative_error'],rel_tol=1e-12,abs_tol=1e-12),
        frozen_tolerance_matches=c['tolerance']==plan['thresholds'][c['metric']],
        numeric_pass=error<=plan['thresholds'][c['metric']], recorded_pass=c['passed'] is True))

test_image_mismatches=[]
for e in splits['test']:
    if sha(OLD / 'data/extracted' / e['image_id']) != e['image_sha256']:
        test_image_mismatches.append(e['image_id'])

protected_preserved = all(freeze['protected_sha256'].get(k)==v for k,v in before['protected_sha256'].items())
env=read(BASE / 'protocol/ENVIRONMENT_SPEC.json')
facts = dict(at=datetime.now().astimezone().isoformat(), reviewer_method='Stdlib read-only SHA256, JSON identity and AST/source checks; no experiment imports',
    hashes={name:sha(BASE/name) for name in ['protocol/PLAN.json','protocol/RETAINED_TEST_CONTINUATION.json',
        'protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json','protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json',
        'scripts/run_retained_test_isolated.py','README_RETAINED_TEST_中文.md','scripts/runner.py','scripts/measure_block.py','scripts/v2_common.py',
        'analysis/VALIDATION_REPORT.json','analysis/recovery_20261010/reviewer/AUDIT_REPORT.json',
        'analysis/retained_test_continuation/run_retained_test_isolated_pre_review.py',
        'analysis/retained_test_continuation/README_RETAINED_TEST_PRE_REVIEW_中文.md']},
    binding_traversal=dict(source=binding(BASE/'scripts',plan['source_sha256']), artifact=binding(BASE,plan['artifact_sha256']),
                           old_readonly=binding(OLD,plan['old_readonly_sha256']), protected=binding(BASE,freeze['protected_sha256'])),
    freeze_bindings=dict(plan=freeze['original_plan_sha256']==sha(BASE/'protocol/PLAN.json'),
        source=freeze['entry_source_sha256']==sha(entry_path), readme=freeze['readme_sha256']==sha(BASE/'README_RETAINED_TEST_中文.md'),
        validation=freeze['validation_report_sha256']==sha(BASE/'analysis/VALIDATION_REPORT.json'),
        confirmation_review=freeze['confirmation_review_sha256']==sha(BASE/'analysis/recovery_20261010/reviewer/AUDIT_REPORT.json')),
    archive=dict(failed_review_passed=historical['passed'], initial_report_json_equal=historical==initial,
        historical_continuation_bound=historical['continuation_sha256']==sha(BASE/'protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json'),
        historical_source_bound=historical['entry_source_sha256']==sha(archive_path), historical_readme_bound=historical['readme_sha256']==sha(archived_readme),
        archived_freeze_source_bound=before['entry_source_sha256']==sha(archive_path), archived_freeze_readme_bound=before['readme_sha256']==sha(archived_readme),
        repair_prior_continuation_bound=freeze['prelaunch_entry_repair']['previous_continuation_sha256']==sha(BASE/'protocol/RETAINED_TEST_CONTINUATION_PRE_ENTRY_REVIEW.json'),
        repair_failed_review_bound=freeze['prelaunch_entry_repair']['failed_review_sha256']==sha(BASE/'protocol/RETAINED_TEST_ENTRY_REVIEW_PRE_REPAIR.json'),
        repair=freeze['prelaunch_entry_repair'], old_protected_count=len(before['protected_sha256']), current_protected_count=len(freeze['protected_sha256']),
        all_old_protected_bindings_preserved=protected_preserved,
        added_protected_paths=sorted(set(freeze['protected_sha256'])-set(before['protected_sha256']))),
    test_pool=dict(exact_original_entries=splits['test']==old_splits['test'], entries=len(splits['test']), unique_images=len(test_ids),
        validation_test_overlap=len(test_ids&{e['image_id'] for e in splits['validation']}),
        reference_test_overlap=len(test_ids&{e['image_id'] for e in splits['reference']}),
        v1_calibration_test_overlap=len(test_ids&{e['image_id'] for e in old_splits['calibration']}),
        actual_jpeg_hashes_checked=len(splits['test']), actual_jpeg_hash_mismatches=test_image_mismatches),
    schedule=dict(exact_plan_sessions=freeze['sessions']==test_sessions, sessions=[x['session'] for x in test_sessions],
        total_jobs=sum(x['jobs'] for x in session_facts), total_rows=sum(x['rows'] for x in session_facts), details=session_facts),
    frozen_settings={k:plan[k] for k in ['frozen_at','spacing_seconds','maximum_idle_wait_seconds','machine_idle_percent',
        'selected_core_idle_percent','minimum_memory_bytes','warmup','thresholds','bootstrap_replicates','confirmation_rows','retained_test_rows','weights_sha256']},
    environment_canonical_id=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],
    acceptance_prerequisites=dict(accepted=validation['accepted'], criteria_count=len(validation['criteria']), criteria_checks=numeric_checks,
        unsupported_inputs=validation['unsupported_inputs'], report_plan_binding=validation['plan_sha256']==sha(BASE/'protocol/PLAN.json'),
        report_model_binding=validation['model_sha256']==sha(BASE/'model/CONDITIONAL_MODEL.json'),
        record_integrity_status=confirmation_review['record_integrity_status'], frozen_six_metric_acceptance=confirmation_review['frozen_six_metric_acceptance']),
    existing_test_evidence=existing_evidence,
    source_line_ranges={name:[node.lineno,node.end_lineno] for name,node in functions.items()},
    current_date_guard_callers=[name for name in functions if any(x['call']=='measurement_day' for x in calls(name))],
    subsequent_processing_calls={name:calls(name) for name in ['frozen_guard','statistics_worker','measured_protocol_guard','audit_all','export_worker']},
    source_revision_diff=''.join(difflib.unified_diff(archived_source.splitlines(True),source.splitlines(True),fromfile='archived_entry_source',tofile='current_entry_source')),
    readme_revision_diff=''.join(difflib.unified_diff(archived_readme.read_text(encoding='utf-8').splitlines(True),(BASE/'README_RETAINED_TEST_中文.md').read_text(encoding='utf-8').splitlines(True),fromfile='archived_readme',tofile='current_readme')))
print(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False))
