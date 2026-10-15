"""Read-only supplemental PDF, exact ranks, timestamps, and provenance checks."""
import csv, json, hashlib, math
from pathlib import Path
from datetime import datetime
from collections import Counter
import pdfplumber

B=Path('X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2')
V=B.parent/'dac_visual_calibration_20261009'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvrows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
data=read(B/'analysis/retained_test_continuation/result_review/INDEPENDENT_RECOMPUTE.json')
raw=csvrows(B/'analysis/v2_complete_service_requests.csv')
rank={}
for profile in ['CPU-1T','CPU-4T']:
    values=sorted(float(r['service_ms']) for r in raw if r['image_split']=='test' and r['profile_id']==profile)
    rank[profile]={}
    for key,percent in [('p95',95),('p99',99)]:
        exact_rank=(percent*len(values)+99)//100
        frozen=data['statistics']['test'][profile]['measured_ms'][key]
        rank[profile][key]={'exact_rank_one_based':exact_rank,'frozen_rank_one_based':values.index(frozen)+1,'exact_ms':values[exact_rank-1],'frozen_ms':frozen,'difference_ms':frozen-values[exact_rank-1]}

pdf=pdfplumber.open(B/'figures/fig_visual_service_v2_complete.pdf')
page=pdf.pages[0]
text=page.extract_text()
labels=[f"{data['statistics'][r][p]['measured_ms'][m]:.2f}" for r in ['validation','test'] for p in ['CPU-1T','CPU-4T'] for m in ['mean','p95','p99']]
drawings=page.rects+page.curves
blue=(0.0,114/255,178/255); orange=(213/255,94/255,0.0)
bars=[]
for d in drawings:
    fill=d.get('non_stroking_color')
    if isinstance(fill,(list,tuple)) and len(fill)==3 and (all(abs(a-b)<1e-6 for a,b in zip(fill,blue)) or all(abs(a-b)<1e-6 for a,b in zip(fill,orange))):
        # Exclude small legend swatches; chart bars span >10 points in width.
        if d['width'] > 10 and d['height'] > 20:
            bars.append(dict(color='predicted' if abs(fill[0])<1e-6 else 'measured',x=float(d['x0']),bottom=float(d['bottom']),height=float(d['height'])))
bars.sort(key=lambda d:(round(d['bottom']),d['x']))
panels=[]
for role,row in [('validation',0),('test',1)]:
    for profile,col in [('CPU-1T',0),('CPU-4T',1)]:
        group=[d for d in bars if (d['bottom'] < page.height/2)==(row==0) and (d['x'] < page.width/2)==(col==0)]
        group.sort(key=lambda d:d['x'])
        expected=[data['statistics'][role][profile][kind+'_ms'][metric] for metric in ['mean','p95','p99'] for kind in ['predicted','measured']]
        scales=[d['height']/v for d,v in zip(group,expected)]
        panels.append(dict(role=role,profile=profile,bar_count=len(group),expected_values_ms=expected,points_per_ms_range=[min(scales),max(scales)],bars_proportional_to_raw_recomputation=len(group)==6 and max(scales)-min(scales)<1e-5))
plan=read(B/'protocol/PLAN.json')
splits=read(B/'protocol/SPLITS.json')
population=read(B/'protocol/POPULATION_PROFILE.json')
def size(e):return sum(int(e['width'])*int(e['height'])>b for b in [250000,500000,1000000,2000000])
weight_bad=[]
for e in splits['validation']:
    key=f"{size(e)}|{e['class_id']}"
    expected=population['size_class_counts'][key]/population['eligible_images']/population['selected_size_class_counts'][key]
    if e['target_weight']!=expected:weight_bad.append(e['image_id'])
starts=[datetime.fromisoformat(s['first_block_start']) for s in data['test_sessions']]
meta_dates=[]
for p in (B/'measurements').rglob('*.json'):
    m=read(p)
    for k in ['sensor_before','sensor_after']:
        if datetime.fromisoformat(m[k]['wall_clock']).date().isoformat()!='2026-10-10':meta_dates.append(str(p))
state=read(B/'protocol/RETAINED_TEST_STATE.json')
result={
 'quantile_rank_audit':rank,
 'PDF':{'pages':len(pdf.pages),'all_twelve_measured_labels_match':all(label in text for label in labels),'expected_labels':labels,'text':text,'panels':panels,'sha256':sha(B/'figures/fig_visual_service_v2_complete.pdf')},
 'confirmation_target_weights':{'mismatches':weight_bad,'sum':sum(e['target_weight'] for e in splits['validation'])},
 'actual_first_block_start_gaps_seconds':[(b-a).total_seconds() for a,b in zip(starts,starts[1:])],
 'all_170_block_metadata_dates_same_day':not meta_dates,
 'V1_original_rule':read(V/'protocol/SERVICE_DISTRIBUTION_VALIDATION_RULE.json'),
 'report_aliases':{'VAL_REPORT_json_exists':(B/'analysis/VAL_REPORT.json').exists(),'actual_validation_report':'analysis/VALIDATION_REPORT.json','analysis_AUDIT_REPORT_json_exists':(B/'analysis/AUDIT_REPORT.json').exists(),'actual_historical_review':'analysis/recovery_20261010/reviewer/AUDIT_REPORT.json'},
 'retained_state':state,
 'PNG_visual_review':'Auditor visually inspected original PNG, matched four panels, twelve displayed labels, positive zero-based axes and scope title; no CI drawn.',
 'result_doc_warning':'analysis/V2_保留测试结果_中文.md:15 uses confirmation target-weighting template wording; retained test is equal image/session weighted.',
 'status_doc_warning':'RUN_STATUS_中文.md:3,5 still running; :19 says test not executed although complete_pending_independent_audit state has 10000 rows; stale status, not numerical mismatch.',
 'readonly_scope':'No original module imported, measurement/probe/export called, inference/model fitting performed or evidence rewritten.'
}
print(json.dumps(result,ensure_ascii=False,indent=2))
