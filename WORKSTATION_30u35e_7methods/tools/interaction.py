"""Same-schedule 2x2 feedback x router interaction; descriptive run bootstrap."""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import read,write_new,quantile

p=argparse.ArgumentParser()
p.add_argument('--analyses',nargs='+',required=True)
p.add_argument('--metric',default='cvar99_s')
p.add_argument('--output',required=True)
a=p.parse_args()
labels={'qla|periodic','qla|coordinated','score_softmax|periodic','score_softmax|coordinated'}
pairs=defaultdict(dict)
for path in a.analyses:
    value=read(path)
    if value['system_id'] not in labels:
        continue
    if not value['infrastructure_valid']:
        raise ValueError('Infrastructure-invalid run cannot enter the interaction analysis')
    key=(value['schedule_sha256'],value['schedule_seed'])
    if value['system_id'] in pairs[key]:
        raise ValueError('Duplicate replicate/condition; analyze each load/propagation block separately')
    pairs[key][value['system_id']]=value
if len(pairs)<3 or any(set(v)!=labels for v in pairs.values()):
    raise ValueError('At least 3 full paired 2x2 repetitions required')
deltas=[]
for block in pairs.values():
    reference=block['qla|periodic']
    for value in block.values():
        if any(value[k]!=reference[k] for k in ('comparison_contract','feedback_budget_contract')):
            raise ValueError('Resource/feature/drain/budget mismatch')
        if value[a.metric] is None:
            raise ValueError('Undefined metric, not zero')
    deltas.append((block['qla|coordinated'][a.metric]-block['qla|periodic'][a.metric])-
                  (block['score_softmax|coordinated'][a.metric]-block['score_softmax|periodic'][a.metric]))
rng=random.Random(20261013)
resamples=[sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(10000)]
write_new(a.output,dict(schema='systemlab-interaction-v1',metric=a.metric,paired_repetitions=len(deltas),
    interaction_differences=deltas,mean_interaction=sum(deltas)/len(deltas),
    descriptive_paired_repetition_bootstrap_95=[quantile(resamples,.025),quantile(resamples,.975)],
    definition='(coordinatedQLA-periodicQLA)-(coordinatedScore-periodicScore)',
    no_task_level_iid_CI=True,lower_is_better_only_for_latency_metrics=True))
