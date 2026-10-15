"""Register the 30-user/35-edge seven-method E1 matrix; does not launch services."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import read,sha,write_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--feature-scales',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--block',choices=['E1'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    scales,cfg=read(args.feature_scales),read(args.config)
    if scales['status']!='frozen_after_calibration_validation':
        raise ValueError('Freeze the pilot settings before registering a test matrix')
    capacity=sum(1./scales['calibration_service_s_by_edge'][x['id']] for x in cfg['edges'])
    cells=[]
    methods=('qla','ppo_mv','ppo_dsr','mfc_di','qeco','jseq','lyfoto_lc')
    for seed in (1,2,3,4,5):
        for method in methods:
            cells.append(dict(id=f'E1_{len(cells)+1:03}', seed=seed,
                aggregate_rate_per_s=62.0, method=method, feedback_mode='periodic',
                injected_feedback_delay_s=0., injected_feedback_loss=0.,
                common_schedule_key=f'62_{seed}',
                feedback_budget_bytes_per_s=2097152))
    random.Random(20261013).shuffle(cells)
    write_new(args.output,dict(schema='systemlab-grid-v2',block=args.block,cells=cells,
        methods=list(methods),seeds=[1,2,3,4,5],users=30,logical_edges=35,
        nominal_capacity_definition='Not used to generate load; 62 requests/s is bound to the completed pilot.',
        reference_capacity_per_s=None,config_sha256=sha(args.config),feature_scales_sha256=sha(args.feature_scales),
        feature_scales_path=str(Path(args.feature_scales).resolve()),
        execution_order='preregistered_seeded_shuffle',started=False))


if __name__=='__main__':
    main()
