"""Register a compact paired matrix; does not launch services."""
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
    p.add_argument('--block',choices=['E1','E2','E4'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    scales,cfg=read(args.feature_scales),read(args.config)
    if scales['status']!='frozen_after_calibration_validation':
        raise ValueError('Freeze the pilot settings before registering a test matrix')
    capacity=sum(1./scales['calibration_service_s_by_edge'][x['id']] for x in cfg['edges'])
    cells=[]
    for seed in (1,2,3):
        for rho in (.6,.9):
            methods=('score_softmax','stale_shortest','qla') if args.block=='E1' else ('score_softmax','qla')
            modes=('periodic',) if args.block!='E4' else ('periodic','queue_trigger','age','coordinated')
            conditions=[(0.,0.)] if args.block!='E2' else [(0.,0.),(.1,0.),(.3,0.),(.1,.1)]
            for method in methods:
                for mode in modes:
                    for delay,loss in conditions:
                        cells.append(dict(id=f'{args.block}_{len(cells)+1:03}',seed=seed,rho=rho,
                            aggregate_rate_per_s=rho*capacity,method=method,feedback_mode=mode,
                            injected_feedback_delay_s=delay,injected_feedback_loss=loss,
                            common_schedule_key=f'{rho}_{seed}',
                            feedback_budget_bytes_per_s=cfg['feedback']['budget_bytes_per_s']))
    random.Random(20261013).shuffle(cells)
    write_new(args.output,dict(schema='systemlab-grid-v1',block=args.block,cells=cells,
        nominal_capacity_definition='sum inverse mean concurrent-pilot service; report achieved load separately',
        reference_capacity_per_s=capacity,config_sha256=sha(args.config),feature_scales_sha256=sha(args.feature_scales),
        execution_order='preregistered_seeded_shuffle',started=False))


if __name__=='__main__':
    main()
