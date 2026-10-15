"""Actual historical actor load/forward with synthetic stale inputs; no run."""
import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import read, write_new
from systemlab.policy import Router


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--bundle',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    cfg=read(args.config)
    router=Router('qla',cfg,args.bundle)
    def fingerprint():
        h=hashlib.sha256()
        for key,tensor in sorted(router.actor.state_dict().items()):
            h.update(key.encode())
            h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint()
    now=time.time_ns()
    states={x['id']:dict(predicted_wait_s=.02*i,service_estimate_s=x['initial_service_s'],
                         source_epoch_ns=now-100000000) for i,x in enumerate(cfg['edges'])}
    weights,receipt=router.portfolio([dict(payload_bytes=100000)],states,now,0)
    after=fingerprint()
    if before!=after:
        raise RuntimeError('Inference modified frozen actor')
    write_new(args.output,dict(status='functional_actor_forward_pass',input_is_synthetic_fixture=True,
        before_parameter_sha256=before,after_parameter_sha256=after,portfolio=weights,
        observation_dim=len(receipt['actor_observation']),receipt=router.receipt,optimizer_calls=0,
        live_system_execution=False,performance_evidence=False))


if __name__=='__main__':
    main()
