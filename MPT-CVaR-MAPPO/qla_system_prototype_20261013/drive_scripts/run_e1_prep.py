"""Fresh E1 schedule/registry preparation; explicit versioned inputs, no launch."""
import argparse
import subprocess
import sys
from pathlib import Path

lab=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(lab))
from systemlab.common import read
from systemlab.admission import check_protocol,register_test_registry


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--grid',required=True)
    p.add_argument('--validation-registration',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--actor-bundle',required=True)
    p.add_argument('--schedule-output-dir',required=True)
    p.add_argument('--registry-output',required=True)
    a=p.parse_args()
    config,grid,registration=map(read,(a.config,a.grid,a.validation_registration))
    check_protocol(config)
    if grid['block']!='E1':
        raise ValueError('This driver prepares only the preregistered E1 matrix')
    if registration['users']!=config['users'] or registration['fixed_drain_s']!=config['fixed_drain_s']:
        raise ValueError('Validation-selected users/drain mismatch')
    destination=Path(a.schedule_output_dir).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    common={}
    for cell in grid['cells']:
        key=cell['common_schedule_key']
        values=(cell['seed'],cell['aggregate_rate_per_s'])
        if key in common and common[key]!=values:
            raise ValueError('Conflicting common matrix key')
        common[key]=values
    schedules=[]
    for index,(key,(seed,rate)) in enumerate(sorted(common.items())):
        path=destination/f'common_{index+1:02d}_s{seed}.json'
        subprocess.run([sys.executable,'-m','systemlab','schedule','--manifest',str(Path(a.manifest).resolve()),
            '--role','test','--users',str(registration['users']),'--duration',str(registration['horizon_s']),
            '--rate',repr(rate),'--seed',str(seed),'--output',str(path)],cwd=lab,check=True)
        schedules.append(path)
    register_test_registry(a.config,a.grid,a.validation_registration,a.manifest,schedules,a.registry_output,
                           generators=[Path(__file__).resolve()],actor_bundles={'qla':a.actor_bundle})
    print('E1 schedules and v2 registry prepared; no experiment started:',a.registry_output)


if __name__=='__main__':
    main()
