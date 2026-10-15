"""Bind an existing grid and complete schedules; does not launch experiments."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.admission import register_test_registry

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--grid',required=True)
p.add_argument('--validation-registration',required=True)
p.add_argument('--manifest',required=True)
p.add_argument('--schedules',nargs='+',required=True)
p.add_argument('--generator',nargs='*',default=[])
p.add_argument('--qla-actor',help='Actual validated frozen formal QLA bundle')
p.add_argument('--output',required=True)
a=p.parse_args()
register_test_registry(a.config,a.grid,a.validation_registration,a.manifest,a.schedules,a.output,
                       [str(Path(__file__).resolve()),*a.generator],
                       actor_bundles={'qla':a.qla_actor} if a.qla_actor else {})
