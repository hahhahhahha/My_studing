"""Create a new pilot configuration with actual calibration scales; no launch."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import read,write_new

p=argparse.ArgumentParser()
p.add_argument('--base-config',required=True)
p.add_argument('--feature-scales',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
config,scales=read(a.base_config),read(a.feature_scales)
if scales['status']!='fitted_calibration_requires_validation_and_registration':
    raise ValueError('Actual calibration output required')
if set(scales['calibration_service_s_by_edge'])!={e['id'] for e in config['edges']}:
    raise ValueError('Calibrated/configured edge membership mismatch')
config['feature_scales']=scales
write_new(a.output,config)
