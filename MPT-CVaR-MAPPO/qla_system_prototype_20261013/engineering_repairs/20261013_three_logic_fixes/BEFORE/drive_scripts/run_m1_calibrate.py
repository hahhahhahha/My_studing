import subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

steps = [
    ["-m", "systemlab", "calibrate", "--client-dir", "runs/M1_rr_s1_attempt_01/client",
     "--edges-dir", "runs/M1_rr_s1_attempt_01/edges", "--config", "configs/local_4edge.json",
     "--manifest", "data/calibration_images.json", "--output", "protocol/calibration_scales_v1.json"],
    ["tools/configure_pilot.py", "--base-config", "configs/local_4edge.json",
     "--feature-scales", "protocol/calibration_scales_v1.json",
     "--output", "protocol/validation_config_v1.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    result = subprocess.run([str(py), *step], cwd=lab)
    if result.returncode != 0:
        sys.exit(result.returncode)

import json
s = json.load(open(lab / "protocol/calibration_scales_v1.json"))
print(json.dumps(s, ensure_ascii=False)[:900])
c = json.load(open(lab / "protocol/validation_config_v1.json"))
print("val config schema:", c["schema"], "scales status:", c["feature_scales"].get("status"))
print("ALL_STEPS_OK")
