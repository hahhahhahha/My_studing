import json, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

steps = [
    ["tools/make_grid.py", "--feature-scales", "protocol/frozen_v2/FEATURE_SCALES.json",
     "--config", "protocol/frozen_v2/FORMAL_CONFIG.json", "--block", "E1",
     "--output", "protocol/E1_grid_v2.json"],
    ["drive_scripts/run_e1_prep.py", "--config", "protocol/frozen_v2/FORMAL_CONFIG.json",
     "--grid", "protocol/E1_grid_v2.json",
     "--validation-registration", "protocol/validation_registration_v1.json",
     "--manifest", "data/test_images.json", "--actor-bundle", "protocol/formal_actor_v2.json",
     "--schedule-output-dir", "data/E1_common_v2", "--registry-output", "protocol/E1_test_registry_v2.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    r = subprocess.run([str(py), *step], cwd=lab)
    print("exit:", r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

g = json.load(open(lab / "protocol/E1_grid_v2.json"))
print("E1_grid_v2 cells:", len(g["cells"]), "| started:", g["started"], "| ref cap:", round(g["reference_capacity_per_s"], 2))
reg = json.load(open(lab / "protocol/E1_test_registry_v2.json"))
print("registry schema:", reg["schema"], "| schedules:", len(reg["common_schedules"]), "| actor bundles:", list(reg["actor_bundles"]))
print("ALL_STEPS_OK")
