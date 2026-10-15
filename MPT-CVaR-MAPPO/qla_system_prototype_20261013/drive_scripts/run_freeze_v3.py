import json, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"
ckpt = repo / "results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"
source = repo / "results_qla_training_repair_pilot_20260930/source_snapshot"

reg = json.load(open(lab / "protocol/validation_registration_v1.json"))
dirs = [f"runs/M2d_{m}_s101_attempt_01/client" for m in reg["method_execution_order"]]

steps = [
    ["-m", "systemlab", "freeze", "--config", "protocol/validation_config_v1.json",
     "--validation-client-dirs", *dirs, "--output-dir", "protocol/frozen_v3"],
    ["-m", "systemlab", "register-actor", "--checkpoint", str(ckpt), "--source-root", str(source),
     "--feature-scales", "protocol/frozen_v3/FEATURE_SCALES.json", "--edges", "4",
     "--allow-edge-count-transfer", "--output", "protocol/formal_actor_v3.json"],
    ["tools/make_grid.py", "--feature-scales", "protocol/frozen_v3/FEATURE_SCALES.json",
     "--config", "protocol/frozen_v3/FORMAL_CONFIG.json", "--block", "E1",
     "--output", "protocol/E1_grid_v3.json"],
    ["drive_scripts/run_e1_prep.py", "--config", "protocol/frozen_v3/FORMAL_CONFIG.json",
     "--grid", "protocol/E1_grid_v3.json",
     "--validation-registration", "protocol/validation_registration_v1.json",
     "--manifest", "data/test_images.json", "--actor-bundle", "protocol/formal_actor_v3.json",
     "--schedule-output-dir", "data/E1_common_v3", "--registry-output", "protocol/E1_test_registry_v3.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    r = subprocess.run([str(py), "-B", *step], cwd=lab)
    print("exit:", r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("frozen_v3 scales status:", json.load(open(lab/"protocol/frozen_v3/FEATURE_SCALES.json"))["status"])
g = json.load(open(lab/"protocol/E1_grid_v3.json"))
print("E1_grid_v3 cells:", len(g["cells"]), "started:", g["started"])
reg2 = json.load(open(lab/"protocol/E1_test_registry_v3.json"))
print("registry schema:", reg2["schema"], "schedules:", len(reg2["common_schedules"]), "actors:", list(reg2["actor_bundles"]))
print("ALL_STEPS_OK")
