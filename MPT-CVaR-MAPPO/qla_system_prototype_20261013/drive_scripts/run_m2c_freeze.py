import json, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"
ckpt = repo / "results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"
source = repo / "results_qla_training_repair_pilot_20260930/source_snapshot"

reg = json.load(open(lab / "protocol/validation_registration_v1.json"))
dirs = [f"runs/M2c_{m}_s101_attempt_01/client" for m in reg["method_execution_order"]]

steps = [
    ["-m", "systemlab", "freeze", "--config", "protocol/validation_config_v1.json",
     "--validation-client-dirs", *dirs, "--output-dir", "protocol/frozen_v2"],
    ["-m", "systemlab", "register-actor", "--checkpoint", str(ckpt), "--source-root", str(source),
     "--feature-scales", "protocol/frozen_v2/FEATURE_SCALES.json", "--edges", "4",
     "--allow-edge-count-transfer", "--output", "protocol/formal_actor_v2.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    r = subprocess.run([str(py), *step], cwd=lab)
    print("exit:", r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

d = json.load(open(lab / "protocol/frozen_v2/FROZEN_SYSTEM_PROTOCOL.json"))
print("frozen_v2 schema:", d["schema"], "| independent_entry_review_required:", d["independent_entry_review_required"])
s = json.load(open(lab / "protocol/frozen_v2/FEATURE_SCALES.json"))
print("frozen_v2 scales status:", s["status"])
a = json.load(open(lab / "protocol/formal_actor_v2.json"))
print("formal_actor_v2 transfer:", a["transfer"], "| target_edges:", a["target_edges"])
print("ALL_STEPS_OK")
