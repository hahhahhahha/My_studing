import json, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

reg = json.load(open(lab / "protocol/validation_registration_v1.json"))
dirs = [f"runs/M2_{m}_s101_attempt_01/client" for m in reg["method_execution_order"]]

steps = [
    ["-m", "systemlab", "freeze", "--config", "protocol/validation_config_v1.json",
     "--validation-client-dirs", *dirs, "--output-dir", "protocol/frozen_v1"],
    ["-m", "systemlab", "register-actor",
     "--checkpoint", str(repo / "results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"),
     "--source-root", str(repo / "results_qla_training_repair_pilot_20260930/source_snapshot"),
     "--feature-scales", "protocol/frozen_v1/FEATURE_SCALES.json", "--edges", "4",
     "--allow-edge-count-transfer", "--output", "protocol/formal_actor_v1.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    result = subprocess.run([str(py), *step], cwd=lab)
    if result.returncode != 0:
        sys.exit(result.returncode)

frozen = lab / "protocol/frozen_v1"
print("frozen files:", sorted(p.name for p in frozen.iterdir()))
for name in ("FROZEN_SYSTEM_PROTOCOL.json", "FEATURE_SCALES.json"):
    p = frozen / name
    if p.exists():
        d = json.load(open(p))
        if name.startswith("FROZEN"):
            print("protocol keys:", list(d)[:12])
        else:
            print("frozen scales status:", d.get("status"), "role:", d.get("fitted_role"))
print("ALL_STEPS_OK")
