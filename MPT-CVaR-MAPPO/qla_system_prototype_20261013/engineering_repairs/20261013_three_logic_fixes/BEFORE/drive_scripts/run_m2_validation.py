import json, subprocess, sys, time
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

reg = json.load(open(lab / "protocol/validation_registration_v1.json"))
order = reg["method_execution_order"]

for method in order:
    out = f"runs/M2_{method}_s101_attempt_01"
    print(f"=== {method} -> {out} ===", flush=True)
    result = subprocess.run([str(py), "-m", "systemlab", "run-local",
        "--config", "protocol/validation_config_v1.json",
        "--manifest", "data/validation_images.json",
        "--schedule", "data/validation_s101.json",
        "--method", method,
        "--actor-bundle", "protocol/validation_actor_v1.json",
        "--output", out], cwd=lab)
    print(f"{method} exit code:", result.returncode, flush=True)
    if result.returncode != 0:
        sys.exit(result.returncode)
    subprocess.run([str(py), "-m", "systemlab", "analyze",
        "--client-dir", f"{out}/client", "--controller-dir", f"{out}/controller",
        "--output", f"{out}/analysis.json"], cwd=lab, check=True)
    a = json.load(open(lab / out / "analysis.json"))
    print(method, "status:", a["status_counts"], "completed:", a["completed"],
          "cvar99:", round(a["cvar99_s"], 4), "infra:", a["infrastructure_valid"], flush=True)
print("ALL_METHODS_OK")
