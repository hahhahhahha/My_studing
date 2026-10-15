import json, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

reg = json.load(open(lab / "protocol/validation_registration_v1.json"))
order = reg["method_execution_order"]
print("pre-registered order:", order)

for method in order:
    out = f"runs/M2d_{method}_s101_attempt_01"
    print(f"=== {method} -> {out} ===", flush=True)
    r = subprocess.run([str(py), "-m", "systemlab", "run-local",
        "--config", "protocol/validation_config_v1.json",
        "--manifest", "data/validation_images.json",
        "--schedule", "data/validation_s101.json",
        "--method", method,
        "--actor-bundle", "protocol/validation_actor_v1.json",
        "--output", out], cwd=lab)
    print(f"{method} exit code:", r.returncode, flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    subprocess.run([str(py), "-m", "systemlab", "analyze",
        "--client-dir", f"{out}/client", "--controller-dir", f"{out}/controller",
        "--output", f"{out}/analysis.json"], cwd=lab, check=True)
    a = json.load(open(lab / out / "analysis.json"))
    c = json.load(open(lab / out / "client/COMPLETE.json"))
    print(method, "status:", a["status_counts"], "cvar99:", round(a["cvar99_s"], 4),
          "infra:", a["infrastructure_valid"], "source_ok:", c["source_continuity_pass"], flush=True)
print("ALL_METHODS_OK")
