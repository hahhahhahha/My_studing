import subprocess, sys, json
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

# C_ref from protocol/calibration_scales_v1.json (low-load round_robin measurement)
s = json.load(open(lab / "protocol/calibration_scales_v1.json"))
c_ref = sum(1.0 / v for v in s["calibration_service_s_by_edge"].values())
rate = round(0.9 * c_ref)
print("C_ref =", round(c_ref, 2), "req/s; stress rate =", rate)

steps = [
    ["-m", "systemlab", "schedule", "--manifest", "data/calibration_images.json",
     "--role", "calibration", "--users", "16", "--duration", "120", "--rate", str(rate),
     "--seed", "1", "--output", "data/calibration_stress_s1.json"],
    ["-m", "systemlab", "run-local", "--config", "configs/local_4edge.json",
     "--manifest", "data/calibration_images.json", "--schedule", "data/calibration_stress_s1.json",
     "--method", "round_robin", "--output", "runs/M1_stress_s1_attempt_01"],
    ["-m", "systemlab", "analyze", "--client-dir", "runs/M1_stress_s1_attempt_01/client",
     "--controller-dir", "runs/M1_stress_s1_attempt_01/controller",
     "--output", "runs/M1_stress_s1_attempt_01/analysis.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    result = subprocess.run([str(py), *step], cwd=lab)
    if result.returncode != 0:
        sys.exit(result.returncode)
print("ALL_STEPS_OK")
