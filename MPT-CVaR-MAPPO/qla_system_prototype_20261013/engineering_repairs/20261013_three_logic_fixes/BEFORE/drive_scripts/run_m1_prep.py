import subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"
splits = repo / "dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json"
root = repo / "dac_visual_calibration_20261009/data/extracted"

steps = [
    ["-m", "systemlab", "images", "--split-file", str(splits), "--image-root", str(root),
     "--role", "calibration", "--limit", "2000", "--output", "data/calibration_images.json"],
    ["-m", "systemlab", "images", "--split-file", str(splits), "--image-root", str(root),
     "--role", "validation", "--limit", "500", "--output", "data/validation_images.json"],
    ["-m", "systemlab", "schedule", "--manifest", "data/calibration_images.json",
     "--role", "calibration", "--users", "16", "--duration", "30", "--rate", "20",
     "--seed", "1", "--output", "data/calibration_rr_s1.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    result = subprocess.run([str(py), *step], cwd=lab)
    if result.returncode != 0:
        sys.exit(result.returncode)
print("ALL_STEPS_OK")
