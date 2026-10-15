import json, subprocess, sys, hashlib
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

splits = repo / "dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json"
image_root = repo / "dac_visual_calibration_20261009/data/extracted"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1024*1024), b""):
            h.update(c)
    return h.hexdigest()

steps = [
    ["-m", "systemlab", "images", "--split-file", str(splits), "--image-root", str(image_root),
     "--role", "test", "--limit", "1000", "--output", "data/test_images.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    r = subprocess.run([str(py), *step], cwd=lab)
    if r.returncode != 0:
        sys.exit(r.returncode)

grid = json.load(open(lab / "protocol/E1_grid.json"))
capacity = grid["reference_capacity_per_s"]
horizon = 120  # frozen H from protocol/validation_registration_v1.json (same window rule)

# One common schedule per (rho, seed); rate read from the grid's capacity definition.
manifest = "data/test_images.json"
schedule_rows = []
for rho in (0.6, 0.9):
    for seed in (1, 2, 3):
        rate = rho * capacity
        out = f"data/E1_rho{int(rho*100):02d}_s{seed}.json"
        r = subprocess.run([str(py), "-m", "systemlab", "schedule",
            "--manifest", manifest, "--role", "test", "--users", "16",
            "--duration", str(horizon), "--rate", f"{rate:.6f}", "--seed", str(seed),
            "--output", out], cwd=lab)
        if r.returncode != 0:
            sys.exit(r.returncode)
        sched = json.load(open(lab / out))
        schedule_rows.append(dict(file=out, common_schedule_key=f"{rho}_{seed}",
            rho=rho, seed=seed, aggregate_rate_per_s=rate, duration_s=sched["duration_s"],
            tasks=len(sched["tasks"]), schedule_sha256=sha(lab / out)))

manifest_sha = sha(lab / manifest)
registry = dict(schema="systemlab-test-registry-v1",
    grid=dict(path="protocol/E1_grid.json", sha256=sha(lab / "protocol/E1_grid.json"),
              config_sha256=grid["config_sha256"], feature_scales_sha256=grid["feature_scales_sha256"],
              reference_capacity_per_s=capacity),
    horizon_s=horizon, fixed_drain_s=30, users=16,
    test_manifest=dict(path=manifest, sha256=manifest_sha, role="test"),
    common_schedules=schedule_rows,
    note="One common schedule per (rho,seed); the same file is shared by all three methods in that cell. "
         "Rates are rho*reference_capacity_per_s from the frozen grid; H is the frozen 120s window.")
out = lab / "protocol/E1_test_registry.json"
out.open("x").write(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
for row in schedule_rows:
    print(f"{row['file']}: rate={row['aggregate_rate_per_s']:.3f}/s tasks={row['tasks']} sha={row['schedule_sha256'][:12]}")
print("test manifest:", len(json.load(open(lab/manifest))["images"]), "images; registry written to protocol/E1_test_registry.json")
print("ALL_STEPS_OK")
