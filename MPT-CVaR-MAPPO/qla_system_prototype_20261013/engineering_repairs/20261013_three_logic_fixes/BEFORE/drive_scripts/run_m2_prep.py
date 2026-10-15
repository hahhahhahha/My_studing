import json, random, subprocess, sys
from pathlib import Path

repo = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
lab = repo / "qla_system_prototype_20261013"
py = repo / "dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"

ckpt = repo / "results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"
source = repo / "results_qla_training_repair_pilot_20260930/source_snapshot"

scales = json.load(open(lab / "protocol/calibration_scales_v1.json"))
c_ref = sum(1.0 / v for v in scales["calibration_service_s_by_edge"].values())
rate = round(0.6 * c_ref)          # 105 req/s pre-registered from calibration capacity only
horizon = 120                       # ceil_to_minute(max(120, 10000/(0.6*C_ref))) = 120
order = ["qla", "score_softmax", "stale_shortest"]
random.Random().shuffle(order)      # order randomized before any validation run

registration = dict(
    schema="systemlab-validation-preregistration-v1",
    c_ref_per_s=c_ref, validation_rate_per_s=rate, horizon_s=horizon,
    fixed_drain_s=30, users=16, schedule_seed=101, role="validation",
    method_execution_order=order,
    rate_rule="0.6*C_ref from protocol/calibration_scales_v1.json (calibration only; test untouched)",
    window_rule="ceil_to_minute(max(120,10000/(0.6*C_ref)))",
    capacity_evidence=["runs/M1_rr_s1_attempt_01/analysis.json", "runs/M1_stress_s1_attempt_01/analysis.json"],
    stress_note="90%C_ref pilot saturated: 3164/18916 queue_rejected, max_waiting=32, no client backpressure",
)
out = lab / "protocol/validation_registration_v1.json"
out.open("x").write(json.dumps(registration, indent=2, ensure_ascii=False) + "\n")
print("pre-registered order:", order, "rate:", rate, "C_ref:", round(c_ref, 2))

steps = [
    ["-m", "systemlab", "register-actor", "--checkpoint", str(ckpt), "--source-root", str(source),
     "--feature-scales", "protocol/calibration_scales_v1.json", "--edges", "4",
     "--allow-edge-count-transfer", "--output", "protocol/validation_actor_v1.json"],
    ["tools/check_actor_transfer.py", "--config", "protocol/validation_config_v1.json",
     "--bundle", "protocol/validation_actor_v1.json", "--output", "protocol/validation_actor_forward_v1.json"],
    ["-m", "systemlab", "schedule", "--manifest", "data/validation_images.json",
     "--role", "validation", "--users", "16", "--duration", "120", "--rate", str(rate),
     "--seed", "101", "--output", "data/validation_s101.json"],
]
for step in steps:
    print("RUN:", " ".join(step), flush=True)
    result = subprocess.run([str(py), *step], cwd=lab)
    if result.returncode != 0:
        sys.exit(result.returncode)
print("ALL_STEPS_OK")
