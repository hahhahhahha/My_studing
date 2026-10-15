"""Prepare/execute the predeclared critic x rollout-bootstrap paired pilot.

Runs smoke -> all train/validation -> validation gate -> freeze -> held-out
test of ALL six arms. No test performance is used to select a repair.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_training_repair_core import sha256, source_digest
from mappo_gp_sf_dsr.run_qla_simplified_rerun import SimplifiedVariant, build_command, find_summary

DEFAULT_OUT = ROOT / "results_qla_training_repair_pilot_20260930"
ARMS = [("Full", r) for r in ("original", "critic", "bootstrap", "both")] + [("NoQL", r) for r in ("original", "both")]


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_row(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def old_directory(policy, seed, age):
    if policy == "Full":
        return ROOT / "results_qla_canonical_v1/qla/main/QLA-MAPPO-NoMargin" / f"seed_{seed}" / f"delay_{age}"
    return ROOT / "results_qla_signal_components_v1/cells/QLA-No-QL" / f"seed_{seed}" / f"delay_{age}"


def legacy_audit():
    """A provenance reference, NOT permission to treat incomplete diagnostics as new runs."""
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "torch")}
    records = []
    for policy in ("Full", "NoQL"):
        for seed in (1, 2, 3):
            for age in (0, 8):
                summary = find_summary(old_directory(policy, seed, age))
                record = {"policy": policy, "seed": seed, "age": age, "eligible_for_reuse": False}
                reasons = []
                if summary is None:
                    reasons.append("missing historical cell")
                else:
                    directory = summary.parent
                    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    config = json.loads((directory / "config_LCC-MAPPO.json").read_text(encoding="utf-8"))
                    row = read_row(summary)
                    record.update(summary=str(summary), source_sha256=manifest["source_sha256"],
                                  historical_runtime={k: manifest[k] for k in ("python", "numpy", "torch")},
                                  current_runtime=runtime)
                    for filename, key in (("config_LCC-MAPPO.json", "config_sha256"), ("checkpoint_LCC-MAPPO.pt", "checkpoint_sha256")):
                        if sha256(directory / filename) != manifest[key]:
                            reasons.append(f"{filename} integrity mismatch")
                    expected = {"seed": seed, "feedback_delay": age, "train_steps": 10000, "test_steps": 1000,
                                "lcc_price_components": "QT" if policy == "Full" else "NoPrice",
                                "evaluation_split_role": "test", "lcc_edgewise_price_actor": True,
                                "lcc_no_z_queue": True, "lcc_price_logit_bias": .5, "users": 30, "n_edges": 35}
                    for key, value in expected.items():
                        if config.get(key) != value:
                            reasons.append(f"historical config mismatch: {key}")
                    for key in ("numpy", "torch"):
                        if runtime[key] != manifest[key]:
                            reasons.append(f"runtime drift: {key} {manifest[key]} -> {runtime[key]}")
                    if float(row["episode_mass_conservation_error"]) != 0.:
                        reasons.append("nonzero historical accounting error")
                    # Legacy runs have no per-update gradient/advantage/boundary
                    # diagnostics. Rerunning both original arms makes diagnostics
                    # paired and runtime-matched, rather than leaving them absent.
                    reasons.append("historical per-update/boundary diagnostics absent; fresh reference required")
                record["reasons"] = reasons
                records.append(record)
    return records


def variant(policy):
    return SimplifiedVariant(policy, {"lcc-price-components": "QT" if policy == "Full" else "NoPrice"})


def command_for(out, policy, repair, seed, age, steps, eval_steps, role, expected_source, checkpoint=None):
    base = build_command(variant(policy), seed, age, out, "cuda",
                         {"train-steps": steps, "test-steps": eval_steps,
                          "evaluation-split-role": role})
    if checkpoint:
        base.extend(["--load-checkpoint", str(checkpoint)])
    return [sys.executable, "-B", str(Path(__file__).with_name("paired_training_repair_worker.py")),
            "--repair", repair, "--diagnostic-dir", str(out / "diagnostics"),
            "--expected-source", expected_source, "--", *base[4:]]


def numerical_gate(out, role):
    summary = find_summary(out)
    if summary is None:
        raise RuntimeError("Missing completed summary: " + str(out))
    row = read_row(summary)
    for name in ("average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate"):
        if not math.isfinite(float(row[name])):
            raise RuntimeError("Non-finite metric: " + name)
    if row["evaluation_split_role"] != role or abs(float(row["episode_mass_conservation_error"])) > 1e-6:
        raise RuntimeError("Split/accounting gate failed: " + str(out))
    if row["drain_finished"].lower() != "true" or int(float(row["n_unfinished_tasks"])) != 0:
        raise RuntimeError("Drain gate failed: " + str(out))
    manifest = json.loads((out / "diagnostics/repair_manifest.json").read_text(encoding="utf-8"))
    updates = out / "diagnostics/updates.csv"
    if updates.exists():
        with updates.open(newline="", encoding="utf-8") as f:
            for update in csv.DictReader(f):
                if not all(math.isfinite(float(v)) for v in update.values()):
                    raise RuntimeError("Non-finite training diagnostic")
    return summary, row, manifest


def run_cell(out, command, root, cell):
    role = "validation" if cell["stage"] != "test" else "test"
    if (out / "COMPLETE.json").exists():
        numerical_gate(out, role)
        print("[reuse-new] " + str(out), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    json_write(root / "STATUS.json", {"status": "running", "cell": cell, "command": command, "started": datetime.now().isoformat()})
    print("[run] " + json.dumps(cell), flush=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started = perf_counter()
    with (out / "console.log").open("a", encoding="utf-8") as f:
        f.write(subprocess.list2cmdline(command) + "\n")
        f.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    summary, _, manifest = numerical_gate(out, role)
    json_write(out / "COMPLETE.json", {"summary": str(summary), "elapsed_s": perf_counter() - started,
                                     "worker_sha256": manifest["worker_sha256"], "source_sha256": manifest["source_sha256"]})
    print(f"[done] elapsed_s={perf_counter() - started:.1f}", flush=True)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "DESIGN.json"
    worker = Path(__file__).with_name("paired_training_repair_worker.py")
    helper = Path(__file__).with_name("paired_training_repair_core.py")
    lock = {"source_sha256": source_digest(ROOT), "worker_sha256": sha256(worker), "helper_sha256": sha256(helper),
            "runner_sha256": sha256(__file__)}
    if design_path.exists():
        design = json.loads(design_path.read_text(encoding="utf-8"))
        if design["code_lock"] != lock:
            raise RuntimeError("Code lock changed; use a new output root, never mix versions")
        return design
    old = legacy_audit()
    json_write(root / "historical_reuse_audit.json", old)
    cells = [dict(policy=p, repair=r, seed=s, age=d) for s in (1, 2, 3) for d in (0, 8) for p, r in ARMS]
    design = {"study": "Paired critic-input x rollout-bootstrap correctness pilot", "created": datetime.now().isoformat(),
              "code_lock": lock, "conditions": cells, "train_steps_per_condition": 10000,
              "validation_steps": 1000, "test_steps": 1000, "new_trainings": 36, "historical_reuse": 0,
              "reuse_reason": "runtime drift and unavailable historical per-update diagnostics; rerun original arms",
              "critic_fix": "zero six current-action portfolio feature columns; same architecture/parameter/init dimensions",
              "bootstrap_fix": "next-state value at every administrative rollout/time-limit boundary; true terminals remain zero",
              "bootstrap_only_arm": "16 antithetic samples estimate policy-expected next-state action-conditioned value using an isolated generator",
              "checkpoint_rule": "final checkpoint after exactly 10000 training slots; no early stopping or test selection",
              "validation_gate": "finite metrics/GAE/gradients; correct split; conservation <=1e-6; completed drain; no performance threshold",
              "freeze_rule": "all six arms frozen together after validation; all proceed to held-out test; no winner selected from test",
              "latency_semantics": "unchanged workload-derived slot-level latency", "nominal_delays": [0, 8], "seeds": [1, 2, 3],
              "training_reward_and_entropy": "unchanged canonical reward, target entropy .99, QL bias .5, V .5, migration weight .05",
              "rollout_counting": "1024 user-transitions (35 slots for 30 users), unchanged",
              "outputs": ["all metrics and seed variability", "queue jump magnitudes", "critic MSE", "raw advantage distributions",
                          "KL", "preclip gradient norms", "rollout boundary values/advantages", "hashes and schedules"],
              "execution": "single local CUDA worker; no recurring monitoring or external notifications"}
    json_write(design_path, design)
    snapshot = root / "source_snapshot"
    for package in ("mappo_gp_sf_dsr", "repro_risk_averse_edge"):
        target = snapshot / package
        target.mkdir(parents=True, exist_ok=True)
        for path in (ROOT / package).glob("*.py"):
            shutil.copy2(path, target / path.name)
    for path in (Path(__file__), worker, helper, Path(__file__).with_name("test_paired_training_repair.py")):
        shutil.copy2(path, snapshot / path.name)
    return design


def aggregate(root, design):
    rows = []
    for cell in design["conditions"]:
        out = root / "test" / f'{cell["policy"]}_{cell["repair"]}' / f'seed_{cell["seed"]}' / f'delay_{cell["age"]}'
        _, row, extra = numerical_gate(out, "test")
        rows.append({**cell, **{k: float(row[k]) for k in
                              ("average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate")},
                     **{k: extra[k] for k in ("mean_queue_jump_service_slots", "p95_queue_jump_service_slots", "p99_queue_jump_service_slots")}})
    with (root / "test_conditions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    metrics = [k for k in rows[0] if k not in ("policy", "repair", "seed", "age")]
    contrasts, summary = [], []
    import numpy as np
    for policy, repair in ARMS:
        selected = [r for r in rows if r["policy"] == policy and r["repair"] == repair]
        for metric in metrics:
            seed_means = [np.mean([r[metric] for r in selected if r["seed"] == seed]) for seed in (1, 2, 3)]
            summary.append({"policy": policy, "repair": repair, "metric": metric,
                            "mean": float(np.mean(seed_means)), "seed_sd": float(np.std(seed_means, ddof=1))})
            if repair != "original":
                for row in selected:
                    baseline = next(r for r in rows if r["policy"] == policy and r["repair"] == "original" and
                                    r["seed"] == row["seed"] and r["age"] == row["age"])
                    contrasts.append({"policy": policy, "repair": repair, "seed": row["seed"], "age": row["age"],
                                      "metric": metric, "repair_minus_original": row[metric] - baseline[metric]})
    for name, data in (("paired_contrasts.csv", contrasts), ("seed_variability_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    design = prepare(root)
    if args.prepare_only:
        print(json.dumps({"design": str(root / "DESIGN.json"), "new_trainings": design["new_trainings"], "historical_reuse": 0}, indent=2))
        return
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
        torch.manual_seed(123)
        x = torch.randn((8, 8), device="cuda")
        result = x @ x
        if not torch.isfinite(result).all():
            raise RuntimeError("CUDA kernel witness failed")
        json_write(root / "runtime.json", {"python": sys.version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
        subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("test_paired_training_repair.py"))], cwd=ROOT, check=True)
        # Smoke does not inspect test data and does not change training knobs.
        for policy, repair in ARMS:
            cell = dict(stage="smoke", policy=policy, repair=repair, seed=1, age=8)
            out = root / "smoke" / f"{policy}_{repair}"
            cmd = command_for(out, policy, repair, 1, 8, 72, 32, "validation", design["code_lock"]["source_sha256"])
            run_cell(out, cmd, root, cell)
        for cell in design["conditions"]:
            p, r, s, d = (cell[k] for k in ("policy", "repair", "seed", "age"))
            out = root / "validation" / f"{p}_{r}" / f"seed_{s}" / f"delay_{d}"
            cmd = command_for(out, p, r, s, d, 10000, 1000, "validation", design["code_lock"]["source_sha256"])
            run_cell(out, cmd, root, {"stage": "train_validation", **cell})
        frozen = {"code_lock": design["code_lock"], "frozen_at": datetime.now().isoformat(),
                  "rule": "all arms passed numerical/split/accounting validation; no arm chosen by performance", "checkpoints": []}
        for cell in design["conditions"]:
            p, r, s, d = (cell[k] for k in ("policy", "repair", "seed", "age"))
            out = root / "validation" / f"{p}_{r}" / f"seed_{s}" / f"delay_{d}"
            summary, row, _ = numerical_gate(out, "validation")
            checkpoint = summary.with_name("checkpoint_LCC-MAPPO.pt")
            frozen["checkpoints"].append({**cell, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
                                          "train_schedule_sha256": row["train_schedule_sha256"],
                                          "validation_schedule_sha256": row["validation_schedule_sha256"]})
        if not (root / "FROZEN_BEFORE_TEST.json").exists():
            json_write(root / "FROZEN_BEFORE_TEST.json", frozen)
        else:
            previous = json.loads((root / "FROZEN_BEFORE_TEST.json").read_text(encoding="utf-8"))
            if previous["code_lock"] != frozen["code_lock"] or previous["checkpoints"] != frozen["checkpoints"]:
                raise RuntimeError("Previously frozen checkpoints changed")
        schedule_blocks = {}
        for cell in frozen["checkpoints"]:
            p, r, s, d = (cell[k] for k in ("policy", "repair", "seed", "age"))
            out = root / "test" / f"{p}_{r}" / f"seed_{s}" / f"delay_{d}"
            cmd = command_for(out, p, r, s, d, 0, 1000, "test", design["code_lock"]["source_sha256"], cell["checkpoint"])
            run_cell(out, cmd, root, {"stage": "test", **{k: cell[k] for k in ("policy", "repair", "seed", "age")}})
            _, row, _ = numerical_gate(out, "test")
            key = (s, d)
            hashes = (cell["train_schedule_sha256"], row["test_schedule_sha256"])
            if key in schedule_blocks and schedule_blocks[key] != hashes:
                raise RuntimeError("Paired exogenous schedules mismatch")
            schedule_blocks[key] = hashes
        aggregate(root, design)
        json_write(root / "STATUS.json", {"status": "completed", "training_conditions": 36, "test_conditions": 36,
                                         "finished": datetime.now().isoformat()})
    except Exception as exc:
        json_write(root / "STATUS.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                         "at": datetime.now().isoformat()})
        raise


if __name__ == "__main__":
    main()
