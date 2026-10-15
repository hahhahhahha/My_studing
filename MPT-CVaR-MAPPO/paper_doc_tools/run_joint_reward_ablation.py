"""Prepare, preflight, smoke, train/freeze, and evaluate one R1 reward ablation.

No canonical code is edited. CUDA is mandatory; failed preflight is archived.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_reward_ablation_core import VARIANT

BASE = ROOT / "results_qla_formal_bothfix_20260930"
ARCHIVE = ROOT / "results_qla_training_repair_pilot_20260930/source_snapshot"
DEFAULT_OUT = ROOT / "results_qla_joint_reward_ablation_20261009"
SEEDS = (1, 2, 3, 4, 5)
AGES = (0, 1, 2, 4, 8)
PRIMARY = ("average_latency", "cvar99", "queue_oscillation", "portfolio_migration_rate")
SECONDARY = ("max_realized_share_p99", "mean_realized_hhi", "mean_queue_jump_service_slots",
             "p95_queue_jump_service_slots", "p99_queue_jump_service_slots",
             "unconditional_queue_spike_rate", "broadcast_window_queue_spike_rate")
SECONDARY = (*SECONDARY, "full_three_slot_update_spike_rate")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_sha(root, package="mappo_gp_sf_dsr"):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / package).glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_row(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))


def summary_path(out):
    matches = list(out.rglob("summary_LCC-MAPPO.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary in {out}, got {len(matches)}")
    return matches[0]


def prepare(out):
    registry = json.loads((BASE / "FROZEN_BEFORE_TEST.json").read_text(encoding="utf-8"))
    expected = registry["code_lock"]["source_sha256"]
    if package_sha(ARCHIVE) != expected:
        raise RuntimeError("Archived R1 package does not match the original source lock")
    references = []
    runtimes = set()
    for item in registry["checkpoints"]:
        if item["variant"] != "Full":
            continue
        checkpoint = Path(item["checkpoint"])
        if sha(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError("Historical Full checkpoint hash changed")
        validation = Path(item["validation"])
        v_summary = summary_path(validation)
        config_path = v_summary.with_name("config_LCC-MAPPO.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        provenance = json.loads(config_path.with_name("manifest.json").read_text(encoding="utf-8"))
        if sha(config_path) != provenance["config_sha256"]:
            raise RuntimeError("Historical Full config hash changed")
        if provenance["checkpoint_sha256"] != item["checkpoint_sha256"] or provenance["source_sha256"] != expected:
            raise RuntimeError("Historical training provenance mismatch")
        train_repair = json.loads((validation / "diagnostics/repair_manifest.json").read_text(encoding="utf-8"))
        if train_repair["repair"] != "both" or not train_repair["critic_action_features_zeroed"] or not train_repair["rollout_bootstrap"]:
            raise RuntimeError("Historical checkpoint was not trained with both required semantics")
        runtimes.add(tuple(provenance[k] for k in ("python", "numpy", "torch", "device")))
        row = csv_row(v_summary)
        if row["train_schedule_sha256"] != item["train_schedule_sha256"]:
            raise RuntimeError("Historical training schedule mismatch")
        if row["evaluation_schedule_sha256"] != item["validation_schedule_sha256"]:
            raise RuntimeError("Historical validation schedule mismatch")
        test = Path(item.get("test", BASE / "test/Full" / f"seed_{item['seed']}" / f"delay_{item['age']}"))
        test_summary = summary_path(test)
        test_row = csv_row(test_summary)
        repair = json.loads((test / "diagnostics/repair_manifest.json").read_text(encoding="utf-8"))
        if not repair["critic_action_features_zeroed"] or not repair["rollout_bootstrap"]:
            raise RuntimeError("Historical Full training fixes are missing")
        expected_config = {"users": 30, "n_edges": 35, "train_steps": 10000, "test_steps": 1000,
                           "feedback_delay": item["age"], "seed": item["seed"],
                           "lcc_price_logit_bias": .5, "lcc_no_z_queue": True,
                           "lcc_no_h_queue": False, "lcc_price_components": "QT"}
        for key, value in expected_config.items():
            if config.get(key) != value:
                raise RuntimeError(f"Historical config mismatch: {key}")
        references.append({"variant": "Full", "seed": item["seed"], "age": item["age"],
                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                           "config": str(config_path), "config_sha256": sha(config_path),
                           "train_schedule_sha256": item["train_schedule_sha256"],
                           "validation_schedule_sha256": item["validation_schedule_sha256"],
                           "test_schedule_sha256": test_row["evaluation_schedule_sha256"],
                           "historical_metrics": {k: float(test_row[k]) for k in PRIMARY}})
    if {(x["seed"], x["age"]) for x in references} != set(itertools.product(SEEDS, AGES)):
        raise RuntimeError("Historical Full grid incomplete")
    if len(runtimes) != 1:
        raise RuntimeError("Historical Full training runtimes are not uniform")
    python_version, numpy_version, torch_version, device = next(iter(runtimes))
    environment = {"python": python_version, "numpy": numpy_version, "torch": torch_version,
                   "device": device, "gpu": "NVIDIA GeForce RTX 4060 Ti"}
    lock = {"source_sha256": expected,
            "repro_source_sha256": package_sha(ARCHIVE, "repro_risk_averse_edge"),
            "repair_worker_sha256": registry["code_lock"]["worker_sha256"],
            "repair_helper_sha256": registry["code_lock"]["helper_sha256"],
            "runner_sha256": sha(__file__),
            "intervention_core_sha256": sha(HERE / "joint_reward_ablation_core.py"),
            "intervention_worker_sha256": sha(HERE / "joint_reward_ablation_worker.py"),
            "tests_sha256": sha(HERE / "test_joint_reward_ablation.py")}
    design = {"study": "Demand-aware shared reward independent ablation",
              "variants": ["Full", VARIANT], "seeds": SEEDS, "nominal_delays": AGES,
              "new_training_conditions": 25, "Full_test_only_replays": 25,
              "train_steps": 10000, "validation_steps": 1000, "test_steps": 1000,
              "checkpoint_rule": "final checkpoint; freeze all 25 before held-out test",
              "reward_intervention": "uniform mean normalized queue penalty with idle gate; H*C omitted only in reward",
              "unchanged": "actor, QT, bias, H/C state update, shared reward, latency, migration, PPO, critic, bootstrap, schedules",
              "reward_scale": "q coefficient 1, q bounds [0,1], V .5, migration .05, same max/tanh; no fitted rescale",
              "source_lock": lock, "environment": environment, "Full_references": references,
              "analysis": "paired seed-block exact bootstrap over complete five-delay profiles; all results retained"}
    path = out / "DESIGN.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(design)):
            raise RuntimeError("Design/code changed; use a fresh output directory")
    else:
        save(path, design)
    source = out / "execution_source"
    for package in ("mappo_gp_sf_dsr", "repro_risk_averse_edge"):
        target = source / package
        target.mkdir(parents=True, exist_ok=True)
        for origin in (ARCHIVE / package).glob("*.py"):
            destination = target / origin.name
            if destination.exists() and sha(destination) != sha(origin):
                raise RuntimeError("Snapshot modified")
            if not destination.exists():
                shutil.copy2(origin, destination)
    target = source / "paper_doc_tools"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("paired_training_repair_worker.py", "paired_training_repair_core.py"):
        origin = BASE / "source_snapshot" / name
        destination = target / name
        if destination.exists() and sha(destination) != sha(origin):
            raise RuntimeError("Frozen training helper modified")
        if not destination.exists():
            shutil.copy2(origin, destination)
    if sha(target / "paired_training_repair_worker.py") != registry["code_lock"]["worker_sha256"]:
        raise RuntimeError("Frozen training worker lock mismatch")
    if sha(target / "paired_training_repair_core.py") != registry["code_lock"]["helper_sha256"]:
        raise RuntimeError("Frozen training helper lock mismatch")
    if package_sha(source, "repro_risk_averse_edge") != lock["repro_source_sha256"]:
        raise RuntimeError("Frozen execution/data package lock mismatch")
    save(out / "ENVIRONMENT_SPEC.json", {
        **environment, "python_executable": sys.executable,
        "device": "cuda", "cpu_threads": 1, "install_or_rebuild": False,
        "fallback_to_cpu": False, "readiness": "requires successful seeded CUDA witness and two arm smoke tests"})
    save(out / "CURRENT_ENVIRONMENT_METADATA.json", {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": importlib.metadata.version("numpy"), "torch": importlib.metadata.version("torch"),
        "metadata_only_not_a_CUDA_readiness_check": True})
    return design


def preflight(out):
    spec = json.loads((out / "ENVIRONMENT_SPEC.json").read_text(encoding="utf-8"))
    code = ("import json,sys,numpy as np; "
            "print(json.dumps({'python':sys.version,'numpy':np.__version__}),flush=True); import torch; "
            f"assert '.'.join(map(str,sys.version_info[:3]))=={spec['python']!r}; "
            f"assert torch.__version__=={spec['torch']!r}; assert np.__version__=={spec['numpy']!r}; "
            "assert torch.cuda.is_available(); "
            "assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 4060 Ti'; "
            "torch.manual_seed(123); x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4); "
            "assert (x@x.T).sum().item()==3680.0; "
            "print(json.dumps({'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,"
            "'gpu':torch.cuda.get_device_name(0),'kernel_witness':'pass'}))")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "preflight.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        save(out / "STATUS.json", {"status": "blocked_preflight", "training_started": False,
                                  "reason": "Historical PyTorch/CUDA runtime witness failed",
                                  "returncode": result.returncode, "log": str(out / "preflight.log"),
                                  "at": datetime.now().isoformat()})
        raise RuntimeError("CUDA/PyTorch preflight failed; no training launched")
    save(out / "runtime.json", json.loads(result.stdout.strip().splitlines()[-1]))
    cli = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvidia-smi.exe"
    usage = subprocess.run([str(cli), "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "gpu_occupancy.log").write_text(usage.stdout, encoding="utf-8")
    first = usage.stdout.strip().splitlines()[0].split(",") if usage.returncode == 0 else []
    if len(first) != 3 or int(first[0]) != 0 or int(first[1]) >= 500:
        save(out / "STATUS.json", {"status": "blocked_preflight", "training_started": False,
                                  "reason": "GPU availability/occupancy check failed", "log": str(out / "gpu_occupancy.log")})
        raise RuntimeError("GPU not verified free; refusing to compete with another job")


def command(out, cell, root, design, role, checkpoint=None, smoke=False):
    source = root / "execution_source"
    path = source / "mappo_gp_sf_dsr/run_qla_simplified_rerun.py"
    spec = importlib.util.spec_from_file_location("frozen_r1_commands", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    overrides = {"train-steps": 72 if smoke else (0 if checkpoint else 10000),
                 "test-steps": 32 if smoke else 1000, "evaluation-split-role": role,
                 "lcc-price-logit-bias": .5, "canonical-max-feedback-delay": 8,
                 "canonical-schedule-root": str(ROOT / "results_qla_canonical_v1/shared_schedules"),
                 "trace-root": str(ROOT / "Traces_TimeXY_10sec_txt"), "queue-constraint": "resample"}
    if checkpoint:
        overrides["load-checkpoint"] = str(checkpoint)
    variant = module.SimplifiedVariant("Full", {"lcc-price-components": "QT"})
    base = module.build_command(variant, cell["seed"], cell["age"], out, "cuda", overrides)
    return [sys.executable, "-B", str(HERE / "joint_reward_ablation_worker.py"),
            "--variant", cell["variant"], "--source-root", str(source),
            "--diagnostic-dir", str(out / "diagnostics"),
            "--expected-source", design["source_lock"]["source_sha256"], "--", *base[4:]]


def validate(out, role, cell, reference, design, smoke=False, checkpoint=None):
    summary = summary_path(out)
    row = csv_row(summary)
    manifest = json.loads((out / "diagnostics/intervention_manifest.json").read_text(encoding="utf-8"))
    repair = json.loads((out / "diagnostics/repair_manifest.json").read_text(encoding="utf-8"))
    if repair["worker_sha256"] != design["source_lock"]["repair_worker_sha256"] or repair["helper_sha256"] != design["source_lock"]["repair_helper_sha256"]:
        raise RuntimeError("Required training wrapper hash changed")
    if manifest["variant"] != cell["variant"] or manifest["source_sha256"] != design["source_lock"]["source_sha256"]:
        raise RuntimeError("Intervention provenance mismatch")
    if manifest["intervention_worker_sha256"] != design["source_lock"]["intervention_worker_sha256"]:
        raise RuntimeError("Worker changed")
    if manifest["intervention_core_sha256"] != design["source_lock"]["intervention_core_sha256"]:
        raise RuntimeError("Intervention changed")
    if sha(Path(manifest["effective_reward_file"])) != manifest["effective_reward_sha256"]:
        raise RuntimeError("Effective reward source artifact changed")
    if row["evaluation_split_role"] != role or manifest["evaluation_split_role"] != role:
        raise RuntimeError("Wrong held-out split")
    if not repair["critic_action_features_zeroed"] or not repair["rollout_bootstrap"]:
        raise RuntimeError("Required training semantics missing")
    for key in PRIMARY:
        if not math.isfinite(float(row[key])):
            raise RuntimeError("Nonfinite performance metric")
    if abs(float(row["episode_mass_conservation_error"])) > 1e-6:
        raise RuntimeError("Mass conservation failed")
    if row["drain_finished"].lower() != "true" or int(float(row["n_unfinished_tasks"])):
        raise RuntimeError("Drain failed")
    actual_config = json.loads(summary.with_name("config_LCC-MAPPO.json").read_text(encoding="utf-8"))
    expected_steps = 72 if smoke else (10000 if role == "validation" else 0)
    if actual_config["train_steps"] != expected_steps or actual_config["test_steps"] != (32 if smoke else 1000):
        raise RuntimeError("Training/evaluation budget changed")
    if role == "test" and (checkpoint is None or Path(actual_config["load_checkpoint"]).resolve() != Path(checkpoint).resolve()):
        raise RuntimeError("Test did not load the exact frozen checkpoint")
    if not smoke:
        expected_schedule = reference["validation_schedule_sha256" if role == "validation" else "test_schedule_sha256"]
        if row["evaluation_schedule_sha256"] != expected_schedule:
            raise RuntimeError("Common evaluation schedule mismatch")
        if role == "validation" and row["train_schedule_sha256"] != reference["train_schedule_sha256"]:
            raise RuntimeError("Common training schedule mismatch")
        if manifest["evaluation_slots"] != 1000:
            raise RuntimeError("Incomplete evaluation")
        baseline_config = json.loads(Path(reference["config"]).read_text(encoding="utf-8"))
        ignored = {"train_steps", "output_dir", "load_checkpoint", "evaluation_split_role", "trace_root"}
        for key, value in baseline_config.items():
            if key not in ignored and actual_config.get(key) != value:
                raise RuntimeError(f"Nonintervention configuration changed: {key}")
        if cell["variant"] == "Full" and role == "test":
            for key in PRIMARY:
                if not math.isclose(float(row[key]), reference["historical_metrics"][key], rel_tol=0., abs_tol=1e-8):
                    raise RuntimeError(f"Historical Full replay differs: {key}")
    return summary, row, manifest


def run_cell(root, cell, role, design, reference, checkpoint=None, smoke=False):
    out = root / ("smoke" if smoke else role) / cell["variant"] / f"seed_{cell['seed']}" / f"delay_{cell['age']}"
    if not (out / "COMPLETE.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        cmd = command(out, cell, root, design, role, checkpoint, smoke)
        save(root / "STATUS.json", {"status": "running", "stage": "smoke" if smoke else role,
                                  "cell": cell, "training_started": bool(checkpoint is None),
                                  "at": datetime.now().isoformat()})
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        with (out / "console.log").open("a", encoding="utf-8") as handle:
            handle.write(subprocess.list2cmdline(cmd) + "\n")
            handle.flush()
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        summary, row, manifest = validate(out, role, cell, reference, design, smoke, checkpoint)
        save(out / "COMPLETE.json", {"cell": cell, "role": role, "summary": str(summary),
                                   "source_lock": design["source_lock"],
                                   "checkpoint_sha256": sha(summary.with_name("checkpoint_LCC-MAPPO.pt")),
                                   "at": datetime.now().isoformat()})
    return validate(out, role, cell, reference, design, smoke, checkpoint)


def analyze(out, rows):
    metrics = (*PRIMARY, *SECONDARY)
    with (out / "test_conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contrasts = {}
    for metric in metrics:
        seed_diffs = []
        for seed in SEEDS:
            get = lambda variant: [float(r[metric]) for r in rows if r["seed"] == seed and r["variant"] == variant]
            seed_diffs.append(sum(get(VARIANT)) / 5 - sum(get("Full")) / 5)
        bootstrap = sorted(sum(seed_diffs[i] for i in ix) / 5 for ix in itertools.product(range(5), repeat=5))
        contrasts[metric] = {"contrast": VARIANT + " minus Full", "mean": sum(seed_diffs)/5,
                             "ci95": [bootstrap[78], bootstrap[3046]], "seed_differences": seed_diffs}
    save(out / "paired_seed_block_contrasts.json", contrasts)
    lines = ["# 联合需求感知奖励消融结果", "", "此处仅报告真实完成的50个测试cell；正差值表示移除奖励塑形后该指标上升。", "",
             "| 指标 | NoJointDemandReward − Full | 配对95%区间 |", "| --- | ---: | --- |"]
    for name, result in contrasts.items():
        lo, hi = result["ci95"]
        lines.append(f"| {name} | {result['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "必须同时查看分延迟结果、奖励分布、H活跃率、丢弃与unfinished；负结果也保留。",
              "H不活跃时不能由此证明集中度反馈项的贡献。此实验没有取消共享目标或合作训练。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    design = prepare(out)
    if args.prepare_only:
        save(out / "STATUS.json", {"status": "prepared_not_started", "training_started": False})
        print(json.dumps({"design": str(out / "DESIGN.json"), "new_trainings": 25, "Full_replays": 25}))
        return
    preflight(out)
    try:
        subprocess.run([sys.executable, "-B", str(HERE / "test_joint_reward_ablation.py")], cwd=ROOT, check=True)
        for variant in ("Full", VARIANT):
            cell = {"variant": variant, "seed": 1, "age": 8}
            reference = next(r for r in design["Full_references"] if r["seed"] == 1 and r["age"] == 8)
            run_cell(out, cell, "validation", design, reference, smoke=True)
        frozen = []
        for reference in design["Full_references"]:
            cell = {"variant": VARIANT, "seed": reference["seed"], "age": reference["age"]}
            summary, _, _ = run_cell(out, cell, "validation", design, reference)
            checkpoint = summary.with_name("checkpoint_LCC-MAPPO.pt")
            frozen.append({**cell, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint)})
        freeze_path = out / "FROZEN_BEFORE_TEST.json"
        registry = {"source_lock": design["source_lock"], "checkpoints": frozen}
        if freeze_path.exists():
            prior = json.loads(freeze_path.read_text(encoding="utf-8"))
            if {k: prior[k] for k in registry} != registry:
                raise RuntimeError("Frozen checkpoint registry changed")
        if not freeze_path.exists():
            save(freeze_path, {**registry, "frozen_at": datetime.now().isoformat()})
        rows = []
        for reference in design["Full_references"]:
            for variant in ("Full", VARIANT):
                cell = {"variant": variant, "seed": reference["seed"], "age": reference["age"]}
                item = reference if variant == "Full" else next(r for r in frozen if r["seed"] == cell["seed"] and r["age"] == cell["age"])
                checkpoint = Path(item["checkpoint"])
                if sha(checkpoint) != item["checkpoint_sha256"]:
                    raise RuntimeError("Frozen checkpoint byte hash changed")
                _, row, manifest = run_cell(out, cell, "test", design, reference, checkpoint=checkpoint)
                rows.append({**cell, **{k: float(row[k]) for k in PRIMARY},
                             **{k: manifest[k] for k in SECONDARY},
                             **{k: float(row[k]) for k in ("n_generated_tasks", "n_accepted_tasks", "n_dropped_tasks", "n_unfinished_tasks")},
                             "h_active_slot_fraction": manifest["h_active_slot_fraction"],
                             "reward_mean": manifest["reward_mean"], "reward_std": manifest["reward_std"]})
        if len(rows) != 50:
            raise RuntimeError("Incomplete formal test grid")
        analyze(out, rows)
        save(out / "STATUS.json", {"status": "completed", "new_training_conditions": 25,
                                  "Full_replays": 25, "test_conditions": 50, "at": datetime.now().isoformat()})
    except Exception as exc:
        save(out / "STATUS.json", {"status": "failed", "error": str(exc), "at": datetime.now().isoformat()})
        raise


if __name__ == "__main__":
    main()
