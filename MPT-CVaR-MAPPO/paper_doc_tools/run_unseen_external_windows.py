"""Prepare and launch a frozen R6 external stress test; never train."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from heldout_external_windows import ROOT, OUT, WINDOWS, prepare_azure, prepare_alibaba2017, save, sha

OLD = ROOT / "results_qla_formal_trace_bothfix_20261001_v5"
METHODS = ("QLA-MAPPO", "PPO-MV-A", "JSEQ-A")
DATASETS = ("KAIST", "NCSU", "NewYork")
SEEDS = (1, 2, 3)
DELAYS = (0, 4, 8)


def package_sha(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((OLD / "FROZEN_BEFORE_TEST.json").read_text(encoding="utf-8"))
    checkpoints = []
    for entry in old["checkpoints"]:
        if entry["method"] not in METHODS[:2] or entry["age"] != 4:
            continue
        path = Path(entry["checkpoint"])
        if sha(path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {path}")
        configs = list(path.parent.glob("config_*.json"))
        if len(configs) != 1:
            raise RuntimeError(f"Ambiguous checkpoint configuration: {path}")
        checkpoints.append({**entry, "config": str(configs[0]), "config_sha256": sha(configs[0])})
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 delay-4 checkpoint entries, found {len(checkpoints)}")
    frozen_root = out / "execution_source"
    package = frozen_root / "mappo_gp_sf_dsr"
    package.mkdir(parents=True, exist_ok=True)
    # Restore each package filename from the original flat R6 source snapshot.
    for current in sorted((ROOT / "mappo_gp_sf_dsr").glob("*.py")):
        archived = OLD / "source_snapshot" / current.name
        if not archived.is_file():
            raise FileNotFoundError(f"R6 source file missing: {archived}")
        destination = package / current.name
        if destination.exists() and sha(destination) != sha(archived):
            raise RuntimeError(f"Execution snapshot changed: {destination}")
        shutil.copy2(archived, destination)
    if package_sha(package) != old["code_lock"]["package_source_sha256"]:
        raise RuntimeError("Restored R6 package does not match frozen source")
    for dependency in ("repro_risk_averse_edge", "repro_risk_averse"):
        origin = ROOT / dependency
        if origin.is_dir():
            target = frozen_root / dependency
            target.mkdir(exist_ok=True)
            for file in origin.glob("*.py"):
                destination = target / file.name
                if destination.exists() and sha(destination) != sha(file):
                    raise RuntimeError(f"Dependency snapshot changed: {destination}")
                shutil.copy2(file, destination)
    helpers = frozen_root / "paper_doc_tools"
    helpers.mkdir(exist_ok=True)
    for name in ("run_real_trace_external_formal.py", "run_cross_method_sync_diagnostic.py"):
        origin = ROOT / "paper_doc_tools" / name
        destination = helpers / name
        if destination.exists() and sha(destination) != sha(origin):
            raise RuntimeError(f"Evaluation helper changed: {destination}")
        shutil.copy2(origin, destination)
    code = {str(p.relative_to(frozen_root)): sha(p) for p in sorted(frozen_root.rglob("*.py"))}
    cells = [{"method": m, "dataset": d, "seed": s, "nominal_delay": a, "window": w[0]}
             for m, d, s, a, w in itertools.product(METHODS, DATASETS, SEEDS, DELAYS, WINDOWS)]
    random.Random(20261009).shuffle(cells)
    design = {"study": "Frozen R6 policies on independent Azure days and Alibaba release",
              "created": datetime.now().isoformat(), "train_steps": 0, "test_steps": 1000,
              "checkpoint_training_nominal_delay": 4, "checkpoint_rule": "pre-existing frozen last checkpoints; no new selection",
              "methods": METHODS, "mobility_datasets": DATASETS, "training_seeds": SEEDS,
              "test_nominal_delays": DELAYS, "windows": WINDOWS, "cells": cells,
              "expected_cells": 243, "checkpoints": checkpoints, "execution_source_sha256": code,
              "runner_sha256": sha(__file__),
              "data_code_sha256": sha(ROOT / "paper_doc_tools/heldout_external_windows.py"),
              "worker_sha256": sha(ROOT / "paper_doc_tools/unseen_external_window_worker.py"),
              "source_registry_sha256": sha(OLD / "FROZEN_BEFORE_TEST.json"),
              "required_alibaba_file": str(ROOT / "datasets/external_traces/extracted/alibaba2017/server_usage.csv"),
              "scope": "external trace-shaped stress test; schema mapping changes from legacy R6, not a time-only causal contrast",
              "jseq_temperature": .25, "primary_metrics": ["average_latency", "cvar99"],
              "secondary_metrics": ["queue_oscillation", "portfolio_migration_rate", "max_realized_share_p99", "mean_queue_jump_service_slots", "p_drop", "unfinished_tasks"]}
    target = out / "DESIGN.json"
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
        if any(prior[key] != design[key] for key in design if key != "created"):
            raise RuntimeError("Frozen design changed; use a new output directory")
        design = prior
    else:
        save(target, design)
    save(out / "FROZEN_BEFORE_TEST.json", {"source_registry": str(OLD / "FROZEN_BEFORE_TEST.json"),
                                          "source_registry_sha256": design["source_registry_sha256"],
                                          "checkpoints": checkpoints, "trained_at_nominal_delay": 4})
    save(out / "ENVIRONMENT_SPEC.json", {"python": "3.13.5", "numpy": "2.2.6",
                                         "torch": "2.11.0+cu128", "device": "cuda",
                                         "cpu_threads": 1, "cuda_witness_seed": 123,
                                         "automatic_cpu_fallback": False,
                                         "policy_update_permitted": False})
    return design


def preflight(out):
    path = ROOT / "paper_doc_tools/unseen_external_window_worker.py"
    with (out / "preflight.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-B", str(path), "--output", str(out), "--preflight"],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--prepare-data", action="store_true")
    p.add_argument("--run", action="store_true")
    opts = p.parse_args()
    design = prepare(opts.output)
    if opts.prepare_data:
        prepare_azure(opts.output)
    ready = preflight(opts.output)
    missing = not Path(design["required_alibaba_file"]).is_file()
    status = {"at": datetime.now().isoformat(), "model_tests_started": False, "training_started": False,
              "environment_ready": ready, "alibaba2017_missing": missing,
              "status": "blocked_preflight_or_data" if not ready or missing else "prepared_not_run",
              "completed_model_cells": 0, "expected_model_cells": 243}
    save(opts.output / "STATUS.json", status)
    if not opts.run or not ready or missing:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    prepare_azure(opts.output)
    prepare_alibaba2017(design["required_alibaba_file"], opts.output)
    # One pair of model smoke cells first. They use the same preregistered W1
    # window; successful complete cells are reusable, not selected by outcome.
    first = [next(c for c in design["cells"] if c["method"] == m and c["dataset"] == "KAIST"
                  and c["seed"] == 1 and c["nominal_delay"] == 4 and c["window"] == "W1") for m in METHODS]
    ordered = first + [c for c in design["cells"] if c not in first]
    status["status"], status["model_tests_started"] = "running", True
    save(opts.output / "STATUS.json", status)
    for index, cell in enumerate(ordered):
        key = f'{cell["window"]}/{cell["dataset"]}/seed_{cell["seed"]}/delay_{cell["nominal_delay"]}/{cell["method"]}'
        cell_out = opts.output / "cells" / key
        cell_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", str(ROOT / "paper_doc_tools/unseen_external_window_worker.py"),
               "--output", str(opts.output), "--cell-json", json.dumps(cell)]
        with (cell_out / "execution.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode:
            status.update(status="failed_cell", failed_cell=cell)
            save(opts.output / "STATUS.json", status)
            raise SystemExit(f"Failed {key}; preserved log, no fallback")
        status["completed_model_cells"] = index + 1
        save(opts.output / "STATUS.json", status)
    status["status"] = "evaluated_pending_independent_audit_and_analysis"
    save(opts.output / "STATUS.json", status)


if __name__ == "__main__":
    main()
