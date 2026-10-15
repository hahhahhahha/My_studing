"""Sequential same-day sessions with frozen rules, checkpoints and honest status."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

from common import BASE, read, save, sha, session_blocks
from measurement_audit import audit_role, EXPECTED_SESSIONS
from pipeline import image_manifest, require_pilot, count_measurement_rows

PLAN = BASE / "protocol/FORMAL_EXECUTION_PLAN.json"
STATE = BASE / "protocol/FORMAL_EXECUTION_STATE.json"
LOCK = BASE / "protocol/FORMAL_EXECUTION.lock"
SOURCES = ("common.py", "windows_runtime.py", "pipeline.py", "measure_block.py",
           "measurement_audit.py", "service_summary.py", "formal_runner.py")


def initialize():
    if PLAN.exists():
        raise FileExistsError("Execution plan already exists; do not silently refreeze")
    for role in ("validation", "test"):
        if list((BASE / "measurements" / role).glob("*.csv")):
            raise RuntimeError("Must freeze distribution criteria before validation/test")
    splits = image_manifest()
    require_pilot(splits)
    first, _ = audit_role("calibration", [1], require_complete=True)
    sequence = [{"role": role, "session": session}
                for role in ("calibration", "validation", "test")
                for session in range(1, EXPECTED_SESSIONS[role] + 1)]
    plan = {"schema": "same_day_cpu_service_measurements_v1",
            "created": datetime.now().isoformat(), "date": datetime.now().date().isoformat(),
            "sequence": sequence, "minimum_session_start_gap_seconds": 900,
            "session_start_definition": "first timing block sensor_before.wall_clock, before warmup",
            "idle_check": {"seconds": 5, "max_machine_CPU_percent": 35,
                           "max_selected_core_CPU_percent": 60,
                           "min_available_memory_bytes": 2 * 1024**3,
                           "max_wait_seconds": 3600},
            "same_machine_same_day": True, "original_plan_cross_day_coverage_not_claimed": True,
            "temperature_measurements_available": False, "continuous_frequency_sampling": False,
            "calibration_session_1_adopted_after_complete_audit": first,
            "environment_spec_sha256": sha(BASE / "protocol/ENVIRONMENT_SPEC.json"),
            "image_split_sha256": sha(BASE / "protocol/IMAGE_SPLITS.json"),
            "distribution_rule_sha256": sha(BASE / "protocol/SERVICE_DISTRIBUTION_VALIDATION_RULE.json"),
            "source_sha256": {f"scripts/{name}": sha(BASE / "scripts" / name) for name in SOURCES},
            "expected_formal_rows": 33000, "pilot_rows_excluded": 200,
            "test_gate": "all calibration/validation sessions audited; frozen service-distribution tolerances accepted; independent source/protocol review passed",
            "out_of_scope": ["RL policy training", "simulator integration", "paper changes", "multi-device deployment"]}
    save(PLAN, plan)
    save(STATE, {"status": "prepared_pending_independent_review", "completed_sessions": [],
                 "datetime": datetime.now().isoformat()})
    print("FORMAL_PLAN_FROZEN")


def check(require_review=True):
    plan = read(PLAN)
    expected_sequence = [{"role": role, "session": session}
                         for role in ("calibration", "validation", "test")
                         for session in range(1, EXPECTED_SESSIONS[role] + 1)]
    expected_idle = {"seconds": 5, "max_machine_CPU_percent": 35,
                     "max_selected_core_CPU_percent": 60,
                     "min_available_memory_bytes": 2 * 1024**3, "max_wait_seconds": 3600}
    if (plan.get("schema") != "same_day_cpu_service_measurements_v1"
            or plan.get("sequence") != expected_sequence
            or plan.get("minimum_session_start_gap_seconds") != 900
            or plan.get("idle_check") != expected_idle
            or plan.get("expected_formal_rows") != 33000
            or plan.get("pilot_rows_excluded") != 200
            or plan.get("same_machine_same_day") is not True):
        raise RuntimeError("Execution plan does not match preregistered session semantics")
    datetime.strptime(plan["date"], "%Y-%m-%d")
    if plan["environment_spec_sha256"] != sha(BASE / "protocol/ENVIRONMENT_SPEC.json") or plan["image_split_sha256"] != sha(BASE / "protocol/IMAGE_SPLITS.json"):
        raise RuntimeError("Measurement environment/image freeze changed")
    if plan["distribution_rule_sha256"] != sha(BASE / "protocol/SERVICE_DISTRIBUTION_VALIDATION_RULE.json"):
        raise RuntimeError("Preregistered service distribution rule changed")
    for relative, digest in plan["source_sha256"].items():
        if sha(BASE / relative) != digest:
            raise RuntimeError(f"Frozen measurement source changed: {relative}")
    require_pilot(image_manifest())
    if require_review:
        review = read(BASE / "protocol/FORMAL_RUNNER_REVIEW.json")
        if (review.get("passed") is not True or review["source_sha256"] != plan["source_sha256"]
                or review.get("execution_plan_sha256") != sha(PLAN)):
            raise RuntimeError("Independent runner/source review absent or stale")
    print("FROZEN_SOURCES_AND_PILOT_OK")
    return plan


def idle_wait(plan, role, session):
    import psutil
    rule = plan["idle_check"]
    deadline = time.monotonic() + rule["max_wait_seconds"]
    cpus = read(BASE / "protocol/CPU_TOPOLOGY.json")["logical_cpu_per_physical_core"][:4]
    probes = []
    while True:
        per_cpu = psutil.cpu_percent(interval=rule["seconds"], percpu=True)
        sample = {"datetime": datetime.now().isoformat(), "per_logical_CPU_percent": per_cpu,
                  "machine_CPU_percent": sum(per_cpu)/len(per_cpu),
                  "available_memory_bytes": psutil.virtual_memory().available}
        probes.append(sample)
        path = BASE / "protocol" / f"idle_{role}_s{session}.json"
        save(path, {"selected_logical_cpus": cpus, "rule": rule, "probes": probes})
        if (sample["machine_CPU_percent"] <= rule["max_machine_CPU_percent"]
                and max(per_cpu[cpu] for cpu in cpus) <= rule["max_selected_core_CPU_percent"]
                and sample["available_memory_bytes"] >= rule["min_available_memory_bytes"]):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("No sufficiently idle window in one hour; preserved status and stopped")
        time.sleep(30)


def child(command, logfile):
    with (BASE / "logs" / logfile).open("x", encoding="utf-8") as output:
        result = subprocess.run([sys.executable, "-B", str(BASE / "scripts" / command[0]), *command[1:]],
                                cwd=BASE, stdout=output, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode:
        raise RuntimeError(f"Child failed ({result.returncode}); see logs/{logfile}")


def require_new_session(role, session):
    """The wrapper rejects orphan artifacts before pipeline can write any file."""
    splits = image_manifest()
    for job in session_blocks(splits[role], role, session):
        stem = f'{role}_s{session}_b{job["block"]}_{job["profile"]}'
        csv_path = BASE / "measurements" / role / f"{stem}.csv"
        for path in (csv_path, csv_path.with_suffix(".json"),
                     BASE / "protocol" / f"job_{stem}.json", BASE / "logs" / f"{stem}.log"):
            if path.exists():
                raise FileExistsError(f"Existing/orphan session artifact requires explicit audit: {path}")
    session_log = BASE / "logs" / f"session_{role}_{session}.log"
    if session_log.exists():
        raise FileExistsError(f"Existing session log requires explicit audit: {session_log}")


def update_status(status, **extra):
    original = read(BASE / "STATUS.json")
    counts = count_measurement_rows()
    original.update({"status": status, "datetime": datetime.now().isoformat(),
                     "measurement_rows_by_role": counts,
                     "formal_measurement_rows": sum(n for role, n in counts.items() if role != "pilot"), **extra})
    save(BASE / "STATUS.json", original)


def run():
    plan = check()
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.close(descriptor)
    state = read(STATE)
    if state["completed_sessions"]:
        LOCK.unlink()
        raise RuntimeError("Automatic rerun/resume is disabled; audit existing sessions explicitly")
    state.update(status="running", worker_pid=os.getpid(), datetime=datetime.now().isoformat(),
                 execution_plan_sha256=sha(PLAN))
    save(STATE, state)
    try:
        previous_start = None
        for task in plan["sequence"]:
            role, session = task["role"], task["session"]
            check()
            if role == "calibration" and session == 1:
                report, _ = audit_role(role, [session], require_complete=False)
                if report != plan["calibration_session_1_adopted_after_complete_audit"]:
                    raise RuntimeError("Adopted first calibration session changed")
            else:
                if previous_start is not None:
                    earliest = previous_start + timedelta(seconds=plan["minimum_session_start_gap_seconds"])
                    state.update(status="waiting_for_spaced_session", next_session=task,
                                 earliest_start=earliest.isoformat(), datetime=datetime.now().isoformat())
                    save(STATE, state)
                    update_status("formal_measurement_waiting", next_session=task,
                                  next_earliest_start=earliest.isoformat())
                    while datetime.now() < earliest:
                        time.sleep(min(30, max(.1, (earliest-datetime.now()).total_seconds())))
                if datetime.now().date().isoformat() != plan["date"]:
                    raise RuntimeError("Same-day plan crossed midnight; stop rather than mislabel sessions")
                idle_wait(plan, role, session)
                if datetime.now().date().isoformat() != plan["date"]:
                    raise RuntimeError("Idle wait crossed midnight; stop before submitting measurement")
                require_new_session(role, session)
                if role == "test" and session == 1:
                    child(["service_summary.py", "validate"], "service_distribution_validation.log")
                    state["service_distribution_accepted"] = True
                    save(STATE, state)
                state.update(status="measuring", current_session=task, datetime=datetime.now().isoformat())
                save(STATE, state)
                child(["pipeline.py", role, "--session", str(session)], f"session_{role}_{session}.log")
                report, _ = audit_role(role, [session], require_complete=False)
            start = datetime.fromisoformat(report["first_block_start"])
            end = datetime.fromisoformat(report["last_block_end"])
            if start.date().isoformat() != plan["date"] or end.date().isoformat() != plan["date"]:
                raise RuntimeError("Measured session crossed planned date; retained raw records and stopped")
            if previous_start is not None and (start - previous_start).total_seconds() < plan["minimum_session_start_gap_seconds"]:
                raise RuntimeError("Session spacing violated")
            previous_start = start
            state["completed_sessions"].append({**task, "audit": report})
            state.update(status="session_audited", datetime=datetime.now().isoformat())
            save(STATE, state)
            print(f"AUDITED {role} session {session}: {report['rows']} rows", flush=True)
        child(["service_summary.py", "summarize"], "formal_service_summary.log")
        audit = read(BASE / "analysis/MEASUREMENT_AUDIT.json")
        if audit["formal_rows"] != plan["expected_formal_rows"] or not audit["passed"]:
            raise RuntimeError("Formal row count/integrity audit failed")
        state.update(status="formal_measurement_complete_audited", datetime=datetime.now().isoformat())
        save(STATE, state)
        update_status("formal_measurement_complete_audited", service_measurements_complete=True,
                      remaining=["simulator integration and queue-execution acceptance", "calibrated policy comparisons"])
        print("FORMAL_SERVICE_MEASUREMENTS_COMPLETE: 33000 audited rows", flush=True)
    except Exception as error:
        state.update(status="stopped_with_error", error=repr(error), datetime=datetime.now().isoformat())
        save(STATE, state)
        update_status("formal_measurement_stopped_with_error", error=repr(error))
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "check", "run", "status"))
    parser.add_argument("--pre-review", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
    elif args.command == "check":
        check(require_review=not args.pre_review)
    elif args.command == "run":
        run()
    else:
        print(__import__("json").dumps(read(STATE), indent=2, ensure_ascii=False))
