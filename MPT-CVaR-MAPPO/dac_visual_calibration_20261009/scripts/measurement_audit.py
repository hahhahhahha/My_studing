"""Strict integrity checks on measured requests, separate from timed execution."""
from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path

from common import BASE, read, sha, session_blocks, validate_roles

EXPECTED_SESSIONS = {"pilot": 1, "calibration": 5, "validation": 3, "test": 5}
BOUNDARY = "decode+RGB+preprocess+FP32 inference+argmax+dict; no disk/network/queue/controller"
STAGES = ("decode_ms", "preprocess_ms", "infer_ms", "postprocess_ms")


def audit_role(role, sessions=None, require_complete=True):
    splits = read(BASE / "protocol/IMAGE_SPLITS.json")
    validate_roles(splits)
    freeze = read(BASE / "protocol/IMAGE_FREEZE.json")
    if sha(BASE / "protocol/IMAGE_SPLITS.json") != freeze["split_sha256"]:
        raise RuntimeError("Image split fingerprint changed")
    versions = read(BASE / "protocol/ENVIRONMENT_SPEC.json")["versions"]
    weights_hash = sha(BASE / "models/resnet18-f37072fd.pth")
    topology = read(BASE / "protocol/CPU_TOPOLOGY.json")
    entries = {e["image_id"]: e for e in splits[role]}
    sessions = list(range(1, EXPECTED_SESSIONS[role] + 1)) if sessions is None else list(sessions)
    fingerprints, rows_out, intervals, jobs_expected = {}, [], [], set()
    for session in sessions:
        for job in session_blocks(splits[role], role, session):
            stem = f'{role}_s{session}_b{job["block"]}_{job["profile"]}'
            path = BASE / "measurements" / role / f"{stem}.csv"
            jobs_expected.add(path.name)
            metadata_path = path.with_suffix(".json")
            metadata = read(metadata_path)
            threads = 1 if job["profile"] == "CPU-1T" else 4
            affinity = topology["logical_cpu_per_physical_core"][:threads]
            job_path = BASE / "protocol" / f"job_{stem}.json"
            if (metadata["job"] != job or read(job_path) != job
                    or metadata["job_sha256"] != sha(job_path)
                    or metadata["versions"] != versions
                    or metadata["weights_sha256"] != weights_hash
                    or metadata["affinity"] != affinity
                    or metadata["warmup_requests"] != 200
                    or metadata["errors"] != 0
                    or metadata["timing_boundary"] != BOUNDARY):
                raise RuntimeError(f"Metadata integrity failure: {stem}")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if ([r["image_id"] for r in rows] != job["image_ids"]
                    or len(rows) != metadata["requests"]):
                raise RuntimeError(f"Missing/reordered/extra requests: {stem}")
            before = datetime.fromisoformat(metadata["sensor_before"]["wall_clock"])
            after = datetime.fromisoformat(metadata["sensor_after"]["wall_clock"])
            last = before
            for order, row in enumerate(rows, 1):
                entry = entries[row["image_id"]]
                if (row["profile_id"] != job["profile"] or row["device"] != "cpu"
                        or int(row["session_id"]) != session
                        or int(row["block_id"]) != job["block"]
                        or int(row["run_order"]) != order
                        or int(row["cpu_threads"]) != threads
                        or __import__("json").loads(row["cpu_affinity"]) != affinity
                        or row["model_id"] != "ResNet18_IMAGENET1K_V1"
                        or row["weights_sha256"] != weights_hash
                        or row["image_sha256"] != entry["image_sha256"]
                        or row["image_split"] != role or row["class_id"] != entry["class_id"]
                        or any(int(row[key]) != entry[key] for key in ("payload_bytes", "width", "height"))
                        or row["error_status"] != "ok"
                        or not 0 <= int(row["predicted_class"]) < 1000):
                    raise RuntimeError(f"Request identity/profile failure: {stem}/{order}")
                values = [float(row[key]) for key in STAGES]
                total = float(row["service_ms"])
                if (not all(math.isfinite(v) and v >= 0 for v in values)
                        or not math.isfinite(total) or total <= 0
                        or not math.isclose(sum(values), total, rel_tol=1e-10, abs_tol=1e-8)):
                    raise RuntimeError(f"Invalid/inconsistent service timing: {stem}/{order}")
                timestamp = datetime.fromisoformat(row["measurement_timestamp"])
                if not before <= timestamp <= after or timestamp < last:
                    raise RuntimeError(f"Invalid timestamp order: {stem}/{order}")
                last = timestamp
            rows_out.extend(rows)
            intervals.append((before, after, stem))
            for artifact in (path, metadata_path):
                fingerprints[str(artifact.relative_to(BASE))] = sha(artifact)
    if require_complete:
        observed = {p.name for p in (BASE / "measurements" / role).glob("*.csv")}
        if observed != jobs_expected:
            raise RuntimeError(f"Unexpected or missing CSV files in {role}")
        observed_meta = {p.name for p in (BASE / "measurements" / role).glob("*.json")}
        if observed_meta != {Path(name).with_suffix(".json").name for name in jobs_expected}:
            raise RuntimeError(f"Unexpected or missing metadata files in {role}")
    intervals.sort()
    for previous, following in zip(intervals, intervals[1:]):
        if previous[1] > following[0]:
            raise RuntimeError(f"Concurrent timing blocks: {previous[2]}/{following[2]}")
    paired = {}
    for row in rows_out:
        key = (int(row["session_id"]), row["image_id"])
        paired.setdefault(key, {})[row["profile_id"]] = row
    if any(set(pair) != {"CPU-1T", "CPU-4T"} for pair in paired.values()):
        raise RuntimeError("Unpaired profile measurements")
    disagree = sum(pair["CPU-1T"]["predicted_class"] != pair["CPU-4T"]["predicted_class"] for pair in paired.values())
    report = {"passed": True, "role": role, "sessions": sessions,
              "rows": len(rows_out), "blocks": len(jobs_expected), "failed_requests": 0,
              "paired_prediction_disagreements": disagree,
              "measurement_sha256": fingerprints,
              "first_block_start": intervals[0][0].isoformat(),
              "last_block_end": intervals[-1][1].isoformat()}
    return report, rows_out


def audit_all():
    reports, rows = {}, {}
    for role in EXPECTED_SESSIONS:
        reports[role], rows[role] = audit_role(role)
    return reports, rows
