"""Download, validate and measure locally; no changes to global packages."""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tarfile

from common import (BASE, setup, save, read, sha, csv_write, safe_member,
                    split_images, validate_roles, session_blocks, dhash, cross_role_near_duplicates)

SOURCES = {
    "imagenette": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", BASE / "data/raw/imagenette2.tgz"),
    "resnet18": ("https://download.pytorch.org/models/resnet18-f37072fd.pth", BASE / "models/resnet18-f37072fd.pth"),
}


def environment_spec():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "torchvision", "Pillow", "psutil"):
        versions[package] = importlib.metadata.version(package)
    spec = {"versions": versions, "device": "cpu", "model": "ResNet18_IMAGENET1K_V1",
            "profiles": {"CPU-1T": {"intraop_threads": 1, "interop_threads": 1, "physical_cores": 1},
                         "CPU-4T": {"intraop_threads": 4, "interop_threads": 1, "physical_cores": 4}},
            "batch_size": 1, "precision": "float32", "warmup_per_process": 200,
            "witness_seed": 20261009, "automatic_package_install": False}
    target = BASE / "protocol/ENVIRONMENT_SPEC.json"
    if target.exists() and read(target) != spec:
        raise RuntimeError("Existing frozen measurement environment differs; don't silently overwrite")
    save(target, spec)
    return spec


def topology():
    import psutil
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("This documented topology reader requires Windows x64")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    func = kernel.GetLogicalProcessorInformationEx
    func.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    func.restype = ctypes.c_int
    length = ctypes.c_ulong()
    func(0, None, ctypes.byref(length))  # RelationProcessorCore
    if not length.value:
        raise OSError(ctypes.get_last_error(), "CPU topology query returned no data")
    buffer = ctypes.create_string_buffer(length.value)
    if not func(0, buffer, ctypes.byref(length)):
        raise OSError(ctypes.get_last_error(), "CPU topology query failed")
    raw, offset, cores = buffer.raw[:length.value], 0, []
    allowed = set(psutil.Process().cpu_affinity())
    while offset < len(raw):
        relation, size = struct.unpack_from("<II", raw, offset)
        if size < 32 or offset + size > len(raw):
            raise ValueError("Malformed Windows CPU topology record")
        if relation == 0:
            groups = struct.unpack_from("<H", raw, offset + 30)[0]
            logical = []
            for group_index in range(groups):
                pos = offset + 32 + group_index * 16
                mask, group = struct.unpack_from("<QH", raw, pos)
                if group != 0:
                    continue
                logical.extend(bit for bit in range(64) if mask & (1 << bit) and bit in allowed)
            if logical:
                cores.append(sorted(logical))
        offset += size
    cores.sort(key=lambda x: x[0])
    if len(cores) < 4:
        raise RuntimeError(f"Only {len(cores)} verified available physical cores")
    value = {"physical_core_logical_cpus": cores,
             "logical_cpu_per_physical_core": [core[0] for core in cores],
             "source": "Windows GetLogicalProcessorInformationEx RelationProcessorCore",
             "process_allowed_affinity": sorted(allowed)}
    target = BASE / "protocol/CPU_TOPOLOGY.json"
    if target.exists() and read(target) != value:
        raise RuntimeError("CPU topology/availability changed; don't mix measurement profiles")
    save(target, value)
    return value


def check():
    # Real CPU kernel, not just import or installed-package metadata.
    import torch
    import torchvision
    import numpy as np
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261009)
    a = torch.randn((64, 64), dtype=torch.float32)
    actual = a @ torch.eye(64)
    if not torch.allclose(a, actual) or not torch.isfinite(actual).all():
        raise RuntimeError("CPU kernel witness failed")
    save(BASE / "protocol/CPU_WITNESS.json", {"passed": True, "torch": torch.__version__,
         "torchvision": torchvision.__version__, "seed": 20261009, "tensor_sha256": __import__("hashlib").sha256(actual.numpy().tobytes()).hexdigest(),
         "numpy": np.__version__, "environment_spec_sha256": sha(BASE / "protocol/ENVIRONMENT_SPEC.json"),
         "datetime": datetime.now().isoformat()})
    print("CPU_KERNEL_WITNESS_OK")


def probe_environment():
    with (BASE / "logs/environment_preflight.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "check"],
                                stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return result.returncode == 0


def validate_asset(name, path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing/empty file")
    digest = sha(path)
    if name == "resnet18" and not digest.startswith("f37072fd"):
        raise ValueError("Weights do not match official SHA256 filename prefix")
    if name == "imagenette":
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                safe_member(member.name.rstrip("/"))
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive contains links or other special entries")
            if not any(m.isfile() and "/train/" in m.name for m in members) or not any(m.isfile() and "/val/" in m.name for m in members):
                raise ValueError("Archive has no expected train/val data")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def download_all():
    statuses = {}
    for name, (url, target) in SOURCES.items():
        if target.exists():
            try:
                statuses[name] = {"status": "verified_existing", "url": url, **validate_asset(name, target)}
            except Exception as error:
                statuses[name] = {"status": "invalid_existing", "url": url, "error": repr(error)}
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        command = ["C:/Windows/System32/curl.exe", "--fail", "--location", "--silent", "--show-error",
                   "--connect-timeout", "15", "--max-time", "900", "--output", str(partial), url]
        with (BASE / "logs" / f"download_{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            statuses[name] = {"status": "download_failed", "url": url, "returncode": result.returncode,
                              "partial_bytes": partial.stat().st_size if partial.exists() else 0}
            continue
        try:
            verified = validate_asset(name, partial)
            partial.replace(target)
            statuses[name] = {"status": "downloaded_verified", "url": url, **verified, "path": str(target)}
        except Exception as error:
            statuses[name] = {"status": "validation_failed", "url": url, "error": repr(error)}
    save(BASE / "protocol/DOWNLOAD_STATUS.json", {"datetime": datetime.now().isoformat(), "assets": statuses})
    return statuses


def prepare():
    setup()
    if shutil.disk_usage(BASE).free < 6 * 1024**3:
        raise RuntimeError("Less than 6 GiB free; stop before downloading original-size images")
    spec = environment_spec()
    try:
        topology()
        topology_ok = True
    except Exception as error:
        save(BASE / "protocol/CPU_TOPOLOGY_FAILURE.json", {"error": repr(error)})
        topology_ok = False
    assets = download_all()
    environment_ok = probe_environment()
    ready = environment_ok and topology_ok and all(a["status"] in ("verified_existing", "downloaded_verified") for a in assets.values())
    prior = read(BASE / "STATUS.json") if (BASE / "STATUS.json").exists() else {}
    observed = count_measurement_rows()
    started = any(observed.values())
    state = {**prior, "datetime": datetime.now().isoformat(),
             "status": (prior.get("status", "prior_measurements_pending_audit") if started and ready
                        else "existing_measurements_preflight_failed" if started
                        else "prepared_pending_pilot" if ready else "blocked_download_or_environment"),
             "downloads": {k: v["status"] for k, v in assets.items()}, "cpu_kernel_ready": environment_ok,
             "topology_verified": topology_ok, "service_measurements_started": started,
             "formal_measurement_rows": sum(n for role, n in observed.items() if role != "pilot"),
             "measurement_rows_by_role": observed, "global_environment_changed": False,
             "remaining": ["pilot", "calibration/validation measurement", "calibration integration and independent acceptance", "test measurement", "calibrated policy comparisons"]}
    save(BASE / "STATUS.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def image_manifest():
    from PIL import Image
    target = BASE / "protocol/IMAGE_SPLITS.json"
    if target.exists():
        frozen = read(BASE / "protocol/IMAGE_FREEZE.json")
        if sha(target) != frozen["split_sha256"]:
            raise RuntimeError("Frozen image split changed")
        roles = read(target)
        if validate_roles(roles) != frozen["role_counts"]:
            raise RuntimeError("Frozen image counts changed")
        return roles
    archive_path = SOURCES["imagenette"][1]
    validate_asset("imagenette", archive_path)
    extracted = BASE / "data/extracted"
    if not (extracted / "imagenette2").exists():
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extracted, filter="data")
    rows, invalid = [], []
    for path in sorted((extracted / "imagenette2").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        parts = path.relative_to(extracted).parts
        if len(parts) != 4 or parts[1] not in ("train", "val"):
            continue
        try:
            with Image.open(path) as source:
                width, height = source.size
                visual_hash = dhash(source)
                source.load()
            rows.append({"image_id": path.relative_to(extracted).as_posix(), "official_split": parts[1],
                         "class_id": parts[2], "image_sha256": sha(path), "payload_bytes": path.stat().st_size,
                         "width": width, "height": height, "dhash64": f"{visual_hash:016x}"})
        except Exception as error:
            invalid.append({"path": str(path), "error": repr(error)})
    splits, duplicates = split_images(rows)
    near = cross_role_near_duplicates(splits)
    save(BASE / "protocol/IMAGE_AUDIT.json", {"valid_images": len(rows), "invalid_images": invalid,
         "sha256_duplicates_excluded": duplicates, "near_duplicate_method": "64-bit dHash, Hamming<=2, selected roles",
         "cross_role_near_duplicates": near, "archive_sha256": sha(archive_path)})
    if near:
        raise RuntimeError("Cross-role near-duplicate candidates require audit before freezing; no outcome-based resampling")
    save(target, splits)
    csv_write(BASE / "protocol/IMAGE_MANIFEST.csv", rows)
    save(BASE / "protocol/IMAGE_FREEZE.json", {"seed": 20261009, "split_sha256": sha(target),
         "role_counts": {k: len(v) for k, v in splits.items()}, "frozen_before_service_measurement": True})
    return splits


def count_measurement_rows():
    counts = {}
    for role in ("pilot", "calibration", "validation", "test"):
        count = 0
        for path in (BASE / "measurements" / role).glob("*.csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                count += sum(1 for _ in csv.DictReader(handle))
        counts[role] = count
    return counts


def audited_role_files(role, sessions, splits):
    fingerprints = {}
    for session in sessions:
        for job in session_blocks(splits[role], role, session):
            stem = f'{role}_s{session}_b{job["block"]}_{job["profile"]}'
            path = BASE / "measurements" / role / f"{stem}.csv"
            metadata = read(path.with_suffix(".json"))
            if metadata["job"] != job or metadata["errors"] != 0 or metadata["warmup_requests"] < 200:
                raise RuntimeError("Measurement metadata incomplete or wrong job")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if [r["image_id"] for r in rows] != job["image_ids"]:
                raise RuntimeError("Measured image list/order differs from the frozen job")
            for row in rows:
                if row["error_status"] != "ok" or not math.isfinite(float(row["service_ms"])) or float(row["service_ms"]) <= 0:
                    raise RuntimeError("Failed/nonpositive/nonfinite service sample")
                if row["profile_id"] != job["profile"] or row["device"] != "cpu":
                    raise RuntimeError("Measurement profile/device changed")
            for artifact in (path, path.with_suffix(".json")):
                fingerprints[str(artifact.relative_to(BASE))] = sha(artifact)
    return fingerprints


def measurement_source_lock():
    return {f"scripts/{name}": sha(BASE / "scripts" / name)
            for name in ("common.py", "pipeline.py", "measure_block.py")}


def require_pilot(splits):
    gate = read(BASE / "protocol/PILOT_GATE.json")
    if not gate.get("passed"):
        raise RuntimeError("Successful pretrained-model pilot required")
    if gate["environment_spec_sha256"] != sha(BASE / "protocol/ENVIRONMENT_SPEC.json") or gate["image_split_sha256"] != sha(BASE / "protocol/IMAGE_SPLITS.json"):
        raise RuntimeError("Pilot does not match current environment/images")
    if gate["source_sha256"] != measurement_source_lock() or gate["measurement_sha256"] != audited_role_files("pilot", [1], splits):
        raise RuntimeError("Pilot source or measurement artifacts changed")


def require_calibration_freeze(splits):
    freeze = read(BASE / "calibration/CALIBRATION_FREEZE.json")
    if freeze.get("accepted") is not True or freeze.get("schema") != "visual_calibration_acceptance_v1":
        raise RuntimeError("Missing valid independent calibration acceptance")
    if freeze.get("environment_spec_sha256") != sha(BASE / "protocol/ENVIRONMENT_SPEC.json") or freeze.get("image_split_sha256") != sha(BASE / "protocol/IMAGE_SPLITS.json"):
        raise RuntimeError("Calibration freeze does not match current environment/images")
    expected = {**audited_role_files("calibration", range(1, 6), splits),
                **audited_role_files("validation", range(1, 4), splits)}
    if freeze.get("measurement_sha256") != expected:
        raise RuntimeError("Calibration acceptance is not bound to all required measured sessions")
    report = BASE / "calibration/VALIDATION_REPORT.json"
    if freeze.get("validation_report_sha256") != sha(report) or read(report).get("accepted") is not True:
        raise RuntimeError("Independent calibration validation report missing or changed")


def run_measurements(role, session):
    setup()
    if not (BASE / "STATUS.json").exists():
        raise RuntimeError("Run prepare first")
    state = read(BASE / "STATUS.json")
    if not state["cpu_kernel_ready"] or not state["topology_verified"]:
        raise RuntimeError("CPU preflight/topology not ready; no simulated replacement for real timing")
    for name, (_, path) in SOURCES.items():
        validate_asset(name, path)
    if role != "pilot" and session is None:
        raise ValueError("Specify --session so measurement sessions can be placed in separate idle time periods")
    session = 1 if role == "pilot" else session
    splits = image_manifest()
    if role != "pilot":
        require_pilot(splits)
    if role == "validation":
        audited_role_files("calibration", range(1, 6), splits)
    if role == "test":
        require_calibration_freeze(splits)
    jobs = session_blocks(splits[role], role, session)
    state.update(status=f"measuring_{role}", service_measurements_started=True)
    save(BASE / "STATUS.json", state)
    for index, job in enumerate(jobs, 1):
        stem = f'{role}_s{session}_b{job["block"]}_{job["profile"]}'
        destination = BASE / "measurements" / role / f"{stem}.csv"
        if destination.exists():
            raise FileExistsError(f"Existing timing block: {destination}; audit before resume, do not overwrite")
        job_path = BASE / "protocol" / f"job_{stem}.json"
        save(job_path, job)
        print(f"Measurement block {index}/{len(jobs)}: {stem}", flush=True)
        with (BASE / "logs" / f"{stem}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", str(BASE / "scripts/measure_block.py"), "--job", str(job_path)],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1" if job["profile"] == "CPU-1T" else "4", "MKL_NUM_THREADS": "1" if job["profile"] == "CPU-1T" else "4"})
        if result.returncode:
            state.update(status="failed_measurement_block", failed_job=job, failed_returncode=result.returncode)
            save(BASE / "STATUS.json", state)
            raise RuntimeError(f"Failed block {stem}; original error/rows retained")
    state.update(status=f"{role}_session_{session}_measured_pending_audit")
    counts = count_measurement_rows()
    state["formal_measurement_rows"] = sum(n for key, n in counts.items() if key != "pilot")
    state["measurement_rows_by_role"] = counts
    state["formal_measurement_row_count_status"] = "actual raw CSV rows; completion still requires integrity audit"
    save(BASE / "STATUS.json", state)
    if role == "pilot":
        fingerprints = audited_role_files("pilot", [1], splits)
        save(BASE / "protocol/PILOT_GATE.json", {"passed": True,
             "environment_spec_sha256": sha(BASE / "protocol/ENVIRONMENT_SPEC.json"),
             "image_split_sha256": sha(BASE / "protocol/IMAGE_SPLITS.json"),
             "measurement_sha256": fingerprints, "source_sha256": measurement_source_lock(),
             "formal_samples": False, "requests": 200, "datetime": datetime.now().isoformat()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "inspect", "check", "pilot", "calibration", "validation", "test"))
    p.add_argument("--session", type=int)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "inspect":
        print(json.dumps(read(BASE / "STATUS.json"), ensure_ascii=False, indent=2) if (BASE / "STATUS.json").exists() else "Not prepared")
    elif args.stage == "check":
        check()
    else:
        run_measurements(args.stage, args.session)


if __name__ == "__main__":
    main()
