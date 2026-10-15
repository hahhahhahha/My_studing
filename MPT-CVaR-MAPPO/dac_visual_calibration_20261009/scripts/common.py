from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

from windows_runtime import initialize_windows_runtime

# Run before importing PyTorch/asyncio or launching NVML/curl subprocesses.
WINDOWS_RUNTIME_RESTORED = initialize_windows_runtime()

BASE = Path(__file__).resolve().parents[1]
DIRS = ("data/raw", "data/extracted", "models", "protocol", "measurements/pilot",
        "measurements/calibration", "measurements/validation", "measurements/test",
        "calibration", "schedules", "policies", "evaluations", "analysis", "figures", "logs")


def setup():
    for name in DIRS:
        (BASE / name).mkdir(parents=True, exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(path, rows):
    if not rows:
        raise ValueError("No rows; refuse to write a successful empty measurement")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_member(name):
    parts = name.replace("\\", "/").split("/")
    if not parts or parts[0] != "imagenette2" or any(p in ("..", ".") or ":" in p for p in parts):
        raise ValueError(f"Unsafe/unexpected archive member: {name}")
    return name


def split_images(rows, seed=20261009):
    rng = random.Random(seed)
    seen, pools = set(), {}
    excluded = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        if row["image_sha256"] in seen:
            excluded.append(row["image_id"])
            continue
        seen.add(row["image_sha256"])
        pools.setdefault((row["official_split"], row["class_id"]), []).append(row)
    classes = sorted({r["class_id"] for r in rows})
    if len(classes) != 10:
        raise ValueError(f"Expected 10 classes, got {len(classes)}")
    result = {role: [] for role in ("pilot", "calibration", "validation", "test")}
    for cls in classes:
        train, test = pools.get(("train", cls), [])[:], pools.get(("val", cls), [])[:]
        rng.shuffle(train)
        rng.shuffle(test)
        if len(train) < 260 or len(test) < 100:
            raise ValueError(f"Insufficient nonduplicate class {cls}: {len(train)} train/{len(test)} val")
        for role, chunk in (("pilot", train[:10]), ("calibration", train[10:210]),
                            ("validation", train[210:260]), ("test", test[:100])):
            result[role].extend({**r, "image_split": role} for r in chunk)
    for values in result.values():
        rng.shuffle(values)
    hashes = [r["image_sha256"] for values in result.values() for r in values]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Image roles overlap")
    return result, excluded


def validate_roles(roles):
    expected = {"pilot": 100, "calibration": 2000, "validation": 500, "test": 1000}
    if {k: len(v) for k, v in roles.items()} != expected:
        raise ValueError("Frozen image role counts differ from the protocol")
    hashes, ids = [], []
    for role, rows in roles.items():
        for row in rows:
            if row["image_split"] != role:
                raise ValueError("Image role label does not match its pool")
            if row["official_split"] != ("val" if role == "test" else "train"):
                raise ValueError("Official image split differs from the protocol")
            safe_member(row["image_id"])
            hashes.append(row["image_sha256"])
            ids.append(row["image_id"])
    if len(set(hashes)) != len(hashes) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate image ID/hash across frozen roles")
    return expected


def session_blocks(rows, role, session):
    maximum = {"pilot": 1, "calibration": 5, "validation": 3, "test": 5}[role]
    if not 1 <= session <= maximum:
        raise ValueError("Session outside the preregistered count")
    ordered = rows[:]
    # Same order for both profiles; only processing order is counterbalanced.
    random.Random(f"20261009:{role}:{session}").shuffle(ordered)
    jobs = []
    first = ("CPU-1T", "CPU-4T") if session % 2 else ("CPU-4T", "CPU-1T")
    for block_id, start in enumerate(range(0, len(ordered), 200), 1):
        profiles = first if block_id % 2 else first[::-1]
        for profile in profiles:
            jobs.append({"role": role, "session": session, "block": block_id,
                         "profile": profile, "image_ids": [r["image_id"] for r in ordered[start:start+200]]})
    return jobs


def dhash(image):
    grey = image.convert("L").resize((9, 8))
    pixels = list(grey.getdata())
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def cross_role_near_duplicates(roles):
    # Hamming<=2 => at least one of three disjoint hash segments is equal.
    index, seen_pairs, hits = {}, set(), []
    for role, rows in roles.items():
        for row in rows:
            value = int(row["dhash64"], 16)
            keys = ((0, value & ((1 << 22)-1)), (1, (value >> 22) & ((1 << 21)-1)), (2, value >> 43))
            candidates = {i for key in keys for i in index.get(key, [])}
            for previous in candidates:
                old_role, old_id, old_value = previous
                pair = tuple(sorted((old_id, row["image_id"])))
                if old_role != role and (value ^ old_value).bit_count() <= 2 and pair not in seen_pairs:
                    hits.append({"image_a": old_id, "role_a": old_role,
                                 "image_b": row["image_id"], "role_b": role,
                                 "dhash_distance": (value ^ old_value).bit_count()})
                    seen_pairs.add(pair)
            item = (role, row["image_id"], value)
            for key in keys:
                index.setdefault(key, []).append(item)
    return hits
