"""Schema-aware, test-only external window preparation (no torch dependency)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_qla_unseen_external_windows_20261009"
AZURE = ROOT / "datasets/external_traces/extracted/azure"
WINDOWS = (("W1", 11, 0), ("W2", 12, 220), ("W3", 13, 440))
MINUTES = 200
HOLD_SLOTS = 5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_azure(path):
    """Sum official minute columns across functions; never average metadata."""
    total = np.zeros(1440, dtype=np.int64)
    rows = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indices = [header.index(str(minute)) for minute in range(1, 1441)]
        contiguous = indices == list(range(indices[0], indices[0] + 1440))
        for row in reader:
            cells = row[indices[0]:indices[-1] + 1] if contiguous else [row[i] for i in indices]
            values = np.fromstring(",".join(cells), dtype=np.int64, sep=",")
            if values.size != 1440 or (values < 0).any():
                raise ValueError(f"Malformed invocation counts in {path}, row {rows + 2}")
            total += values
            rows += 1
    if rows == 0 or total.sum() <= 0:
        raise ValueError(f"Empty Azure day: {path}")
    return total, {"path": str(path), "sha256": sha(path), "function_rows": rows,
                   "minutes": 1440, "total_invocations": int(total.sum()),
                   "count_semantics": "invocations recorded after execution, demand-intensity proxy"}


def old_source_audit(root=ROOT):
    """Trace actual saved external sources, not only names of generated npzs."""
    paths, sources, largest = [], set(), 0
    for result_root in sorted(root.glob("results*")):
        schedule_root = result_root / "external_schedules"
        if not schedule_root.is_dir():
            continue
        for path in schedule_root.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            meta = data.get("protocol_metadata", {})
            for key, value in meta.items():
                if (key.endswith("source") or key.endswith("path")) and isinstance(value, str):
                    sources.add(value)
            largest = max(largest, int(data.get("n_steps", 0)), int(data.get("split_length", 0)))
            paths.append(str(path))
    forbidden = [s for s in sources if any(f"d{day:02d}.csv" in s for day in (11, 12, 13))
                 or "v2017" in s.lower() or "server_usage.csv" in s.lower()]
    if forbidden:
        raise RuntimeError(f"Proposed held-out sources occur in saved old schedules: {forbidden}")
    return {"schedule_manifests_checked": len(paths), "source_files": sorted(sources),
            "maximum_schedule_prefix_length": largest, "proposed_sources_overlap": forbidden,
            "scope": "saved results*/external_schedules manifests; not a claim about every historical manual use"}


def audit_alibaba_legacy(path, prefix=10000):
    timestamps, machines = [], set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # Legacy reader mistakenly treated the first data row as header.
        for _, row in zip(range(prefix), reader):
            timestamps.append(float(row[1]))
            machines.add(row[0])
    return {"path": str(path), "legacy_used_prefix_rows": len(timestamps),
            "timestamp_min_s": min(timestamps), "timestamp_max_s": max(timestamps),
            "machine_ids": sorted(machines), "calendar_holdout_available_from_later_rows": False,
            "reason": "legacy prefix already spans most of the v2018 collection period; rows are not time order"}


def prepare_azure(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    audit = old_source_audit()
    legacy = ROOT / "datasets/external_traces/extracted/alibaba/machine_usage.csv"
    if legacy.is_file():
        audit["alibaba_legacy_prefix"] = audit_alibaba_legacy(legacy)
    save(out / "SOURCE_USAGE_AUDIT.json", audit)
    series, metadata = {}, {}
    for day in (1, 11, 12, 13):
        target = out / "data" / f"azure_d{day:02d}.npz"
        source = AZURE / f"invocations_per_function_md.anon.d{day:02d}.csv"
        if target.is_file():
            meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
            if sha(source) != meta["sha256"]:
                raise RuntimeError(f"Azure source changed: {source}")
            with np.load(target, allow_pickle=False) as loaded:
                values = loaded["invocation_count_per_minute"]
        else:
            print(f"Reading Azure day {day:02d}, official minute columns", flush=True)
            values, meta = aggregate_azure(source)
            target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target, invocation_count_per_minute=values)
            save(target.with_suffix(".json"), meta)
        series[day], metadata[day] = values, meta
    lo, hi = np.percentile(series[1], [5., 95.])
    if hi <= lo:
        raise ValueError("Calibration day has degenerate invocation intensity")
    calibration = {"reference_day": 1, "p05": float(lo), "p95": float(hi),
                   "fit_on_test": False, "formula": "lambda=0.65+4.2*(0.2+0.8*clip((count-p05)/(p95-p05),0,1))",
                   "window_minutes": MINUTES, "slots_per_minute": HOLD_SLOTS,
                   "time_compression": "one raw minute -> five 1-s simulation slots; 12x compression",
                   "test_days": [11, 12, 13], "raw_azure_minute_slice_zero_based": [0, MINUTES]}
    save(out / "data/DEMAND_CALIBRATION.json", calibration)
    reports = []
    for name, day, capacity_minute in WINDOWS:
        raw = series[day][:MINUTES]
        scaled = np.clip((raw - lo) / (hi - lo), 0., 1.)
        lam = np.repeat(0.65 + 4.2 * (0.2 + 0.8 * scaled), HOLD_SLOTS)
        np.savez_compressed(out / "data" / f"{name}_demand.npz", lambda_per_user=lam, raw_minute_counts=raw)
        reports.append({"window": name, "azure_day": day, "azure_minutes": [0, MINUTES],
                        "alibaba2017_relative_minutes": [capacity_minute, capacity_minute + MINUTES],
                        "lambda_min": float(lam.min()), "lambda_mean": float(lam.mean()),
                        "lambda_max": float(lam.max()),
                        "fraction_below_calibration": float(np.mean(raw < lo)),
                        "fraction_above_calibration": float(np.mean(raw > hi)),
                        "source_sha256": metadata[day]["sha256"]})
    save(out / "data/DEMAND_WINDOWS.json", reports)
    return reports


def complete_minutes(records, start, length=MINUTES):
    """No interpolation: require each minute's official valid measurement."""
    values = []
    for minute in range(start, start + length):
        if minute not in records:
            return None
        values.append(float(np.mean(records[minute])))
    return np.asarray(values, dtype=np.float64)


def prepare_alibaba2017(path, out=OUT):
    """2017 server_usage has timestamp,machine_id,cpu_percent,..., no header."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required independent source missing: {path}")
    if path.name != "server_usage.csv":
        raise ValueError("Do not pass Alibaba 2018 machine_usage.csv to the 2017 schema reader")
    machines, bad, count, minimum = {}, 0, 0, None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            try:
                stamp, machine, cpu = float(row[0]), row[1], float(row[2])
            except (ValueError, IndexError):
                bad += 1
                continue
            if not np.isfinite(stamp) or stamp < 0 or not np.isfinite(cpu) or not 0 <= cpu <= 100:
                bad += 1
                continue
            minute = int(stamp // 60)
            machines.setdefault(machine, {}).setdefault(minute, []).append(cpu)
            minimum = minute if minimum is None else min(minimum, minute)
            count += 1
    if minimum is None:
        raise ValueError("No valid Alibaba 2017 CPU measurements")
    # Selection depends only on measurement completeness, never policy outcomes.
    selected, raw_windows = [], {w[0]: [] for w in WINDOWS}
    for machine in sorted(machines, key=lambda x: (int(x) if x.isdigit() else 10**30, x)):
        data = {name: complete_minutes(machines[machine], minimum + start) for name, _, start in WINDOWS}
        if any(value is None for value in data.values()):
            continue
        selected.append(machine)
        for name, values in data.items():
            raw_windows[name].append(values)
        if len(selected) == 35:
            break
    if len(selected) != 35:
        raise ValueError(f"Only {len(selected)} machines cover all preregistered minutes; do not relax or select windows post hoc")
    metadata = {"path": str(path), "sha256": sha(path), "release": "Alibaba cluster trace v2017",
                "schema": ["timestamp_seconds", "machine_id", "cpu_percent"],
                "valid_rows": count, "invalid_rows": bad, "origin_minute": minimum,
                "selected_machine_ids": selected, "selection": "first 35 numeric-ID ordered complete machines",
                "capacity_formula": "clip(1-cpu_percent/100,0.35,1.0)",
                "meaning": "available-service proxy, not measured visual-inference service time",
                "missing_value_policy": "exclude incomplete machines; no gap fill or interpolation"}
    for name, _, start in WINDOWS:
        raw = np.stack(raw_windows[name], axis=1)
        capacity = np.repeat(np.clip(1. - raw / 100., .35, 1.), HOLD_SLOTS, axis=0).astype(np.float32)
        np.savez_compressed(out / "data" / f"{name}_capacity.npz", capacity_scale=capacity, raw_cpu_percent=raw)
    save(out / "data/CAPACITY_SOURCE.json", metadata)
    return metadata


def schedule_arrays(lambda_per_user, capacity_scale, seed, trace_indices):
    n = len(lambda_per_user)
    if n != 1000 or capacity_scale.shape != (1000, 35):
        raise ValueError("Frozen test protocol requires 1000 slots and 35 servers")
    # Separate RNG streams: changes in Poisson intensity must not change task
    # assignment uniforms, feedback loss, or extra-age draws.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(10000 + seed).spawn(5)]
    weights = streams[0].lognormal(0., .25, size=30)
    weights /= weights.mean()
    lam = np.clip(np.asarray(lambda_per_user)[:, None] * weights[None, :], .05, 7.)
    return {"trace_indices": np.asarray(trace_indices, dtype=np.int64),
            "requests": np.clip(streams[1].poisson(lam), 0, 7).astype(np.int64),
            "assignment_uniforms": streams[2].random((n, 30, 7)),
            "capacity_scale": np.asarray(capacity_scale, dtype=np.float32),
            "background_scale": np.ones((n, 35), dtype=np.float32),
            "summary_loss": streams[3].random((n, 35)) < .03,
            "source_age_extra": streams[4].integers(0, 9, size=(n, 35), dtype=np.int64),
            "task_compute_scale": np.ones(n, dtype=np.float32)}
