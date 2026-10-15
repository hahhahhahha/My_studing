"""Independent stdlib-only read-only verification; writes only review diagnostics.

Does not import experiment code, extract files to disk, download, or run policies.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "dac2027_followup_experiments_20261010"
ATTEMPT = HERE.parent
RAW = ROOT / "datasets/external_traces/extracted/alibaba2017/server_usage.csv"
ARCHIVE = ATTEMPT / "alibaba-trace-2017.tar.gz"
OLD_ARCHIVE = BASE / "data_acquisition/alibaba-trace-2017.tar.gz.partial"
EXPECTED = "02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df"
WINDOWS = {"W1": (0, 200), "W2": (220, 420), "W3": (440, 640)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


inputs = [ARCHIVE, OLD_ARCHIVE, RAW, ATTEMPT / "DOWNLOAD_STATE.json",
          ATTEMPT / "SOURCE_PATH_REPAIR.json", BASE / "data_acquisition/DOWNLOAD_STATE.json",
          BASE / "data_acquisition/original_source_probe_20261010_233159_417364.json",
          BASE / "scripts/acquire_original_alibaba2017.py", BASE / "scripts/acquire_alibaba2017.py",
          BASE / "scripts/audit_alibaba2017_schema.py",
          BASE / "protocol/ALIBABA2017_SCHEMA_20261010_234236_101936.json",
          ROOT / "paper_doc_tools/heldout_external_windows.py",
          ROOT / "paper_doc_tools/run_unseen_external_windows.py",
          ROOT / "results_qla_unseen_external_windows_20261009/EXPERIMENT_PLAN_中文.md",
          ROOT / "results_qla_unseen_external_windows_20261009/DESIGN.json"]
before = {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in inputs}
print("Input hashes computed", flush=True)
member_entries = []
archive_inventory = []
with tarfile.open(ARCHIVE, "r|gz") as archive:
    for member in archive:
        archive_inventory.append({"name": member.name, "bytes": member.size,
                                  "is_regular": member.isfile(), "type": member.type.decode("ascii")})
        if PurePosixPath(member.name.replace("\\", "/")).name == "server_usage.csv":
            entry = {"name": member.name, "is_regular": member.isfile(), "bytes": member.size}
            if member.isfile():
                digest = hashlib.sha256()
                actual = 0
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        actual += len(block)
                        digest.update(block)
                entry.update(stream_sha256=digest.hexdigest(), stream_bytes=actual)
            member_entries.append(entry)
print("Archive member stream verified without disk extraction", flush=True)
valid = defaultdict(dict)
widths, empty_fields, full_schema_bad_fields = Counter(), Counter(), Counter()
invalid, full_schema_invalid = [], []
rows = 0
raw_timestamps, timestamp_mod300 = Counter(), Counter()
cpu_min, cpu_max = math.inf, -math.inf
duplicate_count = 0
examples = []
with RAW.open(encoding="utf-8", newline="") as handle:
    for number, row in enumerate(csv.reader(handle), 1):
        rows += 1
        widths[len(row)] += 1
        if len(examples) < 3:
            examples.append({"line": number, "row": row})
        bad_fields = []
        if len(row) != 8:
            bad_fields.append("width")
        for index, cell in enumerate(row):
            if not cell:
                empty_fields[index + 1] += 1
            try:
                value = float(cell)
                if not math.isfinite(value) or (index in (0, 1) and not value.is_integer()):
                    raise ValueError("nonfinite or noninteger mandatory integer")
            except ValueError:
                bad_fields.append(index + 1)
                full_schema_bad_fields[index + 1] += 1
        if bad_fields:
            full_schema_invalid.append({"line": number, "bad_fields": bad_fields, "row": row})
        try:
            timestamp, machine, cpu = float(row[0]), row[1], float(row[2])
            if not math.isfinite(timestamp) or timestamp < 0 or not math.isfinite(cpu) or not 0 <= cpu <= 100:
                raise ValueError("invalid timestamp/CPU according to frozen parser")
        except (IndexError, ValueError) as exc:
            invalid.append({"line": number, "reason": str(exc), "row": row})
            continue
        if timestamp in valid[machine]:
            duplicate_count += 1
        valid[machine][timestamp] = cpu
        raw_timestamps[timestamp] += 1
        timestamp_mod300[timestamp % 300] += 1
        cpu_min, cpu_max = min(cpu_min, cpu), max(cpu_max, cpu)
origin = int(min(raw_timestamps) // 60)
gaps, coverage, coverage_patterns = Counter(), [], Counter()
complete, native_complete = [], []
all_time_coverage = Counter()
for machine in sorted(valid, key=lambda x: (int(x) if x.isdigit() else 10**30, x)):
    times = sorted(valid[machine])
    minutes = {int(t // 60) for t in times}
    gaps.update(b - a for a, b in zip(times, times[1:]))
    all_time_coverage[len(times)] += 1
    counts, absent = {}, {}
    for name, (start, stop) in WINDOWS.items():
        expected = set(range(origin + start, origin + stop))
        counts[name] = len(minutes & expected)
        absent[name] = sorted(x - origin for x in expected - minutes)
    eligible = all(counts[k] == WINDOWS[k][1] - WINDOWS[k][0] for k in WINDOWS)
    if eligible:
        complete.append(machine)
    # Diagnostic only: this does NOT select a revised cohort or prepare schedules.
    native_eligible = all(set(range(origin + a, origin + b, 5)) <= minutes for a, b in WINDOWS.values())
    if native_eligible:
        native_complete.append(machine)
    coverage_patterns[tuple(counts.values())] += 1
    coverage.append({"machine_id": machine, "valid_unique_timestamps": len(times),
                     "coverage_minutes": counts, "complete_all_registered_minute_windows": eligible,
                     "missing_relative_minutes": absent,
                     "first_valid_timestamp": times[0], "last_valid_timestamp": times[-1]})
state = read_json(ATTEMPT / "DOWNLOAD_STATE.json")
old_state = read_json(BASE / "data_acquisition/DOWNLOAD_STATE.json")
repair = read_json(ATTEMPT / "SOURCE_PATH_REPAIR.json")
design = read_json(ROOT / "results_qla_unseen_external_windows_20261009/DESIGN.json")
previous = read_json(BASE / "protocol/ALIBABA2017_SCHEMA_20261010_234236_101936.json")
schema = {"rows": rows, "valid_rows": rows - len(invalid), "invalid_rows": len(invalid),
          "machines": len(valid), "origin_minute": origin, "cpu_percent_min": cpu_min,
          "cpu_percent_max": cpu_max, "duplicate_machine_timestamps": duplicate_count,
          "field_width_histogram": dict(widths), "timestamp_mod_300_histogram": dict(timestamp_mod300),
          "positive_timestamp_gap_seconds_histogram": dict(sorted(gaps.items())),
          "complete_machine_count": len(complete), "first_35_complete_machine_ids": complete[:35]}
simple_comparison = {k: schema[k] == previous[k] for k in schema
                     if k not in ("field_width_histogram", "timestamp_mod_300_histogram", "positive_timestamp_gap_seconds_histogram")}
for key in ("field_width_histogram", "timestamp_mod_300_histogram", "positive_timestamp_gap_seconds_histogram"):
    simple_comparison[key] = {str(k): v for k, v in schema[key].items()} == previous[key]
report = {
    "classification": "dataset-preparation verification only; NOT a completed OOD experiment",
    "at": datetime.now().astimezone().isoformat(), "python": sys.version,
    "script_sha256": sha(__file__), "inputs_before": before,
    "official_sources": {
        "checksum": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/SHA256SUM",
        "description": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/trace_201708.md",
        "schema": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/schema.csv"},
    "official_expected_archive_sha256": EXPECTED,
    "archive_matches_official_checksum": before[str(ARCHIVE)]["sha256"] == EXPECTED,
    "archive_matches_state": before[str(ARCHIVE)]["sha256"] == state["archive_sha256"],
    "archive_bytes_match_state": before[str(ARCHIVE)]["bytes"] == state["bytes_downloaded"],
    "archive_inventory": archive_inventory, "server_usage_members": member_entries,
    "single_regular_member": len(member_entries) == 1 and member_entries[0]["is_regular"],
    "raw_equals_member_stream": len(member_entries) == 1 and member_entries[0].get("stream_sha256") == before[str(RAW)]["sha256"],
    "raw_equals_acquisition_state": before[str(RAW)]["sha256"] == state["server_usage_sha256"],
    "old_failed_archive_preserved_and_matches_record": before[str(OLD_ARCHIVE)]["sha256"] == old_state["archive_sha256"],
    "repair_provenance_checks": {
        "wrapper": repair["wrapper_sha256"] == before[str(BASE / "scripts/acquire_original_alibaba2017.py")]["sha256"],
        "acquisition_source": repair["acquisition_source_sha256"] == before[str(BASE / "scripts/acquire_alibaba2017.py")]["sha256"],
        "historical_failure": repair["historical_failure_sha256"] == before[str(BASE / "data_acquisition/DOWNLOAD_STATE.json")]["sha256"],
        "probe": repair["probe_sha256"] == before[str(BASE / "data_acquisition/original_source_probe_20261010_233159_417364.json")]["sha256"]},
    "design_code_checks": {
        "data_code": design["data_code_sha256"] == before[str(ROOT / "paper_doc_tools/heldout_external_windows.py")]["sha256"],
        "runner": design["runner_sha256"] == before[str(ROOT / "paper_doc_tools/run_unseen_external_windows.py")]["sha256"]},
    "schema": schema, "invalid_cpu_timestamp_records": invalid,
    "full_eight_field_schema_invalid_count": len(full_schema_invalid),
    "full_eight_field_schema_invalid_records": full_schema_invalid,
    "empty_fields_by_1_based_column": dict(empty_fields),
    "full_schema_bad_fields_by_1_based_column": dict(full_schema_bad_fields),
    "first_rows": examples,
    "timestamp_min_seconds": min(raw_timestamps), "timestamp_max_seconds": max(raw_timestamps),
    "unique_global_timestamp_count": len(raw_timestamps),
    "global_timestamp_values": sorted(raw_timestamps),
    "per_machine_unique_valid_timestamp_count_histogram": dict(all_time_coverage),
    "window_coverage_patterns": [{"W1": a, "W2": b, "W3": c, "machines": n}
                                 for (a, b, c), n in sorted(coverage_patterns.items())],
    "registered_windows": WINDOWS, "required_machines": 35, "registered_gate_passed": len(complete) >= 35,
    "complete_on_native_five_minute_grid_machine_count_diagnostic_only": len(native_complete),
    "no_alternative_cohort_selected": True,
    "previous_diagnostic_comparison": simple_comparison,
    "machine_coverage": coverage,
    "no_download_no_disk_extraction_no_gapfill_no_policy_run": True,
}
after = {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in inputs}
report["inputs_unchanged_during_review"] = before == after
report["inputs_after"] = after
target = HERE / "INDEPENDENT_DIAGNOSTIC.json"
with target.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
summary = {k: v for k, v in report.items() if k not in ("inputs_before", "inputs_after", "machine_coverage", "archive_inventory", "global_timestamp_values")}
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("DIAGNOSTIC", target, "SHA256", sha(target))
