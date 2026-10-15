"""Read-only raw schema/coverage diagnostic; never fill gaps or run policies."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
SOURCE = ROOT / 'datasets/external_traces/extracted/alibaba2017/server_usage.csv'
ACQUISITION = BASE / 'data_acquisition/original2017_attempt_20261010/DOWNLOAD_STATE.json'
WINDOWS = {'W1': (0, 200), 'W2': (220, 420), 'W3': (440, 640)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    source_hash = sha(SOURCE)
    acquired = json.loads(ACQUISITION.read_text(encoding='utf-8'))
    if source_hash != acquired['server_usage_sha256']:
        raise RuntimeError('Raw server_usage changed since verified acquisition')
    if acquired['archive_sha256'] != '02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df':
        raise RuntimeError('Not the registered original archive')
    records, invalid, widths, rows, cpu_min, cpu_max = {}, [], Counter(), 0, math.inf, -math.inf
    timestamp_remainders = Counter()
    with SOURCE.open(encoding='utf-8', newline='') as handle:
        for number, row in enumerate(csv.reader(handle), start=1):
            rows += 1
            widths[len(row)] += 1
            try:
                timestamp, machine, cpu = float(row[0]), row[1], float(row[2])
                if not math.isfinite(timestamp) or timestamp < 0 or not math.isfinite(cpu) or not 0 <= cpu <= 100:
                    raise ValueError('Invalid timestamp or CPU percent')
            except (IndexError, ValueError) as exc:
                invalid.append({'line': number, 'reason': str(exc)})
                continue
            records.setdefault(machine, {}).setdefault(timestamp, []).append(cpu)
            timestamp_remainders[timestamp % 300] += 1
            cpu_min, cpu_max = min(cpu_min, cpu), max(cpu_max, cpu)
    if not records:
        raise RuntimeError('No valid raw records')
    origin = int(min(timestamp for series in records.values() for timestamp in series) // 60)
    gap_histogram, duplicates = Counter(), 0
    complete, machine_coverage = [], []
    for machine in sorted(records, key=lambda value: (int(value) if value.isdigit() else 10**30, value)):
        times = sorted(records[machine])
        gap_histogram.update(right - left for left, right in zip(times, times[1:]))
        duplicates += sum(len(values) - 1 for values in records[machine].values())
        minutes = {int(time // 60) for time in times}
        coverage = {name: sum(origin + start <= minute < origin + end for minute in minutes)
                    for name, (start, end) in WINDOWS.items()}
        eligible = all(coverage[name] == end - start for name, (start, end) in WINDOWS.items())
        if eligible:
            complete.append(machine)
        machine_coverage.append({'machine_id': machine, 'valid_unique_timestamps': len(times),
                                 'coverage_minutes': coverage, 'complete_all_registered_minute_windows': eligible})
    report = {
        'at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source': str(SOURCE), 'source_sha256': source_hash,
        'acquisition_state_sha256': sha(ACQUISITION), 'script_sha256': sha(__file__),
        'archive_sha256': acquired['archive_sha256'], 'rows': rows,
        'valid_rows': rows - len(invalid), 'invalid_rows': len(invalid), 'invalid_examples': invalid[:20],
        'field_width_histogram': dict(widths), 'schema_interpreted': ['timestamp_seconds', 'machine_id', 'cpu_percent'],
        'machines': len(records), 'cpu_percent_min': cpu_min, 'cpu_percent_max': cpu_max,
        'duplicate_machine_timestamps': duplicates, 'origin_minute': origin,
        'timestamp_mod_300_histogram': dict(timestamp_remainders),
        'positive_timestamp_gap_seconds_histogram': dict(sorted(gap_histogram.items())),
        'registered_relative_minute_windows': WINDOWS,
        'expected_distinct_minutes_per_window': 200,
        'complete_machine_count': len(complete), 'first_35_complete_machine_ids': complete[:35],
        'registered_data_preparation_gate_passed': len(complete) >= 35 and not invalid,
        'machine_coverage': machine_coverage,
        'no_gap_fill': True, 'windows_changed': False, 'policy_training_or_tests_started': False,
        'scope': 'Raw coverage diagnostic only; sampling-interval mismatch requires explicit protocol decision, not silent interpolation.'
    }
    destination = BASE / 'protocol' / ('ALIBABA2017_SCHEMA_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ('rows', 'valid_rows', 'invalid_rows', 'machines', 'origin_minute',
          'timestamp_mod_300_histogram', 'positive_timestamp_gap_seconds_histogram', 'complete_machine_count',
          'registered_data_preparation_gate_passed')}, ensure_ascii=False, indent=2))
    print('SCHEMA_DIAGNOSTIC_REPORT', destination)


if __name__ == '__main__':
    main()
