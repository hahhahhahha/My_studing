"""Frozen image membership; common real-time schedules; no service-time draws."""
from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

from .common import read, sha, write_new


def import_images(split_file, image_root, role, output, limit=100):
    pools = read(split_file)
    if role not in pools or role not in ('pilot', 'calibration', 'validation', 'test'):
        raise ValueError('Use an existing explicit image role')
    records = sorted(pools[role], key=lambda x: x['image_id'])
    if limit <= 0 or limit > len(records):
        raise ValueError('Requested size must fit the original role')
    # Deterministic content-hash ordering avoids choosing by performance/size.
    records = sorted(records, key=lambda x: x['image_sha256'])[:limit]
    root = Path(image_root).resolve()
    selected = []
    for item in records:
        path = (root / item['image_id']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != item['image_sha256']:
            raise ValueError('Image path/hash mismatch')
        if path.stat().st_size != item['payload_bytes']:
            raise ValueError('JPEG bytes mismatch')
        selected.append({key: item[key] for key in ('image_id', 'image_sha256', 'payload_bytes', 'width', 'height')})
    value = dict(schema='systemlab-images-v1', role=role, source_split_sha256=sha(split_file),
                 selection='lowest_image_sha256_in_original_role', image_root=str(root), images=selected)
    write_new(output, value)
    return value


def task_rows(times, users, images, seed):
    rng = random.Random(seed)
    return [dict(task_id=f't{i:08d}', arrival_s=float(t), user_id=rng.randrange(users),
                 image_id=rng.choice(images)['image_id'], assignment_uniform=rng.random())
            for i, t in enumerate(sorted(times))]


def schedule(manifest, output, *, users=16, duration=30., rate=20., seed=1, role='pilot', azure=None,
             trace_start=None, trace_window=600., time_scale=1., time_unit='seconds'):
    pool = read(manifest)
    if role not in ('pilot', 'calibration', 'validation', 'test') or pool['role'] != role:
        raise ValueError('Schedule role must match image role')
    if users <= 0 or not all(math.isfinite(x) and x > 0 for x in (duration, rate, trace_window, time_scale)):
        raise ValueError('Positive schedule parameters required')
    provenance = {}
    if azure:
        if trace_start is None or not math.isfinite(trace_start):
            raise ValueError('Register an explicit raw-trace start before filtering')
        multiplier = {'seconds': 1., 'milliseconds': .001}[time_unit]
        times = []
        with Path(azure).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not {'app', 'func', 'end_timestamp', 'duration'}.issubset(reader.fieldnames or []):
                raise ValueError('Exact Azure 2021 event schema required')
            for row in reader:
                end, service = float(row['end_timestamp']) * multiplier, float(row['duration']) * multiplier
                if not all(math.isfinite(x) for x in (end, service)) or service < 0:
                    raise ValueError('Invalid event timestamps')
                start = end - service
                if trace_start <= start < trace_start + trace_window:
                    times.append((start - trace_start) * time_scale)
        duration = trace_window * time_scale
        if not times:
            raise ValueError('No event in preregistered window')
        provenance = dict(source='azure2021', source_sha256=sha(azure), start=trace_start,
                          raw_window_s=trace_window, time_scale=time_scale, raw_time_unit=time_unit,
                          selection='all_invocations_starting_in_half_open_window',
                          timestamps_modified_by_dataset_publisher=True)
    else:
        rng, current, times = random.Random(seed), 0., []
        while True:
            current += rng.expovariate(rate)
            if current >= duration:
                break
            times.append(current)
        provenance = dict(source='synthetic_poisson', aggregate_rate_per_s=rate)
    value = dict(schema='systemlab-schedule-v1', role=role, seed=seed, users=users,
                 duration_s=duration, image_manifest_sha256=sha(manifest), provenance=provenance,
                 tasks=task_rows(times, users, pool['images'], seed + 10000))
    write_new(output, value)
    return value


def minute_schedule(csv_file, manifest, output, *, columns, users, rate_scale, seed, role):
    """2019 counts → conditional within-minute uniform draws, explicitly synthetic."""
    if not columns or len(set(columns)) != len(columns) or any(not 1 <= x <= 1440 for x in columns):
        raise ValueError('Register exact minute columns')
    if rate_scale <= 0 or not math.isfinite(rate_scale):
        raise ValueError('Positive, calibration-selected thinning factor required')
    counts = [0] * len(columns)
    with Path(csv_file).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for i, col in enumerate(columns):
                count = int(row[str(col)])
                if count < 0:
                    raise ValueError('Negative invocation count')
                counts[i] += count
    pool = read(manifest)
    if pool['role'] != role:
        raise ValueError('Image role mismatch')
    rng, times = random.Random(seed), []
    for i, count in enumerate(counts):
        # Deterministic total after scaling, stochastic fractional rounding.
        n = count * rate_scale
        total = math.floor(n) + int(rng.random() < n - math.floor(n))
        times.extend(60 * (i + rng.random()) for _ in range(total))
    value = dict(schema='systemlab-schedule-v1', role=role, seed=seed, users=users,
                 duration_s=60 * len(columns), image_manifest_sha256=sha(manifest),
                 provenance=dict(source='azure2019_minute_counts_uniform_reconstruction',
                     source_sha256=sha(csv_file), columns=columns, counts=counts, count_scale=rate_scale,
                     synthetic_subminute_timestamps=True), tasks=task_rows(times, users, pool['images'], seed + 10000))
    write_new(output, value)
    return value
