"""Read-only byte verification of the exported files; no experiment imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def absolute_refs(root, records):
    counts = {}
    samples = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(key)
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, str) and re.match(r'^(?:[A-Za-z]:[\\/]|/)', value):
            yield value

    for row in records:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or path.suffix != '.json' or not path.is_file():
            continue
        try:
            values = list(walk(json.loads(path.read_text(encoding='utf-8-sig'))))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if values:
            counts[row['path']] = len(values)
            if len(samples) < 12:
                samples.append({'file': row['path'], 'example': values[0]})
    return {'files_with_absolute_refs': len(counts),
            'total_refs': sum(counts.values()), 'examples': samples,
            'note': 'Read-only inventory, not path relocation or entry clearance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--absolute-refs', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    with (root / 'SYNC_MANIFEST.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    errors = []
    seen = set()
    for row in records:
        relative = row['path']
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            errors.append({'path': relative, 'error': 'duplicate_or_unsafe_path'})
            continue
        seen.add(relative)
        if not path.is_file():
            errors.append({'path': relative, 'error': 'missing'})
        elif path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            errors.append({'path': relative, 'error': 'byte_or_sha_mismatch'})
    report = {'files': len(records), 'errors': errors, 'pass': not errors,
              'scope': 'Copied source snapshot only; not scientific/entry approval.'}
    if args.absolute_refs:
        report['portability'] = absolute_refs(root, records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
