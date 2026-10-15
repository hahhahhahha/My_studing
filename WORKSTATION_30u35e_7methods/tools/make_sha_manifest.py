"""Write a complete SHA-256 manifest for the additive formal-entry package."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from systemlab.common import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    files = sorted(path for path in root.rglob('*')
                   if path.is_file() and path != output and '__pycache__' not in path.parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('path', 'bytes', 'sha256'))
        for path in files:
            writer.writerow((str(path.relative_to(root)), path.stat().st_size, sha(path)))


if __name__ == '__main__':
    main()
