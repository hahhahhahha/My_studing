"""Create the seven-method registry and five common test schedules."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemlab.common import read, sha, source_receipt, write_new
from systemlab.data import schedule

METHODS = ('qla', 'ppo_mv', 'ppo_dsr', 'mfc_di', 'qeco', 'jseq', 'lyfoto_lc')


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def ensure_artifact(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing artifact differs; choose a new additive directory: ' + str(path))
        return
    write_new(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read(args.config)
    if config['users'] != 30 or len(config['edges']) != 35:
        raise ValueError('Registry requires 30 users and 35 logical edges')
    registration = {
        'schema': 'workstation-30u35e-formal-registration-v1',
        'role': 'validation-derived-test',
        'users': 30, 'horizon_s': 120, 'fixed_drain_s': 30,
        'aggregate_rate_per_s': 62, 'seeds': [1, 2, 3, 4, 5],
        'methods': list(METHODS), 'formal_entry_approved': False,
    }
    registration_path = root / 'protocol' / 'FORMAL_REGISTRATION.json'
    ensure_artifact(registration_path, registration)
    grid_path = root / 'protocol' / 'E1_GRID.json'
    cells = []
    for seed in registration['seeds']:
        for method in METHODS:
            cells.append({
                'id': f'E1_{len(cells) + 1:03}', 'seed': seed, 'method': method,
                'aggregate_rate_per_s': 62.0, 'feedback_mode': 'periodic',
                'injected_feedback_delay_s': 0., 'injected_feedback_loss': 0.,
                'feedback_budget_bytes_per_s': 2097152,
                'common_schedule_key': f'62_{seed}',
            })
    random.Random(20261015).shuffle(cells)
    grid = {
        'schema': 'systemlab-grid-v2', 'block': 'E1',
        'methods': list(METHODS), 'seeds': registration['seeds'],
        'users': 30, 'logical_edges': 35, 'cells': cells,
        'config_sha256': sha(args.config),
        'feature_scales_path': str((root / 'protocol' / 'FEATURE_SCALES.json').resolve()),
        'feature_scales_sha256': sha(root / 'protocol' / 'FEATURE_SCALES.json'),
        'execution_order': 'preregistered_seeded_shuffle', 'started': False,
    }
    ensure_artifact(grid_path, grid)
    schedules = []
    for seed in registration['seeds']:
        path = root / 'schedules' / f'test_seed_{seed}.json'
        if not path.exists():
            schedule(args.manifest, path, users=30, duration=120., rate=62.,
                     seed=seed, role='test')
        schedules.append(path)
    bundles = root / 'protocol' / 'bundles'
    registry = {
        'schema': 'systemlab-test-registry-v3',
        'config': binding(args.config), 'grid': binding(grid_path),
        'protocol': binding(config['frozen_protocol_path']),
        'validation_registration': binding(registration_path),
        'feature_scales': binding(root / 'protocol' / 'FEATURE_SCALES.json'),
        'test_manifest': binding(args.manifest),
        'common_schedules': [
            dict(binding(path), common_schedule_key=f'62_{seed}')
            for seed, path in zip(registration['seeds'], schedules)
        ],
        'generator_sources': [binding(Path(__file__)), binding(root / 'prepare_formal_entry.py')],
        'actor_bundles': {'qla': binding(bundles / 'qla_actor_bundle.json')},
        'baseline_bundles': {
            method: binding(bundles / f'{method}.json') for method in METHODS if method != 'qla'
        },
        'source_sha256': source_receipt(), 'horizon_s': 120,
        'users': 30, 'fixed_drain_s': 30,
        'registration_is_not_independent_approval': True,
        'formal_entry_approved': False,
    }
    ensure_artifact(args.output, registry)


if __name__ == '__main__':
    main()
