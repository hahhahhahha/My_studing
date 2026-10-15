"""Build the additive seven-method formal-entry package without launching a cell."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from systemlab.common import encoded, read, sha, source_receipt, write_new

METHODS = ('qla', 'ppo_mv', 'ppo_dsr', 'mfc_di', 'qeco', 'jseq', 'lyfoto_lc')


def read_external(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--calibrated-config', type=Path, required=True)
    parser.add_argument('--pilot-summary', type=Path, required=True)
    parser.add_argument('--pilot-plan', type=Path, required=True)
    parser.add_argument('--qla-bundle', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--image-split', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_dir = out / 'protocol'
    schedules_dir = out / 'schedules'
    bundles_dir = protocol_dir / 'bundles'
    for path in (protocol_dir, schedules_dir, bundles_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = read_external(args.calibrated_config)
    summary = read_external(args.pilot_summary)
    plan = read_external(args.pilot_plan)
    if config['users'] != 30 or len(config['edges']) != 35:
        raise ValueError('Formal package requires 30 users and 35 logical edges')
    if config['feedback']['budget_bytes_per_s'] != 2097152 or config['feedback']['burst_bytes'] != 1048576:
        raise ValueError('Formal package feedback budget differs from the registered pilot')
    if plan['aggregate_rate_per_s'] != 62 or plan['arrival_horizon_s'] != 120 or plan['fixed_drain_s'] != 30:
        raise ValueError('Pilot load contract is not the requested 62/120/30 contract')
    observed = {item['method'] for item in summary['methods']}
    if observed != set(METHODS) or not summary['all_infrastructure_valid'] or not summary['all_source_continuity_pass']:
        raise ValueError('Pilot adoption requires complete seven-method valid evidence')
    if len({item['schedule_sha256'] for item in summary['methods']}) != 1:
        raise ValueError('Pilot methods do not share one schedule')

    scales = dict(config['feature_scales'])
    scales['status'] = 'frozen_after_calibration_validation'
    write_new(protocol_dir / 'FEATURE_SCALES.json', scales)
    config['feature_scales'] = scales
    config['frozen_protocol_path'] = str((protocol_dir / 'FROZEN_SYSTEM_PROTOCOL.json').resolve())
    config['frozen_protocol_sha256'] = 'PENDING_PROTOCOL_HASH'
    write_new(protocol_dir / 'FORMAL_CONFIG.json', config)

    validation_artifacts = {
        str(args.pilot_summary.resolve()): sha(args.pilot_summary),
        str(args.pilot_plan.resolve()): sha(args.pilot_plan),
    }
    for method in METHODS:
        run = args.pilot_root / 'runs' / method / 'client'
        for name in ('INPUTS.json', 'COMPLETE.json', 'CUTOFF.json', 'tasks.jsonl', 'decisions.jsonl'):
            path = run / name
            if not path.is_file():
                raise ValueError(f'Missing pilot evidence: {path}')
            validation_artifacts[str(path.resolve())] = sha(path)
    formal_plain = {key: value for key, value in config.items()
                    if key not in ('frozen_protocol_path', 'frozen_protocol_sha256')}
    protocol = {
        'schema': 'systemlab-frozen-protocol-v3',
        'formal_config': formal_plain,
        'source_sha256': source_receipt(),
        'validation_artifacts_sha256': validation_artifacts,
        'pilot_adoption': 'same data/algorithm/routing/feedback/queue/feature/metric path; additive entry only',
        'independent_entry_review_required': True,
        'formal_entry_approved': False,
    }
    write_new(protocol_dir / 'FROZEN_SYSTEM_PROTOCOL.json', protocol)
    config['frozen_protocol_sha256'] = sha(protocol_dir / 'FROZEN_SYSTEM_PROTOCOL.json')
    (protocol_dir / 'FORMAL_CONFIG.json').unlink()
    write_new(protocol_dir / 'FORMAL_CONFIG.json', config)

    adoption = {
        'schema': 'systemlab-pilot-adoption-v1',
        'source_project': str(args.pilot_root.resolve()),
        'methods': list(METHODS),
        'shared_schedule_sha256': summary['shared_schedule_sha256'],
        'pilot_summary': binding(args.pilot_summary),
        'pilot_plan': binding(args.pilot_plan),
        'config': binding(args.calibrated_config),
        'source_sha256': source_receipt(),
        'differences': ['new additive protocol/registry/grid/runner only',
                        'test role and five new common schedules are not pilot results'],
        'formal_entry_approved': False,
    }
    write_new(protocol_dir / 'PILOT_ADOPTION.json', adoption)

    for method in METHODS:
        source = args.qla_bundle if method == 'qla' else args.baseline_root / f'{method}.json'
        if not source.is_file():
            raise ValueError(f'Missing bundle: {source}')
        shutil.copy2(source, bundles_dir / source.name)

    manifest = out / 'TEST_IMAGES.json'
    import_images = __import__('systemlab.data', fromlist=['import_images']).import_images
    import_images(args.image_split, args.image_root, 'test', manifest, 500)
    write_new(out / 'FORMAL_PLAN.json', {
        'schema': 'workstation-30u35e-7methods-formal-plan-v1',
        'status': 'pending_independent_entry_review',
        'methods': list(METHODS), 'seeds': [1, 2, 3, 4, 5],
        'cells': 35, 'users': 30, 'logical_edges': 35,
        'aggregate_rate_per_s': 62, 'arrival_horizon_s': 120, 'fixed_drain_s': 30,
        'feedback_budget_bytes_per_s': 2097152, 'feedback_burst_bytes': 1048576,
        'feedback_nominal_period_s': 0.2, 'training': False,
        'test_manifest_sha256': sha(manifest),
        'pilot_adoption_sha256': sha(protocol_dir / 'PILOT_ADOPTION.json'),
        'execution': 'one cell at a time; new directory; stop on first integrity failure',
    })


if __name__ == '__main__':
    main()
