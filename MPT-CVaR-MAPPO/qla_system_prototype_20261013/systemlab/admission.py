"""Pinned test matrix admission, checked before services/model launch and at end."""
from __future__ import annotations

import math
from pathlib import Path

from .common import encoded, read, sha, source_receipt, write_new
from .policy import tree_hashes


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked(binding_record):
    if sha(binding_record['path']) != binding_record['sha256']:
        raise ValueError('Registered artifact changed: ' + binding_record['path'])
    return read(binding_record['path'])


def check_protocol(config):
    if sha(config['frozen_protocol_path']) != config['frozen_protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    protocol = read(config['frozen_protocol_path'])
    plain = {k:v for k,v in config.items() if k not in ('frozen_protocol_path','frozen_protocol_sha256')}
    if encoded(plain) != encoded(protocol['formal_config']) or protocol['source_sha256'] != source_receipt():
        raise ValueError('Configuration/source differs from frozen protocol; fresh validation and binding required')
    for path, fingerprint in protocol['validation_artifacts_sha256'].items():
        if sha(path) != fingerprint:
            raise ValueError('Validation artifact changed: ' + path)
    return protocol


def _schedule_contract(schedule, manifest_sha, registration, key, cells):
    if (schedule['schema'] != 'systemlab-schedule-v1' or schedule['role'] != 'test' or
            schedule['image_manifest_sha256'] != manifest_sha or
            schedule['users'] != registration['users'] or
            schedule['duration_s'] != registration['horizon_s']):
        raise ValueError('Test schedule role/manifest/users/horizon differs from registration')
    matching = [c for c in cells if c['common_schedule_key'] == key]
    if not matching or any(c['seed'] != schedule['seed'] for c in matching):
        raise ValueError('Schedule seed/key not registered in matrix')
    rates = {c['aggregate_rate_per_s'] for c in matching}
    if len(rates) != 1 or schedule['provenance'].get('source') != 'synthetic_poisson':
        raise ValueError('This registered Poisson matrix requires one exact common rate per schedule')
    actual = schedule['provenance'].get('aggregate_rate_per_s')
    if actual != next(iter(rates)):
        raise ValueError('Schedule rate differs from matrix (use full precision, not rounded values)')


def register_test_registry(config_path, grid_path, registration_path, manifest_path, schedules, output, generators=(), actor_bundles=None):
    """Bind existing registered settings; never regenerate/relabel old results."""
    config, grid, registration, manifest = map(read, (config_path,grid_path,registration_path,manifest_path))
    protocol = check_protocol(config)
    if grid['schema'] != 'systemlab-grid-v1' or grid['config_sha256'] != sha(config_path):
        raise ValueError('Grid/config binding mismatch')
    if encoded(read(grid['feature_scales_path'])) != encoded(config['feature_scales']):
        raise ValueError('Grid feature scale contract mismatch')
    if sha(grid['feature_scales_path']) != grid['feature_scales_sha256']:
        raise ValueError('Grid feature scales changed')
    if (registration.get('role') != 'validation' or manifest['role'] != 'test' or
            registration['users'] != config['users'] or registration['fixed_drain_s'] != config['fixed_drain_s'] or
            not math.isfinite(registration['horizon_s']) or registration['horizon_s'] <= 0):
        raise ValueError('Validation-selected horizon/users/drain and test manifest required')
    cutoffs = [read(path) for path in protocol['validation_artifacts_sha256'] if Path(path).name == 'CUTOFF.json']
    if not cutoffs or any(c['protocol_horizon_s'] != registration['horizon_s'] or
                          c['fixed_drain_s'] != registration['fixed_drain_s'] for c in cutoffs):
        raise ValueError('Registered horizon/drain were not actually exercised by frozen validation')
    cells = grid['cells']
    if not cells or len({c['id'] for c in cells}) != len(cells):
        raise ValueError('Nonempty matrix with unique cell IDs required')
    grouped = {}
    for path in schedules:
        schedule = read(path)
        keys = {c['common_schedule_key'] for c in cells if c['seed'] == schedule['seed'] and
                c['aggregate_rate_per_s'] == schedule['provenance'].get('aggregate_rate_per_s')}
        if len(keys) != 1:
            raise ValueError('Schedule does not identify exactly one common matrix key')
        key = keys.pop()
        if key in grouped:
            raise ValueError('Duplicate common schedule key')
        _schedule_contract(schedule,sha(manifest_path),registration,key,cells)
        grouped[key] = dict(binding(path),common_schedule_key=key)
    if set(grouped) != {c['common_schedule_key'] for c in cells}:
        raise ValueError('Every matrix common schedule must be registered exactly once')
    generator_records = [binding(path) for path in generators]
    if not generator_records:
        raise ValueError('Pin the registration/orchestration generator source')
    actor_records = {}
    required_actors = {c['method'] for c in cells if c['method'] in ('qla','qla_no_bias_probe','frozen_noql')}
    for method in required_actors:
        if method not in (actor_bundles or {}):
            raise ValueError('Pin a validated formal actor bundle for method: ' + method)
        path = actor_bundles[method]
        actor = read(path)
        validated = [read(p)['policy_receipt'].get('checkpoint_sha256') for p in protocol['validation_artifacts_sha256']
                     if Path(p).name == 'INPUTS.json' and read(p)['method'] == method]
        if (not validated or set(validated) != {actor['checkpoint_sha256']} or
                sha(actor['checkpoint']) != actor['checkpoint_sha256'] or
                tree_hashes(actor['source_root']) != actor['source_sha256'] or
                encoded(actor['feature_contract']) != encoded(config['feature_scales'])):
            raise ValueError('Formal actor differs from actual frozen validation/checkpoint/source/scales')
        actor_records[method] = binding(path)
    value = dict(schema='systemlab-test-registry-v2',config=binding(config_path),grid=binding(grid_path),
        protocol=binding(config['frozen_protocol_path']),validation_registration=binding(registration_path),
        feature_scales=binding(grid['feature_scales_path']),test_manifest=binding(manifest_path),
        common_schedules=list(grouped.values()),generator_sources=generator_records,actor_bundles=actor_records,
        source_sha256=source_receipt(),horizon_s=registration['horizon_s'],users=registration['users'],
        fixed_drain_s=registration['fixed_drain_s'],registration_is_not_independent_approval=True)
    write_new(output,value)
    return value


def validate_test_admission(config, schedule_path, manifest_path, method, entry_review=None, test_registry=None, cell_id=None, bundle=None):
    schedule = read(schedule_path)
    if schedule['role'] != 'test':
        return None
    if config['feature_scales']['status'] != 'frozen_after_calibration_validation':
        raise ValueError('Calibration/validation freeze required for test')
    check_protocol(config)
    if not test_registry or not cell_id or not entry_review:
        raise ValueError('Test requires --test-registry, --cell-id and fresh --entry-review')
    registry = read(test_registry)
    if registry.get('schema') != 'systemlab-test-registry-v2' or registry['source_sha256'] != source_receipt():
        raise ValueError('Missing/stale v2 test registry source binding')
    registered_config = checked(registry['config'])
    if encoded(config) != encoded(registered_config) or registry['protocol']['sha256'] != config['frozen_protocol_sha256']:
        raise ValueError('Test registry belongs to a different configuration/protocol')
    checked(registry['protocol'])
    grid, registration = checked(registry['grid']), checked(registry['validation_registration'])
    manifest, scales = checked(registry['test_manifest']), checked(registry['feature_scales'])
    if (sha(manifest_path) != registry['test_manifest']['sha256'] or manifest['role'] != 'test' or
            grid['config_sha256'] != registry['config']['sha256'] or
            grid['feature_scales_sha256'] != registry['feature_scales']['sha256'] or
            encoded(scales) != encoded(config['feature_scales']) or
            registration['horizon_s'] != registry['horizon_s'] or
            registration['users'] != config['users'] or registration['fixed_drain_s'] != config['fixed_drain_s']):
        raise ValueError('Registry matrix/manifest/scale/settings mismatch')
    for item in registry['generator_sources']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Registered generator source changed: ' + item['path'])
    records = {item['common_schedule_key']:item for item in registry['common_schedules']}
    for item in records.values():
        checked(item)
    cells = [c for c in grid['cells'] if c['id'] == cell_id]
    if len(cells) != 1 or cells[0]['method'] != method:
        raise ValueError('Unregistered cell ID/method')
    cell = cells[0]
    if method in ('qla','qla_no_bias_probe','frozen_noql'):
        actor_record = registry['actor_bundles'].get(method)
        if not bundle or not actor_record or sha(bundle) != actor_record['sha256']:
            raise ValueError('Unregistered formal actor bundle')
        actor = checked(actor_record)
        if (sha(actor['checkpoint']) != actor['checkpoint_sha256'] or
                tree_hashes(actor['source_root']) != actor['source_sha256'] or
                encoded(actor['feature_contract']) != encoded(config['feature_scales'])):
            raise ValueError('Registered actor/checkpoint/source/scales changed')
    item = records[cell['common_schedule_key']]
    if sha(schedule_path) != item['sha256']:
        raise ValueError('Selected schedule is not the registered cell schedule')
    _schedule_contract(schedule,sha(manifest_path),registration,cell['common_schedule_key'],grid['cells'])
    feedback = config['feedback']
    for field, key in (('feedback_mode','mode'),('injected_feedback_delay_s','injected_delay_s'),
                       ('injected_feedback_loss','injected_loss'),('feedback_budget_bytes_per_s','budget_bytes_per_s')):
        if cell[field] != feedback[key]:
            raise ValueError('Cell feedback condition differs from configuration')
    review = read(entry_review)
    if (review.get('schema') != 'systemlab-independent-entry-v1' or review.get('status') not in ('PASS','WARN') or
            review.get('blockers') != [] or review.get('scope') != 'complete_system_entry' or review.get('self_audit') or
            review.get('protocol_sha256') != config['frozen_protocol_sha256'] or
            review.get('test_registry_sha256') != sha(test_registry) or review.get('source_sha256') != source_receipt() or
            not review.get('reviewer_identity') or not review.get('complete_report') or
            sha(review['report_path']) != review['report_sha256']):
        raise ValueError('Incomplete/stale/blocked/self-audit entry review or unreviewed test registry')
    return dict(test_registry_sha256=sha(test_registry),entry_review_sha256=sha(entry_review),cell_id=cell_id,
                actor_bundle_sha256=sha(bundle) if method in ('qla','qla_no_bias_probe','frozen_noql') else None,
                generator_sources=registry['generator_sources'])
