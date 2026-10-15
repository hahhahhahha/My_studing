"""Artificial metadata-only RNG/descriptor checks, no retained pool draws."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from dataclasses import replace, asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from visual_episode_collector import BASE, write_evidence
from visual_training_transactions import file_sha, digest
from visual_request_schedule import generate_schedule
from retained_test_schedule import _decode_registered_document, SCHEMA
from retained_test_schedule_generation import (RetainedGenerationDesign,
    _generate_descriptor_payload, RealRetainedScheduleGenerator)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_151825_900055_test_grid/RESULT.json'
PRIOR_SHA = '075541a3437960219d1360e73862ec16b9fcb133ae3709a5ee195c822404b39d'
IDS = ('EXPLICIT_ARTIFICIAL_IMAGE_A', 'EXPLICIT_ARTIFICIAL_IMAGE_B', 'EXPLICIT_ARTIFICIAL_IMAGE_C')


class ArtificialMetadataIDs:
    role = 'calibration'
    def image_ids(self):
        return IDS


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError('Invalid pure design or missing generation authority accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_gen')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No original retained timing CSV in artificial generation check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            design = RetainedGenerationDesign(('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH'), (60., 120.),
                8, 1., '0'*64, .1, 2.)
            global_before = np.random.get_state()
            records, total = [], 0
            for load_index in (0, 1):
                for seed in design.seeds:
                    payload, witness = _generate_descriptor_payload(design, load_index, seed, IDS)
                    repeat, again = _generate_descriptor_payload(design, load_index, seed, IDS)
                    assert payload.fingerprint() == repeat.fingerprint() and witness == again
                    legacy = generate_schedule(ArtificialMetadataIDs(), design.users, design.generation_decisions,
                        design.decision_seconds, seed, design.total_rates_per_second[load_index], .1, 2.)
                    assert payload.requests == legacy.requests
                    doc = dict(schema=SCHEMA, payload=payload.to_dict(), semantic_sha256=payload.fingerprint())
                    parsed = _decode_registered_document(doc, frozenset(IDS), '0'*64)
                    assert parsed.requests == payload.requests
                    assert all(1 <= payload.decision_index(row) <= design.generation_decisions for row in payload.requests)
                    changed_pool, changed_witness = _generate_descriptor_payload(design, load_index, seed, IDS[:2])
                    # IDs are the ONLY changed field. Do not let image choice
                    # advance arrival/session/assignment random streams.
                    strip_image = lambda rows: [dict(task_id=r.task_id, user=r.user, arrival_seconds=r.arrival_seconds,
                        measured_session=r.measured_session, assignment_uniform=r.assignment_uniform) for r in rows]
                    assert strip_image(changed_pool.requests) == strip_image(payload.requests)
                    for name in ('counts_and_burst', 'arrival_offsets', 'measured_sessions', 'assignment_uniforms'):
                        assert witness['state_sha256_after'][name] == changed_witness['state_sha256_after'][name]
                    records.append(dict(EXPLICIT_ARTIFICIAL_NOT_REGISTERED_REQUESTS=True,
                        schema=SCHEMA, payload=payload.to_dict(), semantic_sha256=payload.fingerprint(), witness=witness))
                    total += len(payload.requests)
            global_after = np.random.get_state()
            assert global_before[0] == global_after[0] and np.array_equal(global_before[1], global_after[1])
            assert global_before[2:] == global_after[2:]
            empty, _ = _generate_descriptor_payload(replace(design, total_rates_per_second=(1e-12, 2e-12)), 0, 1, IDS)
            assert len(empty.requests) == 0
            large, _ = _generate_descriptor_payload(replace(design, total_rates_per_second=(6000., 12000.)), 0, 1, IDS)
            per_interval_user = {}
            for row in large.requests:
                key = large.decision_index(row), row.user
                per_interval_user[key] = per_interval_user.get(key, 0)+1
            assert max(per_interval_user.values()) > 7
            bad = [lambda: replace(design, seeds=(True, 2, 3, 4, 5)),
                lambda: replace(design, users=31), lambda: replace(design, generation_decisions=True),
                lambda: replace(design, fixed_drain_seconds=0.), lambda: replace(design, burst_probability=1.1),
                lambda: replace(design, total_rates_per_second=(120., 60.)),
                lambda: _generate_descriptor_payload(design, True, 1, IDS),
                lambda: _generate_descriptor_payload(design, 0, True, IDS),
                lambda: _generate_descriptor_payload(design, 0, 1, tuple(reversed(IDS))),
                lambda: _generate_descriptor_payload(design, 0, 1, IDS+IDS[:1]),
                lambda: RealRetainedScheduleGenerator(design, None, attempt/'NO_REAL_GENERATION')]
            for call in bad:
                refused(call)
            assert not (attempt/'NO_REAL_GENERATION').exists()
            tree = ast.parse((BASE/'scripts/retained_test_schedule_generation.py').read_text(encoding='utf-8'))
            called = {node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            assert not {'evaluate', 'act', 'step', 'backward', 'measured_pair_seconds', 'fitting_rows'} & called
            assert 'torch' not in sys.modules and not timing_opens
            write_evidence(attempt/'EXPLICIT_ARTIFICIAL_METADATA_DRAWS_NOT_REGISTERED_SCHEDULES.json',
                dict(records=records, artificial_empty_count=len(empty.requests),
                    artificial_high_rate_count=len(large.requests), artificial_max_user_interval_count=max(per_interval_user.values()),
                    fake_metadata_only_not_real_test_schedules=True))
            sources = {str(path): file_sha(path) for path in (Path(__file__),
                BASE/'scripts/retained_test_schedule_generation.py', BASE/'README_RETAINED_SCHEDULE_GENERATION_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(), artificial_metadata_rng_descriptor_guards_passed=True,
                explicit_artificial_load_seed_descriptors=10, explicit_artificial_task_rows=total,
                repeated_five_stream_RNG_and_existing_calibration_recipe_match=True,
                changed_fake_image_pool_does_not_advance_other_four_streams=True, global_numpy_rng_unchanged=True,
                artificial_empty_schedule_retained=True, artificial_uncapped_user_interval_max=max(per_interval_user.values()),
                invalid_design_and_missing_authority_cases_rejected=len(bad),
                original_retained_pool_or_metadata_draws_performed=False, actual_registered_test_schedules_generated=0,
                actual_generator_constructor_generate_or_register_calls=0, actual_model_policy_queue_or_test_calls=0,
                actual_original_test_timing_CSV_opens=timing_opens, torch_imported=False,
                preparation_permission_and_actual_smoke_pilot_guards_exercised=False,
                independent_entry_review=False, formal_runtime_protocol_frozen=False,
                generation_before_immutable_formal_training_test_bindings_source_prepared=True,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_METADATA_DRAWS_ONLY_NOT_REGISTERED_RETAINED_SCHEDULES', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_registered_test_schedules_generated=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
