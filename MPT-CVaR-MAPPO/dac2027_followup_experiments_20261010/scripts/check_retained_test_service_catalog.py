"""Explicit artificial parser rows + missing-authority rejection only.

No actual retained timing CSV is opened, no actual execution/model is run.
"""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from metadata_service_predictor import ObservableImage, PROFILES
from visual_episode_collector import BASE, write_evidence
from visual_training_transactions import file_sha
from retained_test_service_catalog import (SERVICE_ROOT, SERVICE_EVIDENCE_BINDINGS,
    _registered_inventory, _PairedDurationIndex, RetainedExecutionServiceCatalog)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_131601_349980_test_contract/RESULT.json'
PRIOR_SHA = 'a50abf2342c950172e6fd1737063190a0569ef945ce3a979e56893821d59bd74'


def refused(call):
    try:
        call()
    except (ValueError, RuntimeError, TypeError, KeyError):
        return
    raise AssertionError('Artificial invalid parser/access accepted')


def artificial_row(image, session, profile, metadata, image_sha, milliseconds):
    # Literal artificial durations: unrelated to any measured/test image.
    return dict(image_id=image, image_split='test', session_id=str(session), profile_id=profile,
        error_status='ok', image_sha256=image_sha, width=str(metadata.width),
        height=str(metadata.height), payload_bytes=str(metadata.payload_bytes),
        service_ms=str(milliseconds), decode_ms=str(milliseconds), preprocess_ms='0',
        infer_ms='0', postprocess_ms='0', EXPLICIT_ARTIFICIAL_NOT_MEASURED=True)


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_retained_parser')
    attempt.mkdir(parents=True, exist_ok=False)
    actual_csv_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = (prior['original_protected_files_sha256'] | prior['new_implementation_sha256']
            | SERVICE_EVIDENCE_BINDINGS | {str(PRIOR): PRIOR_SHA})
        original_open = Path.open

        def forbid_actual_timing_open(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to((SERVICE_ROOT/'measurements/test').resolve()):
                actual_csv_opens.append(str(path))
                raise AssertionError('This source/parser check must not open real retained timing CSV')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', forbid_actual_timing_open):
            assert all(file_sha(Path(path)) == sha for path, sha in protected.items())
            # Only original inventory metadata/hashes read, not measured rows.
            from retained_test_service_catalog import RAW_AUDIT
            raw = json.loads(RAW_AUDIT.read_text(encoding='utf-8'))
            inventory = _registered_inventory(raw['measurement_sha256'])
            assert len(inventory) == 100 and sum(Path(name).suffix == '.csv' for name in inventory) == 50
            missing = deepcopy(raw['measurement_sha256'])
            missing.pop(next(iter(missing)))
            refused(lambda: _registered_inventory(missing))
            extra = dict(raw['measurement_sha256']) | {'../outside.csv': '0'*64}
            refused(lambda: _registered_inventory(extra))
            absolute = dict(raw['measurement_sha256']) | {'X:/outside.csv': '0'*64}
            refused(lambda: _registered_inventory(absolute))
            duplicate = dict(raw['measurement_sha256'])
            key = next(iter(duplicate))
            duplicate[key.replace('\\', '/')] = duplicate[key]
            refused(lambda: _registered_inventory(duplicate))
            changed_sha = deepcopy(raw['measurement_sha256'])
            changed_sha[key] = 'not-a-hash'
            refused(lambda: _registered_inventory(changed_sha))
            metadata = {'ARTIFICIAL-A': ObservableImage(80, 60, 900),
                        'ARTIFICIAL-B': ObservableImage(100, 75, 1300)}
            image_sha = {'ARTIFICIAL-A': 'a'*64, 'ARTIFICIAL-B': 'b'*64}
            rows = [artificial_row(image, session, profile, metadata[image], image_sha[image],
                    (1000000000 if image == 'ARTIFICIAL-B' and session == 2 else 20 if profile == 'CPU-1T' else 8))
                    for image in metadata for session in (1, 2) for profile in PROFILES]
            paired = _PairedDurationIndex(metadata, image_sha, (1, 2))
            for row in rows:
                paired.add_row(row, int(row['session_id']), row['profile_id'])
            paired.seal()
            assert paired.pair_seconds('ARTIFICIAL-A', 1) == {'CPU-1T': .02, 'CPU-4T': .008}
            assert paired.pair_seconds('ARTIFICIAL-B', 2) == {'CPU-1T': 1000000., 'CPU-4T': 1000000.}
            copy = paired.pair_seconds('ARTIFICIAL-A', 1)
            copy['CPU-1T'] = -1.
            assert paired.pair_seconds('ARTIFICIAL-A', 1)['CPU-1T'] == .02
            refused(lambda: paired.pair_seconds('ARTIFICIAL-A', True))
            refused(lambda: paired.add_row(rows[0], 1, 'CPU-1T'))
            refused(paired.seal)
            changes = [dict(image_split='calibration'), dict(image_id='OUTSIDE'), dict(image_sha256='c'*64),
                dict(session_id='3'), dict(profile_id='CPU-4T'), dict(error_status='error'),
                dict(payload_bytes='1'), dict(width='0'), dict(service_ms='nan'), dict(service_ms='0'),
                dict(decode_ms='-1'), dict(infer_ms='1')]
            for change in changes:
                broken = _PairedDurationIndex(metadata, image_sha, (1, 2))
                bad = dict(rows[0]) | change
                refused(lambda: broken.add_row(bad, 1, 'CPU-1T'))
                assert broken._failed
                refused(lambda: broken.add_row(rows[0], 1, 'CPU-1T'))
                refused(broken.seal)
            duplicate_rows = _PairedDurationIndex(metadata, image_sha, (1, 2))
            duplicate_rows.add_row(rows[0], 1, 'CPU-1T')
            refused(lambda: duplicate_rows.add_row(rows[0], 1, 'CPU-1T'))
            incomplete = _PairedDurationIndex(metadata, image_sha, (1, 2))
            for row in rows[:-1]:
                incomplete.add_row(row, int(row['session_id']), row['profile_id'])
            refused(incomplete.seal)
            assert incomplete._failed
            refused(lambda: incomplete.pair_seconds('ARTIFICIAL-A', 1))
            # No fabricated PASS/protocol/weights or mock authority bypass.
            refused(lambda: RetainedExecutionServiceCatalog(None))
            refused(lambda: RetainedExecutionServiceCatalog('prepared-descriptor-is-not-authority'))
            assert not actual_csv_opens and 'torch' not in sys.modules
            sources = {str(path): file_sha(path) for path in (Path(__file__),
                BASE/'scripts/retained_test_service_catalog.py', BASE/'README_RETAINED_EXECUTION_CATALOG_中文.md')}
            write_evidence(attempt/'EXPLICIT_ARTIFICIAL_PARSER_ROWS.json', rows)
            report = dict(at=datetime.now().astimezone().isoformat(), parser_and_missing_authority_guards_passed=True,
                artificial_rows_not_measured=8, artificial_images=2, artificial_sessions=2,
                artificially_slow_1000000_second_pairs_retained=True, altered_pair_lookup_copy_not_internal_state=True,
                original_inventory_metadata_entries_checked=100, original_timing_CSVs_parsed=0,
                attempted_original_timing_CSV_open=actual_csv_opens, actual_catalog_constructor_succeeded=False,
                actual_catalog_pair_lookups=0, actual_policy_or_queue_or_training_calls=0,
                actual_test_cells=0, actual_test_schedules_generated=False, torch_imported=False,
                original_protected_files_sha256=protected, new_implementation_sha256=sources,
                independent_repair_or_complete_entry_review=False, formal_protocol_frozen=False)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(path)) == sha for path, sha in (protected | sources).items())
        print('ARTIFICIAL_RETAINED_PAIR_PARSER_ONLY_NO_TEST_TIMING_OR_POLICY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_catalog_or_policy_calls=0,
            attempted_original_timing_CSV_open=actual_csv_opens, partial_evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()
