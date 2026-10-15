"""Explicit artificial schedule documents, no real retained generation/load."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from visual_episode_collector import BASE, write_evidence
from visual_request_schedule import ScheduledRequest, VisualRequestSchedule
from visual_training_transactions import digest, file_sha
from retained_test_schedule import (_PreparedRetainedSchedulePayload, _decode_registered_document,
    _strict_json, SCHEMA, RegisteredRetainedRequestSchedule)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_134402_711447_retained_parser/RESULT.json'
PRIOR_SHA = 'f95e00baea28f693186144a0aff247e9329b3def65726c15e17c86a3e4c129fa'
POOL_SHA = 'a'*64  # explicit artificial identity, never the original1000 pool
ARTIFICIAL_IDS = frozenset({'EXPLICIT-ARTIFICIAL-A', 'EXPLICIT-ARTIFICIAL-B'})


def refused(call):
    try:
        call()
    except (ValueError, RuntimeError, TypeError, KeyError, FrozenInstanceError):
        return
    raise AssertionError('Artificial invalid retained schedule accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_retained_schedule')
    attempt.mkdir(parents=True, exist_ok=False)
    actual_timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA
        prior = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = prior['original_protected_files_sha256'] | prior['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_real_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                actual_timing_opens.append(str(path))
                raise AssertionError('No real retained timing CSV may be opened in this source check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_real_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            rows = (
                ScheduledRequest('ARTIFICIAL-0', 0, 0., 'EXPLICIT-ARTIFICIAL-A', 1, 0.),
                ScheduledRequest('ARTIFICIAL-1', 1, .05, 'EXPLICIT-ARTIFICIAL-B', 5, .2),
                ScheduledRequest('ARTIFICIAL-2', 29, .05000000000000001, 'EXPLICIT-ARTIFICIAL-A', 2, .9),
                ScheduledRequest('ARTIFICIAL-3', 0, .14999999999999997, 'EXPLICIT-ARTIFICIAL-B', 4, .99))
            pure = _PreparedRetainedSchedulePayload('test', 'EXPLICIT_ARTIFICIAL_LOW', 30, .05, 3, 1,
                2., .1, 2., POOL_SHA, rows)
            doc = dict(schema=SCHEMA, payload=pure.to_dict(), semantic_sha256=pure.fingerprint())
            loaded = _decode_registered_document(doc, ARTIFICIAL_IDS, POOL_SHA)
            assert loaded == pure and loaded.horizon_seconds == .15
            assert tuple(loaded.decision_index(row) for row in rows) == (1, 1, 2, 3)
            assert tuple(loaded.decision_time(i) for i in (1, 2, 3)) == (.05, .1, .15)
            refused(lambda: loaded.decision_time(True))
            refused(lambda: loaded.decision_time(4))
            refused(lambda: loaded.decision_index(replace(rows[0], measured_session=3)))
            refused(lambda: loaded.decision_index(replace(rows[0], assignment_uniform=.3)))
            refused(lambda: setattr(loaded, 'seed', 2))
            try:
                loaded._by_task['OTHER'] = rows[0]
            except TypeError:
                pass
            else:
                raise AssertionError('Task index must be immutable')
            copy = loaded.to_dict()
            copy['requests'][0]['assignment_uniform'] = .6
            assert loaded.requests[0].assignment_uniform == 0. and loaded.fingerprint() == pure.fingerprint()
            # Equivalent JSON presentation retains semantic identity, but its
            # external byte SHA is different; real loader requires both.
            a = json.dumps(doc, sort_keys=True, separators=(',', ':'))
            b = json.dumps(dict(reversed(tuple(doc.items()))), indent=3)
            assert hashlib.sha256(a.encode()).hexdigest() != hashlib.sha256(b.encode()).hexdigest()
            assert _decode_registered_document(_strict_json(a), ARTIFICIAL_IDS, POOL_SHA) == loaded
            assert _decode_registered_document(_strict_json(b), ARTIFICIAL_IDS, POOL_SHA) == loaded
            refused(lambda: _strict_json('{"seed":1,"seed":2}'))
            refused(lambda: _strict_json('{"x":NaN}'))
            invalid_docs = []
            def changed(edit):
                value = deepcopy(doc)
                edit(value)
                invalid_docs.append(value)
            changed(lambda d: d.update(hidden_method='QLA'))
            changed(lambda d: d['payload'].update(nominal_delay_seconds=.4))
            changed(lambda d: d['payload']['requests'][0].update(service_ms=1.))
            changed(lambda d: d['payload']['requests'][0].update(image_id='OUTSIDE'))
            changed(lambda d: d['payload']['requests'][0].update(measured_session=True))
            changed(lambda d: d['payload']['requests'][0].update(user=True))
            changed(lambda d: d['payload']['requests'][0].update(arrival_seconds=.15))
            changed(lambda d: d['payload']['requests'][0].update(assignment_uniform=1.))
            changed(lambda d: d['payload']['requests'][0].update(arrival_seconds=float('inf')))
            changed(lambda d: d['payload']['requests'].append(deepcopy(d['payload']['requests'][0])))
            changed(lambda d: d['payload']['requests'].reverse())
            changed(lambda d: d['payload'].update(seed=True))
            changed(lambda d: d['payload'].update(role='validation'))
            changed(lambda d: d['payload'].update(users=True))
            changed(lambda d: d['payload'].update(decisions=2))
            changed(lambda d: d['payload'].update(total_requests_per_second=3.))
            changed(lambda d: d['payload'].update(burst_probability=.2))
            changed(lambda d: d['payload'].update(retained_pool_identity_sha256='b'*64))
            changed(lambda d: d['payload']['requests'][1].update(assignment_uniform=.25))
            changed(lambda d: d['payload']['requests'].pop())
            for value in invalid_docs:
                refused(lambda: _decode_registered_document(value, ARTIFICIAL_IDS, POOL_SHA))
            empty = replace(pure, requests=())
            empty_doc = dict(schema=SCHEMA, payload=empty.to_dict(), semantic_sha256=empty.fingerprint())
            assert _decode_registered_document(empty_doc, ARTIFICIAL_IDS, POOL_SHA).requests == ()
            # Real entry cannot be replaced with an artificial descriptor;
            # no fake certificates/current-weight contracts are constructed.
            refused(lambda: RegisteredRetainedRequestSchedule(None, pure.load_label, 1,
                attempt/'MISSING_NONCERTIFICATE_SCHEDULE.json', '0'*64))
            refused(lambda: RegisteredRetainedRequestSchedule(pure, pure.load_label, 1,
                attempt/'MISSING_NONCERTIFICATE_SCHEDULE.json', '0'*64))
            refused(lambda: VisualRequestSchedule('test', 30, .05, 3, 1, rows, 'artificial'))
            assert not actual_timing_opens and 'torch' not in sys.modules
            sources = {str(path): file_sha(path) for path in (Path(__file__),
                BASE/'scripts/retained_test_schedule.py', BASE/'README_RETAINED_TEST_SCHEDULE_中文.md')}
            write_evidence(attempt/'EXPLICIT_ARTIFICIAL_FOUR_TASK_DOCUMENT.json', doc)
            report = dict(at=datetime.now().astimezone().isoformat(), pure_descriptor_and_missing_authority_guards_passed=True,
                synthetic_tasks_not_generated_or_measured=4, explicitly_artificial_pool_not_original_identity=POOL_SHA,
                altered_field_or_identity_documents_rejected=len(invalid_docs), zero_arrival_artificial_case_retained=True,
                exact_physical_boundary_task_indices=[1, 1, 2, 3], immutable_rows_and_task_index_checked=True,
                semantic_JSON_encoding_identity_and_distinct_byte_SHA_checked=True,
                duplicate_JSON_keys_and_nonfinite_constants_rejected=True,
                actual_registered_schedule_constructor_succeeded=False, actual_retained_schedule_generated=False,
                actual_original_test_timing_CSV_open=actual_timing_opens, actual_model_queue_or_policy_calls=0,
                actual_test_cells=0, torch_imported=False, original_protected_files_sha256=protected,
                new_implementation_sha256=sources, independent_entry_review=False, formal_pilot_or_test_protocol_frozen=False)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_RETAINED_SCHEDULE_DESCRIPTOR_ONLY_NOT_REAL_TEST', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_registered_schedule_or_policy_calls=0, attempted_original_test_timing_CSV_open=actual_timing_opens))
        raise


if __name__ == '__main__':
    main()
