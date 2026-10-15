"""Explicit artificial condition/SHA binding only, NOT a real pilot."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from visual_episode_collector import BASE, write_evidence
from visual_test_contract import FixedTestMatrixCandidate, PROFILES
from visual_training_transactions import digest, file_sha
from visual_upload_link import UploadLinkConfig
from visual_protocol_assembly import (bind_generated_test_environment_schedules,
    compose_common_test_environment_table, PILOT_TEST_TEMPLATE_FIELDS, RealVisualProtocolAssembler)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_172053_789145_learning/RESULT.json'
PRIOR_SHA = '3fcc93a047d79b31d684ba3b8ce9d56a9a704c609832e5442c57b93e785b5d5c'
BEFORE = BASE/'visual_calibrated_policy/engineering_attempts/20261011_174154_pilot_order_repair/SOURCE_BEFORE_REPAIR'
SNAPSHOTS = {
    'visual_protocol_assembly.py': 'c246a7460dbaf75373c5fa27ef91f586be90572c9588d55d4936328915fd840e',
    'README_VISUAL_PROTOCOL_ASSEMBLY_中文.md': 'd68eaaafed126f96990f990311b079019ebf917ea4866deb2055f038da31caa0',
    'retained_test_schedule_generation.py': 'cbf63ac2f253225a85e7a65a6313968af871bc57847629825aa816c16f432c17'}


def refused(call):
    try:
        call()
    except (TypeError, ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid artificial future SHA/template or missing actual generator accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_order')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        changed_sources = (BASE/'scripts/visual_protocol_assembly.py', BASE/'README_VISUAL_PROTOCOL_ASSEMBLY_中文.md')
        # ONLY the two explicitly snapshotted unfrozen preparation files may
        # change. Keep the old bytes under their preserved snapshot identities;
        # no scientific input/raw/paper/result path is waived or rebound.
        transferred = {}
        for source in changed_sources:
            keys = [name for name in protected if Path(name).resolve() == source.resolve()]
            assert len(keys) == 1 and protected[keys[0]] == SNAPSHOTS[source.name]
            transferred[keys[0]] = dict(before_sha256=protected.pop(keys[0]), snapshot_path=str(BEFORE/source.name))
        protected.update({str(BEFORE/name): sha for name, sha in SNAPSHOTS.items()})
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No actual retained timing CSV in pure pilot-order repair check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            common = tuple(tuple(f'{li*5+seed:064x}' for seed in range(1, 6)) for li in range(2))
            candidate = FixedTestMatrixCandidate(('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH'), (30., 60.), 8, 1., common, '0'*64)
            templates = {label: {str(seed): dict(role='test', profiles=list(PROFILES),
                edge_positions=[[0., 0.], [1., 0.], [0., 1.], [1., 1.]], mobility_start_seconds=1000.,
                broadcast_seconds=.1, nominal_delay_seconds=0., waiting_capacity=32, upload_link=asdict(UploadLinkConfig()),
                catalog_source_sha256={'EXPLICIT_ARTIFICIAL_NO_CATALOG_FILE': '0'*64},
                mobility_source_sha256={f'EXPLICIT_ARTIFICIAL_NO_MOBILITY_FILE_{u}': '0'*64 for u in range(30)})
                for seed in candidate.seeds} for label in candidate.load_labels}
            original = deepcopy(templates)
            original_sha = digest(templates)
            bound = bind_generated_test_environment_schedules(candidate, templates)
            assert all(set(t) == PILOT_TEST_TEMPLATE_FIELDS for rows in templates.values() for t in rows.values())
            for li, label in enumerate(candidate.load_labels):
                for si, seed in enumerate(candidate.seeds):
                    b = deepcopy(bound[label][str(seed)])
                    assert b.pop('schedule_sha256') == common[li][si] and b == templates[label][str(seed)]
            table = compose_common_test_environment_table(candidate, bound)
            assert table['descriptor_conditions'] == 30 and table['policy_cells_covered'] == 120
            assert table['actual_environment_or_policy_execution_proven'] == 0
            assert digest(templates) == original_sha and templates == original
            assert bind_generated_test_environment_schedules(candidate, json.loads(json.dumps(templates, indent=7))) == bound
            bound['ARTIFICIAL_LOW']['1']['edge_positions'][0][0] = 999.
            assert templates == original
            bad = []
            for field, value in (('schedule_sha256', '0'*64), ('schedule_sha256', 'GENERATE_LATER'),
                    ('future_request_SHA', 'f'*64), ('role', 'calibration'), ('nominal_delay_seconds', .1),
                    ('broadcast_seconds', .2), ('waiting_capacity', True), ('future_service_seconds', .03)):
                changed = deepcopy(templates)
                changed['ARTIFICIAL_LOW']['1'][field] = value
                bad.append(changed)
            changed = deepcopy(templates)
            changed['ARTIFICIAL_LOW'].pop('5')
            bad.append(changed)
            changed = deepcopy(templates)
            changed['ARTIFICIAL_LOW']['1']['mobility_source_sha256'].pop('EXPLICIT_ARTIFICIAL_NO_MOBILITY_FILE_29')
            bad.append(changed)
            changed = deepcopy(templates)
            changed['ARTIFICIAL_LOW']['1']['upload_link']['bandwidth_hz'] = 20e6
            bad.append(changed)
            for changed in bad:
                refused(lambda: bind_generated_test_environment_schedules(candidate, changed))
            refused(lambda: RealVisualProtocolAssembler(None, None, None))
            tree = ast.parse((BASE/'scripts/visual_protocol_assembly.py').read_text(encoding='utf-8'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RealVisualProtocolAssembler')
            build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_from_actual_inputs')
            statements = [n for n in build.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)]
            callnames = [n.value.func.id for n in statements if isinstance(n.value.func, ast.Name)]
            assert callnames.index('bind_generated_test_environment_schedules') < callnames.index('compose_common_test_environment_table')
            binder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bind_generated_test_environment_schedules')
            forbidden = {'generate_schedule', '_generate_descriptor_payload', 'OriginalRetainedMetadataPool',
                'RealRetainedScheduleGenerator', 'fitting_rows', 'measured_pair_seconds', 'act', 'evaluate', 'assemble_once'}
            calls = [(n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else '')
                for n in ast.walk(binder) if isinstance(n, ast.Call)]
            assert not forbidden & set(calls)
            assert 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), *changed_sources,
                BASE/'README_VISUAL_PILOT_ORDER_REPAIR_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(),
                pure_artificial_pilot_condition_to_generated_SHA_binding_checks_passed=True,
                explicit_artificial_pilot_condition_templates=10, resulting_artificial_environment_conditions=30,
                resulting_artificial_policy_cell_descriptors=120, actual_environment_or_results_proven=0,
                unknown_future_request_SHA_not_required_in_pilot=True,
                original_artificial_pilot_conditions_not_mutated=True, deep_copy_and_encoding_independent=True,
                artificial_future_placeholder_changed_role_delay_link_or_incomplete_cases_rejected=len(bad),
                static_bind_after_authenticated_same_generator_before_environment_composition=True,
                actual_metadata_generation_previews_sampling_or_request_files_created=0,
                actual_pilot_freeze_assembler_or_protocol_write_calls=0,
                actual_source_permission_path_guards_exercised=False,
                actual_test_timing_CSV_opens=timing_opens, torch_imported=False,
                actual_model_training_validation_or_test_cells=0, independent_entry_review=False,
                actual_completed_pilot_generator_weight_review_or_PASS_fixtures_fabricated=False,
                formal_pilot_or_protocol_frozen=False, source_only_not_launch_authority=True,
                two_prepared_source_changes_after_preserved_snapshots=transferred,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_PILOT_ORDER_BINDING_ONLY_NOT_ACTUAL_PILOT', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_pilot_generation_freeze_model_or_test_calls=0, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
