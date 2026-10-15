"""SOURCE/ARTIFICIAL only. Never construct a completed pilot or PASS fixture."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from check_visual_pilot_episode import refused
from check_visual_pilot_freeze_contract import artificial_selection
from visual_episode_collector import BASE, write_evidence
from visual_pilot_export import (RealCompletedPilotExporter, EXPORT_FIELDS,
    pilot_export_scientific_identity, merge_complete_export_bindings,
    selected_pilot_reference_bindings)
from visual_pilot_preparation import PilotPreparationArtifactBindings
from visual_training_transactions import digest, file_sha

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_204831_956531_pilot_freeze_contract/RESULT.json'
PRIOR_SHA = '34912e1e3b67dbc7c5c86591574b231986eb3634ff4492c814ae3989cb74f0ab'
EDITED = BASE/'scripts/retained_test_schedule_generation.py'
BEFORE = Path('X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_211043_070307_pilot_export_reader_source_before')/'retained_test_schedule_generation.py'
BEFORE_SHA = 'cbf63ac2f253225a85e7a65a6313968af871bc57847629825aa816c16f432c17'


def call_names(tree):
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
        for node in ast.walk(tree) if isinstance(node, ast.Call)}


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_pilot_export')
    attempt.mkdir(parents=True, exist_ok=False)
    timing_opens = []
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert file_sha(PRIOR) == PRIOR_SHA and file_sha(BEFORE) == BEFORE_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        assert protected.pop(str(EDITED)) == BEFORE_SHA
        protected[str(BEFORE)] = BEFORE_SHA
        original_open = Path.open
        raw = BASE.parent/'dac_visual_calibration_20261010_v2/measurements/test'
        def no_test_timing(path, *args, **kwargs):
            if path.suffix.lower() == '.csv' and path.resolve().is_relative_to(raw.resolve()):
                timing_opens.append(str(path))
                raise AssertionError('No test timing opens in source-only export check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == value for name, value in protected.items())
            a, b = 'a'*64, 'b'*64
            identity = pilot_export_scientific_identity(a, b)
            # Encoding/byte/path fields are absent from this semantic identity.
            encoded = json.loads(json.dumps({'selection': b, 'registration': a}, indent=8))
            assert identity == pilot_export_scientific_identity(encoded['registration'], encoded['selection'])
            assert identity != pilot_export_scientific_identity(a, 'c'*64)
            assert identity != pilot_export_scientific_identity('c'*64, b)
            refused(lambda: pilot_export_scientific_identity('not SHA', b))
            maps = ({'ART_ORIGINAL_RAW': a}, {'ART_EXTRA_INDEPENDENT_EVIDENCE': b}, {'ART_SELECTION': 'c'*64})
            merged = merge_complete_export_bindings(*maps)
            assert merged == dict(ART_ORIGINAL_RAW=a, ART_EXTRA_INDEPENDENT_EVIDENCE=b, ART_SELECTION='c'*64)
            merged['ART_ORIGINAL_RAW'] = b
            assert maps[0]['ART_ORIGINAL_RAW'] == a
            refused(lambda: merge_complete_export_bindings({'ART_ORIGINAL_RAW': a}, {'ART_ORIGINAL_RAW': b}))
            bad_maps = ({}, {'ART': 'not SHA'}, {'': a}, {3: a})
            for row in bad_maps:
                refused(lambda: merge_complete_export_bindings(row))
            refused(lambda: merge_complete_export_bindings())
            selection = artificial_selection()  # Unfrozen artificial descriptors only.
            refs = selected_pilot_reference_bindings(selection)
            assert refs and selection == artificial_selection()
            missing = deepcopy(refs); missing.pop(next(iter(missing)))
            assert any(missing.get(name) != value for name, value in refs.items())
            rebound = deepcopy(refs); rebound[next(iter(rebound))] = 'd'*64
            assert any(rebound.get(name) != value for name, value in refs.items())
            refused(lambda: RealCompletedPilotExporter(None))
            source = (BASE/'scripts/visual_pilot_export.py').read_text(encoding='utf-8')
            tree = ast.parse(source)
            assert not call_names(tree) & {'build_cuda_model', 'witness', 'act', 'collect_once', 'register_once',
                'train', 'step', 'load_checkpoint', 'generate_and_register', 'fit', 'assemble_once', 'seal_all'}
            reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'require_completed_pilot_export_inputs')
            assert not call_names(reader) & {'_write_once', '_claim_grid_once', 'write_evidence', 'mkdir', 'export_once'}
            for text in ('type(same_actual_complete_evidence) is not RealCompletePilotFreezeEvidence',
                'self._started or self._failed or self._finished', 'reviewed_inputs_for_separate_export',
                'complete_input_files_sha256', 'selected_pilot_reference_bindings(selection)',
                "'gpt-6-astra'", "'xhigh'", "'Type-A'", 'requirement_assessments',
                'additional_independent_checks_performed_not_same_foundation_replay_only',
                'metadata_request_generation_authorized=False', 'formal_training_or_test_launch_authorized=False',
                'immutable_formal_protocol_frozen=False', 'ten_learned_training_or_120_test_completed=False',
                "path/'FAILURE.json'", "lock.stem+'_FAILURE.json'", 'automatic_retry_append_or_partial_merge_allowed=False'):
                assert text in source
            # Actual binding producer uses actual_smoke, not actual_model_smoke.
            binding_fields = set(PilotPreparationArtifactBindings.__dataclass_fields__)
            attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute) and node.value.attr == 'bindings'}
            assert attrs <= binding_fields and {'actual_smoke_path', 'actual_smoke_sha256'} <= attrs
            record = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'dict'
                and any(kw.arg == 'status' and isinstance(kw.value, ast.Name) and kw.value.id == 'EXPORT_STATUS' for kw in node.keywords))
            produced = {kw.arg for kw in record.keywords if kw.arg is not None}
            produced |= {key+'_'+suffix for key in ('source_repair_review', 'pilot_entry_review', 'human_direction_record')
                for suffix in ('path', 'sha256')}
            assert produced == EXPORT_FIELDS
            generator_source = EDITED.read_text(encoding='utf-8')
            generator_tree = ast.parse(generator_source)
            preparation = next(node for node in generator_tree.body if isinstance(node, ast.FunctionDef)
                and node.name == '_require_generation_preparation')
            names = call_names(preparation)
            assert {'require_completed_pilot_export_inputs', 'merge_complete_export_bindings'} <= names
            for text in ('complete_pilot_input_files_sha256', 'pilot_export_scientific_identity_sha256',
                'review.get(\'protected_input_and_source_sha256\') != protected', 'current != self._inputs'):
                assert text in generator_source
            # Original sampling/class code preserved byte-for-byte; only the
            # metadata preparation gate is repaired, not scientific settings.
            old_source = BEFORE.read_text(encoding='utf-8')
            before_gate = 'def _require_generation_preparation'
            after_gate = 'class RealRetainedScheduleGenerator'
            assert generator_source.split(before_gate)[0] == old_source.split(before_gate)[0]
            assert generator_source.split(after_gate)[1] == old_source.split(after_gate)[1]
            assembler = (BASE/'scripts/visual_protocol_assembly.py').read_text(encoding='utf-8')
            assert '_require_generation_preparation(actual_generator.design, actual_generator.bindings)' in assembler
            assert "generated['input_files_sha256']" in assembler and 'self.inputs.items()' in assembler
            assert 'torch' not in sys.modules and not timing_opens
            current = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_pilot_export.py',
                EDITED, BASE/'README_VISUAL_PILOT_EXPORT_中文.md')}
            write_evidence(attempt/'RESULT.json', dict(at=datetime.now().astimezone().isoformat(),
                source_and_artificial_export_identity_input_reference_and_static_reader_checks_passed=True,
                artificial_semantic_identity_cases=3, artificial_invalid_identity_cases=1,
                artificial_invalid_or_empty_or_rebound_map_cases=6, artificial_reference_omission_or_rebind_cases=2,
                artificial_maps_not_actual_pilot_evidence_or_assessment=True,
                static_exact_export_producer_reader_fields_and_original_smoke_binding_names_match=True,
                static_full_original_input_propagation_through_generator_and_assembler_checked=True,
                original_request_sampling_recipe_and_generator_class_source_unchanged=True,
                edited_preparation_source_before_snapshot_path=str(BEFORE), edited_preparation_source_before_sha256=BEFORE_SHA,
                absent_actual_contract_early_type_rejection_exercised=True,
                actual_exporter_reader_or_generator_constructor_export_IO_or_runtime_guards_exercised=False,
                actual_independent_review_user_smoke_MODEL_pilot_completed_or_PASS_fixtures_fabricated=False,
                actual_pilot_export_request_generation_training_or_test_calls=0, torch_imported=False,
                actual_test_timing_CSV_opens=timing_opens, independent_audit=False, environment_or_full_pilot_ready=False,
                original_protected_files_sha256=protected, new_implementation_sha256=current))
            assert all(file_sha(Path(name)) == value for name, value in (protected | current).items())
        print('ARTIFICIAL_EXPORT_IDENTITY_REFERENCE_STATIC_READER_CHECK_ONLY', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), actual_pilot_export_generation_model_or_review_calls=0,
            partial_evidence_preserved=True, attempted_actual_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
