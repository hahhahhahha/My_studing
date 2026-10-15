"""Pure artificial environment identities, not a frozen protocol fixture."""
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
from visual_training_transactions import CommonBudgetCandidate, TrainingTransactionLedger, file_sha
from visual_upload_link import UploadLinkConfig
from visual_protocol_assembly import (compose_common_test_environment_table,
    paired_training_condition, RealVisualProtocolAssembler)

PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_154554_590570_gen/RESULT.json'
PRIOR_SHA = 'fb782b7e5368c89bdb49c60ef0b541bb819d4bebf46ebcf18515d24fc93faab6'


def refused(call):
    try:
        call()
    except (ValueError, TypeError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid artificial environment/missing actual authority accepted')


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_proto')
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
                raise AssertionError('No original retained timing CSV in artificial protocol check')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_test_timing):
            assert all(file_sha(Path(name)) == sha for name, sha in protected.items())
            common = tuple(tuple(f'{li*5+seed:064x}' for seed in range(1, 6)) for li in range(2))
            candidate = FixedTestMatrixCandidate(('ARTIFICIAL_LOW', 'ARTIFICIAL_HIGH'), (60., 120.), 8, 1., common, '0'*64)
            templates = {label: {str(seed): dict(schedule_sha256=common[li][si], role='test', profiles=list(PROFILES),
                edge_positions=[[0., 0.], [1., 0.], [0., 1.], [1., 1.]], mobility_start_seconds=1000.,
                broadcast_seconds=.1, nominal_delay_seconds=0., waiting_capacity=32, upload_link=asdict(UploadLinkConfig()),
                catalog_source_sha256={'EXPLICIT_ARTIFICIAL_NO_FILE': '0'*64},
                mobility_source_sha256={f'EXPLICIT_ARTIFICIAL_NO_MOBILITY_FILE_{u}': '0'*64 for u in range(30)})
                for si, seed in enumerate(candidate.seeds)} for li, label in enumerate(candidate.load_labels)}
            table = compose_common_test_environment_table(candidate, templates)
            assert table['descriptor_conditions'] == 30 and table['policy_cells_covered'] == 120
            assert table['actual_environment_or_policy_execution_proven'] == 0
            assert compose_common_test_environment_table(candidate, json.loads(json.dumps(templates, indent=7))) == table
            for cell in candidate.cells():
                b = table['descriptors'][cell['load_label']][str(cell['seed'])][str(cell['nominal_delay_seconds'])]
                assert b['schedule_sha256'] == cell['common_exogenous_schedule_sha256']
                assert b['nominal_delay_seconds'] == cell['nominal_delay_seconds']
            original = deepcopy(templates)
            table['descriptors']['ARTIFICIAL_LOW']['1']['0.0']['edge_positions'][0][0] = 999.
            assert templates == original
            bad = []
            for field, value in (('role', 'calibration'), ('schedule_sha256', 'f'*64), ('waiting_capacity', True),
                    ('waiting_capacity', 33), ('nominal_delay_seconds', .1), ('broadcast_seconds', .2),
                    ('mobility_start_seconds', float('nan')), ('profiles', ['CPU-1T']),
                    ('edge_positions', [[float('inf'), 0.]]*4), ('future_service_seconds', 0.01)):
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
            for value in bad:
                refused(lambda: compose_common_test_environment_table(candidate, value))
            budget = CommonBudgetCandidate((4, 4), fixed_drain_seconds=1.)
            schedules = common[0][:2]
            pair = paired_training_condition(1, budget, {'EXPLICIT_ARTIFICIAL': True}, {'EXPLICIT_ARTIFICIAL': True},
                {'EXPLICIT_ARTIFICIAL': True}, 'reset_each_independent_episode', original['ARTIFICIAL_LOW']['1'], schedules)
            for method in ('QLA-MAPPO', 'PPO-MV-A'):
                ledger = TrainingTransactionLedger(method, 1, budget, schedules, pair['common_condition_sha256'])
                assert ledger.paired_design_sha256 == pair['paired_design_sha256'] and ledger.phase == 'idle'
            refused(lambda: RealVisualProtocolAssembler(None, attempt/'NO_ACTUAL_PROTOCOL.json', '0'*64))
            assert not (attempt/'NO_ACTUAL_PROTOCOL.json').exists()
            tree = ast.parse((BASE/'scripts/visual_protocol_assembly.py').read_text(encoding='utf-8'))
            called = {node.func.attr for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
            assert not {'evaluate', 'act', 'step', 'backward', 'measured_pair_seconds', 'fitting_rows', 'calibration_scales'} & called
            assert 'torch' not in sys.modules and not timing_opens
            sources = {str(path): file_sha(path) for path in (Path(__file__), BASE/'scripts/visual_protocol_assembly.py',
                BASE/'README_VISUAL_PROTOCOL_ASSEMBLY_中文.md')}
            report = dict(at=datetime.now().astimezone().isoformat(), pure_artificial_environment_descriptor_checks_passed=True,
                explicit_artificial_templates=10, derived_artificial_conditions=30, artificial_policy_cell_descriptors_covered=120,
                invalid_hidden_truth_role_pairing_geometry_link_cases_rejected=len(bad),
                input_template_deep_copy_and_semantic_encoding_checked=True,
                pure_paired_condition_matches_unexecuted_ledger_formula=True,
                actual_assembler_constructor_or_protocol_writes=0, real_pilot_and_generation_permission_guards_exercised=False,
                actual_geometry_or_source_catalog_validation_exercised=False, actual_frozen_protocol_or_PASS_fixtures_created=False,
                actual_model_policy_queue_training_or_test_calls=0, actual_original_test_timing_CSV_opens=timing_opens,
                torch_imported=False, formal_protocol_frozen=False, independent_entry_review=False,
                original_protected_files_sha256=protected, new_implementation_sha256=sources)
            write_evidence(attempt/'RESULT.json', report)
            assert all(file_sha(Path(name)) == sha for name, sha in (protected | sources).items())
        print('ARTIFICIAL_ENVIRONMENT_DESCRIPTORS_ONLY_NOT_FROZEN_PROTOCOL', attempt)
    except Exception as error:
        write_evidence(attempt/'FAILURE.json', dict(error=repr(error), partial_evidence_preserved=True,
            actual_frozen_protocol_writes=0, attempted_original_test_timing_CSV_opens=timing_opens))
        raise


if __name__ == '__main__':
    main()
