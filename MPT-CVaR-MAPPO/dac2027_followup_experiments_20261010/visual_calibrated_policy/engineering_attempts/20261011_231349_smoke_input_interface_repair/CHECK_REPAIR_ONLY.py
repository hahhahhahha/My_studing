"""Author's pure/source fixture checks, NOT an independent verdict or smoke.

Never calls expected_plan/check/read_plan/run/worker, data catalogs, models,
resources, schedule generation or service execution. No raw files are changed.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path.insert(0, str(BASE/'scripts'))
from windows_runtime import initialize
initialize()
import ast
from datetime import datetime
import json
from visual_training_transactions import digest, file_sha
from visual_episode_collector import write_evidence
import visual_model_smoke as entry
from physical_mobility import KAIST, PhysicalMobility, TrainingWindow
from execution_service_catalog import DIRECTORIES

PACKET = BASE/'visual_calibrated_policy/entry_source_reviews/20261011_214124_683383_integrated_source_packet'
MANIFEST = PACKET/'CURRENT_PATHS_AND_BYTES.json'
MAP_KEYS = ('source_files_sha256', 'protected_input_and_source_sha256',
    'preserved_continuity_files_sha256', 'original_protocol_document_and_prior_review_paths_sha256')
manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
before = {}
for key in MAP_KEYS:
    for path, sha in manifest[key].items():
        if path in before and before[path] != sha:
            raise RuntimeError('Original maps conflict')
        before[path] = sha
rebound = {}
for relative in ('scripts/visual_model_smoke.py', 'README_VISUAL_MODEL_SMOKE_中文.md'):
    old_path = str(BASE/relative)
    saved = HERE/'SOURCE_BEFORE'/Path(relative).name
    if file_sha(saved) != before[old_path]:
        raise RuntimeError('Before-source does not match actual independent-reviewed bytes')
    rebound[old_path] = dict(original_sha256=before.pop(old_path), preserved_path=str(saved))
    before[str(saved)] = file_sha(saved)
for name in ('REVIEW_RESPONSE.md', 'REVIEW_RESULT.json'):
    path = PACKET/'repair_rereview'/name
    before[str(path)] = file_sha(path)
assert file_sha(PACKET/'repair_rereview/REVIEW_RESPONSE.md') == '98b319a6e525767e9082e3a5b5a8b334c1c9459d2762732c0f06bc6725281c12'
assert file_sha(PACKET/'repair_rereview/REVIEW_RESULT.json') == '8085ac67bc765e4e3ba4d8ca3673ef73baa2f3a5affc7a13682e80cd3d4c72bc'
assert all(file_sha(Path(path)) == sha for path, sha in before.items())

try:
    protected = entry.protected_bindings()  # Hash only original files; no timing CSV parsing.
    plan_fragment = dict(protected_input_and_source_sha256=protected)
    cal = {path: sha for path, sha in protected.items()
        if Path(path).parent == DIRECTORIES['calibration'] and Path(path).suffix == '.csv'}
    mobility = {path: sha for path, sha in protected.items()
        if Path(path).parent == KAIST and Path(path).suffix == '.txt'}
    assert len(cal) == 100 and len(mobility) == 30
    original_mobility = dict(mobility)
    original_cal = dict(cal)
    exact = entry.assert_consumed_input_maps(plan_fragment, cal, mobility)
    assert exact == dict(calibration_source_sha256=cal, mobility_source_sha256=mobility)
    entry.assert_consumed_input_maps(plan_fragment, dict(reversed(list(cal.items()))), dict(reversed(list(mobility.items()))))
    rejected = []
    def refuses(label, c, m, p=plan_fragment):
        try:
            entry.assert_consumed_input_maps(p, c, m)
        except RuntimeError:
            rejected.append(label)
        else:
            raise AssertionError(f'Fixture not rejected: {label}')
    first = next(iter(mobility))
    later = list(mobility)[-1]
    cfirst = next(iter(cal))
    early = str(KAIST/'!ARTIFICIAL_EARLIER_SORTING_GUARD_ONLY.txt')
    refuses('ARTIFICIAL earlier-sorting substitution, no raw file created', cal,
        {path: sha for path, sha in mobility.items() if path != later} | {early: mobility[first]})
    refuses('ARTIFICIAL added mobility entry, no raw file created', cal, mobility | {early: mobility[first]})
    refuses('Missing mobility entry', cal, {path: sha for path, sha in mobility.items() if path != first})
    refuses('Mobility changed hash mock', cal, mobility | {first: '0'*64})
    refuses('Alias spelling not a new registered path', cal,
        {path: sha for path, sha in mobility.items() if path != first} | {first.replace('\\', '/'): mobility[first]})
    refuses('Missing calibration entry', {path: sha for path, sha in cal.items() if path != cfirst}, mobility)
    refuses('Calibration changed hash mock', cal | {cfirst: '0'*64}, mobility)
    refuses('ARTIFICIAL added calibration path, no raw file created',
        cal | {str(DIRECTORIES['calibration']/'ARTIFICIAL_GUARD_ONLY.csv'): cal[cfirst]}, mobility)
    refuses('Wrong-role substitute, no validation data parsed',
        {path: sha for path, sha in cal.items() if path != cfirst} |
        {str(DIRECTORIES['validation']/Path(cfirst).name): cal[cfirst]}, mobility)
    refuses('Registered mobility list cannot be widened', cal, mobility,
        dict(protected_input_and_source_sha256=protected | {early: mobility[first]}))
    assert original_mobility == mobility and original_cal == cal

    # Actual tuple-producing function, explicitly ARTIFICIAL bounds; no real
    # trajectory or data-derived geometry is reconstructed by this fixture.
    producer = object.__new__(PhysicalMobility)
    producer.training_bounds = lambda window: ((1.25, -2.), (7., 5.))
    edges = producer.server_positions(TrainingWindow(0., 500.), 123, 4)
    assert type(edges) is tuple and all(type(point) is tuple for point in edges)
    try:
        edges.tolist()
    except AttributeError:
        old_interface_fault_reproduced_with_artificial_bounds = True
    else:
        raise AssertionError('Preserved tuple interface fault not reproduced')
    recorded = entry.server_positions_record(edges)
    assert recorded == json.loads(json.dumps(recorded, allow_nan=False))
    assert tuple(map(tuple, recorded)) == edges
    assert recorded is not edges and all(recorded[i] is not edges[i] for i in range(4))

    source = ast.parse((BASE/'scripts/visual_model_smoke.py').read_text(encoding='utf-8'))
    old = ast.parse((HERE/'SOURCE_BEFORE/visual_model_smoke.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    for name in old_functions.keys() - {'worker', 'run'}:
        assert ast.dump(functions[name]) == ast.dump(old_functions[name]), name
    old_settings = next(node for node in old.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SMOKE_SETTINGS' for t in node.targets))
    new_settings = next(node for node in source.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SMOKE_SETTINGS' for t in node.targets))
    assert ast.dump(old_settings) == ast.dump(new_settings)
    worker = functions['worker']
    calls = [node for node in ast.walk(worker) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    lines = lambda name: sorted(node.lineno for node in calls if node.func.id == name)
    assert len(lines('assert_consumed_input_maps')) == 3
    assert lines('assert_consumed_input_maps')[0] < lines('calibration_scales')[0] < lines('generate_schedule')[0] < lines('VisualReceiptEnvironment')[0]
    assert lines('assert_consumed_input_maps')[1] < lines('build_cuda_model')[0]
    assert lines('assert_consumed_input_maps')[-1] > lines('inspect_checkpoint_archive')[0]
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'edges' and node.func.attr == 'tolist'
        for node in ast.walk(worker))
    parent_calls = [node for node in ast.walk(functions['run']) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'assert_consumed_input_maps']
    assert len(parent_calls) == 2
    assert 'torch' not in sys.modules
    assert all(file_sha(Path(path)) == sha for path, sha in before.items())
    report = dict(at=datetime.now().astimezone().isoformat(), status='author_pure_source_boundary_self_check_passed_not_independent_review',
        original_independent_verdict='FAIL', independent_repair_rereview='PENDING',
        old_review_sha256=file_sha(PACKET/'repair_rereview/REVIEW_RESPONSE.md'),
        protected_original_inputs_sha256=protected, preserved_continuity_files_sha256=before,
        edited_original_paths_replaced_by_exact_before_snapshots=rebound,
        current_implementation_sha256={str(BASE/'scripts/visual_model_smoke.py'): file_sha(BASE/'scripts/visual_model_smoke.py'),
            str(BASE/'README_VISUAL_MODEL_SMOKE_中文.md'): file_sha(BASE/'README_VISUAL_MODEL_SMOKE_中文.md'), str(Path(__file__)): file_sha(Path(__file__))},
        current_source_files_sha256=entry.source_lock(),
        fixture_classification='ARTIFICIAL maps and bounds, no raw additions or measured replay; static call-order and readonly byte hashes',
        original_registered_calibration_files=len(cal), original_registered_mobility_files=len(mobility),
        exact_map_and_reordered_map_checks_passed=True, rejected_artificial_cases=rejected,
        old_tuple_fault_reproduced_only_on_artificial_bounds=old_interface_fault_reproduced_with_artificial_bounds,
        producer_tuple_values_exactly_preserved_in_JSON_lists=True, original_settings_and_unmodified_functions_AST_equal=True,
        static_worker_membership_boundaries=3, static_parent_received_map_boundaries=2,
        torch_imported=False, actual_model_or_optimizer_or_load_calls=0,
        old_check_or_expected_plan_or_read_plan_or_run_or_worker_called=False, plan_resealed=False,
        calibration_or_test_timing_CSVs_parsed=False, raw_file_or_membership_mutations=False,
        actual_pilot_or_schedule_generation_or_training_or_test=0, formal_launch_authorized=False,
        retained_warnings=['W3 original QLA finite checks occur after step', 'W4 pre-try/failure evidence and point-in-time non-reserved GPU gate'],
        Type_B_certification=False)
    write_evidence(HERE/'RESULT.json', report)
    assert all(file_sha(Path(path)) == sha for path, sha in before.items())
    print('AUTHOR_REPAIR_CHECK_ONLY_NOT_MODEL_SMOKE_OR_INDEPENDENT_PASS', HERE, 'rejected_cases', len(rejected))
except Exception as error:
    write_evidence(HERE/'FAILURE.json', dict(error=repr(error), actual_model_or_training_or_test=False,
        independent_verdict='NOT_GRANTED', auto_retry=False))
    raise
