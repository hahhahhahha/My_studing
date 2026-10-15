"""Bounded standard-library-only author checks; never launches a model/GPU."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = Path(__file__).resolve().parent
SMOKE = ROOT/'visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py'
REWARD_GATE = ROOT/'visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py'
PILOT = ROOT/'scripts/visual_pilot_preparation.py'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def require_sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError('lowercase hexadecimal SHA256 required')
    return value


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False).encode('utf-8')
    return sha_bytes(raw)


def parse_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    function = next(row for row in tree.body if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef))
        and row.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def check_protected_bindings():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        current_source, raw_input = root/'implementation.py', root/'raw.csv'
        current_source.write_bytes(b'new reviewed implementation')
        raw_input.write_bytes(b'frozen raw bytes')
        prior = root/'prior.json'
        prior_map = {str(current_source): sha_bytes(b'old implementation'),
            str(raw_input): file_sha(raw_input)}
        implementation_map = {str(current_source): sha_bytes(b'old implementation')}
        prior.write_text(json.dumps(dict(protected_input_and_source_sha256=prior_map,
            new_implementation_sha256=implementation_map)), encoding='utf-8')
        prior_sha = file_sha(prior)
        namespace = dict(Path=Path, json=json, PRIOR=prior, PRIOR_SHA=prior_sha,
            file_sha=file_sha, require_sha=require_sha,
            source_lock=lambda: {str(current_source): file_sha(current_source)})
        function = parse_function(SMOKE, 'protected_bindings', namespace)
        locked = function()
        assert locked == {str(raw_input): file_sha(raw_input), str(prior): prior_sha}

        prior.write_text(json.dumps(dict(protected_input_and_source_sha256=prior_map,
            new_implementation_sha256=implementation_map | {str(root/'missing.py'): 'a'*64})), encoding='utf-8')
        namespace['PRIOR_SHA'] = file_sha(prior)
        try:
            function()
        except RuntimeError:
            pass
        else:
            raise AssertionError('uncovered historical implementation must be rejected')

        prior.write_text(json.dumps(dict(protected_input_and_source_sha256=prior_map,
            new_implementation_sha256=implementation_map)), encoding='utf-8')
        namespace['PRIOR_SHA'] = file_sha(prior)
        raw_input.write_bytes(b'mutated raw bytes')
        try:
            function()
        except RuntimeError:
            pass
        else:
            raise AssertionError('mutated frozen raw input must be rejected')
    return 3


def check_review_directory_gate():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        here = root/'engineering_attempts'/'20261012_s4'
        here.mkdir(parents=True)
        current_map = {'current_source.py': 'a'*64}
        def write_review(path):
            path.mkdir(parents=True, exist_ok=True)
            (path/'REVIEW_RESULT.json').write_text(json.dumps(dict(status='WARN', blockers=[],
                review_type='Type-A', reviewed_sources_sha256=current_map)), encoding='utf-8')
            (path/'REVIEW_RESPONSE.md').write_text('ARTIFICIAL AUTHOR FIXTURE ONLY', encoding='utf-8')
        namespace = dict(Path=Path, HERE=here, read=lambda path: json.loads(Path(path).read_text(encoding='utf-8')),
            sha=file_sha, own_source_map=lambda: current_map, require_sha=require_sha)
        function = parse_function(REWARD_GATE, 'require_review', namespace)
        old_dir = here/'independent_command_rereview'
        new_dir = here.parent/'20261013_20mhz_receiver_repair/independent_repair_rereview'
        write_review(old_dir); write_review(new_dir)
        assert set(function()) == {'JSON', 'MD'}
        assert set(function(new_dir)) == {'JSON', 'MD'}
        try:
            function(root/'unregistered_review')
        except ValueError:
            pass
        else:
            raise AssertionError('unregistered review path must be rejected')
    return 3


def make_receipt(method, schedule_sha, calibration, mobility, settings, upload_link):
    token = 'smoke-token-' + ('qla' if method == 'QLA-MAPPO' else 'ppo')
    condition = dict(calibration_source_sha256=calibration, mobility_source_sha256=mobility,
        schedule_sha256=schedule_sha, server_positions=[[i, i+1] for i in range(4)],
        calibration_scales=dict(upload_link=upload_link), smoke_settings=settings)
    condition_sha = digest(condition)
    stats = dict(loss=0.25, update_norm=0.5)
    stats_sha = digest(stats)
    checkpoint_sha = 'c'*64
    reward_semantics = 'QLA-shared-receipt' if method == 'QLA-MAPPO' else 'PPO-MV-A'
    counts = dict(episode_token=token, decisions=8, users=30, behavior_samples=240,
        action_interval_seconds=.4, reward_semantics=reward_semantics)
    paired_design_sha = 'b'*64
    commit = dict(episode_index=1, schedule_sha256=schedule_sha, decisions=8,
        data_role='calibration', episode_token=token, users=30, behavior_samples=240,
        action_interval_seconds=.4, reward_semantics=reward_semantics,
        update_statistics_sha256=stats_sha, checkpoint_sha256=checkpoint_sha)
    descriptor = dict(method=method, seed=1, episode_index=1, committed_environment_decisions=8,
        committed_behavior_samples=240, paired_design_sha256=paired_design_sha,
        data_role='calibration', episode_token=token, snapshot_only_no_resume=True)
    events = [
        dict(event='begin_collect', episode_index=1, schedule_sha256=schedule_sha, decisions=8, data_role='calibration'),
        dict(event='collection_closed', counts=counts),
        dict(event='begin_update', episode_token=token),
        dict(event='update_committed', episode_token=token),
        dict(event='checkpoint_attached', **descriptor, artifact_sha256=checkpoint_sha),
    ]
    ledger = dict(method=method, seed=1, phase='budget_complete_pending_validation',
        candidate_budget=dict(episode_decisions=[8], users=30, decision_seconds=.05, fixed_drain_seconds=30.),
        accounting=dict(environment_decisions=8, user_behavior_samples=240),
        paired_design_sha256=paired_design_sha, committed_episodes=[commit], current_episode=None,
        events=events, failure=None, checkpoint_selection_done=False, formal_protocol_frozen=False,
        model_execution_proven_by_this_ledger=False, automatic_resume_supported=False)
    return dict(method=method, schedule_sha256=schedule_sha, source_files_sha256={'source.py': 'f'*64},
        protected_input_and_source_sha256={}, plan_sha256='a'*64, semantic_identity_sha256='e'*64,
        input_condition=condition, input_condition_sha256=condition_sha,
        actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
        exact_loaded_parameters_and_deterministic_outputs=True, deterministic_Torch_RNG_unchanged=True,
        before_parameter_sha256='1'*64, after_parameter_sha256='2'*64,
        counts=counts, update_statistics=stats,
        checkpoint_archive=dict(artifact_sha256=checkpoint_sha, descriptor=descriptor),
        roundtrip_witness_files_sha256={'ACTUAL_ROUNDTRIP_INPUT_OUTPUT.npz': '3'*64,
            'ACTUAL_READONLY_TORCH_RNG.npz': '4'*64},
        calibration_source_sha256=calibration, mobility_source_sha256=mobility,
        ledger=ledger)


def check_actual_receipt_validator():
    settings = dict(users=30, decisions=8, decision_seconds=.05, fixed_drain_seconds=30.,
        network_condition_id='20MHz_per_user_orthogonal_simulation')
    upload_link = dict(bandwidth_hz=20_000_000.0)
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        calibration_dir, mobility_dir = base/'calibration', base/'KAIST'
        calibration_dir.mkdir(); mobility_dir.mkdir()
        calibration_path = calibration_dir/'service.csv'
        calibration_path.write_bytes(b'time,service\n0,1\n')
        calibration = {str(calibration_path): file_sha(calibration_path)}
        mobility = {}
        for index in range(30):
            path = mobility_dir/f'user_{index:02d}.txt'
            path.write_bytes(f'{index},0,0\n'.encode())
            mobility[str(path)] = file_sha(path)
        protected = calibration | mobility
        service_module = types.ModuleType('execution_service_catalog')
        service_module.DIRECTORIES = {'calibration': calibration_dir}
        mobility_module = types.ModuleType('physical_mobility')
        mobility_module.KAIST = mobility_dir
        previous_service = sys.modules.get('execution_service_catalog')
        previous_mobility = sys.modules.get('physical_mobility')
        sys.modules['execution_service_catalog'] = service_module
        sys.modules['physical_mobility'] = mobility_module
        try:
            namespace = dict(Path=Path, METHODS=('QLA-MAPPO', 'PPO-MV-A'), require_sha=require_sha,
                digest=digest, SMALL_SMOKE_SETTINGS=settings, NETWORK_UPLOAD_LINK=upload_link, math=math)
            validate = parse_function(PILOT, '_validate_small_smoke_plan_and_result', namespace)
            sources = {'source.py': 'f'*64}
            plan = dict(status='prepared_unexecuted_functional_smoke_only', methods=['QLA-MAPPO', 'PPO-MV-A'],
                source_files_sha256=sources, protected_input_and_source_sha256=protected,
                smoke_settings=settings, formal_protocol_frozen=False, test_access_allowed=False,
                actual_model_forward_and_optimizer_executed=False, actual_roundtrip_performed=False)
            plan_sha = 'a'*64
            rows = [make_receipt(method, '8'*64, calibration, mobility, settings, upload_link)
                for method in ('QLA-MAPPO', 'PPO-MV-A')]
            for row in rows:
                row['source_files_sha256'] = sources
                row['protected_input_and_source_sha256'] = protected
                row['plan_sha256'] = plan_sha
                row['semantic_identity_sha256'] = digest(plan)
            smoke = dict(status='actual_cuda_two_arm_smoke_pass', methods=['QLA-MAPPO', 'PPO-MV-A'],
                actual_model_forward_and_optimizer_executed=True, actual_roundtrip_performed=True,
                independent_complete_entry_review=False, formal_protocol_frozen=False,
                policy_training_cells=0, formal_policy_test_cells=0,
                source_files_sha256=sources, protected_input_and_source_sha256=protected,
                smoke_plan_path=str(base/'smoke-plan.json'), plan_sha256=plan_sha,
                semantic_identity_sha256=digest(plan), smoke_settings=settings,
                method_reports=rows, input_condition_sha256=rows[0]['input_condition_sha256'])
            validate(plan, smoke, smoke['smoke_plan_path'], plan_sha, sources, protected)

            rejected = 0
            def reject(change):
                nonlocal rejected
                bad = copy.deepcopy(smoke)
                change(bad)
                try:
                    validate(plan, bad, bad['smoke_plan_path'], plan_sha, sources, protected)
                except (RuntimeError, ValueError, TypeError, KeyError):
                    rejected += 1
                else:
                    raise AssertionError('invalid receipt fixture was accepted')

            reject(lambda value: value['method_reports'][0]['ledger']['committed_episodes'].clear())
            reject(lambda value: value['method_reports'][0]['ledger'].update(failure={'error':'fake'}, phase='failed'))
            reject(lambda value: value['method_reports'][0].update(schedule_sha256=None))
            reject(lambda value: value['method_reports'][0].update(input_condition_sha256=None))
            reject(lambda value: value['method_reports'][0].update(input_condition_sha256='z'*64))
            reject(lambda value: value['method_reports'][0]['ledger']['committed_episodes'][0].update(decisions=7))
            reject(lambda value: value['method_reports'][0]['ledger']['events'].pop())
            reject(lambda value: value['method_reports'][0]['ledger']['committed_episodes'].append(
                copy.deepcopy(value['method_reports'][0]['ledger']['committed_episodes'][0])))
            reject(lambda value: value['method_reports'][1].update(schedule_sha256='9'*64))
            reject(lambda value: value['method_reports'][0]['input_condition']['calibration_source_sha256'].clear())
            reject(lambda value: value['method_reports'][0].pop('input_condition_sha256'))
            reject(lambda value: value.update(plan_sha256='0'*64))
            if rejected != 12:
                raise AssertionError(f'expected 12 rejected receipt mutations, got {rejected}')
        finally:
            if previous_service is None: sys.modules.pop('execution_service_catalog', None)
            else: sys.modules['execution_service_catalog'] = previous_service
            if previous_mobility is None: sys.modules.pop('physical_mobility', None)
            else: sys.modules['physical_mobility'] = previous_mobility
    return rejected


def check_plan_boundary_source():
    tree = ast.parse(PILOT.read_text(encoding='utf-8'), filename=str(PILOT))
    cls = next(row for row in tree.body if isinstance(row, ast.ClassDef)
        and row.name == 'RealCalValidationPilotPreparation')
    methods = {row.name: row for row in cls.body if isinstance(row, ast.FunctionDef)}
    init_text = ast.unparse(methods['__init__'])
    verify_text = ast.unparse(methods['verify_boundary'])
    assert 'smoke_plan_path' in init_text and 'smoke_plan_sha' in init_text
    assert 'self._smoke_plan_path' in init_text and 'self._smoke_plan_sha' in init_text
    assert 'self._smoke_plan_identity' in init_text
    assert 'self._smoke_plan_path' in verify_text and 'self._smoke_plan_sha' in verify_text
    assert '_read_actual(self._smoke_plan_path, self._smoke_plan_sha, output)' in verify_text
    assert '_validate_small_smoke_plan_and_result' in verify_text
    assert 'merge_frozen_file_bindings(files, {str(Path(smoke_plan_path).resolve()): require_sha(smoke_plan_sha)})' in init_text
    return 8


def check_live_small_entry_identity():
    spec = __import__('importlib.util', fromlist=['util']).spec_from_file_location(
        'visual_model_smoke_small_revision2_author_check', SMOKE)
    module = __import__('importlib.util', fromlist=['util']).module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sources, protected = module.source_lock(), module.protected_bindings()
    if len(sources) != 209 or len(protected) != 210 or 'torch' in sys.modules:
        raise AssertionError('Expected exact 209-source/210-protected non-neural current identity')
    inventory = ''.join(path+'\0'+value.lower()+'\n' for path, value in sorted(sources.items()))
    protected_inventory = ''.join(path+'\0'+value.lower()+'\n' for path, value in sorted(protected.items()))
    assignment = ast.parse(PILOT.read_text(encoding='utf-8'), filename=str(PILOT))
    expected_additions = next(ast.literal_eval(row.value) for row in assignment.body
        if isinstance(row, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'SMALL_SMOKE_ADDITIONAL_BINDINGS' for target in row.targets))
    actual_additions = {Path(module.__file__).parent/name: file_sha(Path(module.__file__).parent/name)
        for name in ('visual_model_smoke_small.py', 'reward_smoke_small.py', 'reward_input_integrity.py',
            'FROZEN_BASELINE.json', 'README_SMALL_SMOKE_中文.md', 'USER_REQUEST.json')}
    if ({Path(path).name: value for path, value in expected_additions.items()}
            != {path.name: value for path, value in actual_additions.items()}):
        raise AssertionError('Pilot must pin the exact current six small-smoke additions')
    return dict(source_count=len(sources), protected_count=len(protected),
        source_inventory_sha256=sha_bytes(inventory.encode('utf-8')),
        protected_inventory_sha256=sha_bytes(protected_inventory.encode('utf-8')))


def main():
    for path in (SMOKE, REWARD_GATE, PILOT):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    protected_cases = check_protected_bindings()
    review_path_cases = check_review_directory_gate()
    rejected = check_actual_receipt_validator()
    boundary_checks = check_plan_boundary_source()
    identity = check_live_small_entry_identity()
    result = dict(schema='visual-20mhz-receiver-repair-author-check-v1', status='AUTHOR_FIXTURES_PASS',
        current_source_sha256={str(path): file_sha(path) for path in (SMOKE, REWARD_GATE, PILOT)},
        current_source_inventory=identity,
        actual_model_or_optimizer_or_cuda_calls=0, actual_smoke_plan_generated=False,
        actual_smoke_started=False, actual_pilot_started=False, formal_cells_started=0,
        protected_binding_positive_and_negative_cases=protected_cases,
        review_directory_positive_and_negative_cases=review_path_cases,
        actual_receipt_positive_cases=1, adversarial_receipt_cases_rejected=rejected,
        smoke_plan_boundary_source_assertions=boundary_checks,
        prior_review_failures_preserved=True, no_scientific_or_resource_policy_changes=True,
        note='All receipts and path-change cases in this file are artificial fixtures, never experiment evidence.')
    out = ATTEMPT/'AUTHOR_CHECKS_REVISION_2.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
