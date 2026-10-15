"""Pure transaction/count and explicitly non-model archive-byte fixtures."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime
import json
from pathlib import Path
import sys
import shutil

from visual_training_transactions import (CommonBudgetCandidate, ClosedEpisodeCounts,
    TrainingTransactionLedger, digest, file_sha, inspect_checkpoint_archive,
    validation_checkpoint_reference, write_actual_controller_checkpoint)

BASE = Path(__file__).resolve().parents[1]
PRIOR = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_101524_380119_mv_stream/RESULT.json'
PRIOR_SHA = '2e3b33e737255fff12eb505d89a612a8392c521aed6f9210baff17543fe41397'


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def refused(callback):
    try:
        callback()
    except (ValueError, RuntimeError, TypeError, AttributeError, FrozenInstanceError):
        return
    raise AssertionError('Transaction guard was not enforced')


def ledger(method='QLA-MAPPO', seed=1, budget=None):
    budget = budget or CommonBudgetCandidate((3, 2))
    schedules = tuple(digest({'explicit_synthetic_episode_schedule': n}) for n in range(len(budget.episode_decisions)))
    return TrainingTransactionLedger(method, seed, budget, schedules, digest({'explicit_synthetic_common_condition': True})), schedules


def counts(book, episode):
    n = book.budget.episode_decisions[episode-1]
    return ClosedEpisodeCounts(f'explicit-synthetic-token-{episode}', n, book.budget.users,
        n * book.budget.users, n * book.budget.decision_seconds,
        'QLA-shared-receipt' if book.method == 'QLA-MAPPO' else 'PPO-MV-A')


def begin_closed(book, schedules, episode):
    book.begin_episode(episode, schedules[episode-1], 'calibration')
    book.close_collection(counts(book, episode))


def finish_synthetic(book, schedules):
    for episode in range(1, len(schedules)+1):
        begin_closed(book, schedules, episode)
        token = counts(book, episode).episode_token
        book.begin_update(token)
        # Literal numbers only: no actual optimizer, behavior or learning.
        book.commit_update(token, {'explicit_synthetic_loss_placeholder': .25})
        descriptor = book.checkpoint_descriptor()
        book.attach_checkpoint(descriptor, digest({'nonfile_synthetic_checkpoint_placeholder': episode}))
    return book.snapshot()


def transaction_checks():
    candidate = CommonBudgetCandidate((1000,)*10)
    a = candidate.accounting()
    assert a['environment_decisions'] == 10000 and a['user_behavior_samples'] == 300000
    assert a['generation_seconds'] == 500 and a['fixed_drain_seconds_sum'] == 300
    assert a['independent_episode_horizon_sum'] == 800 and a['action_interval_settlement_seconds'] == .5
    refused(lambda: CommonBudgetCandidate((True,)))
    refused(lambda: CommonBudgetCandidate((3,), decision_seconds=True))
    refused(lambda: setattr(candidate, 'users', 31))
    full, schedules = ledger()
    ppo, _ = ledger('PPO-MV-A')
    assert full.paired_design_sha256 == ppo.paired_design_sha256
    different, _ = ledger(budget=CommonBudgetCandidate((4, 1)))
    assert different.paired_design_sha256 != full.paired_design_sha256
    different_seed, _ = ledger(seed=2)
    assert different_seed.paired_design_sha256 != full.paired_design_sha256
    changed_condition = TrainingTransactionLedger('QLA-MAPPO', 1, full.budget, schedules, digest('changed condition'))
    assert changed_condition.paired_design_sha256 != full.paired_design_sha256
    refused(lambda: full.begin_episode(1, schedules[0], 'validation'))
    refused(lambda: full.begin_episode(1, schedules[0], 'test'))
    refused(lambda: full.begin_episode(2, schedules[1], 'calibration'))
    refused(lambda: full.begin_episode(1, digest('different schedule'), 'calibration'))
    refused(full.checkpoint_descriptor)
    snapshots = [finish_synthetic(book, schedules) for book in (full, ppo)]
    for book in (full, ppo):
        assert book.phase == 'budget_complete_pending_validation'
        snap = book.snapshot()
        assert sum(e['decisions'] for e in snap['committed_episodes']) == 5
        assert sum(e['behavior_samples'] for e in snap['committed_episodes']) == 150
        refused(lambda: book.begin_episode(3, schedules[0], 'calibration'))
        refused(lambda: setattr(book, 'phase', 'idle'))
        snap['committed_episodes'].clear()
        assert len(book.snapshot()['committed_episodes']) == 2
    bad, schedules = ledger()
    bad.begin_episode(1, schedules[0], 'calibration')
    refused(lambda: bad.close_collection(ClosedEpisodeCounts('bad-short-buffer', 3, 30, 89, .15, 'QLA-shared-receipt')))
    assert bad.phase == 'failed'
    refused(lambda: bad.begin_episode(1, schedules[0], 'calibration'))
    failed, schedules = ledger()
    begin_closed(failed, schedules, 1)
    failed.begin_update(counts(failed, 1).episode_token)
    refused(lambda: failed.begin_update(counts(failed, 1).episode_token))
    failed.mark_failed(RuntimeError('Explicit partial-optimizer-failure fixture, no optimizer was called'))
    refused(lambda: failed.commit_update(counts(failed, 1).episode_token, {'loss': .25}))
    assert not failed.snapshot()['committed_episodes']
    nonfinite, schedules = ledger()
    begin_closed(nonfinite, schedules, 1)
    nonfinite.begin_update(counts(nonfinite, 1).episode_token)
    refused(lambda: nonfinite.commit_update(counts(nonfinite, 1).episode_token, {'loss': float('nan')}))
    assert nonfinite.phase == 'failed'
    wrong_reward, schedules = ledger()
    wrong_reward.begin_episode(1, schedules[0], 'calibration')
    refused(lambda: wrong_reward.close_collection(ClosedEpisodeCounts('wrong-reward', 3, 30, 90, .15, 'PPO-MV-A')))
    assert wrong_reward.phase == 'failed'
    duplicate, schedules = ledger()
    begin_closed(duplicate, schedules, 1)
    original_token = counts(duplicate, 1).episode_token
    refused(lambda: duplicate.close_collection(counts(duplicate, 1)))
    duplicate.begin_update(original_token)
    duplicate.commit_update(original_token, {'explicit_synthetic_placeholder': .25})
    duplicate.attach_checkpoint(duplicate.checkpoint_descriptor(), digest('placeholder-only'))
    duplicate.begin_episode(2, schedules[1], 'calibration')
    refused(lambda: duplicate.close_collection(ClosedEpisodeCounts(original_token, 2, 30, 60, .1, 'QLA-shared-receipt')))
    assert duplicate.phase == 'failed'
    return dict(candidate_accounting=a, two_methods_explicit_synthetic_count_snapshots=snapshots,
        wrong_role_schedule_order_truncated_buffer_duplicate_and_fail_stop_checked=True,
        paired_design_excludes_method_but_includes_seed_partition_and_common_schedule=True)


def archive_checks(attempt):
    book, schedules = ledger(budget=CommonBudgetCandidate((3,)))
    begin_closed(book, schedules, 1)
    token = counts(book, 1).episode_token
    book.begin_update(token)
    book.commit_update(token, {'literal_placeholder': .25})
    descriptor = book.checkpoint_descriptor()
    path = attempt/'EXPLICIT_NONMODEL_ARCHIVE_FIXTURE'
    path.mkdir()
    for name in ('MODEL_STATE.pt', 'OPTIMIZER_AND_RNG_STATE.pt'):
        # Deliberately invalid Torch files; ONLY byte integrity is exercised.
        with (path/name).open('xb') as stream:
            stream.write(b'EXPLICIT TEXT BYTE FIXTURE. NOT TORCH, MODEL, OPTIMIZER OR LEARNED WEIGHTS.\n')
    save(path/'TRAINING_STATE.json', dict(explicit_synthetic_count_fixture=True, ledger=book.snapshot()))
    files = {name: file_sha(path/name) for name in ('MODEL_STATE.pt', 'OPTIMIZER_AND_RNG_STATE.pt', 'TRAINING_STATE.json')}
    save(path/'CHECKPOINT_COMPLETE.json', dict(descriptor=descriptor, files_sha256=files,
        model_payload_written_by_actual_controller=False, fixture_only_not_real_checkpoint=True))
    checked = inspect_checkpoint_archive(path, descriptor)
    reference = validation_checkpoint_reference(path, descriptor, checked['artifact_sha256'], 'validation')
    assert reference['payloads_loaded'] is False and reference['optimizer_updates_allowed'] is False
    refused(lambda: validation_checkpoint_reference(path, descriptor, checked['artifact_sha256'], 'test'))
    refused(lambda: validation_checkpoint_reference(path, descriptor, checked['artifact_sha256'], 'calibration'))
    refused(lambda: inspect_checkpoint_archive(path, descriptor | {'seed': 2}, checked['artifact_sha256']))
    refused(lambda: inspect_checkpoint_archive(path, descriptor | {'method': 'PPO-MV-A'}, checked['artifact_sha256']))
    partial = attempt/'EXPLICIT_PARTIAL_NONMODEL_FIXTURE'
    partial.mkdir()
    refused(lambda: inspect_checkpoint_archive(partial, descriptor))
    changed = attempt/'EXPLICIT_TAMPERED_NONMODEL_FIXTURE'
    shutil.copytree(path, changed)  # new fixture; original fixture bytes retained
    with (changed/'MODEL_STATE.pt').open('ab') as stream:
        stream.write(b'explicit tamper fixture')
    refused(lambda: inspect_checkpoint_archive(changed, descriptor, checked['artifact_sha256']))
    # New separate manifest simulates tamper of both file and its inside hash.
    manifest = dict(descriptor=descriptor, files_sha256=files | {'MODEL_STATE.pt': file_sha(changed/'MODEL_STATE.pt')},
        fixture_only_not_real_checkpoint=True)
    copied_original = changed/'PRESERVED_ORIGINAL_COMPLETE.json'
    (changed/'CHECKPOINT_COMPLETE.json').rename(copied_original)
    save(changed/'CHECKPOINT_COMPLETE.json', manifest)
    refused(lambda: validation_checkpoint_reference(changed, descriptor, checked['artifact_sha256'], 'validation'))
    # Wrong object guard executes; real constructor/serialization does not.
    refused(lambda: write_actual_controller_checkpoint(object(), book, attempt/'must-not-be-created'))
    assert not (attempt/'must-not-be-created').exists()
    tree = ast.parse((BASE/'scripts/visual_training_transactions.py').read_text(encoding='utf-8'))
    export = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_actual_controller_checkpoint')
    calls = [n for n in ast.walk(export) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('load', 'load_checkpoint', 'update', 'update_closed_episode') for n in calls)
    return dict(nonmodel_placeholder_archive_byte_guards_passed=True,
        complete_marker_external_sha_required_for_validation=True,
        wrong_role_method_seed_partial_tamper_guards_passed=True,
        actual_controller_checkpoint_serialization_executed=False,
        fixture_manifest_sha256=checked['artifact_sha256'])


def main():
    attempt = BASE/'visual_calibrated_policy/engineering_attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_training_transactions')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and file_sha(PRIOR) == PRIOR_SHA
        old = json.loads(PRIOR.read_text(encoding='utf-8'))
        protected = old['protected_input_and_source_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): PRIOR_SHA}
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
        checks = transaction_checks()
        archive = archive_checks(attempt)
        assert 'torch' not in sys.modules
        assert all(file_sha(Path(path)) == value for path, value in protected.items())
        sources = [Path(__file__), BASE/'scripts/visual_training_transactions.py', BASE/'README_VISUAL_TRAINING_TRANSACTIONS_中文.md']
        report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
            pure_synthetic_count_and_nonmodel_byte_checks_only=True, runtime_python=sys.version.split()[0],
            transactions=checks, archive=archive, protected_input_and_source_sha256=protected,
            new_implementation_sha256={str(path): file_sha(path) for path in sources},
            torch_imported=False, actual_model_or_optimizer_or_real_checkpoint_calls=0,
            actual_training_coordinator_executed=False, common_budget_formally_frozen=False,
            checkpoint_selection_done=False, validation_policy_evaluations=0, test_measurements_accessed=False,
            policy_training_cells=0, formal_policy_test_cells=0, independent_complete_entry_review=False,
            network_parameters_changed=False)
        report['prior_success_and_source_snapshot_preserved'] = 'visual_calibrated_policy/engineering_attempts/20261011_104241_514011_training_transactions'
        save(attempt/'RESULT.json', report)
        print('TRAINING_TRANSACTION_ENGINEERING_GUARDS_OK_NOT_MODEL_EXECUTION', attempt)
    except Exception as error:
        save(attempt/'FAILURE.json', dict(error=repr(error), at=datetime.now().astimezone().isoformat(),
            actual_model_or_optimizer_or_real_checkpoint_calls=0, policy_training_cells=0, formal_policy_test_cells=0))
        raise


if __name__ == '__main__':
    main()
