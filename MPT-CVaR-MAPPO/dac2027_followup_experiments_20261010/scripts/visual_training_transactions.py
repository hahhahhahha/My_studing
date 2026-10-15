"""Candidate common-budget and immutable-checkpoint transactions.

Pure ledger checks are NOT model execution, frozen protocol, or training proof.
No launch, automatic retry, resume, checkpoint selection or test entry exists.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


METHODS = ('QLA-MAPPO', 'PPO-MV-A')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit lowercase SHA256 binding required')
    return value


@dataclass(frozen=True)
class CommonBudgetCandidate:
    episode_decisions: tuple[int, ...]
    users: int = 30
    decision_seconds: float = .05
    fixed_drain_seconds: float = 30.

    def __post_init__(self):
        if (type(self.episode_decisions) is not tuple or not self.episode_decisions
                or any(type(n) is not int or n < 1 for n in self.episode_decisions)
                or type(self.users) is not int or self.users < 1
                or type(self.decision_seconds) not in (int, float)
                or not math.isfinite(self.decision_seconds) or self.decision_seconds <= 0
                or type(self.fixed_drain_seconds) not in (int, float) or not math.isfinite(self.fixed_drain_seconds)
                or self.fixed_drain_seconds < self.decision_seconds):
            raise ValueError('Positive, explicitly partitioned common budget and fixed drain required')

    @property
    def total_decisions(self):
        return sum(self.episode_decisions)

    @property
    def total_behavior_samples(self):
        return self.users * self.total_decisions

    def accounting(self):
        generation = float(self.total_decisions * Fraction(str(self.decision_seconds)))
        drains = float(len(self.episode_decisions) * Fraction(str(self.fixed_drain_seconds)))
        return dict(environment_decisions=self.total_decisions,
            user_behavior_samples=self.total_behavior_samples, generation_seconds=generation,
            fixed_drain_seconds_sum=drains, independent_episode_horizon_sum=generation + drains,
            action_interval_settlement_seconds=float(len(self.episode_decisions) * Fraction(str(self.decision_seconds))),
            settlement_is_inside_fixed_drain=True, formal_budget_frozen=False)


@dataclass(frozen=True)
class ClosedEpisodeCounts:
    """Counts only, not an assertion that actions came from a real model."""
    episode_token: str
    decisions: int
    users: int
    behavior_samples: int
    action_interval_seconds: float
    reward_semantics: str


class TrainingTransactionLedger:
    def __init__(self, method, seed, budget, schedule_sha256, common_condition_sha256):
        if method not in METHODS or type(seed) is not int or seed < 0 or type(budget) is not CommonBudgetCandidate:
            raise ValueError('Declared method, seed and common candidate budget required')
        if type(schedule_sha256) is not tuple or len(schedule_sha256) != len(budget.episode_decisions):
            raise ValueError('One predeclared common schedule binding per episode required')
        self._method, self._seed, self._budget = method, seed, budget
        self._schedules = tuple(require_sha(value) for value in schedule_sha256)
        self._condition = require_sha(common_condition_sha256)
        self._phase, self._committed, self._current = 'idle', [], None
        self._tokens, self._events, self._failure = set(), [], None

    @property
    def method(self):
        return self._method

    @property
    def seed(self):
        return self._seed

    @property
    def budget(self):
        return self._budget

    @property
    def phase(self):
        return self._phase

    @property
    def paired_design_sha256(self):
        # Method deliberately excluded: both learning methods must match.
        return digest(dict(seed=self.seed, budget=asdict(self.budget),
            schedules=self._schedules, common_condition=self._condition))

    def _live(self):
        if self._phase in ('failed', 'budget_complete_pending_validation'):
            raise RuntimeError('Failed/completed ledger cannot append or automatically retry')

    def begin_episode(self, episode_index, schedule_sha256, data_role):
        self._live()
        if (self.phase != 'idle' or type(episode_index) is not int
                or episode_index != len(self._committed) + 1
                or episode_index > len(self.budget.episode_decisions)
                or data_role != 'calibration' or schedule_sha256 != self._schedules[episode_index - 1]):
            raise ValueError('Next ordered calibration-only episode on the predeclared common schedule required')
        self._current = dict(episode_index=episode_index, schedule_sha256=schedule_sha256,
            decisions=self.budget.episode_decisions[episode_index - 1], data_role=data_role)
        self._phase = 'collecting'
        self._events.append(dict(event='begin_collect', **self._current))

    def close_collection(self, counts):
        self._live()
        if self.phase != 'collecting':
            raise RuntimeError('Collection may close exactly once before an update')
        try:
            expected = 'QLA-shared-receipt' if self.method == 'QLA-MAPPO' else 'PPO-MV-A'
            if (type(counts) is not ClosedEpisodeCounts
                    or not isinstance(counts.episode_token, str) or not counts.episode_token.strip()
                    or counts.episode_token in self._tokens
                    or type(counts.decisions) is not int or type(counts.users) is not int
                    or type(counts.behavior_samples) is not int
                    or counts.decisions != self._current['decisions'] or counts.users != self.budget.users
                    or counts.behavior_samples != counts.users * counts.decisions
                    or not math.isfinite(counts.action_interval_seconds)
                    or not math.isclose(counts.action_interval_seconds,
                        float(counts.decisions * Fraction(str(self.budget.decision_seconds))), abs_tol=1e-12, rel_tol=0)
                    or counts.reward_semantics != expected):
                raise ValueError('All T by N behavior samples, physical action duration and method reward required')
            self._tokens.add(counts.episode_token)
            self._current.update(asdict(counts))
            self._phase = 'closed'
            self._events.append(dict(event='collection_closed', counts=asdict(counts)))
        except Exception as error:
            self.mark_failed(error)
            raise

    def begin_update(self, episode_token):
        self._live()
        if self.phase != 'closed' or episode_token != self._current['episode_token']:
            raise RuntimeError('Only the unique just-closed episode may start an optimizer transaction')
        # Burn before optimizer invocation; partial updates must never replay.
        self._phase = 'updating'
        self._events.append(dict(event='begin_update', episode_token=episode_token))

    def commit_update(self, episode_token, finite_update_statistics):
        self._live()
        if self.phase != 'updating' or episode_token != self._current['episode_token']:
            raise RuntimeError('Exactly one matching begun update may commit')
        try:
            if (type(finite_update_statistics) is not dict or not finite_update_statistics
                    or not all(isinstance(key, str) and type(value) in (int, float)
                        and math.isfinite(value) for key, value in finite_update_statistics.items())):
                raise ValueError('Actual finite optimizer statistics required; values here are not independent proof')
            self._current['update_statistics_sha256'] = digest(finite_update_statistics)
            self._committed.append(deepcopy(self._current))
            self._current = None
            self._phase = 'updated_awaiting_checkpoint'
            self._events.append(dict(event='update_committed', episode_token=episode_token))
        except Exception as error:
            self.mark_failed(error)
            raise

    def checkpoint_descriptor(self):
        self._live()
        if self.phase != 'updated_awaiting_checkpoint':
            raise RuntimeError('Checkpoint requires one completed optimizer transaction, not collection alone')
        return dict(method=self.method, seed=self.seed, episode_index=len(self._committed),
            committed_environment_decisions=sum(row['decisions'] for row in self._committed),
            committed_behavior_samples=sum(row['behavior_samples'] for row in self._committed),
            paired_design_sha256=self.paired_design_sha256, data_role='calibration',
            episode_token=self._committed[-1]['episode_token'], snapshot_only_no_resume=True)

    def attach_checkpoint(self, descriptor, artifact_sha256):
        expected = self.checkpoint_descriptor()
        if descriptor != expected:
            raise ValueError('Checkpoint method/seed/budget/schedule descriptor does not match committed update')
        require_sha(artifact_sha256)
        self._committed[-1]['checkpoint_sha256'] = artifact_sha256
        self._events.append(dict(event='checkpoint_attached', **expected, artifact_sha256=artifact_sha256))
        self._phase = ('budget_complete_pending_validation'
            if len(self._committed) == len(self.budget.episode_decisions) else 'idle')

    def mark_failed(self, error):
        if self.phase == 'failed':
            return
        self._failure = dict(error=repr(error), phase_before_failure=self.phase,
            current_episode=deepcopy(self._current), committed_episodes=deepcopy(self._committed))
        self._phase = 'failed'
        self._events.append(dict(event='failed', error=repr(error)))

    def snapshot(self):
        return deepcopy(dict(method=self.method, seed=self.seed, phase=self.phase,
            candidate_budget=asdict(self.budget), accounting=self.budget.accounting(),
            paired_design_sha256=self.paired_design_sha256, committed_episodes=self._committed,
            current_episode=self._current, events=self._events, failure=self._failure,
            checkpoint_selection_done=False, formal_protocol_frozen=False,
            model_execution_proven_by_this_ledger=False, automatic_resume_supported=False))


def inspect_checkpoint_archive(path, expected_descriptor, expected_artifact_sha256=None):
    """Byte integrity only. Never deserializes/executes a checkpoint payload."""
    path = Path(path)
    if path.is_symlink() or path.is_junction():
        raise ValueError('No checkpoint directory redirection')
    path = path.resolve()
    marker = path / 'CHECKPOINT_COMPLETE.json'
    if marker.is_symlink() or not marker.is_file() or (path / 'FAILURE.json').exists():
        raise RuntimeError('Complete immutable archive required; partial/failed archive refused')
    artifact_sha = file_sha(marker)
    if expected_artifact_sha256 is not None and artifact_sha != require_sha(expected_artifact_sha256):
        raise RuntimeError('Completed manifest changed relative to its externally bound fingerprint')
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    if manifest['descriptor'] != expected_descriptor:
        raise ValueError('Archive method, seed, budget or paired design mismatch')
    files = manifest['files_sha256']
    if type(files) is not dict or set(files) != {'MODEL_STATE.pt', 'OPTIMIZER_AND_RNG_STATE.pt', 'TRAINING_STATE.json'}:
        raise ValueError('Exact three-file checkpoint archive required, no path traversal')
    for name, expected in files.items():
        source = path / name
        require_sha(expected)
        if source.is_symlink() or not source.is_file() or file_sha(source) != expected:
            raise RuntimeError('Checkpoint file changed; no replacement or implicit restoration')
    return dict(artifact_sha256=artifact_sha, descriptor=deepcopy(expected_descriptor),
        payloads_loaded=False, byte_integrity_only=True)


def validation_checkpoint_reference(path, expected_descriptor, expected_artifact_sha256, data_role):
    if data_role != 'validation':
        raise ValueError('Validation-only reference here; no training re-entry or final test access')
    result = inspect_checkpoint_archive(path, expected_descriptor, require_sha(expected_artifact_sha256))
    result.update(data_role='validation', optimizer_updates_allowed=False,
        training_state_restore_allowed=False, checkpoint_selection_done=False)
    return result


def write_actual_controller_checkpoint(controller, ledger, new_directory, mv_stream=None):
    """Real export SOURCE ONLY; never invoked by the fixture check.

    No actual model, optimizer, CUDA or real model weights run in this wake.
    Complete reviewed formal trainer must integrate this after its update.
    """
    from visual_episode_collector import BASE, require_actual_cuda_controller, write_evidence
    from visual_learned_bindings import source_bindings
    from visual_mv_stream import PersistentMVStream
    require_actual_cuda_controller(controller)
    if type(ledger) is not TrainingTransactionLedger or controller.method != ledger.method:
        raise ValueError('Own actual controller and matching transaction ledger required')
    if controller.method == 'PPO-MV-A':
        if (type(mv_stream) is not PersistentMVStream or mv_stream.purpose != 'training' or mv_stream.failed
                or mv_stream.state().count != ledger.checkpoint_descriptor()['committed_behavior_samples']):
            raise ValueError('The continuously-held own training MV state must match committed budget')
    elif mv_stream is not None:
        raise ValueError('QLA cannot export a PPO MV statistic')
    descriptor = ledger.checkpoint_descriptor()
    root, path = (BASE / 'visual_calibrated_policy').resolve(), Path(new_directory).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('New checkpoint evidence must remain inside declared visual output')
    path.mkdir(parents=True, exist_ok=False)
    try:
        write_evidence(path / 'STARTED.json', dict(descriptor=descriptor, source_sha256=source_bindings()))
        torch = controller._torch
        if controller.method == 'QLA-MAPPO':
            controller.model.save_checkpoint(path / 'MODEL_STATE.pt')
        else:
            torch.save(dict(method=controller.method, state_dict=controller.model.state_dict()), path / 'MODEL_STATE.pt')
        torch.save(dict(optimizer=controller.optimizer.state_dict(), numpy_seed_not_automatically_restored=True,
            torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state_all()),
            path / 'OPTIMIZER_AND_RNG_STATE.pt')
        write_evidence(path / 'TRAINING_STATE.json', dict(ledger=ledger.snapshot(),
            MV_state=mv_stream.state() if mv_stream is not None else None,
            source_sha256=source_bindings(), snapshot_only_no_resume=True,
            virtual_H_and_complete_trainer_state_not_yet_integrated=True))
        names = ('MODEL_STATE.pt', 'OPTIMIZER_AND_RNG_STATE.pt', 'TRAINING_STATE.json')
        write_evidence(path / 'CHECKPOINT_COMPLETE.json', dict(descriptor=descriptor,
            files_sha256={name: file_sha(path / name) for name in names},
            model_payload_written_by_actual_controller=True, fresh_independent_audit_done=False))
        checked = inspect_checkpoint_archive(path, descriptor)
        ledger.attach_checkpoint(descriptor, checked['artifact_sha256'])
        return checked
    except Exception as error:
        ledger.mark_failed(error)
        controller._failed = True
        if mv_stream is not None:
            mv_stream.mark_failed(error)
        write_evidence(path / 'FAILURE.json', dict(error=repr(error), ledger=ledger.snapshot(),
            automatic_retry_allowed=False, partial_files_preserved=True))
        raise
