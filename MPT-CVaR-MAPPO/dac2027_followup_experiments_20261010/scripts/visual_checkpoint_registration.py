"""Prepared common final-checkpoint registration, never test-driven ranking.

Pure descriptor checks are not weights, model execution or a freeze certificate.
The actual registrar requires the external pilot/smoke/complete-entry gates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path

from visual_episode_collector import BASE, jsonable, require_actual_cuda_controller, write_evidence
from visual_training_coordinator import (FrozenRuntimeArtifactBindings, RealReadonlyValidationEvaluator,
    require_reviewed_runtime, actual_parameter_sha256)
from visual_training_transactions import (METHODS, CommonBudgetCandidate, digest, file_sha,
    require_sha, inspect_checkpoint_archive)


@dataclass(frozen=True)
class FinalCheckpointRuleCandidate:
    budget: CommonBudgetCandidate
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        if (type(self.budget) is not CommonBudgetCandidate or type(self.seeds) is not tuple
                or any(type(seed) is not int for seed in self.seeds) or self.seeds != (1, 2, 3, 4, 5)):
            raise ValueError('Same explicitly partitioned budget and original five paired seeds required')

    def definition(self):
        return dict(strategy='final_checkpoint_after_common_budget_no_outcome_ranking',
            methods=list(METHODS), seeds=list(self.seeds), common_budget=jsonable(asdict(self.budget)),
            checkpoint_episode_index=len(self.budget.episode_decisions), validation_role='validation',
            validation_for_completeness_not_performance_filter=True, test_used_for_selection=False,
            negative_validation_outcomes_retained=True, all_ten_registered_before_test_entry=True)

    def prepared_record(self):
        return dict(status='candidate_not_pilot_frozen_not_weight_selection',
            rule=self.definition(), rule_semantic_sha256=digest(self.definition()),
            actual_checkpoint_selection_done=False, actual_validation_evaluations=0,
            test_launch_authorized=False)


def final_descriptor_from_closed_budget(snapshot, rule):
    """Pure consistency extraction only; does not prove real updates/weights."""
    if type(rule) is not FinalCheckpointRuleCandidate or type(snapshot) is not dict:
        raise TypeError('Explicit common rule and completed ledger description required')
    method, seed = snapshot.get('method'), snapshot.get('seed')
    rows = snapshot.get('committed_episodes', [])
    if (method not in METHODS or type(seed) is not int or seed not in rule.seeds
            or snapshot.get('phase') != 'budget_complete_pending_validation'
            or snapshot.get('failure') is not None or snapshot.get('current_episode') is not None
            or digest(snapshot.get('candidate_budget')) != digest(jsonable(asdict(rule.budget)))
            or len(rows) != len(rule.budget.episode_decisions)):
        raise ValueError('Only the complete ordered common training budget, not partial/failed/best epoch')
    seen_tokens = set()
    semantics = 'QLA-shared-receipt' if method == 'QLA-MAPPO' else 'PPO-MV-A'
    for index, (row, decisions) in enumerate(zip(rows, rule.budget.episode_decisions), 1):
        if (type(row.get('episode_index')) is not int or row.get('episode_index') != index or row.get('data_role') != 'calibration'
                or type(row.get('decisions')) is not int or row['decisions'] != decisions
                or row.get('users') != rule.budget.users or type(row.get('users')) is not int
                or row.get('behavior_samples') != decisions*rule.budget.users
                or type(row.get('behavior_samples')) is not int
                or row.get('reward_semantics') != semantics
                or not isinstance(row.get('episode_token'), str) or not row['episode_token']
                or row['episode_token'] in seen_tokens
                or type(row.get('action_interval_seconds')) not in (int, float)
                or not math.isclose(row['action_interval_seconds'],
                    float(decisions*Fraction(str(rule.budget.decision_seconds))), rel_tol=0., abs_tol=1e-12)):
            raise ValueError('Missing/reordered/truncated/foreign-role training segment')
        seen_tokens.add(row['episode_token'])
        for name in ('schedule_sha256', 'update_statistics_sha256', 'checkpoint_sha256'):
            require_sha(row[name])
    return dict(method=method, seed=seed, episode_index=len(rows),
        committed_environment_decisions=rule.budget.total_decisions,
        committed_behavior_samples=rule.budget.total_behavior_samples,
        paired_design_sha256=require_sha(snapshot['paired_design_sha256']), data_role='calibration',
        episode_token=rows[-1]['episode_token'], snapshot_only_no_resume=True)


def validate_validation_record(record, descriptor, manifest_sha, parameter_sha, schedule_sha):
    """Pure record consistency; preserve adverse/drop/unfinished, no ranking."""
    ref = record.get('reference', {})
    if (ref.get('data_role') != 'validation' or ref.get('descriptor') != descriptor
            or ref.get('artifact_sha256') != require_sha(manifest_sha)
            or record.get('parameter_sha256_before_and_after') != require_sha(parameter_sha)
            or record.get('schedule_sha256') != require_sha(schedule_sha)
            or record.get('initial_online_H') != 0.
            or record.get('optimizer_update_called') is not False
            or record.get('training_H_or_MV_statistics_restored') is not False
            or record.get('torch_rng_changed') is not False
            or record.get('checkpoint_selection_done') is not False):
        raise ValueError('Exact final weights and registered readonly validation, never test selection')
    final = record['final']
    counts, latencies = final['tasks'], final['uncapped_latency_seconds']
    if final.get('schedule_sha256') != schedule_sha:
        raise ValueError('Final validation schedule identity mismatch')
    if (any(type(counts.get(name)) is not int or counts[name] < 0
            for name in ('generated', 'accepted', 'completed', 'dropped', 'unfinished', 'inflight_upload'))
            or counts['generated'] != counts['completed']+counts['dropped']+counts['unfinished']
            or counts['completed'] > counts['accepted']
            or counts['inflight_upload'] > counts['unfinished']
            or counts['accepted']+counts['dropped']+counts['inflight_upload'] != counts['generated']
            or len(latencies) != counts['completed']
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in latencies)
            or list(counts['latency_seconds']) != list(latencies)):
        raise ValueError('Complete uncapped completion/drop/unfinished accounting required')
    # Zero completions and long latencies are retained rather than silently
    # rejecting a checkpoint for unfavorable performance.
    return dict(generated=counts['generated'], completed=counts['completed'],
        dropped=counts['dropped'], unfinished=counts['unfinished'],
        uncapped_latency_seconds=list(latencies), negative_outcomes_retained=True,
        performance_filter_applied=False)


def check_complete_paired_grid(entries, rule):
    """Pure ten-row shape/identity check, not real checkpoint freezing."""
    expected = {(method, seed) for method in METHODS for seed in rule.seeds}
    keys = [(row['descriptor']['method'], row['descriptor']['seed']) for row in entries]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('All two methods by five seeds exactly once before test entry')
    for seed in rule.seeds:
        pair = [row for row in entries if row['descriptor']['seed'] == seed]
        for row in pair:
            d = row['descriptor']
            if (any(type(d[name]) is not int for name in ('seed', 'episode_index', 'committed_environment_decisions', 'committed_behavior_samples'))
                    or d['episode_index'] != len(rule.budget.episode_decisions)
                    or d['committed_environment_decisions'] != rule.budget.total_decisions
                    or d['committed_behavior_samples'] != rule.budget.total_behavior_samples
                    or d['data_role'] != 'calibration' or d['snapshot_only_no_resume'] is not True):
                raise ValueError('Only common final-budget descriptors')
        if (pair[0]['descriptor']['paired_design_sha256'] != pair[1]['descriptor']['paired_design_sha256']
                or pair[0]['validation_schedule_sha256'] != pair[1]['validation_schedule_sha256']):
            raise ValueError('Paired methods need identical training design and ordered validation schedules')
    return dict(paired_seeds=list(rule.seeds), descriptor_rows=len(entries),
        actual_weights_or_model_execution_proven=False, test_launch_authorized=False)


def _read_evidence(path, sha):
    path = Path(path)
    root = (BASE/'visual_calibrated_policy').resolve()
    if (path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root)
            or not path.is_file() or (path.parent/'FAILURE.json').exists()):
        raise ValueError('Bound unfailed evidence inside the intended visual output required')
    if file_sha(path) != require_sha(sha):
        raise RuntimeError('Completed training/validation evidence bytes changed')
    return json.loads(path.read_text(encoding='utf-8'))


class RealFinalCheckpointRegistrar:
    """SOURCE ONLY. No constructor/register/seal has succeeded or executed."""
    def __init__(self, rule, bindings):
        if type(rule) is not FinalCheckpointRuleCandidate or type(bindings) is not FrozenRuntimeArtifactBindings:
            raise TypeError('Common rule and external actual runtime bindings required')
        self.protocol = require_reviewed_runtime(bindings)
        if (self.protocol.get('checkpoint_selection_rule') != rule.definition()
                or self.protocol.get('checkpoint_rule_registered_before_training') is not True
                or self.protocol['source_files_sha256'].get(str(Path(__file__))) != file_sha(Path(__file__))):
            raise RuntimeError('The rule and this source must be pilot-frozen/reviewed before training')
        self.rule, self.bindings = rule, bindings
        self._entries, self._inputs, self._failed, self._sealed = {}, {}, False, False

    def register(self, training_path, training_sha, validation_evidence):
        """Consume actual saved outputs with their completed actual evaluators.

        validation_evidence: ordered (RealReadonlyValidationEvaluator, path,
        external byte SHA) tuples. No model/evaluation is run by this method.
        """
        if self._failed or self._sealed:
            raise RuntimeError('Registrar is fail-stop/seal-once, no automatic append/retry')
        try:
            require_reviewed_runtime(self.bindings)
            training = _read_evidence(training_path, training_sha)
            descriptor = final_descriptor_from_closed_budget(training['ledger'], self.rule)
            key = descriptor['method'], descriptor['seed']
            if key in self._entries:
                raise RuntimeError('Duplicate final checkpoint cell')
            if descriptor['paired_design_sha256'] != self.protocol['paired_design_by_seed'][str(key[1])]:
                raise ValueError('Registered paired training design differs')
            records = training['episodes']
            if len(records) != len(self.rule.budget.episode_decisions):
                raise ValueError('Every committed episode archive must remain present')
            training_root = Path(training_path).resolve().parent
            input_files = {str(Path(training_path)): training_sha}
            for index, row in enumerate(records, 1):
                expected = deepcopy(descriptor)
                expected.update(episode_index=index,
                    committed_environment_decisions=sum(self.rule.budget.episode_decisions[:index]),
                    committed_behavior_samples=self.rule.budget.users*sum(self.rule.budget.episode_decisions[:index]),
                    episode_token=training['ledger']['committed_episodes'][index-1]['episode_token'])
                archive = training_root/f'episode_{index:02d}/checkpoint'
                sha = training['ledger']['committed_episodes'][index-1]['checkpoint_sha256']
                inspect_checkpoint_archive(archive, expected, sha)
                marker = json.loads((archive/'CHECKPOINT_COMPLETE.json').read_text(encoding='utf-8'))
                if (row['descriptor'] != expected or row['checkpoint']['artifact_sha256'] != sha
                        or marker.get('model_payload_written_by_actual_controller') is not True):
                    raise ValueError('Actual-controller archive and budget evidence required, not text fixtures')
                input_files[str(archive/'CHECKPOINT_COMPLETE.json')] = sha
                input_files.update({str(archive/name): value for name, value in marker['files_sha256'].items()})
            final_archive = training_root/f'episode_{len(records):02d}/checkpoint'
            final_manifest = training['ledger']['committed_episodes'][-1]['checkpoint_sha256']
            parameter_sha = require_sha(records[-1]['actual_parameter_sha256'])
            schedules = self.protocol['validation_schedule_sha256_by_seed'][str(key[1])]
            if not schedules or len(set(schedules)) != len(schedules) or len(validation_evidence) != len(schedules):
                raise ValueError('All predeclared ordered common validation schedules required')
            reports = []
            for (evaluator, path, sha), schedule in zip(validation_evidence, schedules):
                if (type(evaluator) is not RealReadonlyValidationEvaluator or not evaluator._started
                        or evaluator.bindings != self.bindings or evaluator.controller.method != key[0]
                        or evaluator._checkpoint[1] != descriptor
                        or Path(evaluator._checkpoint[0]).resolve() != final_archive.resolve()
                        or evaluator._checkpoint[2] != final_manifest):
                    raise TypeError('Actual completed readonly evaluator of this own final checkpoint required')
                require_actual_cuda_controller(evaluator.controller)
                if actual_parameter_sha256(evaluator.controller) != parameter_sha:
                    raise RuntimeError('Actual evaluated weights changed or do not match the final training snapshot')
                record = _read_evidence(path, sha)
                reports.append(validate_validation_record(record, descriptor, final_manifest, parameter_sha, schedule))
                input_files[str(Path(path))] = sha
            self._entries[key] = dict(descriptor=descriptor, checkpoint_path=str(final_archive),
                checkpoint_manifest_sha256=final_manifest, actual_parameter_sha256=parameter_sha,
                validation_schedule_sha256=list(schedules), validation_reports=reports,
                training_complete_path=str(training_path), training_complete_sha256=training_sha,
                selection_is_fixed_final_budget_not_metric_ranking=True)
            self._inputs.update(input_files)
        except Exception:
            self._failed = True
            raise

    def seal_all(self, new_directory):
        if self._failed or self._sealed:
            raise RuntimeError('Failed/sealed registration cannot retry')
        require_reviewed_runtime(self.bindings)
        entries = list(self._entries.values())
        check_complete_paired_grid(entries, self.rule)
        if not all(not (Path(path).parent/'FAILURE.json').exists()
                and file_sha(Path(path)) == value for path, value in self._inputs.items()):
            self._failed = True
            raise RuntimeError('Registered evidence/weights changed before whole-grid freeze')
        path, root = Path(new_directory).resolve(), (BASE/'visual_calibrated_policy').resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError('New whole-grid artifact must stay inside visual output')
        identity = digest(dict(rule=self.rule.definition(), protocol_sha256=self.bindings.protocol_sha256))
        lock_root = root/'checkpoint_registration_locks'
        lock_root.mkdir(exist_ok=True)
        # Canonical experiment identity, not JSON encoding/output-name identity.
        with (lock_root/f'{identity}.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(identity=identity, output=str(path), automatic_retry_allowed=False), stream)
        self._sealed = True
        try:
            path.mkdir(parents=True, exist_ok=False)
            write_evidence(path/'ALL_FINAL_CHECKPOINTS_REGISTERED_PENDING_TEST_ENTRY.json',
                dict(status='all_final_checkpoints_registered_pending_test_entry',
                    rule=self.rule.definition(), identity_sha256=identity, bindings=asdict(self.bindings),
                    entries=entries, input_files_sha256=self._inputs, negative_validation_outcomes_retained=True,
                    test_access_used_for_selection=False, test_launch_authorized=False,
                    independent_final_result_audit_done=False))
        except Exception:
            self._failed = True
            raise
