"""SOURCE ONLY: register all actual fixed-policy cells, never execute them.

Descriptor coverage is not result evidence. Real registration requires the
completed runtime contract, actual completed evaluators and external hashes.
No launch, model factory, schedule generator, fitting, statistics or figures.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path

from retained_test_evaluator import (RealReadonlyRetainedEvaluator,
    _scientific_cell_identity, _expected_environment_sha, _check_reporting_final, _write_once)
from retained_test_schedule import _strict_json, _decode_registered_document
from visual_episode_collector import BASE, write_evidence
from visual_test_contract import (FrozenTestExecutionContract, FixedTestMatrixCandidate,
    OriginalRetainedMetadataPool, verify_cell_matrix)
from visual_training_coordinator import require_reviewed_runtime
from visual_training_transactions import digest, file_sha, require_sha

OUTPUT_NAMES = frozenset(('EXECUTION_REPORTING_ONLY.json', 'OBSERVATIONS_AND_DECISIONS.json',
                         'STARTED.json', 'STATE.json'))


def check_complete_descriptor_coverage(rows, candidate):
    """Pure exact coverage; explicitly NOT a claim that any cell ran."""
    if type(rows) is not list or type(candidate) is not FixedTestMatrixCandidate:
        raise TypeError('Explicit descriptor list and common matrix required')
    verify_cell_matrix(rows, candidate)
    groups = {}
    for row in rows:
        key = row['load_label'], row['seed']
        groups.setdefault(key, []).append(row)
    if len(groups) != 10 or any(len(group) != 12 or
            len({row['common_exogenous_schedule_sha256'] for row in group}) != 1
            for group in groups.values()):
        raise ValueError('All four methods and three delays must share each load/seed input')
    return dict(descriptor_cells=120, common_load_seed_groups=10, cells_per_common_group=12,
        actual_result_cells_proven=0, descriptor_check_not_completed_results=True)


def _safe_path(path, root):
    path, root = Path(path), Path(root).resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError('Evidence must remain inside its intended root')
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError('No substituted evidence/parent symlink or junction')
        if part.resolve() == root:
            break
    if (path.parent/'FAILURE.json').exists():
        raise RuntimeError('Failed/partial evidence cannot register as successful')
    return path


def _read_actual(path, sha, root):
    path = _safe_path(path, root)
    if not path.is_file() or file_sha(path) != require_sha(sha):
        raise RuntimeError('Externally bound actual evidence bytes changed/missing')
    return _strict_json(path.read_text(encoding='utf-8'))


def _claim_grid_once(root, identity, record):
    root = Path(root)
    require_sha(identity)
    root.mkdir(parents=True, exist_ok=True)
    path = root/(identity+'.json')
    _write_once(path, dict(scientific_grid_identity_sha256=identity, **record))
    return path


class RealRetainedGridRegistrar:
    """SOURCE PREPARED ONLY. Actual construction/register/seal count is zero."""
    def __init__(self, contract, new_directory):
        if type(contract) is not FrozenTestExecutionContract:
            raise TypeError('Completed actual frozen-weight test contract required')
        rows = contract.cell_descriptors()
        protocol = require_reviewed_runtime(contract._bindings)
        check_complete_descriptor_coverage(rows, contract._candidate)
        required = [Path(__file__), *(BASE/'scripts'/name for name in (
            'retained_test_evaluator.py', 'retained_test_schedule.py',
            'retained_test_service_catalog.py', 'visual_test_contract.py', 'visual_checkpoint_registration.py'))]
        if any(protocol['source_files_sha256'].get(str(source)) != file_sha(source) for source in required):
            raise RuntimeError('Current registrar source must be in the fresh complete Type-A entry')
        root = BASE/'visual_calibrated_policy'
        path = _safe_path(new_directory, root/'test_grid_results')
        if path.exists():
            raise ValueError('New registration only; never overwrite/append old evidence')
        identities = sorted(_scientific_cell_identity(protocol, row, contract._weights['entries']) for row in rows)
        if len(set(identities)) != 120:
            raise ValueError('Exactly 120 distinct scientific cell identities required')
        self.identity_sha256 = digest(dict(scientific_cell_identities=identities))
        self.contract, self.protocol, self.path = contract, protocol, path
        self._entries, self._inputs, self._failed, self._sealed = {}, {}, False, False
        self._lock = _claim_grid_once(root/'test_grid_registration_locks', self.identity_sha256,
            dict(at=datetime.now().astimezone().isoformat(), output_path=str(path),
                 automatic_append_resume_or_retry_allowed=False, no_cell_execution_by_registrar=True))
        self._owned = False
        try:
            path.mkdir(parents=True, exist_ok=False)
            self._owned = True
            _write_once(path/'STARTED.json', dict(at=datetime.now().astimezone().isoformat(),
                scientific_grid_identity_sha256=self.identity_sha256, bindings=asdict(contract._bindings),
                negative_cells_will_not_be_filtered=True, independent_final_result_audit_done=False))
            self._merge_inputs({str(contract._registration_path): contract._registration_sha,
                **contract._weights['input_files_sha256'], **protocol['source_files_sha256'],
                contract._bindings.protocol_path: contract._bindings.protocol_sha256,
                contract._bindings.entry_review_path: contract._bindings.entry_review_sha256,
                protocol['actual_model_smoke_path']: protocol['actual_model_smoke_sha256']})
            self._verify_inputs()
        except Exception as error:
            self._fail(error)
            raise

    def _merge_inputs(self, files):
        for name, sha in files.items():
            name, sha = str(Path(name)), require_sha(sha)
            if name in self._inputs and self._inputs[name] != sha:
                raise RuntimeError('One input path cannot be rebound between cells')
            self._inputs[name] = sha

    def _verify_inputs(self):
        self.contract.cell_descriptors()
        require_reviewed_runtime(self.contract._bindings)
        for name, sha in self._inputs.items():
            path = _safe_path(name, BASE.parent)
            if not path.is_file() or file_sha(path) != sha:
                raise RuntimeError('Input/raw/weight/actual evidence bytes changed; preserve failure')

    def _fail(self, error):
        self._failed = True
        evidence = dict(error=repr(error), at=datetime.now().astimezone().isoformat(),
            scientific_grid_identity_sha256=self.identity_sha256, registered_entries=list(self._entries.values()),
            input_files_sha256=self._inputs, partial_evidence_preserved=True,
            automatic_append_resume_or_retry_allowed=False, independent_final_result_audit_done=False)
        if self._owned and not (self.path/'FAILURE.json').exists():
            _write_once(self.path/'FAILURE.json', evidence)
        failed_lock = self._lock.with_name(self._lock.stem+'_FAILURE.json')
        if not failed_lock.exists():
            _write_once(failed_lock, evidence)

    def register(self, evaluator, complete_path, external_complete_sha256):
        """Consume an actual completed evaluator; never call a policy or queue."""
        if self._failed or self._sealed:
            raise RuntimeError('Fail-stop/seal-once; no partial append, retry or remeasurement')
        try:
            self._verify_inputs()
            if (type(evaluator) is not RealReadonlyRetainedEvaluator or evaluator.contract is not self.contract
                    or not evaluator._started or not evaluator._finished or evaluator._failed):
                raise TypeError('Actual successfully completed evaluator of this exact contract required')
            evaluator._check_weights()
            cell, identity = evaluator.cell, evaluator.semantic_identity_sha256
            if (identity != _scientific_cell_identity(self.protocol, cell, self.contract._weights['entries'])
                    or sum(digest(row) == digest(cell) for row in self.contract.cell_descriptors()) != 1
                    or identity in self._entries):
                raise ValueError('Foreign/duplicate/reconfigured cell; no outcome-based replacement')
            path = _safe_path(complete_path, BASE/'visual_calibrated_policy/test_cells')
            if path.name != 'TEST_CELL_COMPLETE_PENDING_AUDIT.json':
                raise ValueError('Exact actual cell completion artifact, not an aggregate/table/fixture')
            marker = _read_actual(path, external_complete_sha256, BASE/'visual_calibrated_policy/test_cells')
            neural = cell['method'] in ('QLA-MAPPO', 'PPO-MV-A')
            if (marker.get('status') != 'actual_cell_complete_pending_independent_audit'
                    or digest(marker.get('cell')) != digest(cell)
                    or marker.get('semantic_identity_sha256') != identity
                    or set(marker.get('outputs_sha256', {})) != OUTPUT_NAMES
                    or marker.get('source_protocol_sha256') != self.contract._bindings.protocol_sha256
                    or marker.get('fixed_horizon_and_drain') is not True
                    or type(marker.get('optimizer_updates')) is not int or marker['optimizer_updates'] != 0
                    or marker.get('actual_parameter_sha256_before_and_after') != evaluator._parameter_sha
                    or marker.get('deterministic_Torch_RNG_changed') is not (False if neural else None)
                    or marker.get('complete_120_grid_or_final_audit_claim') is not False):
                raise ValueError('Exact unchanged weights/RNG/zero-update/fixed-end pending-audit marker required')
            files = {str(path): require_sha(external_complete_sha256)}
            documents = {}
            for name, sha in marker['outputs_sha256'].items():
                output = path.parent/name
                documents[name] = _read_actual(output, sha, BASE/'visual_calibrated_policy/test_cells')
                files[str(output)] = require_sha(sha)
            started, state = documents['STARTED.json'], documents['STATE.json']
            lock = _safe_path(started['lock'], BASE/'visual_calibrated_policy/test_cell_locks')
            if (lock.name != identity+'.json' or lock.with_name(identity+'_FAILURE.json').exists()):
                raise RuntimeError('Original permanent successful cell lock required')
            lock_sha = file_sha(lock)
            locked = _read_actual(lock, lock_sha, BASE/'visual_calibrated_policy/test_cell_locks')
            binding = started['environment_binding']
            if (digest(started['cell']) != digest(cell) or started['semantic_identity_sha256'] != identity
                    or started.get('optimizer_update_allowed') is not False
                    or started.get('training_state_restore_allowed') is not False
                    or digest(binding) != _expected_environment_sha(self.protocol, cell)
                    or locked['semantic_cell_identity_sha256'] != identity or digest(locked['cell']) != digest(cell)
                    or Path(locked['output_path']).resolve() != path.parent.resolve()
                    or locked.get('automatic_retry_allowed') is not False
                    or state.get('phase') != 'actual_cell_complete_pending_independent_audit'
                    or digest(state['cell']) != digest(cell) or type(state.get('optimizer_updates')) is not int
                    or state['optimizer_updates'] != 0 or type(state.get('completed_decisions')) is not int
                    or state['completed_decisions'] != cell['generation_decisions']):
                raise ValueError('Original complete output, scientific lock and environment/past-only execution binding required')
            files[str(lock)] = lock_sha
            observed = documents['OBSERVATIONS_AND_DECISIONS.json']
            final = documents['EXECUTION_REPORTING_ONLY.json']['final']
            records = observed['records']
            if (len(records) != cell['generation_decisions'] or len(observed['observations']) != len(records)
                    or observed.get('initial_online_H') != 0. or observed.get('training_H_MV_or_RNG_restored') is not False
                    or observed.get('optimizer_update_called') is not False
                    or final.get('schedule_sha256') != cell['common_exogenous_schedule_sha256']):
                raise ValueError('All decisions/common schedule/online-H-only readonly records required')
            for index, (row, obs) in enumerate(zip(records, observed['observations']), 1):
                if (type(row['index']) is not int or row['index'] != index or obs['index'] != index
                        or not math.isclose(row['physical_seconds'], index*cell['decision_seconds'], abs_tol=1e-10, rel_tol=0.)
                        or row['physical_seconds'] != obs['physical_seconds']):
                    raise ValueError('Complete ordered physical-time observation/action coverage required')
                require_sha(row['actor_sha256'])
                require_sha(row['current_action_free_critic_sha256'])
            schedule = self.protocol['test_schedule_artifacts_by_load_seed'][cell['load_label']][str(cell['seed'])]
            if schedule['semantic_sha256'] != cell['common_exogenous_schedule_sha256']:
                raise ValueError('No per-method/delay changed exogenous inputs')
            # Reconstruct the generated count from the already frozen common
            # request file, not the result's self-reported generated count.
            # This reads request metadata only, never measured service labels.
            pool = OriginalRetainedMetadataPool()
            payload = _decode_registered_document(_read_actual(schedule['path'], schedule['file_sha256'],
                BASE/'visual_calibrated_policy'), frozenset(pool.image_ids()), pool.identity_sha256)
            if (payload.fingerprint() != cell['common_exogenous_schedule_sha256']
                    or payload.seed != cell['seed'] or payload.load_label != cell['load_label']
                    or payload.decisions != cell['generation_decisions']
                    or payload.horizon_seconds != cell['generation_seconds']):
                raise ValueError('Entire original registered common request count/identity/horizon required')
            summary = _check_reporting_final(final, len(payload.requests), len(records), cell)
            if digest(summary) != digest(marker['final_summary']):
                raise RuntimeError('Cell summary no longer reconstructs from frozen requests/raw counts/uncapped latencies')
            files.update({schedule['path']: require_sha(schedule['file_sha256'])})
            files.update(binding['catalog_source_sha256'])
            files.update(binding['mobility_source_sha256'])
            self._merge_inputs(files)
            self._verify_inputs()
            self._entries[identity] = dict(cell=cell, semantic_cell_identity_sha256=identity,
                complete_path=str(path), complete_sha256=external_complete_sha256,
                outputs_sha256=marker['outputs_sha256'], final_summary=summary,
                actual_parameter_sha256_before_and_after=evaluator._parameter_sha,
                negative_zero_completed_drop_unfinished_cells_retained=True)
            write_evidence(self.path/'REGISTRATION_STATE.json', dict(status='partial_registration_not_results_complete',
                scientific_grid_identity_sha256=self.identity_sha256, registered_cells=len(self._entries),
                expected_cells=120, entries=list(self._entries.values()), input_files_sha256=self._inputs,
                independent_final_result_audit_done=False, automatic_append_resume_or_retry_allowed=False))
        except Exception as error:
            self._fail(error)
            raise

    def seal_all(self):
        if self._failed or self._sealed:
            raise RuntimeError('Failed/sealed registration cannot append/retry')
        try:
            self._verify_inputs()
            entries = sorted(self._entries.values(), key=lambda row: row['semantic_cell_identity_sha256'])
            check_complete_descriptor_coverage([row['cell'] for row in entries], self.contract._candidate)
            if digest(dict(scientific_cell_identities=[row['semantic_cell_identity_sha256'] for row in entries])) != self.identity_sha256:
                raise RuntimeError('Scientific whole-grid identity changed')
            record = dict(status='all_120_actual_cells_registered_pending_independent_final_audit',
                scientific_grid_identity_sha256=self.identity_sha256, bindings=asdict(self.contract._bindings),
                matrix_definition=self.contract._candidate.definition(), entries=entries,
                input_files_sha256=self._inputs, actual_completed_cells=120, optimizer_updates=0,
                negative_zero_completed_drop_unfinished_cells_retained=True, test_fitting_or_checkpoint_selection=False,
                independent_final_result_audit_done=False, statistics_or_figures_generated=False,
                automatic_append_resume_or_retry_allowed=False)
            self._sealed = True
            _write_once(self.path/'ALL_120_ACTUAL_CELLS_PENDING_FINAL_AUDIT.json', record)
            return record
        except Exception as error:
            self._fail(error)
            raise
