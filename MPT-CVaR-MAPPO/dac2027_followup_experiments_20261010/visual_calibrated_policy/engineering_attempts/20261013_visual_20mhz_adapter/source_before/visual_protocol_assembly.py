"""SOURCE ONLY: immutable shared visual protocol, pending actual entry review.

No fitting, measured timing parsing, queue, policy or model execution. Only an
actual completed common-request generator plus its real pilot may be consumed.
Pure environment hashes are descriptors, not calibration or result evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
import math
from pathlib import Path

from execution_service_catalog import DIRECTORIES
from metadata_service_predictor import MODEL, MODEL_SHA
from physical_mobility import PhysicalMobility, TrainingWindow
from retained_test_evaluator import _write_once
from retained_test_grid_registration import _safe_path, _read_actual, _claim_grid_once
from retained_test_schedule import _decode_registered_document
from retained_test_schedule_generation import (RealRetainedScheduleGenerator,
    _require_generation_preparation)
from retained_test_service_catalog import RAW_AUDIT, SERVICE_EVIDENCE_BINDINGS, _registered_inventory
from visual_checkpoint_registration import FinalCheckpointRuleCandidate
from visual_episode_collector import BASE, EpisodeClosureCandidate, jsonable
from visual_method_features import VisualFeatureScales
from visual_neural_controller import VisualOptimizerSettings
from visual_request_schedule import VisualRequestSchedule, ScheduledRequest
from visual_test_contract import FixedTestMatrixCandidate, PROFILES
from visual_training_transactions import CommonBudgetCandidate, digest, file_sha, require_sha
from visual_upload_link import UploadLinkConfig

ENV_FIELDS = frozenset(('schedule_sha256', 'role', 'profiles', 'edge_positions', 'mobility_start_seconds',
    'broadcast_seconds', 'nominal_delay_seconds', 'waiting_capacity', 'upload_link',
    'catalog_source_sha256', 'mobility_source_sha256'))
PILOT_TEST_TEMPLATE_FIELDS = ENV_FIELDS - {'schedule_sha256'}
SEEDS = frozenset(str(seed) for seed in range(1, 6))
COMPONENT_FIELDS = frozenset(('feature_scales', 'closure_settings', 'common_budget', 'optimizer_settings',
    'learner_kwargs_by_method', 'virtual_H_episode_rule', 'test_rule_policy_settings',
    'training_environment_descriptors_by_seed', 'validation_environment_descriptors_by_seed',
    'training_schedule_artifacts_by_seed', 'validation_schedule_artifacts_by_seed',
    'test_environment_templates_by_load_seed', 'catalog_input_files_sha256_by_role',
    'placement_training_window_seconds', 'placement_seed', 'mobility_role_gap_seconds'))


def _hash_map(value):
    if type(value) is not dict or not value or any(type(name) is not str or not name for name in value):
        raise ValueError('Explicit complete input SHA mapping required')
    for sha in value.values():
        require_sha(sha)


def merge_frozen_file_bindings(existing, incoming):
    """Pure additive SHA merge; a known path can never be rebound."""
    _hash_map(existing)
    _hash_map(incoming)
    merged = dict(existing)
    for name, sha in incoming.items():
        if name in merged and merged[name] != sha:
            raise RuntimeError('Protected input identity cannot be replaced by a later catalog/registration map')
        merged[name] = sha
    return merged


def validate_environment_descriptor(binding, role, schedule_sha, broadcast_seconds):
    """Pure exact pre-action environment identity, never timing/queue access."""
    if type(binding) is not dict or set(binding) != ENV_FIELDS:
        raise ValueError('Complete known environment schema only, no extra future truth/hidden conditions')
    if (binding['role'] != role or role not in ('calibration', 'validation', 'test')
            or binding['schedule_sha256'] != require_sha(schedule_sha)
            or binding['profiles'] != list(PROFILES) or type(binding['waiting_capacity']) is not int
            or binding['waiting_capacity'] != 32 or type(binding['edge_positions']) is not list
            or len(binding['edge_positions']) != 4 or any(type(point) is not list or len(point) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in point)
                for point in binding['edge_positions'])):
        raise ValueError('Original role/common schedule/four action-aligned profiles/32 waiting/finite geometry required')
    for name in ('mobility_start_seconds', 'broadcast_seconds', 'nominal_delay_seconds'):
        x = binding[name]
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError('Physical finite seconds, not sample index or unknown source age')
    if binding['broadcast_seconds'] != broadcast_seconds or broadcast_seconds <= 0:
        raise ValueError('Same explicit physical broadcast period required')
    if binding['upload_link'] != asdict(UploadLinkConfig()):
        raise ValueError('Nondefault link needs separately repaired/reviewed empty-request feature adapter; no silent change')
    _hash_map(binding['catalog_source_sha256'])
    _hash_map(binding['mobility_source_sha256'])
    if len(binding['mobility_source_sha256']) != 30:
        raise ValueError('All original thirty physical mobility source identities required')
    return deepcopy(binding)


def compose_common_test_environment_table(candidate, templates):
    """Pure 30-condition identity table shared by four methods, not results."""
    if type(candidate) is not FixedTestMatrixCandidate or type(templates) is not dict or set(templates) != set(candidate.load_labels):
        raise TypeError('Exact two-load matrix and ten common metadata environment templates required')
    descriptors, hashes = {}, {}
    for li, label in enumerate(candidate.load_labels):
        if type(templates[label]) is not dict or set(templates[label]) != SEEDS:
            raise ValueError('Exactly five paired seeds, no method-specific test environment')
        descriptors[label], hashes[label] = {}, {}
        for si, seed in enumerate(candidate.seeds):
            template = validate_environment_descriptor(templates[label][str(seed)], 'test',
                candidate.common_schedule_sha256_by_load_seed[li][si], candidate.broadcast_seconds)
            if template['nominal_delay_seconds'] != 0.:
                raise ValueError('Explicit zero-delay template; all three registered delays derived from matrix')
            descriptors[label][str(seed)], hashes[label][str(seed)] = {}, {}
            delays = sorted({row['nominal_delay_seconds'] for row in candidate.cells()})
            for delay in delays:
                binding = dict(template, nominal_delay_seconds=delay)
                descriptors[label][str(seed)][str(delay)] = binding
                hashes[label][str(seed)][str(delay)] = digest(binding)
    return dict(descriptors=descriptors, sha256=hashes, descriptor_conditions=30,
        policy_cells_covered=120, actual_environment_or_policy_execution_proven=0)


def bind_generated_test_environment_schedules(candidate, pilot_templates):
    """Pure mechanical binding AFTER common requests are actually generated.

    A cal-validation-only pilot freezes source/physical condition templates,
    not unknown future request-file identities. No placeholder/old/precomputed
    request SHA is accepted in the pilot. This never samples IDs or generates
    requests, rewrites the pilot, fits labels or executes a test environment.
    The real caller has already authenticated the actual same-generator output.
    """
    if (type(candidate) is not FixedTestMatrixCandidate or type(pilot_templates) is not dict
            or set(pilot_templates) != set(candidate.load_labels)):
        raise TypeError('Exact matrix and pilot-frozen metadata-only two-load template table required')
    bound = {}
    for li, label in enumerate(candidate.load_labels):
        if type(pilot_templates[label]) is not dict or set(pilot_templates[label]) != SEEDS:
            raise ValueError('All five common seeds, no method-specific/partial template')
        bound[label] = {}
        for si, seed in enumerate(candidate.seeds):
            template = pilot_templates[label][str(seed)]
            if type(template) is not dict or set(template) != PILOT_TEST_TEMPLATE_FIELDS:
                raise ValueError('Pilot freezes condition ONLY: no future/placeholder request SHA or extra fields')
            binding = deepcopy(template)
            binding['schedule_sha256'] = candidate.common_schedule_sha256_by_load_seed[li][si]
            binding = validate_environment_descriptor(binding, 'test', binding['schedule_sha256'], candidate.broadcast_seconds)
            if binding['nominal_delay_seconds'] != 0.:
                raise ValueError('Zero-delay common template; matrix derives all three physical test delays')
            bound[label][str(seed)] = binding
    return bound


def paired_training_condition(seed, budget, scales, closure, optimizer, H_rule, environments, schedules):
    """Pure shared-design identity matching TrainingTransactionLedger recipe."""
    if type(seed) is not int or seed not in (1, 2, 3, 4, 5) or type(budget) is not CommonBudgetCandidate:
        raise TypeError('Explicit paired seed/common budget required')
    condition = digest(dict(scales=scales, closure=closure, optimizer=optimizer,
        virtual_H_episode_rule=H_rule, environments=environments))
    return dict(common_condition_sha256=condition,
        paired_design_sha256=digest(dict(seed=seed, budget=asdict(budget), schedules=tuple(schedules), common_condition=condition)))


def _complete_dataclass(cls, raw, tuple_fields=()):
    if type(raw) is not dict or set(raw) != {field.name for field in fields(cls)}:
        raise ValueError('All candidate dataclass fields explicit; do not silently fill defaults')
    values = deepcopy(raw)
    for name in tuple_fields:
        values[name] = tuple(tuple(row) if isinstance(row, list) else row for row in values[name])
    return cls(**values)


def _read_engineering_schedule(registration, role, decisions, dt):
    if type(registration) is not dict or set(registration) != {'path', 'file_sha256', 'semantic_sha256'}:
        raise ValueError('External path/byte/semantic cal-validation schedule registration required')
    raw = _read_actual(registration['path'], registration['file_sha256'], BASE/'visual_calibrated_policy')
    if type(raw) is not dict or set(raw) != {field.name for field in fields(VisualRequestSchedule)}:
        raise ValueError('Bare complete engineering request schedule required')
    raw['requests'] = tuple(ScheduledRequest(**row) for row in raw['requests'])
    schedule = VisualRequestSchedule(**raw)
    if (schedule.role != role or schedule.users != 30 or schedule.decisions != decisions
            or schedule.decision_seconds != dt or schedule.fingerprint() != require_sha(registration['semantic_sha256'])):
        raise RuntimeError('Role/grid/full horizon/common schedule changed before protocol assembly')
    return schedule


class RealVisualProtocolAssembler:
    """SOURCE ONLY. Actual constructor/assemble count zero, no launch API."""
    def __init__(self, actual_generator, complete_path, external_complete_sha256):
        if (type(actual_generator) is not RealRetainedScheduleGenerator or not actual_generator._finished
                or actual_generator._failed):
            raise TypeError('Actual successfully completed generator, not artificial descriptors or a false certificate')
        actual_generator._verify_boundary()
        self.generator = actual_generator
        self.pilot, self.inputs = _require_generation_preparation(actual_generator.design, actual_generator.bindings)
        path = _safe_path(complete_path, BASE/'visual_calibrated_policy/test_schedules')
        if path.resolve() != (actual_generator.path/'GENERATED_COMMON_REQUESTS_PENDING_TEST_ENTRY.json').resolve():
            raise ValueError('This exact completed generator output required')
        generated = _read_actual(path, external_complete_sha256, BASE/'visual_calibrated_policy/test_schedules')
        if (generated.get('status') != 'ten_common_request_artifacts_generated_pending_formal_test_entry'
                or generated.get('preparation_bindings') != asdict(actual_generator.bindings)
                or generated.get('generation_design') != actual_generator.design.definition()
                or generated.get('test_labels_or_outcomes_used') is not False
                or generated.get('actual_policy_cells') != 0 or generated.get('actual_training_cells') != 0
                or generated.get('formal_training_or_test_launch_authorized') is not False):
            raise ValueError('Complete real generation before formal work, never post-result protocol revision')
        self.inputs = merge_frozen_file_bindings(self.inputs, {str(path): require_sha(external_complete_sha256)})
        definition = dict(generated['matrix_definition'])
        for name in ('load_labels', 'total_rates_per_second', 'profiles', 'seeds'):
            definition[name] = tuple(definition[name])
        definition['common_schedule_sha256_by_load_seed'] = tuple(tuple(row) for row in definition['common_schedule_sha256_by_load_seed'])
        self.candidate = FixedTestMatrixCandidate(**definition)
        design = actual_generator.design
        original_matrix = FixedTestMatrixCandidate(design.load_labels, design.total_rates_per_second,
            design.generation_decisions, design.fixed_drain_seconds,
            tuple(tuple(actual_generator._artifacts[label][str(seed)]['semantic_sha256']
                for seed in design.seeds) for label in design.load_labels),
            design.retained_pool_identity_sha256, design.decision_seconds, design.broadcast_seconds, design.users)
        if (self.candidate.definition() != original_matrix.definition()
                or generated['test_schedule_artifacts_by_load_seed'] != actual_generator._artifacts
                or generated['input_files_sha256'] != {key: value for key, value in self.inputs.items() if key != str(path)}):
            raise RuntimeError('Completion artifact must retain the actual original generator matrix/files/inputs')
        self.generated, self.components = generated, deepcopy(self.pilot['formal_protocol_components'])
        if type(self.components) is not dict or set(self.components) != COMPONENT_FIELDS:
            raise ValueError('Complete cal-validation-frozen components, no invented test-driven replacement fields')
        self._failed, self._written = False, False

    def _build_from_actual_inputs(self):
        c, candidate = self.components, self.candidate
        scales = _complete_dataclass(VisualFeatureScales, c['feature_scales'],
            ('location_low', 'location_span', 'edge_positions', 'profiles', 'mean_profile_service_seconds'))
        closure = _complete_dataclass(EpisodeClosureCandidate, c['closure_settings'])
        budget = _complete_dataclass(CommonBudgetCandidate, c['common_budget'], ('episode_decisions',))
        optimizer = _complete_dataclass(VisualOptimizerSettings, c['optimizer_settings'])
        if (scales.profiles != PROFILES or scales.decision_seconds != candidate.decision_seconds
                or budget.users != 30 or budget.decision_seconds != candidate.decision_seconds
                or budget.fixed_drain_seconds != closure.fixed_drain_seconds
                or closure.fixed_drain_seconds != candidate.fixed_drain_seconds
                or c['virtual_H_episode_rule'] not in ('carry_across_training_episodes', 'reset_each_independent_episode')
                or set(c['learner_kwargs_by_method']) != {'QLA-MAPPO', 'PPO-MV-A'}
                or set(c['test_rule_policy_settings']) != {'JSEQ-A', 'Score-softmax'}):
            raise ValueError('Single shared physical scales/budget/closure/explicit H rule and declared methods required')
        for rule in c['test_rule_policy_settings'].values():
            if type(rule) is not dict or set(rule) != {'temperature'} or type(rule['temperature']) not in (int, float) or not math.isfinite(rule['temperature']) or rule['temperature'] <= 0:
                raise ValueError('Both rule parameters explicitly cal-validation-frozen')
        catalogs = c['catalog_input_files_sha256_by_role']
        if set(catalogs) != {'calibration', 'validation', 'test'}:
            raise ValueError('Three original role-specific execution catalog identities required')
        for name, sha in SERVICE_EVIDENCE_BINDINGS.items():
            if file_sha(Path(name)) != sha:
                raise RuntimeError('Original service audit bytes changed, never recompute/replace')
        audit = _read_actual(RAW_AUDIT, SERVICE_EVIDENCE_BINDINGS[str(RAW_AUDIT)], BASE.parent)
        expected_test = dict(SERVICE_EVIDENCE_BINDINGS) | _registered_inventory(audit['measurement_sha256'])
        if catalogs['test'] != expected_test:
            raise RuntimeError('Exactly original paired retained inventory, no test labels read for fitting')
        for role in ('calibration', 'validation'):
            if set(catalogs[role]) != {str(path) for path in DIRECTORIES[role].glob('*.csv')}:
                raise RuntimeError('Whole original cal/validation catalog input files required, not filtered rows/files')
        for mapping in catalogs.values():
            _hash_map(mapping)
            self.inputs = merge_frozen_file_bindings(self.inputs, mapping)
        if file_sha(MODEL) != MODEL_SHA:
            raise RuntimeError('Original calibration-only metadata predictor changed')
        self.inputs = merge_frozen_file_bindings(self.inputs, {str(MODEL): MODEL_SHA})
        mobility = PhysicalMobility(users=30)
        window = c['placement_training_window_seconds']
        if type(window) is not list or len(window) != 2 or type(c['placement_seed']) is not int:
            raise ValueError('Explicit calibration geometry window/seed, never test-fitted placement')
        training_window = TrainingWindow(*window)
        edges = mobility.server_positions(training_window, c['placement_seed'], 4)
        low, high = mobility.training_bounds(training_window)
        if (edges != scales.edge_positions or low != scales.location_low
                or tuple(max(b-a, 1.) for a, b in zip(low, high)) != scales.location_span):
            raise RuntimeError('Only training-window positions/location normalization, never test geometry fitting')
        self.inputs = merge_frozen_file_bindings(self.inputs, mobility.source_hashes)
        bound_test_templates = bind_generated_test_environment_schedules(candidate, c['test_environment_templates_by_load_seed'])
        test_table = compose_common_test_environment_table(candidate, bound_test_templates)
        role_windows, train_hashes, val_hashes, training_schedules, validation_schedules = [], {}, {}, {}, {}
        # Geometry/location normalization is itself calibration data access;
        # its full placement window must not overlap held-out role windows.
        role_windows.append(('calibration', training_window.start_seconds, training_window.end_seconds))
        validation_by_schedule, paired, common_conditions = {}, {}, {}
        def check_actual_environment(binding, role, sha, duration):
            binding = validate_environment_descriptor(binding, role, sha, candidate.broadcast_seconds)
            if (binding['catalog_source_sha256'] != catalogs[role] or binding['mobility_source_sha256'] != mobility.source_hashes
                    or binding['edge_positions'] != jsonable(scales.edge_positions)):
                raise RuntimeError('Shared original catalog/mobility/action order/geometry differs')
            start, end = binding['mobility_start_seconds'], binding['mobility_start_seconds']+duration
            if start < mobility.common_start_seconds or end > mobility.common_end_seconds:
                raise ValueError('Entire fixed physical window/drain must be supported, no wrap or time remapping')
            role_windows.append((role, start, end))
            return digest(binding)
        for key in ('training_environment_descriptors_by_seed', 'validation_environment_descriptors_by_seed',
                    'training_schedule_artifacts_by_seed', 'validation_schedule_artifacts_by_seed'):
            if type(c[key]) is not dict or set(c[key]) != SEEDS:
                raise ValueError('All five paired seeds before any formal model work')
        for seed in range(1, 6):
            key = str(seed)
            train_env, val_env = c['training_environment_descriptors_by_seed'][key], c['validation_environment_descriptors_by_seed'][key]
            train_reg, val_reg = c['training_schedule_artifacts_by_seed'][key], c['validation_schedule_artifacts_by_seed'][key]
            if (len(train_env) != len(budget.episode_decisions) or len(train_reg) != len(train_env)
                    or not val_env or len(val_reg) != len(val_env)):
                raise ValueError('Every ordered full training segment and common validation condition required')
            train_hashes[key], val_hashes[key], training_schedules[key], validation_schedules[key] = [], [], [], []
            for role, envs, regs, target, sha_target in (
                    ('calibration', train_env, train_reg, train_hashes[key], training_schedules[key]),
                    ('validation', val_env, val_reg, val_hashes[key], validation_schedules[key])):
                for index, (binding, reg) in enumerate(zip(envs, regs)):
                    raw = _read_actual(reg['path'], reg['file_sha256'], BASE/'visual_calibrated_policy')
                    decisions = budget.episode_decisions[index] if role == 'calibration' else raw['decisions']
                    schedule = _read_engineering_schedule(reg, role, decisions, candidate.decision_seconds)
                    sha = schedule.fingerprint()
                    if binding['nominal_delay_seconds'] != candidate.broadcast_seconds:
                        raise ValueError('This registered train/validation prototype uses common nominal P, no silent grid change')
                    target.append(check_actual_environment(binding, role, sha, schedule.horizon_seconds+closure.fixed_drain_seconds))
                    sha_target.append(sha)
                    self.inputs = merge_frozen_file_bindings(self.inputs, {str(Path(reg['path'])): require_sha(reg['file_sha256'])})
                    if role == 'validation':
                        if sha in validation_by_schedule and validation_by_schedule[sha] != target[-1]:
                            raise ValueError('One validation schedule cannot silently bind multiple environments')
                        validation_by_schedule[sha] = target[-1]
            pair = paired_training_condition(seed, budget, c['feature_scales'], c['closure_settings'],
                c['optimizer_settings'], c['virtual_H_episode_rule'], train_env, training_schedules[key])
            paired[key], common_conditions[key] = pair['paired_design_sha256'], pair['common_condition_sha256']
        for li, label in enumerate(candidate.load_labels):
            for si, seed in enumerate(candidate.seeds):
                reg = self.generated['test_schedule_artifacts_by_load_seed'][label][str(seed)]
                doc = _read_actual(reg['path'], reg['file_sha256'], BASE/'visual_calibrated_policy/test_schedules')
                payload = _decode_registered_document(doc, frozenset(self.generator._ids), candidate.retained_pool_identity_sha256)
                if payload.fingerprint() != candidate.common_schedule_sha256_by_load_seed[li][si]:
                    raise RuntimeError('Actual whole common request file differs from the bound matrix')
                self.inputs = merge_frozen_file_bindings(self.inputs, {str(Path(reg['path'])): reg['file_sha256']})
                for binding in test_table['descriptors'][label][str(seed)].values():
                    check_actual_environment(binding, 'test', payload.fingerprint(), candidate.cells()[0]['fixed_end_seconds'])
        gap = c['mobility_role_gap_seconds']
        if type(gap) not in (int, float) or not math.isfinite(gap) or gap < 0:
            raise ValueError('Explicit cal-validation pilot role gap in physical seconds required')
        for i, (role, start, end) in enumerate(role_windows):
            for other, left, right in role_windows[i+1:]:
                if role != other and not (end+gap <= left or right+gap <= start):
                    raise ValueError('Disjoint physical role windows/gap, no hidden reuse/remapping')
        record = dict(status='frozen_after_cal_validation_only_pilot', source_files_sha256=self.pilot['source_files_sha256'],
            protected_input_and_source_sha256=self.pilot['protected_input_and_source_sha256'],
            actual_model_smoke_path=self.pilot['actual_model_smoke_path'], actual_model_smoke_sha256=self.pilot['actual_model_smoke_sha256'],
            feature_scales=jsonable(asdict(scales)), closure_settings=jsonable(asdict(closure)), common_budget=jsonable(asdict(budget)),
            optimizer_settings=jsonable(asdict(optimizer)), learner_kwargs_by_method=c['learner_kwargs_by_method'],
            virtual_H_episode_rule=c['virtual_H_episode_rule'], checkpoint_selection_rule=FinalCheckpointRuleCandidate(budget).definition(),
            checkpoint_rule_registered_before_training=True, paired_design_by_seed=paired,
            training_common_condition_sha256_by_seed=common_conditions, training_environment_sha256_by_seed=train_hashes,
            training_schedule_sha256_by_seed=training_schedules, training_schedule_artifacts_by_seed=c['training_schedule_artifacts_by_seed'],
            validation_schedule_sha256_by_seed=validation_schedules, validation_schedule_sha256=sorted(validation_by_schedule),
            validation_environment_sha256_by_schedule=validation_by_schedule,
            validation_schedule_artifacts_by_seed=c['validation_schedule_artifacts_by_seed'],
            test_matrix_definition=candidate.definition(), test_matrix_registered_before_test=True,
            test_schedule_schema=self.generated['test_schedule_schema'], test_schedules_registered_before_policy_tests=True,
            test_arrival_design=self.generated['test_arrival_design'], test_rule_policy_settings=c['test_rule_policy_settings'],
            test_schedule_artifacts_by_load_seed=self.generated['test_schedule_artifacts_by_load_seed'],
            test_environment_binding_sha256_by_load_seed_delay=test_table['sha256'],
            test_environment_descriptors_by_load_seed_delay=test_table['descriptors'],
            test_request_SHA_bound_after_actual_metadata_generation_without_pilot_rewrite=True,
            cal_validation_frozen_components=c, input_files_sha256=self.inputs,
            protocol_file_is_not_completed_entry_review_or_launch_permission=True,
            independent_complete_entry_review=False, actual_formal_training_or_test_cells=0,
            test_labels_or_outcomes_used_for_fitting_or_selection=False)
        return record

    def assemble_once(self, new_directory):
        if self._failed or self._written:
            raise RuntimeError('One-shot protocol assembly, no append/reseal/retry after partial failure')
        owned, lock = False, None
        try:
            self.generator._verify_boundary()
            record = self._build_from_actual_inputs()
            self.generator._verify_boundary()
            _require_generation_preparation(self.generator.design, self.generator.bindings)
            for name, sha in self.inputs.items():
                path = _safe_path(name, BASE.parent)
                if file_sha(path) != require_sha(sha):
                    raise RuntimeError('Original inputs/metadata/pilot/source changed before protocol freeze')
            output = _safe_path(new_directory, BASE/'protocol')
            if output.exists():
                raise ValueError('New protocol directory only, never overwrite an old or weight-bound protocol')
            identity = digest(dict(components=self.components, matrix=self.candidate.definition(),
                test_environment_descriptors=record['test_environment_descriptors_by_load_seed_delay']))
            lock = _claim_grid_once(BASE/'visual_calibrated_policy/protocol_assembly_locks', identity,
                dict(output_path=str(output), automatic_append_reseal_retry_allowed=False))
            output.mkdir(parents=True, exist_ok=False)
            owned = True
            self._written = True
            _write_once(output/'FROZEN_PROTOCOL_PENDING_COMPLETE_ENTRY.json', record)
            return output/'FROZEN_PROTOCOL_PENDING_COMPLETE_ENTRY.json'
        except Exception as error:
            self._failed = True
            failure = dict(error=repr(error), input_files_sha256=self.inputs, partial_evidence_preserved=True,
                actual_formal_training_or_test_cells=0, automatic_append_reseal_retry_allowed=False,
                at=datetime.now().astimezone().isoformat())
            if owned:
                _write_once(output/'FAILURE.json', failure)
            if lock is not None:
                _write_once(lock.with_name(lock.stem+'_FAILURE.json'), failure)
            raise
