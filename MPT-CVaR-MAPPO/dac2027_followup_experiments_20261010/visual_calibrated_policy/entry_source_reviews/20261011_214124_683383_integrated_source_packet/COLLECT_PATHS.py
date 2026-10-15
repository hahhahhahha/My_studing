"""Mechanical paths/SHA collection only. No review judgement or experiment."""
from __future__ import annotations
import sys
from pathlib import Path
BASE = Path('X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010')
sys.path.insert(0, str(BASE/'scripts'))
from windows_runtime import initialize
initialize()

from datetime import datetime
import json
from visual_model_smoke import source_lock, protected_bindings
from visual_training_transactions import file_sha, digest

OUT = Path(__file__).resolve().parent
ROOT = BASE.parent
PRIOR = BASE/'visual_calibrated_policy/engineering_attempts/20261011_211624_446835_pilot_export/RESULT.json'
EXPECTED_PRIOR = '3a6ba1dd6e64c6b11a69de94a00c4b6ca6ce782c685e4cd8f45fafd2cb87ed69'
assert file_sha(PRIOR) == EXPECTED_PRIOR
old = json.loads(PRIOR.read_text(encoding='utf-8'))
continuity = old['original_protected_files_sha256'] | old['new_implementation_sha256'] | {str(PRIOR): EXPECTED_PRIOR}
assert all(file_sha(Path(path)) == sha for path, sha in continuity.items())
sources, protected = source_lock(), protected_bindings()
docs = sorted(BASE.glob('README*.md'))
fixed = [ROOT/'paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md',
    ROOT/'results_qla_joint_reward_ablation_20261009/PROTOCOL_中文.md',
    ROOT/'results_qla_unseen_external_windows_20261009/EXPERIMENT_PLAN_中文.md',
    BASE/'protocol/QLA_ENVIRONMENT_SPEC.json']
review = BASE/'visual_calibrated_policy/entry_source_reviews/20261011_115016_650790_independent_typeA_source'
trace = ROOT/'.aris/traces/experiment-audit/2026-10-11_visual_smoke_entry_source'
review_files = sorted(path for root in (review, trace) for path in root.rglob('*') if path.is_file())
reports = sorted((BASE/'visual_calibrated_policy/engineering_attempts').glob('*/RESULT.json'))
aux = {str(path): file_sha(path) for path in docs+fixed+review_files+reports}
manifest = dict(status='paths_and_current_byte_fingerprints_only_not_review_or_runtime_permission',
    at=datetime.now().astimezone().isoformat(), source_files_sha256=sources,
    protected_input_and_source_sha256=protected, preserved_continuity_files_sha256=continuity,
    original_protocol_document_and_prior_review_paths_sha256=aux,
    source_map_identity_sha256=digest(sources), protected_map_identity_sha256=digest(protected),
    original_spec_sha256='75aa3aced99d49fa90a7318cd2b5dc8b6d8d320cd89c7ae740aff5f50cacbc88',
    old_smoke_plan_resealed_or_read=False, actual_model_smoke_pilot_or_policy_calls=0,
    actual_request_generation_or_completed_pilot_export_calls=0, reviewer_invoked_by_collector=False,
    independent_verdict_created=False, Torch_imported='torch' in sys.modules,
    actual_test_timing_CSV_parsed=False, collector_source_path=str(Path(__file__)),
    collector_source_sha256=file_sha(Path(__file__)))
assert 'torch' not in sys.modules
with (OUT/'CURRENT_PATHS_AND_BYTES.json').open('x', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
print(json.dumps(dict(packet_path=str(OUT), source_files=len(sources), protected_files=len(protected),
    prior_continuity_files=len(continuity), auxiliary_files=len(aux),
    manifest_sha256=file_sha(OUT/'CURRENT_PATHS_AND_BYTES.json'), actual_policy_calls=0, judgement=False)))
