"""Exercise revision 5 routing and the real 8-source + 1-resource map shape."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path


FOLLOWUP = Path(__file__).resolve().parents[3]
HELPER = FOLLOWUP / "visual_calibrated_policy" / "engineering_attempts" / "20261012_s4" / "reward_input_integrity.py"
OUT = Path(__file__).resolve().parent / "AUTHOR_CHECKS_REVISION_5.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_review(directory: Path, sources: dict[str, str], protected: dict[str, str], own: dict[str, str], *, include_own=True):
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "blockers": [],
        "review_type": "Type-A",
        "reviewed_sources_sha256": sources,
        "protected_input_and_source_sha256": protected,
    }
    if include_own:
        result["independent_own_resource_pins_sha256"] = own
    (directory / "REVIEW_RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "REVIEW_RESPONSE.md").write_text("synthetic fixture only\n", encoding="utf-8")


def _reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (RuntimeError, ValueError, KeyError):
        return True
    return False


def main() -> None:
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "require_review")

    with tempfile.TemporaryDirectory(prefix="codex-20mhz-review-gate-r4-") as temp:
        base = Path(temp) / "followup"
        here = base / "visual_calibrated_policy" / "engineering_attempts" / "20261012_s4"
        resource_path = base / "protocol" / "QLA_ENVIRONMENT_SPEC.json"
        resource_path.parent.mkdir(parents=True)
        spec = {"schema": "qla-followup-environment-v2-user-gpu1800", "gpu_occupancy_max_mib_exclusive": 1800}
        resource_path.write_text(json.dumps(spec), encoding="utf-8")
        baseline_path = here / "FROZEN_BASELINE.json"
        baseline_path.parent.mkdir(parents=True)
        baseline_path.write_text(json.dumps({"environment_spec_canonical_sha256": _digest(spec)}), encoding="utf-8")
        allowed = here.parent / "20261013_20mhz_receiver_repair" / "independent_repair_rereview_revision_5"
        legacy = here / "independent_command_rereview"

        source_keys = [str((base / "scripts" / f"source_{i:03d}.py").resolve()) for i in range(209)]
        protected_keys = [str((base / "protected" / f"input_{i:03d}.bin").resolve()) for i in range(210)]
        source_map = {key: hashlib.sha256(key.encode()).hexdigest() for key in source_keys}
        protected_map = {key: hashlib.sha256(key.encode()).hexdigest() for key in protected_keys}
        resource_key = str(resource_path.resolve())
        own_map = {key: source_map[key] for key in source_keys[:8]}
        own_map[resource_key] = _hash(resource_path)

        namespace = {
            "Path": Path,
            "HERE": here,
            "BASE": base,
            "read": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
            "sha": lambda path: _hash(Path(path)),
            "digest": _digest,
            "own_source_map": lambda: own_map,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), "exec"), namespace)
        require_review = namespace["require_review"]
        _write_review(allowed, source_map, protected_map, own_map)
        _write_review(legacy, own_map, {}, own_map)

        exact_maps_pass = bool(require_review(allowed, reviewed_source_map=source_map, protected_map=protected_map)["JSON"])
        smoke_source = FOLLOWUP / "visual_calibrated_policy" / "engineering_attempts" / "20261012_s4" / "visual_model_smoke_small.py"
        smoke_tree = ast.parse(smoke_source.read_text(encoding="utf-8"))
        caller_dir_values = [node.value.right.value for node in smoke_tree.body
                             if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "INDEPENDENT_REVIEW_DIR" for target in node.targets)
                             and isinstance(node.value, ast.BinOp)
                             and isinstance(node.value.right, ast.Constant)
                             and isinstance(node.value.right.value, str)]
        actual_caller_revision5 = bool(caller_dir_values and caller_dir_values[0].endswith("independent_repair_rereview_revision_5"))
        source_mutation = dict(source_map)
        source_mutation[source_keys[-1]] = "f" * 64
        protected_mutation = dict(protected_map)
        protected_mutation[protected_keys[-1]] = "e" * 64
        missing_own_field = allowed / "REVIEW_RESULT.json"
        original = json.loads(missing_own_field.read_text(encoding="utf-8"))
        original.pop("independent_own_resource_pins_sha256")
        missing_own_field.write_text(json.dumps(original), encoding="utf-8")
        missing_own_pin_rejected = _reject(require_review, allowed, reviewed_source_map=source_map, protected_map=protected_map)
        _write_review(allowed, source_map, protected_map, own_map)
        bad_resource_review = dict(own_map)
        bad_resource_review[resource_key] = "0" * 64
        _write_review(allowed, source_map, protected_map, bad_resource_review)
        resource_pin_mismatch_rejected = _reject(require_review, allowed, reviewed_source_map=source_map, protected_map=protected_map)
        _write_review(allowed, source_map, protected_map, own_map)

        spec["gpu_occupancy_max_mib_exclusive"] = 500
        resource_path.write_text(json.dumps(spec), encoding="utf-8")
        baseline_path.write_text(json.dumps({"environment_spec_canonical_sha256": _digest(spec)}), encoding="utf-8")
        own_map[resource_key] = _hash(resource_path)
        _write_review(allowed, source_map, protected_map, own_map)
        old_resource_policy_rejected = _reject(require_review, allowed, reviewed_source_map=source_map, protected_map=protected_map)

        # Restore valid 1800 MiB pins before testing map-only negative cases.
        spec["gpu_occupancy_max_mib_exclusive"] = 1800
        resource_path.write_text(json.dumps(spec), encoding="utf-8")
        baseline_path.write_text(json.dumps({"environment_spec_canonical_sha256": _digest(spec)}), encoding="utf-8")
        own_map[resource_key] = _hash(resource_path)
        _write_review(allowed, source_map, protected_map, own_map)
        _write_review(legacy, own_map, {}, own_map)

        checks = {
            "schema": "visual-20mhz-complete-review-gate-fixture-check-v1",
            "status": "PASS",
            "synthetic_only": True,
            "source_map_count": len(source_map),
            "protected_map_count": len(protected_map),
            "own_source_and_resource_pin_count": len(own_map),
            "own_source_members": 8,
            "own_resource_spec_members": 1,
            "actual_small_smoke_caller_targets_revision5": actual_caller_revision5,
            "exact_209_210_plus_separate_own_map_passed": exact_maps_pass,
            "changed_source_rejected": _reject(require_review, allowed, reviewed_source_map=source_mutation, protected_map=protected_map),
            "changed_protected_rejected": _reject(require_review, allowed, reviewed_source_map=source_map, protected_map=protected_mutation),
            "missing_independent_own_resource_map_rejected": missing_own_pin_rejected,
            "changed_resource_pin_rejected": resource_pin_mismatch_rejected,
            "old_500_resource_policy_rejected": old_resource_policy_rejected,
            "legacy_reward_default_path_unchanged": bool(require_review()["JSON"]),
            "torch_cuda_model_or_experiment_calls": 0,
        }
        if not all(value is True for key, value in checks.items() if key not in {"schema", "status", "source_map_count", "protected_map_count", "own_source_and_resource_pin_count", "own_source_members", "own_resource_spec_members", "torch_cuda_model_or_experiment_calls"}):
            checks["status"] = "FAIL"
        OUT.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(checks, sort_keys=True))
        if checks["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
