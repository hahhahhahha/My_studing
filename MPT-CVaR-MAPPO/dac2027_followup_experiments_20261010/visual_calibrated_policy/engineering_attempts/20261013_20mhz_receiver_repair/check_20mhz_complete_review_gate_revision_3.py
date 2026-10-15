"""Exercise only the new 20 MHz review gate with synthetic maps and temp files."""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path


FOLLOWUP = Path(__file__).resolve().parents[3]
HELPER = (FOLLOWUP / "visual_calibrated_policy" / "engineering_attempts" / "20261012_s4"
          / "reward_input_integrity.py")
OUT = Path(__file__).resolve().parent / "AUTHOR_CHECKS_REVISION_3.json"


def _load_review_function():
    tree = ast.parse(HELPER.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == "require_review")
    return node


def _write_review(directory: Path, sources: dict[str, str], protected: dict[str, str], *, legacy=False):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "blockers": [],
        "review_type": "Type-A",
        "reviewed_sources_sha256": sources,
        "protected_input_and_source_sha256": protected,
    }
    if legacy:
        payload["reviewed_sources_sha256"] = sources
    (directory / "REVIEW_RESULT.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8")
    (directory / "REVIEW_RESPONSE.md").write_text("synthetic test fixture\n", encoding="utf-8")


def _expect_reject(function, *args, exception=RuntimeError, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return True
    return False


def main():
    helper_text = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(helper_text)
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == "require_review")
    source_keys = [f"C:\\fixture\\src\\{i:03d}.py" for i in range(209)]
    protected_keys = [f"C:\\fixture\\protected\\{i:03d}.bin" for i in range(210)]
    source_map = {key: hashlib.sha256(key.encode()).hexdigest() for key in source_keys}
    protected_map = {key: hashlib.sha256(key.encode()).hexdigest() for key in protected_keys}
    own_map = dict(list(source_map.items())[:9])

    with tempfile.TemporaryDirectory(prefix="codex-20mhz-review-gate-") as temp:
        root = Path(temp)
        here = root / "20261012_s4"
        allowed = root / "20261013_20mhz_receiver_repair" / "independent_repair_rereview_revision_3"
        legacy = here / "independent_command_rereview"
        namespace = {
            "Path": Path,
            "HERE": here,
            "read": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
            "sha": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "own_source_map": lambda: own_map,
        }
        function_module = ast.Module(body=[node], type_ignores=[])
        exec(compile(function_module, str(HELPER), "exec"), namespace)
        require_review = namespace["require_review"]
        _write_review(allowed, source_map, protected_map)
        _write_review(legacy, own_map, {})

        exact_pass = require_review(allowed, reviewed_source_map=source_map,
                                    protected_map=protected_map)
        changed_source = dict(source_map)
        changed_source[source_keys[-1]] = "f" * 64
        changed_protected = dict(protected_map)
        changed_protected[protected_keys[-1]] = "e" * 64
        source_mutation_rejected = _expect_reject(
            require_review, allowed, reviewed_source_map=changed_source,
            protected_map=protected_map)
        protected_mutation_rejected = _expect_reject(
            require_review, allowed, reviewed_source_map=source_map,
            protected_map=changed_protected)
        missing_protected_rejected = _expect_reject(
            require_review, allowed, reviewed_source_map=source_map)
        nine_file_map_rejected = _expect_reject(
            require_review, allowed, reviewed_source_map=own_map,
            protected_map=protected_map)
        changed_own = dict(source_map)
        changed_own[source_keys[0]] = "d" * 64
        own_pin_mismatch_rejected = _expect_reject(
            require_review, allowed, reviewed_source_map=changed_own,
            protected_map=protected_map)
        wrong_directory_rejected = _expect_reject(
            require_review, root / "wrong", reviewed_source_map=source_map,
            protected_map=protected_map, exception=ValueError)
        legacy_reward_pass = require_review()

    checks = {
        "schema": "visual-20mhz-complete-review-gate-fixture-check-v1",
        "status": "PASS",
        "synthetic_only": True,
        "source_count": len(source_map),
        "protected_count": len(protected_map),
        "exact_full_maps_passed": bool(exact_pass["JSON"] and exact_pass["MD"]),
        "changed_uncovered_source_rejected": source_mutation_rejected,
        "changed_protected_input_rejected": protected_mutation_rejected,
        "missing_protected_map_rejected": missing_protected_rejected,
        "nine_file_map_rejected": nine_file_map_rejected,
        "changed_own_pin_rejected": own_pin_mismatch_rejected,
        "wrong_review_directory_rejected": wrong_directory_rejected,
        "legacy_reward_default_path_unchanged": bool(legacy_reward_pass["JSON"] and legacy_reward_pass["MD"]),
        "torch_cuda_or_model_calls": 0,
        "smoke_plan_or_experiment_calls": 0,
    }
    if not all(value is True for key, value in checks.items()
               if key not in {"schema", "status", "source_count", "protected_count",
                              "torch_cuda_or_model_calls", "smoke_plan_or_experiment_calls"}):
        checks["status"] = "FAIL"
    OUT.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))
    if checks["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
