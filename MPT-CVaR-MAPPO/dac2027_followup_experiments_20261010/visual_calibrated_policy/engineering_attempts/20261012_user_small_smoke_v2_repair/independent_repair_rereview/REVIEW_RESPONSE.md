# Independent Type-A B3 changed-source rereview

Date: 2026-10-12, Asia/Shanghai. Reviewer: `/root/user_small_smoke_v2_integrity`. Requested configuration carried from the original review: gpt-6-astra / xhigh. Provider-side routing telemetry was not independently inspected.

## Verdict and scope

**WARN, with no current blockers in this bounded changed-source scope. B3 is resolved at source.** The actual reward caller/private worker is now checked independently of the separate CUDA witness, and its actual imported modules, device and both thread counts are checked before the frozen model-bearing worker executes.

This verdict can satisfy only the new small-entry source-review prerequisite for the exact nine-file map in the accompanying JSON. It is not actual smoke success, numerical success, resource readiness, an actual-evidence audit, pilot or formal authorization, full integrated-entry certification or Type-B certification. The previous first-V2 FAIL report remains correct for its preserved source bytes and is not converted into a passing report. Existing human authorization, fresh runtime gates, one-shot rules and downstream independent evidence/adoption requirements remain separate.

This is one bounded B3 rereview of the three changed files against `SOURCE_BEFORE_FIRST_V2_REVIEW`, not a new comprehensive review of every unchanged dependency. The original B1/B2 conclusions and scientific-path findings are carried forward only with explicit unchanged-source and byte-continuity evidence below. The author supplied no expected verdict; this reviewer made no source repair.

Absolute path legend:

- R = `X:/Yi-one/workspace/MPT-CVaR-MAPPO`
- F = `R/dac2027_followup_experiments_20261010`
- D = `F/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v2_repair`
- S = `F/scripts`
- E = `R/results_qla_joint_reward_ablation_20261010_repaired/execution_source`

All source line references below are one-based and refer to the current files. The exact nine Windows `Path` strings and byte SHA256 values are in REVIEW_RESULT.json.

## 1. Actual process identity now precedes guarded work

`D/reward_input_integrity.py:66-81` adds `actual_environment(spec)` using only standard-library metadata and path/version inspection. It requires the actual `sys.executable` to resolve to the declared isolated environment's `Scripts/python.exe`, exact Python3.13.5, NumPy distribution2.2.6, Torch distribution2.11.0+cu128, and the NumPy distribution's package directory to lie within that isolated environment. Missing metadata raises instead of allowing an unknown version. It does not import NumPy/Torch, construct a model or allocate CUDA.

`source_guard():129-132` first verifies the exact current canonical specification and strict 1800 threshold, then unconditionally calls this actual-process check. The existing guarded entry paths are unchanged:

| Path | Relevant guard before scoped work |
| --- | --- |
| reward prepare | `prepare():51`, then `build_schedules_once():176`, before its NumPy load and canonical import |
| expected/read plan | `expected_plan():27`; `read_plan():45` requires exact equality with that guarded plan |
| reward parent | `run():177` checks plan before lock/process launch; each arm checks again at191 |
| private worker | `worker():133` checks plan before command construction, then147-149 after witness |
| original runner/command builder | `original_module():63`, `command():69`, and `load_frozen():155` before dynamic imports |
| validator/completion | validator86/123 and final parent read_plan210 retain guards |

The parent still spells the worker command with `sys.executable`, but that caller executable is now constrained before launch, and the fresh worker independently repeats the same condition. Consequently the original B3 ordinary wrong-caller-environment counterexample no longer passes the source contract. Preparation also performs metadata/interpreter/NumPy-distribution ownership checks before its first schedule-related NumPy use.

This is not a hostile-import sandbox. Preparation does not separately compare an already imported/shadowed NumPy object's `__file__` and `__version__`; it relies on the isolated executable, distribution metadata and ordinary import resolution. The actual training worker adds the direct module checks below. Deliberately spoofed package metadata, injected `sys.modules`/`PYTHONPATH` modules, package-byte tampering and filesystem races remain within the inherited installed-package/TOCTOU warning, not a claimed new attestation guarantee.

## 2. Actual NumPy/Torch, CUDA and thread checks precede model-bearing imports

The worker keeps the existing resource/kernel call at `D/reward_smoke_small.py:145-146`. Its new block is after that call and after the renewed plan/source/schedule guards, at152-169. Therefore the current strict `<1800 MiB` gate and the separate seeded kernel must succeed before the worker reaches its own Torch import.

The new block repeats `actual_environment(spec)`, obtains the actual NumPy/Torch module objects, compares their `__version__` values to the fixed spec, and requires the actual NumPy module file to resolve inside the isolated environment (154-158). It then sets intraop threads to the declared value1, sets interop to1 only when necessary, and checks both getters equal1 (159-163). Failure to set interop propagates; an ineffective setting is rejected by the final getter check. CUDA must be available and actual device0 must be the specified RTX4060Ti. There is no CPU or alternative-device fallback.

Only after these checks and evidence save does the source re-run source_guard and execute the frozen worker with runpy at169-172. That frozen worker is the path which imports the training/model packages. No frozen model-bearing import was moved ahead of these checks. Early original-runner command construction can import non-neural NumPy reward utilities, as before; this does not mean a Torch/model import occurred before the gate. The separate witness remains a resource/kernel check, not evidence substituted for this worker's runtime.

Torch's package directory is recorded but is not required to be inside the venv. This is consistent with the unchanged specification permitting read-only system-site package reuse; NumPy alone must be owned by the isolated environment. Version equality is not package-byte attestation. The declared runtime change enforces already specified threads1/1; it does not change a scientific hyperparameter or numerical acceptance criterion.

## 3. Actual-runtime evidence semantics

`worker():164-168` creates `<arm>_ACTUAL_RUNTIME.json` in the new parent output directory using the existing exclusive `save()` helper. It contains the checked executable/Python/distribution versions, NumPy distribution location, actual NumPy/Torch module locations, actual GPU identity and measured thread getter values. The mock accepted cases produced exactly one in-memory record; all tested rejected cases produced none.

`runtime_is_actual_worker=true` means the values are obtained in the same worker which is about to invoke the frozen model path. `model_started=false` is a **pre-model snapshot at the time this file is saved**, not a permanent statement that no training later happened. It is not an actual-smoke-pass marker. An existing evidence file makes exclusive creation fail before model invocation rather than overwrite evidence.

Evidence limitations remain: this record does not itself include PID, plan/spec SHA, or a direct witness-artifact link, and its bytes are not added to the reduced parent completion report's independent hash checks. It must be interpreted with the new output directory, command/start logs and plan, and be included in a later actual-evidence audit. That narrower reporting contract does not nullify the in-process enforced checks, but no stronger end-to-end provenance claim is made here.

## 4. Exact changed scope and continuity

Full text comparison and AST comparison show:

- `reward_input_integrity.py`: one standard-library metadata import; new `actual_environment`; one call in source_guard; require_review's two paths changed to `independent_repair_rereview`.
- `reward_smoke_small.py`: only worker's post-witness actual-runtime check/evidence block and extra guard added. Every other function is AST-identical to first V2.
- `README_SMALL_SMOKE_中文.md`: B3 correction/preservation explanation and distinct new rereview path. No experiment budget or numerical threshold changed.

All six preserved files match the exact hashes in the first-V2 review. Current visual_model_smoke_small.py, FROZEN_BASELINE.json and USER_REQUEST.json match that first review unchanged. Public qla_environment.py, windows_runtime.py and QLA_ENVIRONMENT_SPEC.json also remain unchanged. Current nine-file identities are recorded exactly in the JSON.

This rereview independently rehashed all12 immutable baseline entries, all115 frozen executable files and all30 consumed KAIST files; all matched. Exact executable-tree membership and sorted-first30 consumed raw membership also matched. Because the baseline and the complete names/bytes are unchanged, the original B1/B2 source analysis continues to apply; this is not a newly run schedule/closure execution test. No local schedules or SMALL_REWARD_PLAN existed at inventory.

Schedule generator arguments, raw selection, new local schedule-root policy, old shared-root preservation, guarded membership/bytes/content, both frozen package digests/helpers, original reward/critic/bootstrap/final-buffer path, validation/config/checkpoint/reliability acceptance and visual algorithms remain unchanged. Reward stays32train/8validation,30users/35servers,seed1 and evaluation10001; visual stays8decisions,30users/4servers,50ms and full30s drain. No q/H formula, update, precision, tolerance, reward range, finite check, performance criterion or formal budget was modified by this B3 correction.

## 5. Distinct new review prerequisite

`D/reward_input_integrity.py:142-150` now reads only `D/independent_repair_rereview/REVIEW_RESULT.json` and `REVIEW_RESPONSE.md`. It still requires Type-A, PASS/WARN, empty blockers, exact current nine-source map and MD existence. The first FAIL path is not a fallback. The new helper is in both launchers' source maps, so an old plan cannot silently absorb this change.

Reward's expected plan binds the new report hashes. Both actual visual run/worker and reward prepare/plan/run/worker use the existing review gate paths. The unchanged visual file obtains the changed helper at runtime; it does not need an algorithm edit to consume the new report. No old FAIL, quota error or earlier WARN is used as current approval.

## 6. Bounded independent fixtures

Only stdlib AST extraction and pure in-memory fake metadata/module objects were used. Neither full worker nor actual source_guard, witness, check, prepare or run was called. Current two changed Python files and their two preserved counterparts were AST-parsed for this fixture; unchanged files were continuity-hashed, not claimed as a new209-file semantic review.

Metadata fixtures accepted the exact mock environment and rejected wrong interpreter, Python, NumPy metadata version, Torch metadata version, NumPy distribution outside the isolated root and missing metadata. The isolated post-gate worker block was evaluated with fake NumPy/Torch objects through an in-memory import hook, not real NumPy/Torch. It accepted exact runtime and already-one interop cases; the latter correctly avoided calling the setter again. It rejected wrong actual NumPy/Torch versions, NumPy module outside the isolated root, unavailable CUDA, wrong GPU, ineffective intraop setting, ineffective interop setting and an interop setter exception. No rejecting case wrote a mock runtime record. AST ordering verified witness precedes actual module checks and frozen runpy follows them.

These fixtures validate branch logic and source ordering only. They say nothing about actual CUDA initialization, installed metadata validity, API compatibility, real thread setters, training numerics or schedule generation on this host.

## 7. Warnings retained; no silent scientific repair

The first report's W1-W10 remain applicable with B3 removed and the new review-path update. In particular:

- W1: actual resource/runtime/model/update/checkpoint/roundtrip outcomes are unverified.
- W2: historical QLA finite statistics can be rejected only after optimizer steps; PPO's before-step checks remain different.
- W3: pre-try and abrupt failures may lack complete local snapshots; resource gates are point-in-time, not reservations. New thread-setting failure must remain a real failure, not be bypassed.
- W4: old16-decision/72-32 plans are not current identity or authority.
- W5: parent validation is not an independent rehash of every child artifact; the new runtime record also needs later audit. Locks/PIDs are workflow controls, not hostile-filesystem/liveness proofs.
- W6: hash-then-read, imported-module/metadata/installed-package bytes, filesystem aliasing and concurrent mutation are not atomically attested. Preparation's metadata check should not be described as a direct loaded-module attestation.
- W7: reward smoke still lacks visual-style weight roundtrip/pre-post parameter proof and updates.csv finite-gradient acceptance; the mass-error NaN gap and unchanged four PRIMARY finite checks remain documented. No numerical condition was relaxed or strengthened here.
- W8: new review gating is enforced, but review authenticity/lifetime attestation is not cryptographic; visual checks it at entry, reward binds hashes into its plan.
- W9: tiny functional budgets cannot prove convergence, tails, H activation or full feedback-window coverage.
- W10: inherited visual check parses the original visual entry and loader, not a fresh independent review of the relocated wrapper.

Actual smoke outcomes must be retained even if unfavorable. Full integrated entry, independent actual-evidence adoption, pilot/protocol/network/OOD decisions and formal frozen Full replay requirements remain downstream. None is waived by WARN.

## 8. First report and quota provenance preservation

First reports remain byte-identical to their archived copies:

- MD SHA256: `01b7e8a963a25c09706a251283140fee1873940bb9af45dba885f152b9f4a148`
- JSON SHA256: `441d896a96730c1270d7ad4976042ccffa9cdee7619e51f92f80526bb4bce399`

I read the preserved trace metadata and terminal error at `R/.aris/traces/experiment-audit/2026-10-12_user_small_smoke_v2_first/`. QUOTA_ERROR_VERBATIM.txt SHA256 is `826de1e2d18833a6c9b2f4d03d81a60e1f3d196f81179092830a8f66dea40c47`. The trace records a terminal quota error after complete artifact delivery. It is preserved historical delivery evidence, not an experiment result or an instruction to purchase/reset anything. The first report's earlier `quota_interrupted=false` field described its save-time observation; this later terminal error is acknowledged here rather than rewriting the old JSON. Account availability reported by the parent was not re-probed by this reviewer, and no credits, resets or purchases were performed.

Only this new rereview MD and JSON were created. No first report/source snapshot, author source, raw input, schedule, environment/package, old plan/design/binding, automation or external process was modified. No extra reviewer was spawned.
