# Independent B4 command-source rereview

Date: 2026-10-12 (Asia/Shanghai). Review type: **Type-A**. Verdict: **WARN**, with **no blockers in this bounded changed-source scope**. B4 is resolved at source level. Prior warnings remain; this is not an actual smoke result, runtime readiness certificate, full integrated-entry review, Type-B review, or pilot/formal authorization.

## Scope and retained evidence

`D3` denotes `X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v3_windows_parent`.

I compared the current integrity helper, reward entry, visual entry and README directly against `D3/SOURCE_BEFORE_COMMAND_REPAIR` and the complete first V3 FAIL report. All six preserved pre-repair files match the first report's exact source map. Its complete MD and JSON remain unchanged:

- `independent_repair_rereview/REVIEW_RESPONSE.md`: SHA256 `6ccd9fcbf49025c88bdb4b371670499ddd34cdca06e549135d963585a8c461be`.
- `independent_repair_rereview/REVIEW_RESULT.json`: SHA256 `65d50b9166eef905ff262caa493f856bd85a15d37b5293541defa9ddda515d9d`.

AST comparison shows only `require_worker_parent` and `require_review` changed in the helper, and only `worker` changed in each entry. There are no added/removed functions. Both parent `run` functions and the receipt validator are unchanged. The frozen baseline, user request, and three public environment files still match their previous reviewed hashes. The accompanying JSON binds exactly the six current D3 files plus the three current public environment files using absolute Windows paths.

## B4 disposition and bounded tests

The reward worker now constructs its canonical invocation from the entry path and parsed plan, plan SHA, resolved output, arm and remaining frozen arguments (`reward_smoke_small.py:146`). Remaining arguments must first equal the frozen `command(output, arm)[3:]`; the complete command must still match the parent's pre-Popen command artifact. The visual worker independently constructs its invocation from the entry path and parsed method, plan, plan SHA and resolved output (`visual_model_smoke_small.py:215`). Neither canonical assignment expands `sys.argv`.

The helper still requires program arguments to equal the independently reconstructed canonical suffix. It now also reads `current.cmdline()` and compares the observed OS command to that canonical command **before either direct-parent or single-launcher acceptance** (`reward_input_integrity.py:102`). The executable argv[0] is compared as a resolved path; all remaining tokens are compared exactly, preserving order and multiplicity. Thus this is exact token checking after argv[0] path normalization, not a claim that original shell quoting or equivalent executable path spelling is preserved. The receipt additionally stores `actual_OS_worker_command`, separate from the canonical command.

The existing one-launcher branch continues to check the actual launcher executable and its full command. This does not require the native current worker executable to be the isolated venv executable: the Windows venv redirection may use the base native executable while the declared argv[0]/`sys.executable` identify the isolated environment. A fake base-native current worker with the correct isolated command was accepted in the bounded fixtures. No live Windows diagnostic was executed by this reviewer.

I extracted only the relevant helper functions and the two canonical assignment statements into an isolated standard-library fixture, with fake process objects, a stubbed source guard, and a fake `psutil` import. Outcomes:

- 4 intended exact cases accepted: reward and visual, each with direct logical parent and one declared launcher.
- 24 malformed current-worker cases rejected: missing `-B`, extra interpreter flag, reordered interpreter flags, extra script option, reordered script options and duplicate script option, across both entries and both parent structures.
- 12 malformed launcher-command cases rejected: the same six token variations for both entries.
- 8 parent/receipt regression cases rejected: changed logical-parent creation time and wrong Popen PID for each intended case.

All expected fixture outcomes passed. In particular, the two previous B4 counterexamples no longer pass. These are in-memory source fixtures, not calls to actual workers or a real process/resource guard.

The unchanged ancestry mechanism retains the recorded live logical PID and creation time, same entry and `run` action prefix, direct-parent consistency, exactly one permitted Windows launcher, and no arbitrary ancestor search. The unchanged parent receipt checks still bind `launched_pid` to the actual Popen object and distinguish launcher PID from native worker PID. The new comparison adds no dependency on a parent file created after Popen; the prior race-avoidance arrangement remains intact. The logical-parent check remains an entry/action-prefix check rather than equality of every trailing parent-run argument, as recorded in the first review.

## New review binding, invariants and limits

`require_review` now reads only `D3/independent_command_rereview/REVIEW_RESULT.json` and its MD, so the preserved first FAIL cannot serve as the changed-source prerequisite. Its status/type/empty-blockers/exact-nine-hash-map/MD-existence conditions are unchanged. A separate mocked reader test accepted a current WARN/empty-blockers result and rejected FAIL, nonempty blockers, Type-B, a stale source map and a missing MD; all reads targeted the new directory. The reader still returns both report hashes for plan binding.

No algorithm, small budget, numerical acceptance, input selection or environment guard changed in this delta. Reward remains 32 train/8 validation, 30 users/35 servers; visual remains 8 decisions, 30 users/4 servers and the full 30-second drain. The existing source-closure/raw/schedule guards, caller/interpreter/package checks, current strict `<1800` policy and model-import ordering, actual worker runtime/threads/GPU checks, scientific criteria, one-shot locks and no-retry/no-merge policy are inherited unchanged. This is a comparison finding, not a fresh full audit or execution of the frozen closure. New source-bound plans and independent output paths remain necessary; old failure evidence, locks and plans are not reusable approvals.

Retain the complete limitations and warnings in `D3/independent_repair_rereview/REVIEW_RESPONSE.md` and the D2 reports it cites. In particular, runtime/numerical success remains unestablished, small budgets do not establish formal scientific performance, and point-in-time process/resource/hash checks retain their TOCTOU and evidence limits. No new broad protocol audit or numerical repair was undertaken.

Only the new review MD/JSON were written. No source, old review, old artifact or schedule was modified; no GPU/witness/check/prepare/run/private-worker/model invocation, real experiment import, real process-chain diagnostic, schedule generation or additional reviewer occurred. No quota interruption occurred.

**Final disposition: WARN; B4 closed at source level; blockers empty.** This report can serve only as the requested changed-source review evidence under the existing user-scoped small-test workflow. It does not itself assert live readiness, authorize expanded execution, certify a completed smoke, or grant full-entry/Type-B/pilot/formal approval.
