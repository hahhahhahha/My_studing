# Independent Type-A small-smoke V2 repair source review

Date: 2026-10-12, Asia/Shanghai. Reviewer: `/root/user_small_smoke_v2_integrity`. Requested reviewer configuration: gpt-6-astra / xhigh; provider-side model-routing telemetry was not independently inspected.

## Verdict

**FAIL — one current blocker, B3 (actual reward-process environment is not bound).** Original B1 (consumed raw/schedule continuity) and B2 (frozen executable closure) are resolved at the bounded source-review level, subject to the explicit point-in-time limitations below. Visual V2 changed-scope verdict: WARN. Reward V2 changed-scope verdict: FAIL. The strict 1800 MiB predicate itself is correct at source; this is not a live resource-readiness finding.

This report is one bounded independent Type-A review, not Type-B, actual smoke evidence, numerical success, an integrated-entry certification, pilot/formal authority, or permission to execute a visual-only subset. Both current launchers consume this combined review and should reject its FAIL status. The author/parent did not supply an expected verdict, and this reviewer did not modify the reviewed source to make it pass.

## Scope and path legend

All line references are one-based. The following prefixes are absolute:

- R = `X:/Yi-one/workspace/MPT-CVaR-MAPPO`
- F = `R/dac2027_followup_experiments_20261010`
- D = `F/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v2_repair`
- V1 = `F/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1`
- S = `F/scripts`
- O = `R/results_qla_joint_reward_ablation_20261010_repaired`
- E = `O/execution_source`
- P = `R/paper_doc_tools`
- BEFORE = `F/environment/attempts/20261012_1940_user_authorized_gpu1800/BEFORE`

The machine-readable report contains the exact nine current Windows `Path` strings and byte SHA256 values required by `require_review()`. The six D files, active resource entry/runtime helper/spec, original V1 files and its complete review delivery, frozen reward runner/intervention/repair chain, canonical schedule/data/training producers, visual dependencies, resource authority and original scientific protocols were inspected directly. Hash-only closure coverage is not a line-by-line scientific certification of all 115 files.

## B3 — Actual reward parent/private-worker interpreter and versions are not enforced

Priority: 1. Status: FAIL. This is the sole current blocker.

`D/reward_smoke_small.py:174-179` launches each private worker with the caller's `sys.executable`. Neither `run()` nor `worker()` requires that executable to equal the declared isolated `qla_np226/Scripts/python.exe`. `D/reward_input_integrity.py:110-112` checks only the canonical environment specification and the value 1800; it does not check the running interpreter or its Python/NumPy/Torch versions.

The private worker calls `witness()` at `D/reward_smoke_small.py:145-146`, but `S/qla_environment.py:94,106-107` runs the seeded kernel and version assertions in a **separate subprocess using the specification's isolated interpreter**. It then returns to the original private worker, which executes the frozen reward worker in-process through `runpy.run_path` at `D/reward_smoke_small.py:150-152`. That worker imports `mappo_gp_sf_dsr.train` at `P/joint_reward_ablation_worker.py:44-47`; train imports the model/Torch chain. No intervening actual-worker version/path assertion exists. The training manifest merely records Python/NumPy/Torch at `E/mappo_gp_sf_dsr/train.py:1696-1703`; the reduced validator does not compare those recorded versions to the spec.

A caller using another Python environment can therefore satisfy all file hashes, create the expected commands and private parent binding, obtain a successful witness from the isolated environment, and still reach training imports using the caller's different packages. This is a static counterexample, not a performed launch. The documented isolated command is a useful instruction but does not enforce the source contract. Similarly, `prepare()` can use caller NumPy when registering schedules without validating that process's declared version. The isolated witness's interop-thread setting is also not inherited by the worker: the frozen repair worker sets intraop threads at line 53 but does not establish its own interop setting.

Required disposition: repair the additive entry's actual-process identity/version enforcement before any model-bearing import, and verify the schedule-preparation process as well. A separate witness must remain a resource/kernel check, not be presented as proof of another process's package identity. Preserve the frozen source and old evidence; do not change scientific/numerical criteria to address this software boundary. This report implements no repair and grants no launch approval.

## Original B1 re-evaluation — resolved at source

`D/reward_input_integrity.py:16,65-75` pins FROZEN_BASELINE byte SHA256 `742d529b7f02301734a24f70de42e540a4b4ad5080870179762846174cf1387d`. Its 30-file KAIST map exactly equals the original independent review's recorded consumed map, rather than a newly selected dataset. Independent rehash found all 30 bytes equal and exact current sorted-first30 membership equal. `check_raw()` at lines 54-62 repeats the original loader's sorted-first30 rule (`E/repro_risk_averse_edge/data.py:41-51`), checking both resolved membership and bytes. An extra earlier-sorting member, renamed/missing consumed member, or modified consumed file fails. A later-sorting unconsumed trace is intentionally not rejected; it does not alter this loader's first30 consumption.

`build_schedules_once()` at lines 155-182 requires both the local schedule root and registry to be absent, creates a new root exclusively, uses the pinned original canonical implementation, and demands exactly four output files. It will not repair an existing/partial registration. The source fixes train seed1/32 and validation seed10001/8; users30, edges35, max requests7, feedback delay8, broadcast2, Poisson-bursty lambda3.5/burst0.1/multiplier2.0. The original loader's minimum trajectory length feeds the same canonical split. `canonical_protocol.py:46-60,120-188` retains guard10, 60/20/20 split, sequential modulo trace indices, `default_rng(seed)`, original request/assignment RNG ordering and capacity/background/loss arrays. `train.py:1105-1113,1325-1348` consumes those same settings. Only schedule storage relocates from the old shared root to D/schedules; the root is not part of the original metadata/content identity.

The NPZ/JSON pair file-byte map, per-role content SHA and schedule IDs enter SCHEDULE_REGISTRY and the new complete plan. `schedule_records()` at lines 144-152 requires exact whole file membership and hashes. `read_plan()` binds both the registry's bytes and complete value through `expected_plan()`; it runs before each arm, inside the private worker, after the resource witness/worker, and at final completion. `validate():120-124` compares actual training/validation content SHA fields to registration, not merely the other arm. Stable missing pairs, orphan counterparts, extra files, changed bytes or registry drift cannot enter either arm unnoticed. Pure-memory fixtures exercised exact/missing/orphan/extra/modified cases without touching actual data or generating schedules.

The old shared root is replaced only after an exact old-argument check (`reward_smoke_small.py:74-79`). In the stable checked state, the frozen loader takes its load-existing branch for both registered pairs; no old shared artifact is created or overwritten. This is a point-in-time workflow property, not atomic read-only filesystem enforcement. Concurrent deletion/replacement after checking could still race the old loader's create-on-missing branch; no claim of atomic locking is made.

No D/schedules, SCHEDULE_REGISTRY.json or SMALL_REWARD_PLAN.json existed at this review's inventory. Thus schedule creation/content equality has been reviewed as source, not observed as an actual generated artifact. B3 prevents treating the preparation/runtime environment as enforced.

## Original B2 re-evaluation — resolved at source

The 115-member execution-source map exactly equals the original review's 115-member map. Independent whole-tree membership comparison and byte rehash found no missing/extra/changed Python files. Exact Python `sorted(Path.glob('*.py'))` package hashing produced:

- mappo_gp_sf_dsr: `028c265e80d6b2f868a3e62179bb0cfd93c4ee78c581d76e6ef9bfca2872c4d9`
- repro_risk_averse_edge: `b7df6ba8f49f72711bb9d3879c1fcde3fffb4d785a1d81678ed4a61a8c1a2fe5`

Both match DESIGN. All 12 immutable baseline entries, six helper/runner/intervention/test digests through their fixed maps, V2 binding's two files and its original-binding hash were consistent. DESIGN remains `a9049bf9ba5d8b7ee1b8d1f0906222bc466bd4e09a2316ec389842e4d9734eb2`; the exact 25 historical Full references retain all 50 checkpoint/config hashes.

`source_guard():65-113` verifies the immutable maps, exact closure membership/bytes, authorized current and BEFORE resource bytes, original binding provenance, both package digests, helpers, consumed raw, reference count/bytes and canonical environment spec. `load_frozen():133-141` guards before and after dynamic loading. `original_module()` and `command()` guard before importing the runner and before/after its command-builder chain; thus the formerly unchecked `run_qla_simplified_rerun.py` is covered before it executes. The private worker checks the exact new plan, exact output arm, recorded parent PID, plan SHA and exact reconstructed arguments, then rechecks plan/source/schedules after witness and before runpy. Guarding recurs at arm validation, arm boundaries and completion. A mock rejected-source fixture confirmed `load_frozen` never reached the import hook when its guard failed.

The immutable original runner's 500 MiB formal preflight remains unchanged and is not invoked by this small command-builder use. The new shared 1800 gate is a separately identified authorized path, not a relabeling of the old entry. B2's frozen executable-byte gap is closed; the different actual-environment gap is B3. Guards remain hash-then-read, not an import sandbox or protection against hostile `sys.path`/installed-package substitution.

## Visual delta, science and producer/consumer audit

Full V1→V2 textual comparison and AST function comparison find only source_lock (two additional helper/baseline files), worker (require_review) and run (require_review) changed. SMOKE_SETTINGS is AST-identical. There is no algorithm, precision, optimizer, checkpoint, reward, numerical acceptance or budget change from visual small V1.

Visual retains both actual QLA-MAPPO/PPO-MV-A methods, 30 users, four servers, eight decisions at 50 ms, full fixed 30-second drain, 30 requests/s, schedule seed209, model seed1, geometry seed123, 0..500s training geometry, two CPU-1T/two CPU-4T profiles, existing 2 MHz upload and initial H=0. Dimensions remain actor19/critic705/users30. The 240 time-user samples yield 0.4s generation and 30.4s terminal horizon; the final 50ms settlement is inside the fixed drain. These are source-budget arithmetic, not measured execution counts.

The source retains pre-action critic features, per-user physical-time GAE before flattening, the historical QLA update hook and separate PPO loss, persistent MV stream and shared-receipt QLA q/H reward. Fixed-drain terminal accounting distinguishes completed/dropped/unfinished; unfinished training cost is not fabricated completion or a reliability pass. Exact time-user fields, finite arrays and terminal mask checks remain in `visual_training_coordinator.py:91-107`. `prepared_episode`, closed counts, ledger transactions and `METHOD_SMOKE_PASS` producer/parent keys align. No new deterministic output-key mismatch was found.

The visual worker checks unchanged collection parameters, performs a genuine closed-episode update, rejects unchanged post-update parameters, exports own three-payload checkpoint, reloads into the actual class, saves first PRE-action input and deterministic output witnesses, requires exact output equality and unchanged Torch RNG/parameters. `visual_training_transactions.py:224-256` validates descriptor/hash/named payloads; `visual_checkpoint_load.py:25-57` loads only the bound own snapshot, not optimizer/H/MV/RNG resume state. These are future runtime acceptance paths, not successful runtime evidence.

Reward remains Full versus NoJointDemandReward at 32 training/8 validation slots, seed1/delay8, 30 users/35 servers, original 1s slot and two-slot broadcast. The exact original command fixes CUDA and the frozen scientific arguments; only reduced counts and local schedule storage differ from original small V1. `joint_reward_ablation_core.py:13-46` changes only demand-weighted q to idle-gated uniform q and removes H*C from the reward; H recurrence, other terms, max/tanh scaling and model paths remain. Bounded [-1,0] reward checks and q/H diagnostics remain in the intervention worker. The original action-feature critic mask and per-user bootstrap are retained. 32×30=960 is below rollout1024, but `train.py:1308-1309` genuinely flushes the nonempty final buffer through the repair helper's second exact update anchor.

The reduced validator consumes produced `evaluation_split_role`, schedule SHA fields, actual config train/test counts, intervention `evaluation_slots`, repair flags, effective-reward hash, checkpoint/config names and 32/8 reward CSV counts consistently. It preserves finite PRIMARY metrics, mass tolerance 1e-6, drain_finished and zero unfinished checks. Across arms it compares actual registered schedule content and all config fields except output_dir. It does not acquire additional scientific coverage by being called V2; the limitations below remain.

## Resource authority and preservation

USER_AUTHORITY.json records the human request to change occupied GPU memory from strictly <500 to strictly <1800 MiB. Active qla_environment.py differs from BEFORE only in the canonical spec hash/comment; the spec differs only in schema and 500→1800. The two BEFORE files rehash to the old recorded bytes. Current spec canonical SHA is `8deaf4b6fbbf27332c795dcb5f08febde21269417f7879348e032643a9fe46f3`.

`qla_environment.py:100-107` rejects failed query, missing/malformed/unknown occupancy, wrong GPU identity and memory.used >=1800 before its seeded CUDA subprocess. Pure-expression fixtures confirmed 1799 accepts, 1800 rejects, N/A rejects, wrong-device and missing-fields reject. Python3.13.5, NumPy2.2.6, Torch2.11.0+cu128, RTX4060Ti, thread settings and seed123 kernel arithmetic remain in the witness. The visual worker/factory additionally requires the actual isolated interpreter before constructing a CUDA model (`visual_model_smoke_small.py:200-203`; `visual_learned_bindings.py:50-76`). B3 explains why the reward path does not have that same protection.

The old V1 JSON remains SHA `287499c7d1d61b189dc871349c389742dd2c2c78938bb731218feb41cf7cf6fb`, its completed MD remains `91bdbafd5118b6faa50eb7d7b2cfc52a9e1cbcc93bf984c3d1de46debd814549`, and DELIVERY_COMPLETION remains `6cd4c9885dcdda918129bdcb81602483d365c7abfff2b98d1568468c7d3e1986`. Its FAIL and recorded quota interruption are preserved, not reclassified as success. All four V1 author files and old 043949 plan/result remain equal to their recorded hashes. Of the original 207-source map, only the explicitly revised active qla_environment.py differs. All 232 protected bindings rehash unchanged. Of the original 50 primary files, differences are the two authorized resource files and the mutable VISUAL_INTEGRATION_STATE; the latter is not claimed unchanged. This is bounded preservation evidence, not a new universal audit of every historic failure/log file.

Old approval is not inherited. `require_review():122-130` now requires this new scope's PASS/WARN, empty blockers, Type-A, exact nine-file map and an existing MD. Reward embeds report hashes into its plan. Both actual visual and reward entry paths gate on the current review. FAIL is intentionally written here so the gate remains closed.

## Retained warnings and limitations

1. **W1, runtime unverified:** no actual smoke, model, optimizer, checkpoint load/export, CUDA probe or numerical result was run by this reviewer. Source/fixture success is not readiness.
2. **W2, finite-check asymmetry:** unchanged historical QLA optimizer steps can occur before the visual wrapper rejects nonfinite returned statistics; PPO checks loss/gradient before step. No numerical repair is authorized by this report.
3. **W3, early/abrupt failure evidence:** plan/guard/lock/mkdir errors can precede local try blocks; a local FAILURE.json is not guaranteed. Crashes/serialization failures may leave incomplete snapshots. Locks/partial evidence remain; resource witnesses are point-in-time, not GPU reservations. Reward V2 improves from one parent witness to a separate witness before each arm, but that does not resolve B3.
4. **W4, old plans have old scope:** old 16-decision/72-32 plans and their source WARN do not authorize this revision. No old small or formal completion flag is satisfied by the new small result name/root.
5. **W5, parent audit is narrower:** visual parent trusts detailed fixed-source worker assertions and does not independently rehash all child checkpoint/NPZ evidence. The 232 protected map is not a universal preservation inventory. PID/lock checks are workflow controls, not cryptographic proof that a parent is continuously live or protection from a hostile filesystem.
6. **W6, TOCTOU and installed packages:** hashes and later file loads are separate; transient concurrent changes, filesystem alias/inode substitution and package/driver bytes are not fully attested. Rechecks reject stable drift but do not make the workflow atomic. B3 is an actionable missing ordinary process-identity check, not merely this general limitation.
7. **W7, reward acceptance is limited:** checkpoint hash/existence is not a reward weight-roundtrip or proof of changed parameters; reduced reward validation does not check updates.csv finite loss/gradient fields. `abs(float(mass_error)) > 1e-6` does not by itself reject NaN. Four PRIMARY metrics are explicitly finite-checked. Generated/dropped/unfinished evidence and q/H records must be independently audited; no numerical criterion was silently changed.
8. **W8, prior documentary review gap is improved:** run/worker now programmatically consume the new review. This is a source workflow guard, not certification authenticity. Reward pins the report hashes into its plan; visual reads the review at run/worker entry but does not make the review itself a permanent lifetime source attestation. Current FAIL must be honored.
9. **W9, reduced scientific coverage:** eight visual decisions and eight reward validation slots do not establish tail, convergence, H activation, complete feedback-window or full-budget coverage. Reward exercises final flush rather than the old multiple-rollout-boundary pattern. No favorable-performance requirement should be added or failures explained away.
10. **W10, inherited check scope:** visual check still parses the original visual entry and loader, not its own relocated wrapper. This review AST-parsed the new wrapper, but a future source_entry_check_passed flag is not a fresh full independent API audit.

Full integrated entry, actual-evidence adoption, calibration/validation pilot freezing, human network/OOD decisions, and the 25 frozen Full 1e-8 replay requirement remain downstream boundaries. This review does not waive any of them or authorize formal training/test.

## Reviewer operations and fixture classification

209 unique Python files were AST-parsed without importing project entry modules, NumPy or Torch. Bounded extracted-function mocks exercised raw membership/bytes, schedule membership/bytes, incomplete/wrong review rejection and guarded dynamic import ordering. The resource condition was evaluated as an isolated AST expression. No real raw/schedule file was modified, no actual schedule was generated, and no check/prepare/run/_worker method was invoked.

Reviewer-harness corrections were kept separate from experiment findings: early PowerShell package-digest attempts used ordering not equivalent to Python Path sorting and were discarded; exact Python ordering verified both expected digests. One initial AST-only read rejected a UTF-8 BOM; decoding was corrected to utf-8-sig and all 209 source parses then succeeded. These were reviewer tooling issues, not experiment failures or quota events. No quota interruption occurred in this review delivery.

Only this new independent Markdown and JSON report are written by the reviewer. No author source, raw measurement, old source/plan/design/binding, environment installation, GPU process, automation or other chat was changed. The final JSON records FAIL with B3 and must not be edited by an executor into a passing verdict.
