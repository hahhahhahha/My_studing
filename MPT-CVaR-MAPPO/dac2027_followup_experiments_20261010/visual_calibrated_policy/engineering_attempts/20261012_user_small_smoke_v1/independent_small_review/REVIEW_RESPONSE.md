# Independent Type-A small-v1 source review — delivery completion

Original review date: 2026-10-12, Asia/Shanghai. Reviewer: /root/user_small_smoke_source_review. Requested configuration: gpt-6-astra / xhigh; provider-side routing telemetry was not independently inspected.

**Original combined source verdict: FAIL. Visual small-source verdict: WARN. Reward small-source verdict: FAIL. Two blockers: B1 and B2.**

This is the completion of the previously interrupted original review delivery, not a new review of the subsequent resource-policy revision. The original findings, severities and warnings are retained. This report grants no model execution, GPU readiness, full integrated entry, formal experiment or Type-B certification.

## 1. Delivery history and immutable scope cutoff

At resumed inspection on 2026-10-12T19:45:51.9535012+08:00, REVIEW_RESULT.json existed and REVIEW_RESPONSE.md did not. The saved JSON's SHA256 is 287499c7d1d61b189dc871349c389742dd2c2c78938bb731218feb41cf7cf6fb; its observed last-write time is 2026-10-12T16:20:16.1567565+08:00. The original reviewed state was materialized by that save, before the later resource revision. File modification time is a delivery observation, not proof of an exact instant at which every source was read.

The saved JSON contains complete original findings and hash maps but its complete_report=true field was premature as a two-artifact delivery claim. Its input_continuity_second_pass field also remains PENDING_REPORT_BACKED_READONLY_REHASH. Neither field is changed or silently treated as proof of absent work. This Markdown file is first created during the present completion. A separate DELIVERY_COMPLETION.json records actual artifact existence, immutable JSON hash, the bounded continuity verification and the earlier quota-delivery interruption. The earlier quota interruption is not a scientific result or an independent verdict.

The preserved prior error is at X:/Yi-one/workspace/MPT-CVaR-MAPPO/.aris/traces/experiment-audit/2026-10-12_user_small_smoke_v1_delivery/QUOTA_ERROR_VERBATIM.txt, with adjacent TRACE_METADATA.json. I read both during completion. Provenance is the original collaboration final-status payload copied by the parent, not provider logs; exact provider error timestamp is unavailable. The payload reported a usage limit and failed turn. Its credit/upgrade wording is preserved as historical error text, not followed as an instruction. The adjacent record observed the Markdown absent at 2026-10-12T19:43:45.2144156+08:00, consistent with my later independent absence observation.

The completion rehashed all four small-v1 author files and found exact matches to the saved original review. For the old environment entry/specification, it verified the preserved BEFORE files against the original hashes. It did not rerun the original full 203/232/115-file hash verification at the later time, and does not claim the active tree is still the original reviewed identity.

The resource change recorded after the original review — memory.used <1800 MiB — is explicitly OUTSIDE this original verdict. The original scope used <500 MiB. Current active qla_environment.py and QLA_ENVIRONMENT_SPEC.json now have different hashes, as separately recorded in DELIVERY_COMPLETION.json. The old bytes remain available under BEFORE and alone support the original resource-source observations below. Reading USER_AUTHORITY.json and README_RESOURCE_GATE_1800_中文.md here is contextual provenance, not approval or independent verification of the changed 1800 implementation. The old small-v1 source identity must not be relabeled as current launch approval.

## 2. Exact path legend

All line references are one-based. Abbreviations expand to these absolute paths:

- N/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/
- S/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/
- D/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/
- P/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/
- E/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/
- O/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/
- H/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_training_repair_pilot_20260930/source_snapshot/
- OLD/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_043949_046814_model_smoke_entry/
- PRIOR/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_231349_smoke_input_interface_repair/independent_rereview/
- BEFORE/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/attempts/20261012_1940_user_authorized_gpu1800/BEFORE/

IMPORTANT: every original-scope citation to S/qla_environment.py is now to the identical preserved BEFORE/qla_environment.py; every original-scope citation to D/protocol/QLA_ENVIRONMENT_SPEC.json is now to BEFORE/QLA_ENVIRONMENT_SPEC.json. The original JSON hash-map keys remain unchanged to preserve that historical record.

## 3. Review method and strict non-execution boundary

In the original bounded review I read primary sources and documents directly, compared the entire original/small visual source, traced consumers/producers and independently hashed original source and input maps. Author summaries and old source WARN were not substitutes for this new-scope review. I used static Python parsing and bounded in-memory extraction of non-neural command/budget/identity functions. I did not import the project entry modules or execute their check, prepare, expected_plan, read_plan, run or worker APIs.

No Torch import, GPU/resource probe, CUDA allocation, model forward, optimizer update, checkpoint load/export, actual schedule generation, experiment training, model validation or final test was performed by this reviewer. No raw timing labels were parsed for fitting. Raw trace files were byte-hashed, not used in a model. Author inputs, original artifacts, plans, bindings, designs, data and gates were not edited. Only new independent review artifacts were written.

The first in-memory analysis had an explicit UTF-8 decoding omission and exited during reading; the corrected bounded fixture completed. A later oversized Windows command request was rejected before process creation. These are reviewer-harness events, not failures of the experiment. The original all-input second-pass rehash was therefore not completed before interruption. This completion verifies only the four author files and two preserved environment files required by its narrow scope.

## 4. Per-check disposition

| Check | Status | Source conclusion |
|---|---|---|
| 1 — Source layout and reduced-budget plumbing only | PASS | Exact visual additive delta confirmed; bootstrap paths resolve correctly; isolated in-memory command extraction changes only 72→32 and 32→8; parser/config/manifest/reward-row consumers inspected. No new deterministic field mismatch found. |
| 2 — Unchanged scientific/model semantics at source, no runtime claim | WARN | Both arms/seeds/config paths and original visual update/checkpoint/exact roundtrip are retained; reward training/validation/drain and intervention unchanged. Reward runtime evidence coverage is narrower; finite/gate/runtime limitations retained. |
| 3 — Continuity and one-shot safeguards | FAIL | Visual exact source/protected/consumed membership contract and permanent lock retained. Reward canonical one-shot lock retained but reward source/raw/schedule closure is incomplete: B1/B2. No corruption observed in current files. |
| 4 — Claim and old-entry separation only | PASS | Preparation flags remain false for actual execution; small reward result name/root cannot satisfy original 72/32 completion/entry checks; formal budgets and old plan bytes unchanged. No retained-test fitting or fabricated runtime result produced. |
| 5 — Independent-review and authority boundaries | WARN | Prior WARN remains prior-scope WARN; this combined scope is FAIL, visual-only source delta WARN. Real runtime evidence audit and fresh integrated/formal entry prerequisites remain. Review prerequisites in small launchers are documentary, not programmatically consumed. |
| 6 — Independent report and complete hash manifest | PASS | Complete independent Markdown and JSON reports saved; all reviewed primary inputs have full hashes, plus 207 visual source/additive files, 232 protected bindings, 115 reward execution-source files and current 30 raw mobility hashes. Hash-only coverage is not substantive certification of every file. |

PASS in this table means a bounded source property, never executed-smoke PASS. The combined verdict is FAIL because check 3 fails.

## 5. Exact additive deltas, command flow and budgets

The full textual comparison found only the following visual changes: a three-line bootstrap to the original scripts directory; SMOKE_SETTINGS.decisions=8 instead of16; four adjacent additive files in source_lock; and propagation of decisions through schedule construction, common budget, assembly, collection loop, closed-count acceptance and result counts (N/visual_model_smoke_small.py:7-45,235-253,256-270,316). AST function-body comparison found source_lock and worker changed, with no other function body changed. The separate top-level decisions setting and bootstrap are additionally accounted for. The original file was not edited.

The relocated paths resolve correctly: visual __file__.parents[3]/scripts and reward HERE.parents[2]/scripts both point to D/scripts (N/visual_model_smoke_small.py:7-11; N/reward_smoke_small.py:19-24). The review parsed90 Python files without importing Torch, including both new Python files. That is syntax coverage, not an import/runtime test.

Visual dimensions stay30 users,4 servers, actor19, critic705 and IDs0..29. The formulas are3M+7 and N(actor_dim+M)+3M+3 (S/visual_episode_collector.py:89-91; S/visual_method_features.py:123-138); the worker supplies VisualPolicyDimensions(30,4,19,705,30). All8x30=240 behavior rows are retained, followed by a full fixed30-second drain, one update transaction and own checkpoint. In-memory CommonBudgetCandidate((8,)) accounting yields0.4 seconds of generation and30.4 seconds terminal horizon. Last settlement is0.45 seconds and lies inside that fixed drain, not an extra drain extension (S/visual_receipt_environment.py:70-101; S/visual_training_transactions.py:64-71). Four epochs with batch128 imply8 nominal optimizer minibatch steps per visual arm, not an observed runtime count.

Reward command(out,design) calls the frozen original command builder and requires exactly one old train72 and test32 flag, replacing only their values with32 and8 (N/reward_smoke_small.py:65-77). An independent in-memory AST extraction, using an artificial output string and no dynamic project import, produced only those two token differences. It retained CUDA, Full, seed1, nominal delay8, validation role,30 users,35 servers, slot1.0s, broadcast2, lr.001, rollout1024, batch128, QT, edge-wise actor, bias.5, no stale margin, no Z, original load/arrival/network configuration, shared canonical schedule root and raw trace root. No load-checkpoint flag was added.

The actual source chain is P/run_joint_reward_ablation.py:220-239 -> P/joint_reward_ablation_worker.py:29-48,116-123 -> E/paper_doc_tools/paired_training_repair_worker.py:37-60,167-179 -> E/mappo_gp_sf_dsr/train.py. The parser accepts integer train/test budgets at train.py:1749-1750. Training iterates the requested32 slots, with30 user samples per slot.960 samples is below rollout1024; the final nonempty-buffer flush at train.py:1308-1309 still calls the real update hook. The helper rewrites both normal-boundary and final-flush update sites (E/paper_doc_tools/paired_training_repair_core.py:53-67). Thus reduced training is not a dead/no-update path, conditional on runtime reaching it.

Validation consumes8 slots with seed10001 and validation role at train.py:1325-1348. Config fields retain actual args and requested budgets; checkpoint/config/manifest are exported at train.py:1684-1713. The new validator accepts exactly32 train and8 test config slots,8 evaluation slots and32/8 reward row counts (N/reward_smoke_small.py:105-116). It compares actual train/validation schedule hashes and all non-output config fields across arms (:155-160). No deterministic producer/consumer key mismatch was found in this bounded source chain.

## 6. Scientific conditions, models and evidence checks

Both visual methods remain QLA-MAPPO and PPO-MV-A on a method-independent calibration schedule,30 requests/s, schedule seed209, model seed1, geometry seed123, training window0..500s, profiles CPU-1T/CPU-1T/CPU-4T/CPU-4T,50ms decisions,100ms broadcast and nominal delay, waiting capacity32, original2MHz upload and initial H=0 (N/visual_model_smoke_small.py:29-36,229-253). The actual two-arm schedule and input-condition comparison remains in the parent at :376-388. Physical mobility uses raw10s timestamps with interpolation; calibration normalization uses only calibration measurements and the training window (S/physical_mobility.py:40-79,84-103; S/visual_method_features.py:39-58).

QLA keeps its historical edge-wise actor, centralized pre-action critic and fixed optimizer settings; PPO keeps its historical local critic/global log_std, not a relabeled QLA reward (S/visual_learned_bindings.py:69-79; S/visual_neural_controller.py:43-76,101-103). Per-user physical-time GAE precedes flattening with gamma/lambda raised to interval duration, and exact finite time/user axes are checked (S/visual_temporal_gae.py:7-51; S/visual_training_coordinator.py:91-107). Sampled float32 portfolios are retained while the existing float64 unit-sum execution bridge is unchanged (S/visual_episode_collector.py:124-145). Neither numerical tolerances nor precision were changed by the small visual delta.

QLA's shared receipt reward, prior-H recursion and terminal accounting remain; PPO uses the persistent MV stream, not the obsolete reset-per-episode path (S/visual_shared_reward.py:72-103; S/visual_streaming_episode.py:20-60). Reduction does not alter these formulas. The old per-episode collector still exists as a separate class; the small worker intentionally instantiates CompactCoordinatorAssembly -> StreamingEpisodeAssembly.

The visual worker checks unchanged parameters during collection, consumes a real closed-episode update, rejects equal before/after parameter hashes, exports own checkpoint and invokes the original own-weight loader (:263-280). It saves actual first pre-action input plus original/loaded deterministic outputs and requires exact array equality; it separately persists CPU/all-CUDA RNG arrays around a deterministic action and checks RNG and parameter immutability (:281-305). Checkpoint descriptor, exact three-file payload membership, bound marker hash, model class and no training-state resume constraints remain (S/visual_training_transactions.py:187-205,224-256,260-311; S/visual_checkpoint_load.py:25-57). These are retained future execution checks, not existing runtime evidence.

Reward Full/NoJointDemandReward still changes only demand-weighted q to idle-gated uniform q and removes H*C from reward, with unique AST anchors; H state update, bounded max/tanh scale, latency/migration terms and model/QL paths stay unchanged (P/joint_reward_ablation_core.py:13-46; P/joint_reward_ablation_worker.py:55-85; E/mappo_gp_sf_dsr/rewards.py:248-349). Critic current-action features are zeroed and both regular/final bootstrap are retained (E/paper_doc_tools/paired_training_repair_worker.py:59-72,93-127,162-179). This is the old35-server/1s-slot reward study, not the4-server/50ms visual setting.

The reduced reward validator retains finite PRIMARY metrics, mass-error tolerance1e-6, drain_finished and zero unfinished checks (N/reward_smoke_small.py:95-104). The original drain disables arrivals and drains queues under the fixed original horizon; finalize_evaluation retains uncapped metrics and drop/unfinished accounting (E/mappo_gp_sf_dsr/env.py:904-943; E/mappo_gp_sf_dsr/metrics.py:291-305). Limitations of acceptance are W7 below, not silently strengthened claims.

The original resource-source scope retains exact isolated interpreter/package versions, CUDA device identity, <500MiB check before the seeded CUDA kernel, single-thread settings and no CPU fallback (BEFORE/qla_environment.py:39-47,90-108; BEFORE/QLA_ENVIRONMENT_SPEC.json; S/visual_learned_bindings.py:50-68). The later1800 revision is excluded, and no live resource measurement was made.

## 7. Blocking findings

### B1 — Reward consumed raw membership and schedule identity are not bound to the one-shot plan

Status: FAIL; priority 1.

The reward plan binds four new files and three old binding/design files, not the actually consumed sorted-first-30 KAIST trace paths/bytes or exact 32/8 schedule contents. prior.check verifies bindings/design and historical checkpoint bytes, not those inputs. The loader discovers raw traces anew; schedule metadata hashes length and dimensions, not coordinates or selected filenames. A stable changed trace with the same length, or a same-shape earlier-sorting replacement, can change geometry/inputs without changing the plan or both-arm schedule equality. This is a static counterexample, not an executed mutation. The canonical schedule generator also writes into the historical shared root and overwrites an orphan counterpart if only one of NPZ/manifest exists. No matching 32/8 manifest was found in the current bounded inventory, so no actual overwrite is asserted.

Evidence:

- X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/reward_smoke_small.py:46-56,119-162
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py:54-62,78-90
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/run_joint_reward_ablation.py:220-239
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/repro_risk_averse_edge/data.py:41-59
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/canonical_protocol.py:124-188

Required disposition, not work authorized by this report: Before this reward revision may pass its stated continuity contract, separately authorize a new additive correction that binds and checks the original consumed trace membership/bytes and exact schedule creation/consumption policy before either arm and at completion, rejects changed or partial existing artifacts, and preserves all old evidence. Do not implement or reseal under this review.

### B2 — Reward execution-source closure is incompletely checked before execution

Status: FAIL; priority 1.

The stored DESIGN includes both package digests and repair helper digests, but prior.check does not rehash execution_source. command dynamically executes run_qla_simplified_rerun.py before the worker's mappo-only package digest. The worker digest excludes repro_risk_averse_edge, which supplies data/config/metrics, and the new validator never checks its stored digest. Repair worker/helper hashes are verified only after their execution. Therefore altered executable inputs can run before detection, and a changed repro package can remain undetected. Current package and entry bytes independently match the design; this is a missing runtime enforcement boundary, not observed source corruption.

Evidence:

- X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/reward_smoke_small.py:46-56,65-77,87-94,134-154
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py:54-62,78-90
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/run_joint_reward_ablation.py:220-239
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/joint_reward_ablation_worker.py:38-47,116-123
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/paper_doc_tools/paired_training_repair_core.py:45-50

Required disposition, not work authorized by this report: A separate additive correction and independent rereview must verify the complete consumed execution-source closure and helper hashes against the existing frozen design before dynamic import/command construction/worker launch, and recheck at completion. Preserve the original bindings and design unchanged.

The B1 counterexample needs no adversarial concurrent writer: changing selected coordinate values while preserving file length/shape leaves schedule metadata and random schedule generation unchanged, while geometry and model inputs differ. Both arms can agree with one another and still fail original-input continuity. A directory insertion can likewise replace a first30 selection. No raw file was changed to execute this scenario.

B2 is also not a claim that today's package bytes are wrong: independent byte checks found both package digests equal the stored design. The defect is that the future execution path does not enforce the complete stored closure before running it. Self-reported post-run helper hashes do not replace a pre-execution guard.

## 8. One-shot identity and artifact preservation

The visual complete parsed plan is compared canonically to expected source/input/settings values; extra fields and distinct numeric/boolean JSON values fail. A bounded artificial in-memory identity fixture accepted whitespace/key-order equivalence and rejected three altered cases. This did not create, register or load any actual plan. Visual source locking adds exactly four author files to the original203-source map, yielding207. The232 protected map is retained from pinned original preparation. Consumed calibration/mobility maps must equal registered paths and hashes, not merely agree across arms (N/visual_model_smoke_small.py:39-100,229-247,305-307,381-388).

Visual permanent semantic lock creation precedes worker/model execution and output directory existence is exclusive; worker parent PID/output/method identity is checked. Sequential worker failure stops aggregate success and no retry/merge branch is provided (:191-210,336-400). Update tokens are burned before optimization. Reward save uses exclusive creation and its canonical semantic lock is written before witness/allocation; output mkdir is exclusive and each failed arm stops rather than merging existing arms (N/reward_smoke_small.py:39-43,119-168). These positive one-shot controls do not cure B1/B2's incomplete identity contents.

The new reward wrapper does not call original run_cell, which contains append/reuse behavior; instead it launches fresh child output paths. The small result is SMALL_TWO_ARM_COMPLETE.json in D/reward_smoke_attempts, not O/TWO_ARM_SMOKE_COMPLETE.json or O/SMOKE_V2_COMPLETE.json. Original entry-check still requires those old artifacts and actual72/32 validation (S/reward_followup.py:106-125; S/reward_followup_v2.py:66-78; P/run_joint_reward_ablation.py:269-290). Small completion cannot automatically pass old72/32 or full entry. Neither old design nor old result/status is written by the new wrapper.

Visual output keeps independent_complete_entry_review=false, formal_protocol_frozen=false and formal cell counts zero. The separate formal coordinator requires scope complete_visual_entry_after_actual_model_smoke, a frozen pilot protocol, exact source maps and actual audited service/smoke evidence (S/visual_training_coordinator.py:47-88). This review's different scope, FAIL status and absence of runtime cannot satisfy it.

## 9. Complete warning list

### W1 — No actual runtime, CUDA, update, checkpoint or roundtrip evidence (prior W5-runtime)

All reviewer checks were source/byte checks or bounded non-neural in-memory fixtures. New small plans were absent at inspection. README's GPU statement is historical author context, not a live reviewer probe. Runtime/import compatibility and numerical success remain unverified.

### W2 — QLA finite statistics are checked after original optimizer steps (prior W3)

visual_neural_controller.py:118-125 wraps historical model update; models.py:630-641 can backward/clip/step before the wrapper rejects nonfinite returned stats. PPO checks loss and gradient before step at visual_neural_controller.py:158-165. This unchanged asymmetry is not repaired or upgraded.

### W3 — Early failures, partial snapshots and point-in-time resources (prior W4)

Visual plan/guard/lock/mkdir validations precede local try; reward lock and output mkdir precede try. Existing locks remain even if mkdir fails, but a local FAILURE.json is not guaranteed. Abrupt termination, serialization of nonfinite partial data and process crashes do not guarantee complete partial model snapshots. GPU witness is point-in-time, not a reservation; reward witnesses once before both workers, visual per method. No gate relaxation is authorized.

### W4 — Old 203-source/16-decision plan remains valid only for its own old scope (prior W5-plan)

The old 043949 artifacts rehash correctly but are not the 207-source/8-decision small plan. They were not executed, resealed, relabeled or overwritten. Small-source review cannot convert original preparation or prior WARN into small runtime PASS.

### W5 — Visual aggregate and preservation coverage are narrower than a fresh evidence audit (prior W6)

The parent relies on fresh fixed-source workers for detailed flags and witness correctness; it does not independently rehash every child NPZ/checkpoint or validate every result field. The 232 protected bindings are not a universal enforcement of the old 548-file preservation inventory. One-shot files are workflow controls, not hostile-filesystem attestation.

### W6 — Non-atomic hash-then-read and installed-package limits (prior W7)

Catalog hashes bytes then reopens CSV; several checkpoint/predictor readers separately hash and load. Stable changes are checked on visual boundaries, but transient concurrent mutation is not excluded. Installed packages, drivers and filesystem links/inodes are not fully attested.

### W7 — Reward acceptance is not the visual weight-roundtrip contract

Reward training has a final nonempty-buffer update and checkpoint export, but the small reward validator checks checkpoint existence/bytes only; it neither compares pre/post parameter hashes nor reloads a checkpoint/read-only outputs nor checks updates.csv finite loss/gradient fields. This is inherited limited reward-smoke coverage, not a regression of an original reward roundtrip check. Mass conservation abs(float(value)) > 1e-6 also does not reject NaN by itself; only PRIMARY metrics are explicitly finite-checked. Actual evidence must be audited without claiming visual roundtrip guarantees for reward.

### W8 — Fresh source-review prerequisite is documentary in the small wrappers

Both new wrappers state review requirements but do not read this review result to gate run. Human/orchestrator control must honor FAIL and remaining runtime authority. The distinct formal coordinator does require a complete_visual_entry_after_actual_model_smoke review and frozen protocol; this source-only scope cannot satisfy it.

### W9 — Reduced coverage must remain a functional smoke only

Visual 8 decisions gives 240 samples, 0.4 s generation, 30.4 s terminal horizon and nominal 8 minibatch optimizer steps/arm, not an observed update count. Reward 32x30=960 samples falls below rollout1024, so the final-buffer flush exercises one update transaction rather than the old multiple-boundary pattern. Eight validation slots at nominal delay8 are particularly shallow. No requirement that H be active or complete receipt windows occur is proved. No performance, tail, convergence, full-budget, formal-entry or Type-B conclusion follows.

### W10 — Inherited check parses original visual entry, not the new entry

visual_model_smoke_small.py:126-128 still AST-parses BASE/scripts/visual_model_smoke.py and the loader. The new file is source-hashed, and this independent review parsed both new Python files, but a future check's old source_entry_check_passed flag alone is not a separate independent syntax/API review of the relocated wrapper.

W2-W6 preserve the substantive earlier W3-W7 limitations; they are not upgraded to new-code PASS. The original prior report remains an independent statement about its own earlier source identity, not this reward/scale-reduction scope.

## 10. Original independent hash evidence and completion continuity

The original independent rehash results saved in REVIEW_RESULT.json were:

| Map | Count | Mismatches observed at original inspection |
|---|---:|---:|
| Old visual source map | 203 | 0 |
| Protected original visual input/source map | 232 | 0 |
| Small visual expected source/additive map | 207 | Inherits matched203 plus4 freshly hashed author files |
| Reward execution Python inventory | 115 | Both aggregate package digests matched design |
| Current selected reward mobility files | 30 | Recorded current bytes; not falsely called a reward runtime binding |
| Primary directly reviewed/supporting artifacts | 50 | Full hashes saved |

The old visual plan remains the16-decision/203-source plan, with byte SHA4871dc429bb7d2bdb96f22aaae2beccbeaa6921325aaaedce302f356c9014baa. Its old RESULT reports preparation-only and zero actual CUDA/model/roundtrip calls. Neither was rewritten or run. At original inspection no new small reward plan or same-directory small visual plan existed; current later filesystem state is not inferred from that earlier observation.

The reward package digests observed were mappo_gp_sf_dsr=028c265e80d6b2f868a3e62179bb0cfd93c4ee78c581d76e6ef9bfca2872c4d9 and repro_risk_averse_edge=b7df6ba8f49f72711bb9d3879c1fcde3fffb4d785a1d81678ed4a61a8c1a2fe5. Both existing additive entry bindings' mapped files matched. This independent static verification does not insert missing runtime checks.

Completion-time continuity (all matches are exact SHA256 equality to saved original review):

| Original small-v1 author file | SHA256 |
|---|---|
| USER_REQUEST.json | 577575609ff2939258aeacc5b8efbc0931a799c5b5c05b1fc044d0c54e576415 |
| README_SMALL_SMOKE_中文.md | 0817961b23f84a6e8056fd77b62f3a460b535e5c2fc06d5c31f5dcd41f1eb484 |
| visual_model_smoke_small.py | 5c8028fbde9a2b7b706ef1acc8b12f3f46d8da5eb24256cfd2f32884418c2d9d |
| reward_smoke_small.py | 353b3f95f433185e64b4dd4edf633ed91682ac50e8d46a8de8a21142016de0d3 |

| Preserved original environment file | SHA256 |
|---|---|
| X:\Yi-one\workspace\MPT-CVaR-MAPPO\dac2027_followup_experiments_20261010\environment\attempts\20261012_1940_user_authorized_gpu1800\BEFORE\qla_environment.py | f20497f0725fcef76e3581d4d423c684cc2b115d750e55df85d72fdca62d1d91 |
| X:\Yi-one\workspace\MPT-CVaR-MAPPO\dac2027_followup_experiments_20261010\environment\attempts\20261012_1940_user_authorized_gpu1800\BEFORE\QLA_ENVIRONMENT_SPEC.json | a6a440c36c4a0ebe188145c29affbdc0fd83353838f9c745b523d79f556e9cf4 |

## 11. Primary artifact hashes at the original review cutoff

Full original maps, including source closure, protected input bytes and current trace inventory, are retained in the immutable REVIEW_RESULT.json. Hash-only inventory coverage does not mean every algorithm or every possible entry in those files received substantive certification. The two environment paths in the table below are historical keys; the identical old bytes are now verified at BEFORE, not the changed active paths.

| Original reviewed path | SHA256 |
|---|---|
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/USER_REQUEST.json | 577575609ff2939258aeacc5b8efbc0931a799c5b5c05b1fc044d0c54e576415 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/README_SMALL_SMOKE_中文.md | 0817961b23f84a6e8056fd77b62f3a460b535e5c2fc06d5c31f5dcd41f1eb484 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/visual_model_smoke_small.py | 5c8028fbde9a2b7b706ef1acc8b12f3f46d8da5eb24256cfd2f32884418c2d9d |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v1/reward_smoke_small.py | 353b3f95f433185e64b4dd4edf633ed91682ac50e8d46a8de8a21142016de0d3 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_model_smoke.py | 1fe79512edac9023cce2dcffee063aba5891f14c34f12c2a0518a82e8bf31059 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_checkpoint_load.py | 02c54e06e95f10e1febfb0829c30c28a3ad1d88220d29cc7b9608b0fb6a0ff8e |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_transactions.py | 5f8b19f28b9757194bb88e142bb1db639bc4067c01ae9c36475de27e016832b6 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_coordinator.py | bcda7f4dadba74246ab563d85a8b116fe5b2e25bc08cfe6c4c17a9a7660df3bf |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_neural_controller.py | 9a5bae58b66665e58cde564aa231587b92963cea41bbc2f1937b223aa6f3ea5e |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_episode_collector.py | 06cdc1ca534407c6d0e2feb177dfaafa2b9984186282398b8b67181d6448ef63 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_request_schedule.py | 52fa48eeef2ca3715a2441276d92dbbd376c9c44ad0c784eb907ba4d157aa90e |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_learned_bindings.py | 9fb8399b73547ad8e525a543f7ce1f4aa1941e84da8d1c8b23a871bba2019416 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py | f20497f0725fcef76e3581d4d423c684cc2b115d750e55df85d72fdca62d1d91 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py | b3351e64ac522664d2cfed3f796560be36e0066311ce25504385ac8a4c853917 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py | 8090e9de0271156a4ee7d14558630b4f9c4329e5dac98b036cafd63271fba63f |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/windows_runtime.py | 279e70e8b78ea62b397da75b20177adeb36849073d17541f6e4bd8b4d43b07a0 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/physical_mobility.py | 7b187f792cd3306a699a2116e692e388882f7955fe6f46a810b64b9d022a80ab |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/execution_service_catalog.py | 94f353ed9159e52ccc5a85e29fe6acdab8a73ccc45261f6cfd768d77357b9424 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_streaming_episode.py | 4c222338e892fd1e6d7496c37f70ad4adcddb1e1c8af2a4dc7bb5b33a97086d6 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_method_features.py | b74ed6336c6dc44c0041d1427f904a22b022cc4fd6a57dfc086b226aa58bd722 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_shared_reward.py | 46b6fd5ba9adfae286d736c31b64841915ba17bfeb91b8f86852f2bd8a764faf |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_receipt_environment.py | ae933775ca365e0301eef1a14ad8f118d59adc2b43bef0af3c8f5e94be998036 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_temporal_gae.py | 53aaf0a77eb55437fef3e408e09c4ff4df7bba7e4ab0acf1daeb7f0208c4e46d |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_learning_buffer.py | 1192e72c2e7c77986108f3c7c623082b531e7f85bc41e405944630b82dbc259c |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_mv_stream.py | aacedf9b8acd063e80b1c2a6391aed0b9f6e45dec118ba33a7d0f331327b94ec |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_environment_adapter.py | 66efc2c045f2894b7d2a800c3e269ad61718b4b73cb22c148a8a711b30603723 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/run_joint_reward_ablation.py | 7022349105cae59caffd6e9fd04aa12fc6abf6a775675cfcb32fc48af3c2b4a4 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/joint_reward_ablation_worker.py | dc5df7f043144f481ec14d6342a4145e7884475982566eec67eb7210ee87b630 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/joint_reward_ablation_core.py | 5a4d6584cbca59c5e247176b10d4391cbed2281a569da9d045926a342ca1b8a5 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/ADDITIVE_ENTRY_BINDING.json | 6aeddbc363876fbb03021be01050a1e9490e6eae3ea3d69c9c7f9edd93322ebf |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/ADDITIVE_V2_BINDING.json | ca255421821ee5fc07fc88ad7c65be511deba482b543661c5f3375b2943456ba |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/DESIGN.json | a9049bf9ba5d8b7ee1b8d1f0906222bc466bd4e09a2316ec389842e4d9734eb2 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/STATUS.json | bc25014b7eb106d72c9e72374eca82865cb8214526b364120e159ca4daa871a2 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json | a6a440c36c4a0ebe188145c29affbdc0fd83353838f9c745b523d79f556e9cf4 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/VISUAL_INTEGRATION_STATE.json | 590f42aab621e0986494409bc277a147a12c51505cb110a70e521d28c7397e82 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_043949_046814_model_smoke_entry/RESULT.json | 9fc61040fca578c2308bb841c59a834bcc81c9d8a58970fd0b2168893ee4d963 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_043949_046814_model_smoke_entry/SMOKE_CANDIDATE_PLAN.json | 4871dc429bb7d2bdb96f22aaae2beccbeaa6921325aaaedce302f356c9014baa |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_231349_smoke_input_interface_repair/independent_rereview/REVIEW_RESPONSE.md | 68fdcd6e6350377c8a76f1bfa50553938c0d0d14848e216598393169ce0eb119 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_231349_smoke_input_interface_repair/independent_rereview/REVIEW_RESULT.json | 16827cbc26dc2f45a9b0c392f5789ea6def3619ca3abfb828eebdedf2e5fc143 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261009/PROTOCOL_中文.md | 44cfa1a0df369c2e2085692a3d6ecc269cb60fe19da1ef8567034d5262f635ea |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md | e85b0ce792b47480f9e69ee1eed2e335118815f8112603628a1f261011341d59 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/paper_doc_tools/paired_training_repair_worker.py | 4992925825afb20c9176e9d6742bc039ae296c650fa14526217c9bd8e65c4d02 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/paper_doc_tools/paired_training_repair_core.py | b42b3ebbb881f3586b5c76f058daeac4f8c4a97351554a7cb3131770e4371357 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/run_qla_simplified_rerun.py | b63fdd99ce7ebfb575bbdaf34f0b4c9b1954083567e7db30646d4ee55bc48e4f |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/train.py | a6487286efe80269781634cf477fbc740e0886caa8e0286377e9c4783771e25d |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/canonical_protocol.py | 7c3bc4bf82d17ca17309f0b8cb2be23dbb0c7f4faa1cf1ce1c864c76d27b77c2 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/rewards.py | f786cca5c985395a272ca91a3ab97ed6e240229e9f68544d3dee160d5778b3d4 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/evaluate.py | 440c0d76edfda3430244ab89f1804c6dce958b16f0dfbee0fa98a0fd4e09783a |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/mappo_gp_sf_dsr/models.py | 770d4791b4a1ddc86e89469d49dc1aeb84f69493b5c43b8b327509bc8a528b85 |
| X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired/execution_source/repro_risk_averse_edge/data.py | 6256e434d424523d8a520b8a3c5130d9da3e82ec82887ccc77e504c4bdb909e6 |

## 12. Decision and limits

Retain original combined **FAIL**, with visual-only source **WARN** and reward source **FAIL**. B1 and B2 require separately authorized additive correction and a fresh independent review; this completion performs neither. Do not overwrite/reseal old plans, bindings, source or failure evidence. No runtime result is fabricated to fill the absence of execution.

The original reward formal budgets remain10000 training/1000 validation/1000 test slots; the visual proposal's ten trainings and120 test cells are not covered by this functional smoke (original reward protocol:24-52; visual calibration proposal:487-516,624-659). No performance superiority, convergence, tail-quality, test selection, deployment or mechanism-contribution claim is established. Negative or incomplete numerical outcomes must remain evidence, not a reason for retries or tolerance changes.

Delivery completion creates this previously absent Markdown report and a separate factual delivery record while retaining REVIEW_RESULT.json byte-for-byte. It does not change original verdict, erase warnings, certify the later1800MiB resource policy, revive old launch identity, approve runtime/full entry, or provide Type-B certification.
