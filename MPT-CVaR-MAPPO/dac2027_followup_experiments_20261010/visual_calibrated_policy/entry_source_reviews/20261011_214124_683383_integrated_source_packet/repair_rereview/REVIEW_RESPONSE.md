# Independent Type-A source repair rereview

Date: 2026-10-11, Asia/Shanghai  
Reviewer: /root/visual_smoke_repair_rereview_v2  
Model / reasoning: gpt-6-astra / xhigh  
Scope: visual_smoke_source_repair_after_preserved_FAIL  
Overall status: **FAIL — source repair is not yet cleared.**

This is an independent, bounded source-integrity rereview, not a runtime result, full pilot/formal-entry review, or Type-B certification. I read the primary source, original independent FAIL, retained source snapshots, preparation artifacts and trace records directly. I did not accept the packet collector's or repair author's conclusions as proof. I did not run the project's check, run, _worker, generation, pilot, training, testing, model, optimizer, checkpoint loader, GPU/resource probe, environment provisioning, or network/driver operations. No implementation source, raw input, old evidence, scientific gate or plan was edited. Only this new review's artifacts are written.

Two launch-blocking source findings remain: a deterministic tuple/array interface defect introduced in the repaired worker (B3), and an incomplete input-set binding left in the B1 repair (B1-R). B2's JSON-serialization retry loophole is resolved at source level. W1/W2 and the original PPO initialization-seed note have source-level repairs; W3/W4 remain limitations.

## 1. Exact reference roots

Every abbreviated reference below expands to an exact absolute path and one-based line number:

- S/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/
- D/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/
- H/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_training_repair_pilot_20260930/source_snapshot/mappo_gp_sf_dsr/
- R/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/entry_source_reviews/20261011_115016_650790_independent_typeA_source/
- P/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/entry_source_reviews/20261011_214124_683383_integrated_source_packet/
- E/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_121100_687420_model_smoke_entry/
- A/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_115010_522869_smoke_launch_guard_repair/SOURCE_SNAPSHOT/
- T/ = X:/Yi-one/workspace/MPT-CVaR-MAPPO/.aris/traces/experiment-audit/2026-10-11_visual_smoke_entry_source/
- O = X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md

## 2. All original blockers and warnings

| Original finding | Rereview disposition | Direct current-source evidence |
|---|---|---|
| B1: preparation-time raw/prior evidence not checked by run | PARTIALLY RESOLVED; retain FAIL as B1-R | S/visual_model_smoke.py:44 reconstructs and checks the pinned 232-entry map; :54 includes it in the plan; :76, :162, :212, :272, :303, :327, :350 revalidate it. However S/physical_mobility.py:34 discovers the actual first 30 files afresh and S/visual_model_smoke.py:204 does not bind that discovered map back to an expected set. |
| B2: raw JSON-byte hash used as one-shot identity | RESOLVED at source level | S/visual_model_smoke.py:61 compares the complete canonical plan to the exact expected plan; S/visual_training_transactions.py:20 fixes sorted compact JSON with nonfinite values forbidden; S/visual_model_smoke.py:315 uses semantic identity for the permanent exclusive lock. |
| W1: historical transitive Python imports not locked | RESOLVED for the advertised local Python tree, not installed packages | S/visual_model_smoke.py:36 includes all followup scripts and SOURCE.rglob('*.py'); S/visual_learned_bindings.py:11 identifies SOURCE as the whole historical source_snapshot. The manifest contains both historical packages, including imported buffer/config/data/env/metrics/train/evaluate/rewards/feedback modules. |
| W2: deterministic roundtrip arrays not persisted | RESOLVED in the future execution source; still UNEXECUTED | S/visual_model_smoke.py:251–256 saves actual first-pre-action inputs and both models' output arrays; :264–266 saves Torch RNG arrays; :283–284 binds both NPZ files by SHA. B3 presently prevents reaching this code. |
| W3: nonfinite QLA detection later than PPO | RETAINED WARN | H/models.py:630–641 still performs backward/clip/step before any wrapper finite-statistics check at S/visual_neural_controller.py:124. PPO checks loss and gradient norm before its step at :158–165. |
| W4: weaker pre-try failure evidence; point-in-time resource gate | RETAINED WARN | S/visual_model_smoke.py:162–180 precedes worker try; :303–319 precedes parent try. The occupancy check at S/qla_environment.py:99–106 does not reserve the GPU. The failure branches at S/visual_model_smoke.py:287–298 and :360–364 do not rehash the protected maps. |
| Unnumbered PPO clone seed note | RESOLVED at source level | S/visual_checkpoint_load.py:43 now calls torch.manual_seed(initialization_seed) before constructing the PPO clone. Constructors remain explicitly RNG-affecting; no training state is resumed. |

The original FAIL is not overwritten or relabeled. R/REVIEW_RESPONSE.md:72–111 defines B1/B2, :157–173 defines W1–W4 and the seed note. The exact original report and its trace copy both hash to e70537fd0a330631d6ce0f42c2c3fb80115dcec0af91f109ebcf12a5b36ae5ab.

## 3. Blocker B3: repaired worker calls an array method on a tuple

Severity: launch-blocking functional source defect. Status: FAIL.

The producer S/physical_mobility.py:97–103 returns a tuple of tuples from server_positions(). The consumer S/visual_model_smoke.py:201 assigns that return directly to edges. calibration_scales at :202 does not reassign edges and its implementation at S/visual_method_features.py:54–58 merely constructs tuple-valued scales. Nevertheless S/visual_model_smoke.py:206 calls edges.tolist() while constructing input_condition.

A Python tuple has no tolist method. On a launch that gets past preceding input preparation, this raises AttributeError before env construction at :207, before assert_runtime_bindings at :212, and before build_cuda_model at :213. It is not a GPU problem or a failed numerical acceptance criterion. It was introduced on the repaired input-condition-reporting path: the preserved A/visual_model_smoke.py:139–150 passes the same tuple directly to the environment without calling tolist.

Expected failure flow, inferred from source and NOT executed here: worker STARTED exists (:183); controller/stream/assembly/ledger are still None (:181); worker FAILURE records the error with no model pass (:287–298); parent sees nonzero exit at :338–339 and does not launch the second arm or write overall PASS. The permanent semantic lock is already consumed. Thus the current repaired entry cannot reach its advertised two-arm forward/update/roundtrip success path.

Required source resolution if separately authorized: make this producer/consumer serialization contract consistent, preserve the old bytes/finding, and re-review the new exact source identity. No live launch or scientific threshold change is necessary to establish this defect, and none is authorized by this report.

## 4. Blocker B1-R: checked file bytes do not pin the actual mobility input set

Severity: launch-blocking residual provenance invariant. Status: FAIL.

The main B1 repair correctly detects changed bytes of every already registered protected file and of the pinned prior RESULT. However, it checks only entries in that fixed dictionary (S/visual_model_smoke.py:44–51). It does not check that the files actually selected by consumers are exactly the registered input set.

The actual consumer uses sorted(Path(folder).glob('*.txt'))[:users] (S/physical_mobility.py:34), not a declared list. The smoke invokes PhysicalMobility(users=30) at S/visual_model_smoke.py:198. Its newly discovered source_hashes are recorded in input_condition at :204–206, but never compared to the expected 30 protected KAIST entries. The parent compares two arms to each other at :343–345 and separately checks each report's copied protected map at :347–349; these are not an equality check between consumed input identities and the reviewed expected set.

Concrete source-level counterexample, not performed: add one valid, earlier-sorting .txt trace to the same KAIST directory while leaving all registered files untouched. The new first-30 selection can substitute that trace for the previously selected 30th trace. protected_bindings still passes, source_lock is unchanged because the addition is not Python, and the accepted plan/semantic one-shot key is unchanged. Both arms would discover the same substituted raw condition, so pairwise equality would also pass. This changes user geometry and potentially calibration bounds/server positions without a newly reviewed input identity. A copy of a valid trace under an earlier name is sufficient for the membership defect; it does not require changing any protected bytes or inventing timing labels.

I checked current directory membership read-only: the actual current first 30 paths are all in the protected map, and all hashes match. I did NOT observe or simulate this substitution. B3 currently masks later success, but fixing B3 alone would leave this independent integrity defect.

S/execution_service_catalog.py:33 also uses directory discovery; its row-count, duplicate and role checks at :38–64 constrain additions but do not substitute for explicit membership binding. The demonstrated material counterexample is the mobility selection.

Required invariant: bind exact consumed paths and hashes to the registered expected input set before preparing/using the environment and before claiming completion, or consume a fixed registered file list. This must cover additions/substitutions, not merely modifications to existing entries. Preserve original evidence and do not widen data or scientific acceptance gates.

## 5. Independent byte verification and continuity boundaries

I independently used filesystem byte hashing and a separate canonical-map serializer, without importing project code or parsing retained-test labels/timings. At 2026-10-11T22:41:05+08:00:

| Mapping in P/CURRENT_PATHS_AND_BYTES.json | Entries rehashed | Mismatches |
|---|---:|---:|
| source_files_sha256 (:4) | 203 | 0 |
| protected_input_and_source_sha256 (:209) | 232 | 0 |
| preserved_continuity_files_sha256 (:443) | 508 | 0 |
| original_protocol_document_and_prior_review_paths_sha256 (:953) | 97 | 0 |

Counts are per map, not unique-file totals; maps overlap. A second full byte recheck at 2026-10-11T22:50:49+08:00 again found zero mismatches in all four maps and the same manifest bytes. Recomputed identities:

- Source map: b9c401029b2fafd69bae66105f56c6fea5eacda5e622e0551fcc49c4ee676d3c
- Protected map: 4f726571169fb2251481cfbd30335414bcda066570c0c28d2fedc347bae19a10
- Continuity map: 08220c2705a3cfdc00c296b5905788791cc4217663010a338e7c8ed288942026
- Auxiliary map: 8dbea99b20f78e62f7516c5515a5ea087a140ae7309d00cfb893f4aa57b35838
- Packet manifest bytes: c9a2b8ee97074bb13ebe3f0530a80d2e506cf5939c342b0b4d49877dd60b7556

The full independently checked source/protected maps are in REVIEW_RESULT.json. Supplementary hashing confirmed the predictor file matches its hard-coded SHA fc5a5311c6f61572480987ccb485724b3a3eeb78a028b3ffc7885d495edc295e (S/metadata_service_predictor.py:15–16, :39–46); it was not deserialized by me. IMAGE_SPLITS.json and the original v2 PLAN.json are preserved in the 508-entry continuity map (P/CURRENT_PATHS_AND_BYTES.json:829, :831); I hashed, not parsed, these data-bearing protocol files.

Do not confuse those four inventory maps with runtime coverage. The smoke runtime reconstructs only its 232-entry protected map plus current Python sources. The larger 508-entry continuity map and auxiliary docs/reviews are not arguments to assert_runtime_bindings. The split/original-plan files are preserved at audit time, but are not individually revalidated by this worker. In this bounded functional path, calibration role and geometry settings are instead fixed by locked source and calibration CSVs; the worker does not select data by parsing those split files. This is not a complete-pipeline continuity certificate.

Runtime boundary trace:

- Parent and child read_plan: file SHA, rooted plan path and exact complete semantic plan are checked (S/visual_model_smoke.py:76–85, :162, :303).
- Before resources/model: named raw/prior/source hashes are rechecked (:212); predictor is independently checked during environment construction via S/visual_environment_adapter.py:58 and S/metadata_service_predictor.py:39.
- Before each arm and successful completion: rechecks occur at :327, :272 and :350. Completion reports bind raw maps/physical conditions, subject to B1-R and B3.
- Checkpoint read: S/visual_checkpoint_load.py:25–30 checks actual controller, historical selected source and externally bound own archive before loading; :54 rechecks the archive afterward. S/visual_training_transactions.py:224–246 verifies exact descriptor, manifest hash and the exact three payload hashes, rejects failed/partial/symlinked archives, and does no deserialization itself.
- Failure exits: there is no full raw/source/model/split/plan rehash in the worker/parent except blocks. No successful end-of-failure byte-continuity proof may be claimed. Early validation failures can occur before a local FAILURE artifact exists. This retains and extends W4, not a fabricated pass.

Original source preservation was checked against the original preparation's source map: A/visual_model_smoke.py SHA 1050a485f3773c7e7229852a61080e2b4e2b77ca450682f6db85cd85e0226050 and A/visual_checkpoint_load.py SHA 4d69b5fbbcf90c8e4f0a6776d3bcbb72f18cceeeecbc33d7b41f353aa377befa exactly match the reviewed old bytes. The archive and trace retain the original FAIL despite the recorded delivery error (T/001-prepared-entry.meta.json; T/004-tool-final-delivery-error.response.md). No old report was resealed.

E/SMOKE_CANDIDATE_PLAN.json still hashes to 094caefc9bc78a689769ff56317492364011e35d441f3d2ea6bb1b06e2cf521c. It contains 170 source entries, whereas today's map contains 203; none of those original 170 hashes changed, but 33 scripts have been added. Therefore it is a preserved preparation artifact, NOT a currently accepted launch plan under the exact schema/map comparison. I did not generate a replacement plan or run check.

## 6. One-shot identity, aliases and partial attempts

For B2 specifically the repair is substantive. digest uses sorted keys, compact separators, UTF-8 and allow_nan=False (S/visual_training_transactions.py:20–22). semantic_plan_identity compares the digest of the entire parsed plan to expected_plan and returns only the expected canonical digest (S/visual_model_smoke.py:61–67).

Consequences derived from source, not adversarial test execution:

- Whitespace, JSON key order or escape spelling that parse to the same values cannot create a different semantic lock.
- Numeric spellings that canonicalize to the expected numeric representation share the lock. Type-changing representations such as an integer field expressed as a float, boolean-for-integer, or negative zero where the expected serialization differs are rejected; they are not new accepted experiment identities.
- Extra fields affect the complete digest and are rejected. Duplicate JSON keys, if accepted by the parser, collapse to a single effective dictionary and likewise do not create a new lock for the same accepted values.
- Renaming/copying a plan or changing its path spelling does not enter the identity. Any accepted alias still uses the same BASE-derived lock root. Aliased strings inside source/protected maps do not equal the expected canonical strings and are rejected.
- New output paths cannot evade the permanent exclusive semantic lock (:313–318). Resolved paths must remain within the declared root and existing parent/child/checkpoint output directories are refused (:307–319, :170–180; S/visual_training_transactions.py:279–282).
- The child checks plan byte SHA, semantic identity, parent output, method directory name and os.getppid() before creating its directory (:174–180). This is ordinary same-run parent binding, not a cryptographic process attestation.
- The parent waits for each worker, stops on the first nonzero exit, and has no retry/replacement loop (:326–340). A first-arm partial success cannot be assembled with a later separately launched arm into the same parent pass.
- The collection/update tokens and failed controller/MV/ledger state cannot be reused through the intended APIs (S/visual_training_transactions.py:119–181, :207–221; S/visual_neural_controller.py:95–130; S/visual_mv_stream.py:60–77).

The lock is an execution-integrity mechanism, not proof that a human or independent reviewer has authorized every newly generated source identity. A new current source map is not, by itself, independent launch approval. This report grants no such approval.

## 7. Model, numerical and scientific contract checks

These are inspected downstream source contracts, not reached-runtime observations; B3 prevents their present execution.

**Environment/resources.** The worker verifies the isolated executable (S/visual_model_smoke.py:166–169); the factory repeats it, then invokes the original witness before model creation (S/visual_learned_bindings.py:50–55). The pinned spec still states Python 3.13.5, NumPy 2.2.6, Torch 2.11.0+cu128, one compute/inter-op thread, RTX 4060 Ti and memory.used < 500 MiB (D/protocol/QLA_ENVIRONMENT_SPEC.json:5–17). S/qla_environment.py:99–106 retains the index/name/strict threshold and seeded CUDA kernel whose declared sum is 3680.0. There is no CPU fallback in the factory/controller. I did not measure live availability or package/runtime compatibility. Windows restoration remains process-local environment defaults/read-only registry lookup plus local cache directories, and rejects unsafe KMP override (S/windows_runtime.py:7–31); it does not repair drivers or change system settings.

**Original architectures and layouts.** Source selects 30 users, four servers, actor 19 and critic 705 (S/visual_model_smoke.py:209). The feature order is location2, distances4, request3, scores4, age1, prices4, scale1 (S/visual_method_features.py:123); critic dimension is 30*(19+4)+3*4+3 (:130–136). H/models.py:210–260 slices this actor layout; :277–282 supports 705 critic coordinates plus user embedding. QLA returns portfolio/latent/logp with a separate value call; PPO returns latent/logp/value/portfolio (H/models.py:400–423; H/paper_policy_baselines.py:117–131; S/visual_learning_buffer.py:27–47). QLA factory fixes the edgewise structure and 0.5 logit bias; controller rejects incompatible class/config/device and mismatched optimizer settings (S/visual_learned_bindings.py:69–79; S/visual_neural_controller.py:43–76). PPO retains its historical local critic/global log_std, QLA its centralized critic. This is a disclosed visual adaptation, not identical architectures or an unchanged historical experimental protocol.

**Actual counts and timing.** If reached, the worker performs 16 stochastic act calls for 30 users and closes fixed drain before update (S/visual_model_smoke.py:222–239). The buffer refuses truncation/reordered or mismatched actions and stores all time-by-user samples (S/visual_learning_buffer.py:74–139). closed_episode_counts requires exactly 16*30=480 finite samples, correct user/slot axes, final-row terminal flags and .8 seconds of action duration (S/visual_training_coordinator.py:91–107). Generation horizon is .8 seconds; decisions occur .05 through .8, action settlements through .85, and final end is .8+30=30.8 seconds. The last .05 action settlement is inside the fixed drain, not an added decision or an extension of the fixed end (S/visual_receipt_environment.py:93–101; S/visual_training_transactions.py:64–71).

One update transaction is not one optimizer step: four epochs over 480 samples at batch size 128 gives four minibatches per epoch (128,128,128,96), hence 16 nominal optimizer.step calls per arm on the completed path (S/visual_neural_controller.py:20–27, :144–165; H/models.py:503–505, :641). No extra entropy optimizer step runs in this declared QLA configuration. Actual step counts have not been measured.

**GAE and terminal/reliability accounting.** Gamma and lambda are per-second powers of each physical interval; recurrence is independently per user before flattening, with final bootstrap zero (S/visual_temporal_gae.py:7–51; S/visual_learning_buffer.py:118–134). FIFO conserves generated/uploading/waiting/running/completed/dropped tasks and service-seconds (S/fifo_execution.py:85–132). Terminal closure checks exact task identities/totals and uncapped latency arrays; unfinished elapsed bounds remain labeled training penalties, not completion times (S/visual_terminal_accounting.py:46–109). Tail costs fold into the final reward without new action/GAE rows (:112–169). PPO MV state is one persistent training-purpose object and commits exactly 480 ascending-user updates, not reset per user (S/visual_model_smoke.py:218; S/visual_streaming_episode.py:24–54; S/visual_mv_stream.py:76–107). These arithmetic contracts do not establish long-run learning or independent performance.

**Finiteness.** Actions, values, features, GAE and update statistics have explicit finite checks (S/visual_learning_buffer.py:37–41, :88, :105–116; S/visual_temporal_gae.py:15–32; S/visual_training_transactions.py:174–177). PPO loss/gradient checks precede steps. Historical QLA still checks later through returned aggregate statistics (W3); parameter byte hashing alone is not a finiteness test (S/visual_training_coordinator.py:149–165). Subsequent actual deterministic action/value validation would reject nonfinite visible outputs. No numerical success or recovery is inferred. On update failure the worker retains behavior/ledger evidence, but its except branch does not write a new full failed optimizer/weight snapshot; do not claim all in-memory corruption is durably archived.

**Checkpoint and roundtrip.** Only a committed actual update can create the fresh archive (S/visual_training_transactions.py:187–195, :260–302), with model, optimizer/RNG and training-state files hashed in the completion marker. The loader uses the matching historical class, strict PPO schema/state_dict or historical QLA loader, and checks the loaded parameter hash (S/visual_checkpoint_load.py:25–54; H/models.py:425–469). It loads weights only, not optimizer/H/MV/RNG training state; constructors may create unused optimizers and reseed global RNG. No resume follows. QLA construction reseeds Torch and NumPy (H/models.py:310–311); PPO clone now explicitly seeds Torch. The schedule uses independent generators (S/visual_request_schedule.py:83–85) and was created before clone loading.

The first actual pre-action packed input is retained at S/visual_model_smoke.py:223–226. The post-update original/loaded deterministic portfolio, latent, logp and value arrays are compared exactly and persisted, then another loaded deterministic call is checked for Torch CPU/all-CUDA RNG and parameter immutability (:249–270). This is meaningful conditional source logic; there are no actual witness arrays from this smoke to validate today.

## 8. Success/failure output and parent/child contracts

The source-only check's mocked hash-change rejection and JSON examples (S/visual_model_smoke.py:112–132) are expressly artificial validation exercises, not CUDA/model evidence. E/RESULT.json:3–6 records source_entry_check_passed=true while actual_cuda_smoke_pass=false, torch_imported=false and actual model/optimizer/roundtrip calls=0. :428 denies complete-entry review. E/SMOKE_CANDIDATE_PLAN.json:440–443 denies frozen protocol, test access and actual roundtrip. A read-only file inventory found no METHOD_SMOKE_PASS.json or ACTUAL_TWO_ARM_SMOKE_PASS.json in the scoped visual output tree. This is not a search of unrelated projects.

If the downstream code becomes reachable, a worker success report at S/visual_model_smoke.py:273–286 carries method, source/protected maps, schedule, full input_condition and its digest, semantic/byte identities, actual-call flags, parameter hashes, finite statistics/counts, archive/loader record, witness hashes and ledger. The parent consumes schedule, input_condition, condition digest, protected map and semantic identity at :341–349, then writes an aggregate report embedding both full method reports at :351–357. There is no absent-field producer/consumer mismatch in those listed JSON keys. The demonstrated mismatch is earlier: the tuple serialization at :206.

The parent does not independently recompute every child witness hash/array equality, nor enforce an exact schema/status/method/source/plan-byte check for every child report field. In the intended controlled fresh-directory/fixed-worker path, the worker itself supplies those checks; the aggregate is not an independent result audit. This limitation must remain explicit for later runtime review.

Failure inside worker try marks controller/MV/ledger failed and writes error/partial behavior/ledger, false smoke flags and zero formal cells (:287–298). Parent failure writes false overall pass and failed state (:360–364). Earlier errors rely on lock or parent logs; failure exits do not provide a complete end-boundary rehash (W4). The generic write_evidence helper overwrites mutable state paths (S/visual_episode_collector.py:69–71); fresh immutable attempt directories and the permanent lock prevent intended rerun reuse. This is not a claim that every individual generated file is protected from arbitrary external writes.

## 9. Leakage, normalization, gates and scope

Calibration-only catalog and schedule guards exclude retained-test data (S/execution_service_catalog.py:18–28; S/visual_request_schedule.py:35–37). Feature scales are fitted from calibration measured data and the declared training geometry window (S/visual_method_features.py:39–58). Actor inputs contain arrived metadata, predicted service, current location/upload wait and retrieved stale feedback; future arrivals are rejected (S/visual_environment_adapter.py:94–109). Measured per-task service is accessed for execution only after action selection (:142–145); feedback sums predicted outstanding service, not future actual durations (S/predicted_fifo_feedback.py:66–80). Critic features remain pre-action observable centralized information (S/visual_method_features.py:130–138).

Feature price scaling, advantage standardization, bounded training rewards and historical MV ratio are training/input transformations, not a self-normalized evaluation score. Completed latency remains uncapped; drops/unfinished retain separate totals. No source condition requires QLA superiority. The 30 requests/s, 16-decision smoke is explicitly low-load functionality, not the scientific rho=.75 training condition, upload feasibility acceptance, original 10,000-step candidate budget, ten trainings or 120 tests (D/README_VISUAL_MODEL_SMOKE_中文.md:11, :21; O:500–516). Negative/no-improvement outcomes and numerical acceptance failures are not authority to alter scientific gates, select favorable data, extend only one arm or discard unfavorable evidence (O:538–578).

The original 2 MHz physical upload constants remain in S/visual_upload_link.py:9–35; bytes convert to bits with the explicit factor of 8. KAIST interpolation retains physical 10-second trace timestamps (S/physical_mobility.py:50, :68–79). Model, split and plan bytes were preserved as described; no replacement data or phantom weights were created.

This audit reads the functional-smoke use closure, selected actual class/update/load bodies and relevant transitive import declarations. Hashing all 203 source files does NOT mean every wider pipeline algorithm has been substantively reviewed. Unreviewed for certification: the complete pilot/preparation/order/episode/driver/registration/freeze/export pipeline; metadata/retained-test schedule generation; final checkpoint selection; complete multi-episode coordinator execution; ten-training learning grid; 120-cell retained-test registration/execution/evaluation; full statistics and publication claims. Historical general-purpose training/evaluation/plotting functions imported transitively were checked for dependency coverage, not certified as executed paths.

S/visual_training_coordinator.py:47–87 requires a different complete_visual_entry_after_actual_model_smoke scope, actual two-arm smoke and frozen pilot protocol for formal use. This source-repair report cannot satisfy that scope. All broader pilot/formal readiness remains unsupported.

## 10. Checklist verdicts and decision

| Requested check | Verdict | Conclusion |
|---|---|---|
| 1. Revisit every original finding | FAIL | B2/W1/W2/seed note source-repaired; B1 retains B1-R; W3/W4 remain. |
| 2. Hashes and entry/failure/checkpoint continuity | FAIL | All audit-time hashes match; runtime consumed mobility membership is unbound; complete failure-end continuity proof is absent. |
| 3. One-shot semantics and partial attempts | PASS within source scope | Canonical identity closes the original encoding loophole; intended fresh-directory/token/lock path forbids replay or merging. Not launch authority. |
| 4. Runtime/model/GAE/budget/roundtrip source contracts | FAIL | B3 makes the current worker fail before model construction. Downstream contracts were inspected but not executed. |
| 5. Output and evidence honesty | WARN | Preparation is honestly unexecuted; persistence additions are conditional source; W4 and aggregate/child-audit limitations remain. |
| 6. Bounded helper/import closure and exclusions | PASS within declared scope | Functional helpers inspected; wider pipeline explicitly not certified. |
| 7. Preservation/leakage/scientific gates | PASS for present evidence, subject to B1-R | No current hash mismatch, phantom result, scientific gate edit or retained-test parsing by reviewer; membership invariant still needs repair. |
| 8. No prohibited operations | PASS | No probes/models/check/run/worker/pilot/generation/training/testing/packages/network/source edits executed. |

Overall FAIL is a source verdict, not an observed CUDA smoke failure. It means the repaired source is not yet safe/functional enough to clear this bounded rereview. The old FAIL and all current evidence must remain preserved. Any separately authorized repair must address B3 and B1-R under a new preserved, independently reviewed source/input identity; it must not reuse a consumed attempt, fabricate runtime evidence, reinterpret negative scientific results as bugs, or relax resource/data/scientific gates.

actual_models_smoke_training_test_executed_by_reviewer=false  
runtime_or_full_pilot_or_formal_launch_authorized=false  
Type_B_certification=false

