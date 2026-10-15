# Independent Type-A prepared-entry source audit

Date: 2026-10-11 (Asia/Shanghai)
Reviewer task: /root/visual_smoke_entry_source_audit
Requested reviewer: gpt-6-astra, xhigh
Review type: Type-A, source/prepared-entry safety and evidence honesty only
Overall verdict: FAIL
Actual CUDA smoke certified: NO
Formal experiment readiness/results certified: NO
Source launch blockers: B1 and B2 below

This is the full independent review response. I read the named source and artifacts directly and followed the experiment-audit integrity checklist. No author-supplied interpretation was accepted as evidence. The skill's integrity checks informed the provenance, phantom-result and scope sections; they do not turn source review into a model execution certificate.

## Path convention

All S references below expand to this exact directory:
X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/

All H references expand to:
X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_training_repair_pilot_20260930/source_snapshot/mappo_gp_sf_dsr/

All D references expand to:
X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/

These abbreviations only keep the report readable; each reference identifies an exact file and one-based source line.

## Checklist verdicts

| Check | Verdict | Independent conclusion |
|---|---|---|
| 1. Authorized readonly check only | PASS | Exactly the documented check was executed once, successfully, producing a fresh additive directory. No run, worker, CUDA, model, update, export or actual loading was invoked. |
| 2. Prepared-result honesty and fingerprint preservation | FAIL | Current prepared statements and all checked fingerprints are truthful/intact. However, the raw-input/prior-evidence bindings checked during preparation do not survive into run-time acceptance; B1 blocks launch under the advertised binding claim. |
| 3. Actual call layouts, counts, checkpoint, RNG, resources | WARN | No direct dimension/tuple/default-config mismatch found in the selected path. Runtime compatibility and numerical behavior remain unexecuted. W1-W4 describe source/evidence limitations. |
| 4. Role/leakage boundary and fixed roundtrip input | PASS | The reviewed smoke path uses calibration only and an actual first pre-action packed bundle, not fabricated diagnostics or future execution truth. This is a source-level PASS, subject to the input-byte continuity defect B1. |
| 5. One-shot/duplicate/parent/failure rules | FAIL | Ordinary exact-byte replay is prevented and workers are parent-bound, but an accepted semantic plan can be reserialized to receive a different permanent-lock identity; B2. |
| 6. Scope separation and no phantom certification | PASS | Prepared artifacts and documentation explicitly deny actual model smoke/formal results; this review also does not certify either. |

## Execution and immutable evidence observations

The sole project execution was the documented command:

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_model_smoke.py' check
```

It exited 0 and printed:

```text
MODEL_SMOKE_ENTRY_PREPARED_ONLY_NOT_EXECUTED X:\Yi-one\workspace\MPT-CVaR-MAPPO\dac2027_followup_experiments_20261010\visual_calibrated_policy\engineering_attempts\20261011_114728_198054_model_smoke_entry
```

Fresh check directory:
X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261011_114728_198054_model_smoke_entry

Fresh RESULT.json SHA256:
e97852cb279213d8626386022be6de420c94b73425f6129784470d8c1491b705

Original prepared RESULT.json SHA256:
2d794295eef582c510303c474574bf4198a69208b400f3ca4e794f918abb81d1

Both original and freshly generated SMOKE_CANDIDATE_PLAN.json have exactly this SHA256:
264d973d491ddab7629325db45690684c50035ccd7ec93231ae7c74361624cff

I independently recomputed every entry in both reports' 232-entry protected-input/source mapping and 56-entry current-source mapping using file hashing. There were zero mismatches in either report at audit time. The source lock contains 53 followup scripts plus three historical files. The original/fresh plans' identical bytes demonstrate that ordinary documented check regeneration currently preserves lock identity. This does not cure B2's acceptance of other serializations.

Both RESULT files state actual_cuda_smoke_pass=false, torch_imported=false and actual_model_or_optimizer_or_roundtrip_calls=0. The check code asserts no torch module at S/visual_model_smoke.py:81; the loader rejection at lines 74-80 stops at the exact-controller type check before any load. Imports reachable by the check were inspected for lazy model behavior. No independent GPU probe was performed, so GPU availability is not newly established by this audit.

The readonly label means no mutation of protected experiment inputs/source, not zero filesystem writes: check creates its new output (S/visual_model_smoke.py:59-60,86-97), and Windows initialization ensures a process-local runtime-cache directory and environment defaults (S/windows_runtime.py:26-31). No source/raw/config/gate edits were made by this audit.

A filename inventory of visual_calibrated_policy found no METHOD_SMOKE_PASS.json or ACTUAL_TWO_ARM_SMOKE_PASS.json. The only MODEL_STATE.pt files found there were explicitly named NONMODEL/TAMPERED_NONMODEL archive fixtures from older transaction checks. They were not deserialized and are not actual model evidence. The smoke_locks directory did not exist at this audit.

## Launch blocker B1: the prepared raw-input/evidence lock is not enforced by run

Severity: launch-blocking provenance defect. Verdict: FAIL.

Exact evidence:

- S/visual_model_smoke.py:62-65 checks the pinned PRIOR result and its protected/new implementation mappings only inside check().
- S/visual_model_smoke.py:82-85 writes the candidate plan with source_files_sha256 and smoke_settings but omits the protected raw-input/prior-evidence mapping and any bound check-result identity.
- S/visual_model_smoke.py:42-55 read_plan() verifies plan bytes, selected semantic fields and source_lock(); it does not revalidate PRIOR or protected data.
- S/visual_model_smoke.py:139-145 constructs the measured catalog, mobility, calibration scales and schedule afresh during actual execution.
- S/execution_service_catalog.py:33-35 computes current CSV hashes, but no expected-hash comparison occurs in the constructor; lines 38-64 validate shape/roles/stages, not historical byte identity.
- S/physical_mobility.py:34-41 reads and hashes currently selected trace files without comparing to the preparation manifest.
- S/visual_request_schedule.py:67-72 hashes the schedule structure; that structure includes image/session IDs and assignment uniforms, not measured service durations or mobility bytes.
- S/visual_model_smoke.py:208-209 records the new raw hashes only after the method succeeds.
- S/visual_model_smoke.py:262-265 compares only schedule identity between methods and the source lock; it never requires equal raw-data/mobility hashes between the two method reports.

Consequence: a structurally valid measured CSV or mobility file changed after check can be consumed by run while all source/plan checks continue to pass. Changing service durations (and matching component sums) without changing IDs leaves schedule SHA unchanged; both arms can then pass on data other than the reviewed prepared inputs. A change between arms can even yield two different physical input conditions while the parent's same-schedule check passes. This is not a claim that any current file was changed: audit-time hashes all match. It is a concrete missing launch invariant.

The metadata predictor is separately pinned at S/metadata_service_predictor.py:15-16,39-46; this protection does not pin the execution catalog or physical traces.

Required resolution before actual launch: carry or reconstruct an externally bound complete protected-input/prior-evidence manifest in the accepted launch identity, verify it before resources/models are used and again on completion, and compare exact raw-condition bindings across arms. Preserve existing source/evidence; revisions must be additive and independently reviewed. Do not reinterpret this check-only audit as acceptance of changed inputs.

## Launch blocker B2: one-shot identity is the JSON encoding, not the accepted plan semantics

Severity: launch-blocking retry-integrity defect. Verdict: FAIL.

Exact evidence:

- S/visual_model_smoke.py:47 binds the caller-supplied expected SHA to the plan FILE bytes.
- S/visual_model_smoke.py:49 parses JSON, and lines 50-53 validate only selected fields.
- There is no canonical-encoding check, exact complete schema check, trusted check-output registration, or fixed semantic run-ID computation in read_plan().
- S/visual_model_smoke.py:238-241 creates the permanent exclusive lock using that raw file SHA.
- S/visual_model_smoke.py:121-125 binds the private worker to the same raw SHA lock, output and parent PID.
- D/README_VISUAL_MODEL_SMOKE_中文.md:13 claims regenerating the same plan cannot bypass the lock.

Consequence: JSON whitespace/key-order changes preserve every value accepted by read_plan() but change file SHA, and hence the lock filename. Extra ignored fields likewise change the lock identity while keeping the accepted experiment unchanged. A caller can supply the new SHA and a fresh output path after a partial/successful attempt, without editing any locked source, settings or raw data. This does not require deleting the old lock. Such a plan is not rejected as non-check-generated despite the error-message wording at line 46.

I did not create a modified plan, call run, launch a worker or consume a lock to demonstrate this. The conclusion follows directly from JSON parsing and the differing identities used by validation versus locking. The ordinary documented check path did regenerate byte-identical plans, so renaming/copying the untouched file alone cannot evade the lock; the defect is specifically the accepted alternative encoding/schema.

Required resolution before actual launch: derive the one-shot lock key from a strict canonical semantic experiment identity (including source and input bindings), or reject all noncanonical/extraneous encodings against a trusted check-generated record. Record both byte hash and semantic identity for audit. Repairing a failed attempt must require a genuinely new reviewed source/design identity, not a new filename or serialization.

## Source findings supporting check 3

### Layouts/classes/configuration: source PASS, not runtime PASS

- S/visual_model_smoke.py:147 selects N=30, M=4, actor=19, critic=705, embedding capacity=30.
- S/visual_method_features.py:123 makes actor order [loc2, distances4, request3, scores4, age1, prices4, scale1], totaling 19; lines 130-136 produce 30*(19+4)+3*4+3 = 705 critic coordinates. User IDs supplied by S/visual_neural_controller.py:87 are 0..29.
- H/models.py:210-213,232-260 expects and slices the same edge-wise actor layout. H/models.py:277-282 accepts the 705-vector plus an 8-dimensional user embedding.
- S/visual_learning_buffer.py:27-47 correctly maps QLA (portfolio, latent, logp) plus separately computed values and PPO (latent, logp, values, portfolio). H/models.py:400-423 and H/paper_policy_baselines.py:117-131 confirm those orders and shapes.
- S/visual_learned_bindings.py:69-79 forces QLA edgewise_price_actor=true and bias=.5; defaults in H/config.py:74-110 match the new controller's learning rate=.001, epochs=4, batch=128, clip=.2, entropy=.01, value=.5, grad norm=.5. S/visual_neural_controller.py:61-76 rejects incompatible QLA settings, wrong class and non-CUDA parameters.
- PPO deliberately retains its historical global log_std behavior and local critic; QLA retains its historical structured actor and centralized critic. Matched input/budget does not mean identical architectures or an exact unmodified historical experimental protocol.

### Complete collection/update/GAE: source PASS

- The source runs 16 actual act calls, each over 30 users, then fixed drain before update: S/visual_model_smoke.py:159-175.
- S/visual_learning_buffer.py:74-93 rejects over-capacity, misordered intervals and mismatched action tuples rather than truncating; lines 122-139 retain time-by-user arrays.
- S/visual_temporal_gae.py:21-30 applies gamma^duration and lambda^duration over each user's time series, then S/visual_temporal_gae.py:38-51 flattens. The final bootstrap is zero via S/visual_learning_buffer.py:118-134.
- S/visual_training_coordinator.py:91-107 requires 16x30 = 480 finite behavior samples, correct user/slot IDs, last-row-only terminal flags, and total physical action duration .8 seconds.
- S/visual_receipt_environment.py:93-101 includes the final .05-second settlement within the fixed drain, not as an extra decision. Generation horizon=.8 and final finish=.8+30=30.8 seconds; no drain-only action/GAE rows are appended.
- One update transaction is burned before execution: S/visual_training_transactions.py:161-181 and S/visual_neural_controller.py:98-104. The QLA facade substitutes already prepared GAE while retaining the historical loss/update, and restores its instance hook even on failure (S/visual_neural_controller.py:106-129).
- One transaction means four epochs of four mini-batches (128+128+128+96), thus 16 optimizer.step calls per method on the normal completed path, not a single gradient step. Both methods have the same nominal exposure count. H/models.py:503-505,641 and S/visual_neural_controller.py:144-165 support that count.
- Persistent PPO MV state is created once and required to count 480 observations before checkpoint: S/visual_model_smoke.py:155-156; S/visual_streaming_episode.py:24-29,46-54; S/visual_training_transactions.py:272-275. This is one-episode smoke, not evidence of long-run recurrence performance.

### Checkpoint roundtrip and RNG: source PASS with warnings

- Save is reached only after collection closure and committed update; unchanged parameter hash fails the smoke (S/visual_model_smoke.py:174-183).
- S/visual_training_transactions.py:278-302 saves own real model, optimizer/RNG snapshot and training-state JSON in a new directory, then hashes the exact three payloads into a completion manifest.
- S/visual_training_transactions.py:224-257 rejects failed/partial archives, wrong descriptor, wrong manifest SHA and payload hash mismatches before loading.
- S/visual_checkpoint_load.py:25-49 requires an actual matching controller and verified archive, constructs the historical class, uses the original QLA loader or strict PPO state_dict loading, and compares loaded parameter SHA.
- The selected QLA save/load keys match H/models.py:425-469. The clone uses a deep copy of the reference dimensions/config. PPO uses the reference actor dimension and log_std bounds.
- S/visual_model_smoke.py:186-198 compares portfolio, latent, logp and value arrays exactly, then checks a further deterministic call leaves CPU/all-CUDA Torch RNG unchanged and loaded parameters unchanged. This is genuinely tied to actual calls in run-source, not an unconditional pass generator.
- QLA constructor reseeds global Torch and NumPy RNG (H/models.py:310-311); PPO construction advances Torch RNG. This occurs after training in the smoke worker, no resumed training follows, and a separate process is used for each arm. Request generation uses independent generators (S/visual_request_schedule.py:83-85), so the source does not let clone reseeding alter the already-created schedule.
- The loader does not load optimizer/H/MV/RNG training state. It still constructs optimizer objects through historical QLA construction and through NeuralVisualController for PPO (H/models.py:388-398; S/visual_neural_controller.py:76). The README correctly distinguishes no optimizer UPDATE/restore from no optimizer OBJECT (D/README_VISUAL_COORDINATOR_中文.md:11).

### Resource/own-allocation handling: source PASS, live availability UNTESTED

- S/visual_learned_bindings.py:50-55 requires the exact environment and performs the original witness before building the initial model.
- S/qla_environment.py:99-106 checks GPU index/name and memory.used <500 MiB, then runs the seeded 3680.0 CUDA-kernel witness. The pinned threshold remains D/protocol/QLA_ENVIRONMENT_SPEC.json:12.
- Each method runs in a fresh child; the parent waits for exit before starting the next (S/visual_model_smoke.py:248-260).
- The roundtrip clone is constructed inside the already verified worker without calling witness/build_cuda_model again (S/visual_checkpoint_load.py:33-48). It therefore does not mistake its own model allocation for external occupancy.
- CUDA is explicitly required, and no CPU fallback branch exists in the new construction path (S/visual_learned_bindings.py:67-68; S/visual_neural_controller.py:57-59).
- No live GPU occupancy/health was measured by this audit. Passing source checks says nothing about available memory, actual import compatibility or numerical success.

## Warnings, not additional demonstrated launch failures

W1. Historical transitive imports are not covered by the advertised source lock.

S/visual_learned_bindings.py:12-25 hashes only models.py, config.py and paper_policy_baselines.py. S/visual_model_smoke.py:39 adds all current followup scripts, not the rest of the historical snapshot. Yet H/models.py:11 imports buffer.py, H/config.py:6 imports repro_risk_averse_edge.config, and H/paper_policy_baselines.py:29-35 imports data/canonical_protocol/env/metrics/train. The latter imports still more modules, including plots and rewards. Those local Python files execute at import time and can change without the present source lock changing. This undercuts the comment "every current dependency" (S/visual_model_smoke.py:37-39), even though the three selected model files are correctly pinned. Extend the reviewed import-closure manifest or explicitly narrow the claimed lock scope. No transitive module tampering was observed or simulated.

W2. Detailed deterministic roundtrip witness arrays are not persisted.

The input itself is defensible: first_packed is assigned immediately after the first env.observe()/features_for() and before act/apply (S/visual_model_smoke.py:159-165), uses read-only feature arrays (S/visual_method_features.py:76-79,137-138), and the first actor/critic inputs also enter the saved behavior NPZ (S/visual_learning_buffer.py:90-93; S/visual_model_smoke.py:171-172). However the original/loaded deterministic output arrays at S/visual_model_smoke.py:186-190 are not saved; only the boolean claim and parameter hashes survive at lines 202-209. Preserve a dedicated input/output witness and its fingerprint if future audit requires independently checking the exact equality. Current source supports the comparison but no actual output exists yet.

W3. Nonfinite-update detection is not equally immediate across methods.

PPO checks loss and gradient norm before each step (S/visual_neural_controller.py:158-165). The original QLA update performs backward/clip/step without checking finiteness (H/models.py:630-641); the wrapper checks aggregate returned statistics afterward (S/visual_neural_controller.py:124-125). Later actual act output validation also rejects nonfinite results (S/visual_learning_buffer.py:37-41). Thus no intentional fake-pass route is established, but QLA may retain more partially corrupted optimizer state before failure. Preserve that state as failed evidence and never retry it. No source repair is authorized by this audit.

W4. Failure evidence is weaker before the try blocks; resource probes are point-in-time.

Worker plan/guard/mkdir/parent validation occurs before its try (S/visual_model_smoke.py:107-127); parent lock creation and output mkdir occur before its try (lines 227-243). Such failures may leave only lock/start/stdout/stderr or a traceback rather than a local FAILURE.json. Once a lock is acquired it is nevertheless preserved. The GPU occupancy check and witness do not reserve the GPU against another process allocating later. These are evidence/coordination limitations, not license to bypass the gate or run a replacement worker.

The PPO loader's initialization_seed argument is validated but not used by its constructor branch (S/visual_checkpoint_load.py:27-29,40-47). Because the full strict state_dict is immediately loaded and construction is declared RNG-affecting, this does not invalidate weight equality, but the seed must not be described as a reproducible PPO clone-initialization seed without adjustment.

## Check 4: leakage/provenance analysis

- The worker explicitly builds ExecutionServiceCatalog('calibration') and physical KAIST geometry over TrainingWindow(0,500), never validation/test catalogs (S/visual_model_smoke.py:139-145). Role-restricted catalogs and schedules refuse test (S/execution_service_catalog.py:18-28; S/visual_request_schedule.py:35-37).
- Calibration normalization fits only calibration-role measured data (S/visual_method_features.py:39-58).
- Actual current-task measured service durations are read only after action selection during execution (S/visual_environment_adapter.py:142-145); actor features are constructed from current arrived metadata, a hash-pinned metadata predictor, current location/upload wait, and retrieved feedback (lines 94-109).
- A future request is explicitly rejected at S/visual_environment_adapter.py:95-96. Task assignment uniform/session IDs and future raw queues are not passed to NeuralVisualController.act; it receives only packed arrays.
- Critic inputs are centralized pre-action observable features/previous portfolios/predicted backlog and prior H, not the current action or future durations (S/visual_method_features.py:130-138).
- Calibration reuse in a functional training smoke is not an independent generalization evaluation. Physical interpolation uses the declared trace model. Neither establishes scientific out-of-sample performance.
- The selected first pre-action roundtrip bundle is actual, deterministic in provenance and retained; it is not synthesized from the trained model or chosen using its success/failure result. See W2 for persistence limitations.

## Check 5: safeguards that do work, and their boundaries

An untouched plan copy/new output name retains the exact same SHA, so exclusive open('x') prevents that ordinary replay (S/visual_model_smoke.py:238-241). The child compares recorded parent PID with os.getppid(), child method name and resolved parent output (lines 121-125). A command launched directly from a different parent fails before model creation. Workers are sequential and a first-arm failure prevents the second (lines 248-260). There is no automatic retry loop and no reuse of a completed partial first arm.

Fresh directories use exist_ok=False. The update token is burned before invoking the optimizer; failed controller/MV/ledger state is marked and partial behavior is written (S/visual_model_smoke.py:211-222; S/visual_neural_controller.py:104-130; S/visual_training_transactions.py:161-213). The parent only creates an overall actual pass after both newly launched method reports exist and its own comparisons pass (S/visual_model_smoke.py:261-270). These are meaningful safeguards, but B2 permits a fresh accepted plan encoding to evade the intended experiment-level one-shot identity.

Source repairs change the current-source manifest and invalidate the old plan. Edits to previously protected implementation also break check's pinned prior chain. Preserve the reviewed bytes, old passes/failures and traces; do not reseal or overwrite them or assemble a new overall pass from prior partial arms.

## Experiment-audit A-F cross-check

A. Ground-truth provenance: PASS within source scope. Planned execution latencies derive from measured calibration durations plus the declared queue/upload/mobility simulator, not model-generated ground truth. Not a real-world end-to-end deployed policy evaluation. B1 remains an input-byte continuity defect.

B. Score normalization: PASS within this smoke scope. There is no accuracy/performance success threshold. Advantage standardization and actor-feature price scaling are training/input transformations, not an evaluation score divided by its own prediction maximum. A finite update/changed parameters/serialization comparison does not demonstrate improvement. Binary MV reward is not reported as an accuracy rate.

C. Result-file existence/phantom results: PASS. Original RESULT.json:3-6 and :305-307 explicitly retain preparation-only, zero actual calls, zero formal cells and no complete-entry review. SMOKE_CANDIDATE_PLAN.json:2 and :92-95 agree. No actual smoke pass artifact was found in the scoped visual output tree.

D. Dead/unexecuted code: WARN, intentional. Model constructors, act/update/export/load and formal coordinator paths remain unexecuted. Static compatibility is not execution evidence. No unexecuted metric is represented as measured performance.

E. Scope assessment: PASS. D/README_VISUAL_MODEL_SMOKE_中文.md:9,11,17 and D/README_VISUAL_COORDINATOR_中文.md:9,13,19 deny formal readiness, final checkpoint selection and 120-test-cell completion. The 30 requests/s setting is explicitly low-load functionality, not rho=.75 or upload feasibility acceptance.

F. Evaluation type: NO EVALUATION EXECUTED in this audit. The future smoke is simulation_only with real measured service-time and physical-trace inputs; it is not real_gt policy-performance certification.

## Decision and claim impact

Overall FAIL is a prepared-entry source-safety verdict because B1 and B2 contradict the launch's provenance and no-retry contract. It is not a finding that an actual CUDA smoke failed, and not a claim of falsified current measurements.

Supported now:
- A fresh documented source-only check succeeds without actual model work.
- Current original and fresh check input/source fingerprints match.
- Current artifacts honestly say prepared/unexecuted.
- The reviewed selected source layouts are internally plausible.

Not supported now:
- Safe experiment-level one-shot launching under all accepted plan encodings.
- Guaranteed preservation of the preparation-time raw inputs during run.
- Actual CUDA model/import/optimizer/checkpoint roundtrip success.
- Scientific pilot completion, formal protocol freeze, final checkpoint selection, performance improvement, or any formal train/test cell completion.

Launch blockers B1/B2 must be resolved and re-reviewed under a new preserved source/plan identity before treating this entry as safe to execute. The original GPU/witness gate and separate later actual-smoke/pilot/full Type-A conditions remain in force. This report cannot satisfy the coordinator's complete_visual_entry_after_actual_model_smoke scope.

