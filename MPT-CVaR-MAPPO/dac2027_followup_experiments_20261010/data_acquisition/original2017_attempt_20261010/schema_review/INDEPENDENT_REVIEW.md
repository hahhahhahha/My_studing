# Alibaba 2017 acquisition and schema integrity review

Overall verdict: **FAIL — preregistered data-preparation readiness**, with **PASS — acquisition provenance and raw preservation**.

Classification: **dataset-preparation verification only; not a completed OOD experiment and not RL performance evidence**. The underlying CPU records are observational trace data; any later policy test remains trace-shaped simulation, not measured visual-inference deployment.

Review computation: 2026-10-10; finalization: 2026-10-11 (Asia/Shanghai). This fresh reviewer read the primary artifacts and independently rehashed and rescanned them using Python standard-library code only. No experiment-module import, Torch/CUDA import, model launch, download, disk extraction, original data-preparation invocation, gap fill, revised cohort selection, or input-file edit was performed. Archive member bytes were streamed for hashing, not re-extracted to disk. Before/after input hashes match. After a quota interruption, all 15 inputs were read-only rehashed at approximately 2026-10-11 04:37 Asia/Shanghai and still matched the original independent diagnostic; no scan, download, extraction, or policy was rerun. Cross-model-family independence is not verifiable from the reviewer runtime; do not label this audit GPT-5.5 or verified cross-family review.

## Evidence paths

All file:line references below are relative to `X:/Yi-one/workspace/MPT-CVaR-MAPPO/`.

- `F/` abbreviates `dac2027_followup_experiments_20261010/`.
- `A/` abbreviates `F/data_acquisition/original2017_attempt_20261010/`.
- `R/` abbreviates `A/schema_review/`.
- `P/` abbreviates `results_qla_unseen_external_windows_20261009/`.

## 1. Acquisition provenance and preservation — PASS

The independently hashed registered archive `A/alibaba-trace-2017.tar.gz` is 243,323,702 bytes with SHA-256 `02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df`, exactly matching the official [SHA256SUM](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/SHA256SUM) entry for `trace_201708.tgz`. The local filename is not being treated as proof: equality is by content hash.

There is exactly one regular archive member named `server_usage.csv`, size 19,030,775 bytes. Its independently streamed SHA-256 and the extracted target's independently computed SHA-256 are both `032fa4193f4b426941fb1eecef5add3d8b0632aaf7cbd7663b0051ceb8a4637a`. The bytes match the acquisition record. Evidence: `A/DOWNLOAD_STATE.json:26-33`; `R/INDEPENDENT_DIAGNOSTIC.json:74-127`; acquisition checksum and single-member gates at `F/scripts/acquire_alibaba2017.py:87-105`.

The earlier failed archive remains at `F/data_acquisition/alibaba-trace-2017.tar.gz.partial`, 243,321,159 bytes, independently hashing to `b7efe9dc9caa75352de41b9ac34b329602cde3414fd7cab87c1d21a9a9871d66`, matching the preserved failure record (`F/data_acquisition/DOWNLOAD_STATE.json:25-27`). The replacement attempt changes the source URL and expected length but not the registered SHA gate (`F/scripts/acquire_original_alibaba2017.py:13-29`). All four repair provenance hashes—wrapper, original acquisition script, historical failure state, and HEAD probe—match independently computed values (`R/INDEPENDENT_DIAGNOSTIC.json:128-133`). No evidence of v2018 substitution exists in this acquired raw file: it is byte-identical to the registered v2017 archive member.

This is a point-in-time verification of retained artifacts, not proof of every historical file operation. All 15 reviewed input files have identical hashes before and after this review (`R/INDEPENDENT_DIAGNOSTIC.json:656156`).

## 2. Schema and raw-record validity — WARN

Independent scan results:

| Property | Observed value |
|---|---:|
| CSV rows, all with eight fields | 187,963 |
| Rows valid under the frozen timestamp/CPU filter | 187,962 |
| Invalid rows | 1 |
| Machines with valid records | 1,310 |
| Duplicate machine/timestamp pairs | 0 |
| Valid CPU percentage range | 0.5–89.71999969482 |
| Minimum / maximum timestamp | 39,600 / 82,500 seconds |
| Distinct global valid timestamps | 144 |
| Earliest valid minute / protocol origin | 660 |

The official [schema.csv](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/schema.csv) identifies timestamp, numeric machine ID, CPU percentage, memory percentage, disk percentage, and three load averages as the eight mandatory fields. The frozen reader correctly uses the first three fields without dropping the first record (`paper_doc_tools/heldout_external_windows.py:153-174`). This is not the old v2018 column order.

The sole invalid record is raw `datasets/external_traces/extracted/alibaba2017/server_usage.csv:37562`: `49200,618,,2.29999995232,,,,`. CPU and columns 5–8 are empty. Independent full-eight-field validation found no other malformed or nonfinite fields. Evidence: `R/INDEPENDENT_DIAGNOSTIC.json:138-206`.

The existing frozen parser explicitly skips invalid timestamp/CPU records (`heldout_external_windows.py:163-174`). Thus this one missing CPU record is not itself a global rejection rule, and changing its CPU to a guessed value is not justified. Leave it raw and preserve the registered incomplete-machine exclusion policy.

## 3. Fixed-window / 35-machine compatibility — FAIL (blocking)

The frozen protocol requires `[0,200)`, `[220,420)`, `[440,640)` minutes relative to minute 660, and the first 35 numeric-ID-ordered machines complete at **every minute of all three windows**, with no filling or post-hoc window changes (`P/EXPERIMENT_PLAN_中文.md:63-79`; `paper_doc_tools/heldout_external_windows.py:13-15,143-150,177-189`).

All 187,962 valid timestamps are multiples of 300 seconds. Within-machine positive gaps are:

| Gap | Count |
|---|---:|
| 300 seconds | 185,982 |
| 600 seconds | 668 |
| 1,800 seconds | 2 |

Observed valid-minute counts per machine in W1/W2/W3 are:

| W1 | W2 | W3 | Machines |
|---:|---:|---:|---:|
| 40 | 40 | 40 | 975 |
| 39 | 39 | 40 | 332 |
| 38 | 39 | 40 | 1 |
| 36 | 40 | 40 | 1 |
| 35 | 39 | 40 | 1 |

Consequently, **0 machines—not 35—satisfy the original 200-valid-minute requirement in each window**. Every machine misses at least 160 required minutes in each 200-minute window. This is structural sampling incompatibility, not the effect of the one malformed CPU row. A longer or differently selected 200-minute window cannot create minute observations absent from the released grid. Evidence: `R/INDEPENDENT_DIAGNOSTIC.json:138-160,255-257,411-459`; the original schema report agrees on every recomputed summary (`R/INDEPENDENT_DIAGNOSTIC.json:461-476`).

The [official description](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/trace_201708.md) says usages were measured at 60-second intervals and averaged over 300 seconds; it also says only twelve consecutive hours of usage are disclosed. This wording does not establish that the released CSV contains one row each minute. The verified bytes contain a five-minute timestamp grid. Do not silently reinterpret a five-minute aggregate as five independently observed minute measurements.

The original preparation **cannot produce the registered capacity arrays as written**: its loop finds zero complete machines and raises at `heldout_external_windows.py:188-189`, before writing arrays at lines 197-201. This conclusion was established by direct source inspection plus an independent scan; the original preparation was not rerun. `DESIGN.json:2103-2104` matches the current runner and preparation source hashes, so this is the frozen code's behavior, not a later unregistered implementation.

## 4. Preparation entry point and diagnostic predicate — WARN

`paper_doc_tools/run_unseen_external_windows.py:132-146` calls only `prepare_azure` for `--prepare-data`, then launches `preflight`, and tests only whether the Alibaba path exists. It can label the study `prepared_not_run` without validating Alibaba schema or completeness. The actual Alibaba preparation call occurs only after `--run` is requested and preflight passes. This is a misleading readiness state, not evidence that capacity preparation succeeded. The full run still hits the completeness failure before its policy-cell loop (`run_unseen_external_windows.py:146-162`). Do not invoke this entry point to perform a supposedly model-free schema audit.

Separately, `F/scripts/audit_alibaba2017_schema.py:81` defines readiness as `len(complete) >= 35 and not invalid`, stricter than the frozen parser, which skips invalid records and checks whether 35 machines remain complete. The stricter diagnostic would incorrectly reject some future files with an irrelevant invalid row. It does **not** change today's verdict, because the true eligible count is zero. A software-only correction can align diagnostics/readiness checks with the frozen rule; it cannot solve the sampling mismatch.

## 5. Protocol decision required; no automatic repair

Conservative choices, with no selection by policy outcomes:

1. **Retain the original registration as a data-availability failure.** Preserve the raw archive, existing protocol, source hashes, and failed status. Make no OOD robustness or policy-performance claim from this block.
2. **Authorize a separately versioned protocol amendment before any policy test.** A native-five-minute formulation is feasible in principle: 975 machines have all forty native-grid observations in each of the same physical windows. This count is only a diagnostic; no revised cohort was selected. Specify the exact bin semantics, Azure alignment, capacity hold/resampling rule, complete-machine eligibility, missing values, temporal compression, and 1,000-slot mapping in advance. Changing 200 observed minute values to forty five-minute values (or holding/interpolating them) changes the input process and missing-data policy. It is an experimental-design revision even if the physical windows, 35-machine count, methods, or 243-cell layout stay unchanged. Preserve the old registration and create a new locked output/version.
3. If actual per-minute observations are indispensable, discuss a new verified source and new registration. Do not silently substitute v2018, use a different archive merely because it has a similar filename, relax the checksum, shift windows, or fill gaps.

Model-free validation and honest readiness labels are software concerns; resampling, changing eligibility/window duration, or changing release/source are design concerns. Any source edit must respect `run_unseen_external_windows.py:99-104`, which rejects frozen design hash changes rather than overwriting the existing registration.

## Claim ceiling and required action

Supported: the registered original v2017 archive was acquired and preserved correctly, and its raw schema/coverage has been independently verified.

Unsupported: that the preregistered data preparation is ready, that a capacity schedule passed, that any policy cell completed, or that OOD robustness has been demonstrated. No performance metrics were audited or generated here. The next required user decision is whether to retain the failed registration or authorize a distinct amended protocol. Do not launch the policy experiment under the unchanged protocol.

## Reproducibility and trace hashes

- `R/INDEPENDENT_DIAGNOSTIC.json`: `a8724dc0289d3daf4d3548aca6038fb4f06385f04e016fdec7ff8e34482cf566`
- `R/independent_verify.py`: `7af42cd851126d9402378e179c509aeccf9f6a7d5da6b9ffd45e94434bc7c372`
- Exact received request, `R/001-integrity.request.json`: `06dde1c08b676d0e20c906c09ff84a055d5fd1e631cd924c550b7eaf32c0fc6a`
- `R/run.meta.json`: `7208d81e868f0721b755efd74ce4c857ca5626cd39f32f1f2ecb21b304513b73`

The review directory contains the full final-response trace and metadata separately. Trace storage is restricted to the explicitly authorized directory, not an outside `.aris` path.
