Overall: **FAIL for original data-preparation readiness; PASS for acquisition/raw preservation.** This is dataset-preparation verification—not a completed OOD experiment or RL-results audit.

- **PASS — provenance:** independently rehashed the 243,323,702-byte archive against the [official checksum](https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2017/SHA256SUM). SHA-256: `02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df`. Its single regular `server_usage.csv` member and the extracted 19,030,775-byte file both hash to `032fa4193f4b426941fb1eecef5add3d8b0632aaf7cbd7663b0051ceb8a4637a`. The prior failed archive and source-repair provenance are preserved and match their records. No v2018 substitution was found.
- **FAIL — original windows:** 187,963 rows; 187,962 valid; 1,310 machines; origin minute 660. Every valid timestamp lies on the five-minute grid. Within-machine gaps: 300 seconds × 185,982; 600 seconds × 668; 1,800 seconds × 2. Each 200-minute window has at most 40 observed minutes, so **0 machines satisfy the original requirement of 200 observed minutes in all three windows**, versus 35 required. The frozen preparation necessarily raises at [heldout_external_windows.py:188](/X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/heldout_external_windows.py:188); I did not execute it.
- **WARN — invalid row, not a new global gate:** raw line 37562 contains a missing CPU value for machine 618 at timestamp 49200. The original reader skips invalid rows. The later diagnostic’s `and not invalid` at [audit_alibaba2017_schema.py:81](/X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/audit_alibaba2017_schema.py:81) is stricter than that rule; it must not be imposed as a new protocol requirement. Removing that extra condition would not change today’s failure: eligible machines remain zero.
- **WARN — readiness label:** `--prepare-data` prepares Azure and invokes preflight, but checks only Alibaba file existence; actual Alibaba completeness validation occurs under `--run`. Therefore `prepared_not_run` is not evidence of completed capacity preparation. See [run_unseen_external_windows.py:133](/X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/run_unseen_external_windows.py:133).

Conservative next decision: retain the original registration as a data-availability failure, or authorize a separately versioned, pre-test protocol using explicitly defined native-five-minute bins/alignment/slot mapping and machine eligibility. Although 975 machines have all forty native-grid observations in each physical window, none was selected for a revised experiment. Holding/interpolating observations, changing eligibility/windows, or replacing the source changes experimental design—not just software. Do not silently fill gaps, shift windows, or substitute v2018.

All 15 reviewed inputs were rehashed after resumption and remain unchanged. No downloads, disk extraction, original preparation, Torch/CUDA imports, or policies were run during finalization. Verified cross-model-family independence is not claimed.

Full report and evidence: [INDEPENDENT_REVIEW.md](/X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/data_acquisition/original2017_attempt_20261010/schema_review/INDEPENDENT_REVIEW.md).

SHA-256 hashes:

- Final report: `50a77bfada76ed2a2c952f97165f90161b377bdae66c49ff9eef5f9c2e953b36`
- Independent diagnostic: `a8724dc0289d3daf4d3548aca6038fb4f06385f04e016fdec7ff8e34482cf566`
- Exact original request: `06dde1c08b676d0e20c906c09ff84a055d5fd1e631cd924c550b7eaf32c0fc6a`
- Resume request: `ed20470a4d75eac7c8f6cdae30c90f1a713a7a6d60a81a65c642d2cf53870a5d`
- Resume hash recheck: `335bcfabdefdc3053e8771d6f27960e36afcd339c502908112caf41e9ae94d6f`

Exact request/response traces and hash metadata are saved in the same `schema_review` directory.
