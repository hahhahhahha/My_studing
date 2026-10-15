# Independent agent-follows-doc preflight review

2026-10-10. Fresh reviewer ran the documented inspect and CPU check commands from the new experiment directory, read-only; no downloads, installations, timing or training.

Verdict: **BLOCKED / NOT READY**.

Inspect exited0 and accurately reported both downloadsfailed, CPU witnessnotready, topologyverified and allfour image-role measurement counts0. Check exited1 at PyTorch import via asyncio/_overlapped withWinError10106; the actual CPU kernel never executed.

The initial review identified four entry-integrity defects. Following repairs, independent recheck confirmed:

1. Cached image split hash, counts, official roles and cross-role ID/hash isolation are verified.
2. Formal measurement requires an audited real-pretrained-model pilot, bound to environment/image/source/measurement hashes.
3. Test calibration freeze verifies schema, acceptance, environment/image hashes, allfive calibration and allthree validation session artifacts, plus the independent validation-report hash and acceptance.
4. Re-prepare counts actualCSV request rows and preserves prior measurement state rather than resetting it to0.

The documented check command, session examples and non-interactive missing-session error now match the source.

This is a source and blocked-entry audit, **not** a real-data integration test. Missing data/weights, failedCPU witness and absentrealmodelpilot remain actual blockers. No performance measurements or simulation results are certified.
