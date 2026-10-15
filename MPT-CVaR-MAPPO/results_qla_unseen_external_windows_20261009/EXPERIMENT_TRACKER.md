# Experiment Tracker

| ID | Purpose | Status | Artifact / stop-go gate |
|---|---|---|---|
| E0 | Actual saved source usage and legacy parser audit | Complete | SOURCE_USAGE_AUDIT.json; 189 saved schedule manifests |
| E1 | Azure d01 calibration and d11/d12/d13 unseen demand windows | Complete | data/*.npz and source hashes; test data never fit normalization |
| E2 | Schema and random-stream isolation tests | Complete | Six pure-data tests; not a model smoke test |
| E3 | Freeze 18 original R6 nominal-delay-4 checkpoint/config entries | Complete | FROZEN_BEFORE_TEST.json; all hashes match original registry |
| E4 | Acquire/check Alibaba2017 and extract three capacity windows | Blocked | Download endpoint DNS failure; original v2018 is not a silent substitute |
| E5 | Matching runtime and seeded CUDA witness | Blocked | PyTorch WinError10106; NumPy mismatch; see preflight.log |
| E6 | W1/KAIST/seed1/delay4 three-method smoke | Not started | End-to-end inference and mass conservation; no model updates |
| E7 | Remaining preregistered frozen-checkpoint tests, 243 total | Not started | No retraining; every method sees paired exogenous randomness |
| E8 | Audit, seed-level analysis and figures | Not started | Do not publish performance conclusions before E7/audit |
