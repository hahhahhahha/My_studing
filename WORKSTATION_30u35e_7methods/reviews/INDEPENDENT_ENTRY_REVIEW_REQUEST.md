# Type-A independent review request

## Scope

Review only the additive `WORKSTATION_30u35e_7methods_v2` entry. Do not start a
test cell. The review must cover the seven-method 30-user/35-logical-edge
contract, pilot adoption continuity, complete bundle and asset binding, common
schedule pairing, pre-launch admission, process shutdown, and the exact
PowerShell runner.

## Required evidence

- `protocol/FORMAL_CONFIG.json`
- `protocol/FROZEN_SYSTEM_PROTOCOL.json`
- `protocol/PILOT_ADOPTION.json`
- `protocol/E1_GRID.json`
- `protocol/E1_TEST_REGISTRY.json`
- `protocol/FORMAL_REGISTRATION.json`
- `TEST_IMAGES.json`
- all five `schedules/test_seed_*.json`
- `protocol/bundles/*.json`
- `FORMAL_PLAN.json`
- `run_formal_30u35e.ps1`
- `tools/*.py` and `systemlab/*.py`
- the referenced v3 calibration, pilot receipts and source dictionaries

The reviewer must issue a machine-readable `protocol/INDEPENDENT_ENTRY_REVIEW.json`
with `status` `PASS` or unblocked `WARN`, `blockers: []`, `complete_report: true`,
`reviewer_identity`, `protocol_sha256`, `test_registry_sha256`, complete
`source_sha256`, `report_path`, and `report_sha256`. A self-audit is not approval.
Formal cells remain forbidden until this receipt is complete and the registry is
rebound with `formal_entry_approved: true`.
