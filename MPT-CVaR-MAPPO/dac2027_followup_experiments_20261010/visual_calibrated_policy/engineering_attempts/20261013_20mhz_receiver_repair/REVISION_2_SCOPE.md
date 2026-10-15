# 20 MHz smoke receiver repair — revision 2

This additive revision responds only to the preserved complete FAIL report in `independent_rereview/REVIEW_RESPONSE.md` and `REVIEW_RESULT.json` (status `FAIL`, three blockers). It does not overwrite either report or the revision-1 before-source snapshots.

Revision-2 source snapshots were copied before edits to `SOURCE_BEFORE_REVISION_2/`:

| File | SHA256 |
|---|---|
| `visual_model_smoke_small.py` | `cc2b14cafde350511322f6111b0fdaf442bf1d40d0554a94df76909448b26bdd` |
| `visual_pilot_preparation.py` | `ae934106b9ae438874b7e4dfdd28a46c3d6c1d79950802336a7ac55fbf2576e2` |
| `reward_input_integrity.py` | `1f774d8322a72b8fdb82fb3f5c2a49353e22d4fb8f45cf5864ec872926837cf3` |

Scoped repairs:

1. Derive the smoke's immutable protected-input map from its full current source lock, retaining and byte-checking only prior raw/evidence files not already covered as executable source. Require all historical implementation entries to remain inside that source lock.
2. Validate each actual smoke receipt against its completed ledger, actual counts, unique committed episode, schedule/input-condition SHA256 identities, finite update-statistics digest, attached checkpoint, and exact transaction-event sequence. Require exact shared schedule and input-condition content across methods.
3. Bind the exact smoke plan path and bytes into the pilot object's persistent input boundary; reread and revalidate the same plan and actual smoke receipt at each boundary check.
4. Allow the visual smoke entry to require its separately designated new-scope Type-A report at `independent_repair_rereview/`; keep the reward smoke's default old path unchanged. A prior review of different source bytes cannot admit the revised visual entry.

No changes are authorized to the 20 MHz link, retained 2 MHz evidence, service/raw inputs, split, model/weights, algorithm, reward, horizon, terminal/H rules, budget, numerical criteria, strict `<1800 MiB` gate, or formal training/test protocol. This source repair does not itself authorize a smoke, pilot, formal training, or test run. The isolated adversarial checks are author fixtures only and must not generate or label any actual result.
