# Paper Alignment Audit After Fixes

Audited against `Towards_Risk-Averse_Edge_Computing_With_Deep_Reinf.pdf` with local `pdftotext` extraction.

## Fixed In This Pass

- PPO actor changed from categorical logits to diagonal Gaussian latent output followed by softmax portfolio.
- PPO transition log-prob now comes from the Gaussian latent portfolio action used to produce the weights.
- DSR/MV/PPO-L rewards now use per-user per-slot mean latency for `T(t,u_i)`.
- PPO collection now updates after the replay buffer reaches the paper buffer size of 20,000 transitions, with a final partial update only when enough samples exist.
- Default paper-scale training steps changed to 80,000, matching the learning-curve scale in the paper.
- Edge-server positions changed from jittered grid to seeded uniform sampling inside the selected trace bounding box.
- Queue upper-bound enforcement now accounts for earlier sampled tasks from the same user in the same slot.
- Added `audit_against_paper.py` for parameter and trace-interval checks.

## Alignment Status

| Item | Paper | Current code | Status |
| --- | --- | --- | --- |
| System size | `m=35`, users in `{5,10,20,30}` | `SystemConfig.n_edges=35`, experiment users match | Match |
| Slot length | `tau=10s` | `slot_s=10.0` | Match, with trace caveat |
| State | `s=(Loc,Dis,Req,Que)` | `env.observe()` concatenates current location, distances, request set encoding, queues | Match |
| Queue meaning | computing load, not task count | `queues_mb`, task load in MB | Match |
| User decision order | random order per slot | `rng.permutation(n_users)` | Match |
| Actor output | m-dimensional Gaussian output, softmax to weights | `ActorCritic.gaussian()` then `softmax()` | Match |
| Execution action | sample edge assignment per task from weights | `env.act_for_current_user()` samples every task | Match |
| Transmission latency | sum over all tasks | per-task `tran_s`, per-user totals/means from task list | Match |
| Queue latency | FCFS by shorter upload/transmission completion time, plus initial queue | per-edge sort by `upload_s`, waiting load starts from initial queue | Match |
| Computation latency | `p/c` | `task_compute_mb / edge_compute_mb_s` | Match |
| Queue update | `max(0, q + arrivals - c*tau)` | `_finish_slot()` update | Match |
| Queue upper bound | reject/reselect when full | resamples feasible servers using initial queue plus current slot load | Match |
| CVaR | evaluation/risk metric | `metrics.empirical_cvar()` only in evaluation | Match |
| PPO-DSR | DSR reward from return `mu - T(t,u_i)` | `DSRReward` | Match |
| PPO-MV | binary reward when SR improves | `MVReward` | Match |
| PPO-L baseline | latency-oriented reward | PPO with `-latency` reward | Match to paper adaptation |
| Baseline rewards | irrelevant objectives omitted, use negative latency | DQN/DRQN/PG/SAC/PPO-L use `-latency` | Match |
| Test steps | 10,000 | default `test_steps=10000` | Match |

## Remaining Caveats

- The bundled trajectory folder is `Traces_TimeXY_30sec_txt`; the raw KAIST/NCSU/NewYork files have 30-second intervals. The paper states the adopted records are 10 seconds apart. Current code linearly interpolates 30-second traces to 10-second slots. This is the largest remaining data-fidelity gap.
- The paper does not include exact edge-server coordinates in the PDF or bundled traces. Current code follows the paper statement that servers are uniformly distributed in the environment area, but regenerated coordinates will not reproduce the authors' exact hidden layout.
- PPO's objective is computed on the Gaussian latent portfolio action. The environment then samples task-level edge assignments from the portfolio. This implements the paper's Gaussian-plus-softmax portfolio mechanism, but the PDF is not explicit about whether the PPO log-prob should additionally include the task assignment sampling probability.
- PG, SAC, DQN, and DRQN are adapted local baselines in the same MEC environment, consistent with the paper's statement that irrelevant reward terms are omitted. They are not complete standalone reproductions of every referenced external baseline paper.
- Smoke-test outputs are not convergence evidence. Use the 80,000-step paper-scale run for learning-curve comparison and the 10,000-step test phase for reported metrics.

## Verification Commands Run

```powershell
python -B -m repro_risk_averse_edge.audit_against_paper

python -B -m repro_risk_averse_edge.run_experiments `
  --datasets KAIST `
  --users 5 `
  --distributions uniform `
  --methods PPO-DSR,PPO-MV,DQN,DRQN,PG,SAC,PPO-L `
  --train-steps 20 `
  --test-steps 20 `
  --output-dir results_repro_smoke_final_audit

python -B -m repro_risk_averse_edge.run_experiments `
  --datasets KAIST `
  --users 5 `
  --distributions uniform `
  --methods PPO-DSR `
  --train-steps 150 `
  --test-steps 20 `
  --output-dir results_repro_smoke_ppo_update_final_audit
```

