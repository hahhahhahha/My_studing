# Towards Risk-Averse Edge Computing With DRL Reproduction

This package is a clean reproduction scaffold for **Towards Risk-Averse Edge Computing With Deep Reinforcement Learning**.

Important scope notes:

- This is **single-agent/shared-policy PPO**, not MAPPO and not independent multi-agent PPO.
- State is exactly the paper form `s=(Loc, Dis, Req, Que)`. It does not include historical latency vectors.
- The actor emits an `m=35` portfolio distribution. The environment samples one edge server per task from that distribution.
- PPO-DSR/PPO-MV use a diagonal Gaussian latent actor followed by softmax, so the portfolio is continuous even though the environment executes sampled edge-server assignments.
- Queue length is computing load in MB, not number of tasks.
- FCFS queueing uses upload completion order inside the current time slot and includes the initial queue load.
- CVaR is reported as an evaluation metric. It is not used as `-latency - lambda * CVaR` training reward.
- PPO-DSR and PPO-MV follow the paper reward definitions; PPO-L is the latency-oriented baseline.
- Training rewards use per-user per-slot average latency `T(t,u_i)`. Reported system average latency uses total task count as the denominator.

## Files

- `config.py`: paper parameters and PPO hyperparameters.
- `data.py`: KAIST/NCSU/NewYork trace loader with 30s-to-10s interpolation.
- `env.py`: MEC offloading environment, portfolio sampling, communication model, FCFS queues, queue update.
- `rewards.py`: DSR, MV, and latency rewards.
- `agents.py`: shared-policy PPO plus DQN/DRQN/PG/discrete-SAC baseline interfaces.
- `metrics.py`: average latency and empirical CVaR@95/@99.
- `run_experiments.py`: training, testing, CSV output, learning curves, latency PDF plots.
- `audit_against_paper.py`: parameter and trace-interval audit helper.

## Quick Smoke Test

From `workspace/MPT-CVaR-MAPPO`:

```powershell
python -m repro_risk_averse_edge.run_experiments `
  --datasets KAIST `
  --users 5 `
  --distributions uniform `
  --methods PPO-DSR,PPO-MV,DQN,DRQN,PG,SAC,PPO-L `
  --train-steps 20 `
  --test-steps 20 `
  --output-dir results_repro_smoke_all
```

## Full Paper Grid

This runs the requested datasets, user counts, request distributions, and baselines. It is computationally much larger:

```powershell
python -m repro_risk_averse_edge.run_experiments `
  --datasets KAIST,NCSU,NewYork `
  --users 5,10,20,30 `
  --distributions uniform,gaussian `
  --methods PPO-DSR,PPO-MV,DQN,DRQN,PG,SAC,PPO-L `
  --train-steps 80000 `
  --test-steps 10000 `
  --output-dir results_repro
```

Outputs:

- `results_repro/summary.csv`: average latency, CVaR@95, CVaR@99, task count, average inference time.
- `results_repro/<dataset>/<users>u/<distribution>/learning_curves.png`.
- `results_repro/<dataset>/<users>u/<distribution>/latency_pdf.png`.
- Per-method learning CSV files in the same folder.

## Reproducibility Detail

The paper specifies real user trajectories and `m=35` edge servers uniformly distributed in the environment area, but does not provide edge server coordinates in the bundled traces. This implementation samples server locations uniformly inside the selected trace bounding box with a fixed seed. User mobility still comes from the KAIST/NCSU/NewYork GPS traces.

The bundled folder is named `Traces_TimeXY_30sec_txt`, and the raw files use 30-second timestamps. `data.py` interpolates these to the paper's 10-second time slots. If the original 10-second CRAWDAD traces are available, replace `Traces_TimeXY_30sec_txt` with those files and keep `slot_s=10.0`.

The PPO-L/PG/SAC entries are local baselines adapted to this reproduction environment. They preserve the requested algorithm families and output metrics, but they are not complete reproductions of every external baseline paper.

Run the audit helper with:

```powershell
python -B -m repro_risk_averse_edge.audit_against_paper
```
