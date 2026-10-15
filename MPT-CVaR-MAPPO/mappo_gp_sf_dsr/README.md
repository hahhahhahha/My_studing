# MAPPO-GP-SF-DSR

This package is a user-side MAPPO extension of the paper reproduction in
`repro_risk_averse_edge`. The original reproduction code is kept unchanged; this
package copies the paper system constants and data-loading semantics, then
changes the learner and information flow for the multi-agent study.

## Algorithm Positioning

`MAPPO-GP-SF-DSR` means:

- `MAPPO`: every mobile user is an agent; users share one actor.
- `GP`: Gaussian latent portfolio, `z ~ Normal(mu, sigma)`, `w = softmax(z)`.
- `SF`: actor sees server feedback `Score`, not raw global queue `Que`.
- `DSR`: MAPPO-DSR uses a global Differential Sharpe Ratio risk signal.

The environment uses a joint slot step: all users produce portfolios first, each
task samples an edge server from its user's portfolio, then delays and queues are
computed jointly.

## Modules

- `config.py`: paper parameters plus MAPPO dimensions and hyperparameters.
- `env.py`: joint MEC environment, ServerScore, portfolio statistics.
- `models.py`: shared Gaussian actor and agent-specific centralized critic.
- `buffer.py`: per-user MAPPO transition buffer with GAE support.
- `rewards.py`: `MAPPO-L`, `MAPPO-TeamL`, V2.1 DSR variants, V3 Q-penalty variants, and the V4 SAC-style variant.
- `metrics.py`: paper metrics plus multi-agent and portfolio metrics.
- `train.py`: training/evaluation CLI.
- `evaluate.py`: deterministic evaluation helpers.
- `check_alignment.py`: smoke-level design compliance checks.
- `compare_summaries.py`: compare an old MAPPO-DSR summary with the risk-DSR summary.

## Key Design Choices

Actor input:

```text
o_i = [Loc_i, Dis_i, Req_i, Score_all]
```

`Req_i` is `[num_tasks, total_data_size, total_compute_load]` normalized by the
per-slot maximum. The actor does not receive raw `Que`.

Server feedback:

```text
Score_j = wr * R_avail_norm_j + wc * c_norm_j
          - wn * Nq_norm_j - ww * Wwait_norm_j
```

Default weights are `wr=1.0`, `wc=0.5`, `wn=0.5`, `ww=1.0`, with no `tanh`.

Critic input includes all server feedback vectors, system task summary, current
user feature, user embedding, and:

```text
mean_portfolio_entropy
mean_portfolio_concentration
global_demand_concentration
max_server_expected_demand_share
user_i_entropy
user_i_concentration
```

Reward ablations:

```text
MAPPO-L            : -T_i_norm - kappa_i*C_i - kappa_g*C_global
MAPPO-TeamL        : -T_global_norm - kappa_i*C_i - kappa_g*C_global
DSR-MTG            : alpha*DSR_norm + (1-alpha)*(-T_i_norm) - concentration penalty
DSR-MTG-Q          : V2.1, DSR-MTG plus Q_pressure in the DSR input
DSR-MTG-Q-noConc   : DSR-MTG-Q without concentration penalty
DSR-MTG-Qpen-v3    : V3, latency-only DSR plus portfolio-weighted Q penalty
DSR-MTG-noQpen-v3  : V3, latency-only DSR without direct Q penalty
DSR-MTG-Qpen-v4    : V3 reward plus SAC-style adaptive portfolio entropy and twin critic
```

For V3 methods, the normalized DSR input is latency-only:

```text
X = beta_mean*T_mean_norm
  + beta_tail*T_tail_norm
  + beta_gap*Gap_norm

T_tail = mean(TopK(active per-user latency))
Gap = max(active per-user latency) - min(active per-user latency)
DSR_norm = tanh(DSR(mu_g - X) / dsr_temperature)
```

For `DSR-MTG-Qpen-v3`, congestion is an immediate portfolio-weighted penalty:

```text
Q_j = (queue_before_j + expected_incoming_load_j) / (c_j*tau)
Q_norm_j = clip(Q_j, 0, q_pressure_max) / q_pressure_max
Q_port_i = sum_j w_ij * Q_norm_j

r_i = alpha*DSR_norm + (1-alpha)*(-T_i_norm)
      - kappa_i*C_i - kappa_g*C_global - q_penalty_weight*Q_port_i
```

`DSR-MTG-Qpen-v4` keeps the same reward and enables two SAC-inspired training
stabilizers:

```text
portfolio_entropy_coef is learned automatically toward target_portfolio_entropy
centralized value uses twin critics, with min(V1, V2) for rollout values
```

Defaults:

```text
beta_mean = 0.4
beta_tail = 0.4
beta_gap  = 0.2
beta_qpressure = 0.0
alpha = 0.3
kappa_i = kappa_g = 0.01
q_penalty_weight = 0.05
target_portfolio_entropy = 0.95
latency_norm = 3.0
dsr_mu = 1.5
dsr_eta = 0.01
dsr_temperature = 5.0
log_std_min = -1.5
```

By default `TopK` uses the top 60% of active users. This avoids the 5-user case
collapsing to only the single worst user. Use `--tail-topk` to set a fixed K.
Use `--cvar-alpha` to set `tail_fraction = 1 - cvar_alpha`.

## Run Checks

```powershell
python -B -m mappo_gp_sf_dsr.check_alignment
```

## Smoke Training

```powershell
python -B -m mappo_gp_sf_dsr.train `
  --dataset KAIST `
  --users 5 `
  --distribution uniform `
  --method DSR-MTG-Qpen-v4 `
  --train-steps 20 `
  --test-steps 20 `
  --rollout-size 20 `
  --batch-size 20 `
  --output-dir results_mappo_smoke
```

Outputs are written under:

```text
results_mappo_smoke/KAIST/5u/uniform/
```

including `learning_<method>.csv`, `latency_samples_<method>.csv`,
`summary_<method>.csv`, `config_<method>.json`, `learning_<method>.png`,
`latency_pdf_<method>.png`, and `entropy_vs_latency_<method>.png`.

## Compare Old vs Risk DSR

```powershell
python -B -m mappo_gp_sf_dsr.compare_summaries `
  --old results_mappo_20u/KAIST/20u/uniform/summary_MAPPO-DSR.csv `
  --new results_mappo_risk_dsr_20u/KAIST/20u/uniform/summary_MAPPO-DSR.csv `
  --out results_mappo_risk_dsr_20u/KAIST/20u/uniform/compare_old_vs_risk_dsr.csv
```
