from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from repro_risk_averse_edge.config import ROOT, TRACE_ROOT


@dataclass(frozen=True)
class MAPPOSystemConfig:
    """Paper parameters plus MAPPO observation dimensions.

    The numerical system constants mirror ``repro_risk_averse_edge``. The
    extension changes only the information flow and learner: actor observations
    receive server feedback scores rather than raw global queues, and the critic
    receives centralized state plus user identity.
    """

    n_users: int = 5
    n_edges: int = 35
    slot_s: float = 10.0
    edge_compute_mb_s: float = 63.0
    queue_upper_mb: float = 630.0
    max_requests_per_user: int = 7
    task_package_mb: float = 270.0 / 1024.0
    task_compute_mb: float = 10.0
    bandwidth_hz: float = 2e6
    noise_w: float = 1e-10
    antenna_gain: float = 4.11
    carrier_hz: float = 915e6
    tx_power_w: float = 0.25
    light_speed_m_s: float = 3e8
    min_distance_m: float = 1.0
    max_supported_users: int = 30
    include_feedback_age: bool = False
    feedback_age_norm: float = 8.0
    extra_actor_obs_dim: int = 0
    critic_obs_override_dim: int | None = None
    # Canonical protocol: finite buffers use explicit, task-atomic admission.
    # Legacy silent truncation is available only through an explicit override.
    finite_buffer_protocol: bool = True
    finite_buffer_tolerance: float = 1e-5
    drain_max_horizon: int = 2_000
    latency_deadline_s: float = 10.0
    drop_penalty_s: float = 10.0

    @property
    def actor_obs_dim(self) -> int:
        # Loc(2) + Dis(m) + Req(num/data/load) + ServerScore(m)
        # + optional normalized feedback age.
        return 2 + self.n_edges + 3 + self.n_edges + int(self.include_feedback_age) + int(self.extra_actor_obs_dim)

    @property
    def critic_obs_dim(self) -> int:
        if self.critic_obs_override_dim is not None:
            return int(self.critic_obs_override_dim)
        # ServerScore, queue task count, available resource, capacity, wait
        # vectors (5m) + system summary(6) + portfolio stats(6) +
        # current user feature(5).
        return 5 * self.n_edges + 6 + 6 + 5


@dataclass(frozen=True)
class ServerScoreConfig:
    resource_weight: float = 1.0
    capacity_weight: float = 0.5
    task_count_weight: float = 0.5
    wait_weight: float = 1.0
    eps: float = 1e-8


@dataclass(frozen=True)
class MAPPOConfig:
    hidden: int = 128
    user_embedding_dim: int = 8
    buffer_size: int = 20_000
    rollout_size: int = 1_024
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    batch_size: int = 128
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_std_min: float = -1.5
    log_std_max: float = 2.0
    twin_critic: bool = False
    critic_ensemble_count: int = 1
    auto_portfolio_entropy: bool = False
    feedback_age_gate: bool = False
    residual_actor: bool = False
    residual_actor_prior_conditioned: bool = False
    residual_amplitude_init_bias: float = 0.0
    residual_amplitude_penalty: float = 0.0
    residual_proposal_temperature: float = 1.0
    residual_beta_max: float = 0.05
    residual_beta_gamma_delta: float = 1.0
    residual_beta_gamma_volatility: float = 1.0
    residual_aux_kl_weight: float = 0.0
    residual_aux_proposal_kl_weight: float = 0.0
    residual_aux_amplitude_weight: float = 0.0
    residual_beta_penalty: float = 0.0
    edgewise_price_actor: bool = False
    flat_price_bias_actor: bool = False
    feedback_gate_eta: float = 0.25
    target_portfolio_entropy: float = 0.95
    auto_entropy_lr: float = 1e-3
    auto_entropy_init: float = 0.01

    # MAPPO-DSR reward weights.
    dsr_mu: float = 1.5
    dsr_eta: float = 0.01
    dsr_alpha: float = 0.3
    dsr_temperature: float = 5.0
    dsr_beta_mean: float = 0.4
    dsr_beta_tail: float = 0.4
    dsr_beta_gap: float = 0.2
    dsr_beta_qpressure: float = 0.0
    dsr_tail_topk: int = 0
    dsr_tail_fraction: float = 0.6
    latency_norm: float = 3.0
    q_pressure_max: float = 3.0
    q_penalty_weight: float = 0.05
    kappa_i: float = 0.01
    kappa_g: float = 0.01

    # Lyapunov-coordinated cooperative MAPPO (LCC-MAPPO).
    # These parameters are deliberately compact: V controls latency-vs-drift
    # pressure and lambda_m controls switching cost. zeta/cbar are service
    # constraints rather than free reward-shaping knobs.
    lcc_enabled: bool = False
    lcc_v: float = 1.0
    lcc_migration_weight: float = 0.02
    lcc_cvar_alpha: float = 0.99
    lcc_cvar_budget: float = 1.30
    lcc_var_lr: float = 0.05
    lcc_concentration_factor: float = 2.0
    lcc_stale_margin: float = 1.0
    lcc_use_z_queue: bool = True
    lcc_use_h_queue: bool = True
    lcc_use_stale_margin: bool = True
    lcc_static_reward: bool = False
    lcc_price_observation: bool = False
    lcc_price_clean_observation: bool = False
    lcc_price_scale_observation: bool = False
    lcc_edgewise_price_actor: bool = False
    lcc_flat_price_bias_actor: bool = False
    lcc_price_components: str = "Full"
    lcc_price_shuffle: bool = False
    lcc_price_logit_bias: float = 0.0
    lcc_local_critic: bool = False

    score: ServerScoreConfig = field(default_factory=ServerScoreConfig)


@dataclass(frozen=True)
class MAPPOExperimentConfig:
    dataset: str = "KAIST"
    request_distribution: str = "uniform"
    train_steps: int = 80_000
    test_steps: int = 10_000
    seed: int = 42
    users: tuple[int, ...] = (5, 10, 20, 30)
    datasets: tuple[str, ...] = ("KAIST", "NCSU", "NewYork")
    distributions: tuple[str, ...] = ("uniform", "gaussian")
    methods: tuple[str, ...] = ("MAPPO-L", "MAPPO-TeamL", "MAPPO-DSR")
    output_dir: Path = field(default_factory=lambda: ROOT / "results_mappo_gp_sf_dsr")


__all__ = [
    "ROOT",
    "TRACE_ROOT",
    "MAPPOSystemConfig",
    "ServerScoreConfig",
    "MAPPOConfig",
    "MAPPOExperimentConfig",
]
