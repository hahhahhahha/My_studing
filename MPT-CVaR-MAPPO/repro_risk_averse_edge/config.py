from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "Traces_TimeXY_10sec_txt"


@dataclass(frozen=True)
class SystemConfig:
    # Paper system parameters.
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
    mu_return: float = 11.0
    dsr_eta: float = 1e-3

    @property
    def obs_dim(self) -> int:
        # Loc(2) + Dis(m) + Req(max count + per-task package/load pairs) + Que(m)
        return 2 + self.n_edges + (1 + 2 * self.max_requests_per_user) + self.n_edges


@dataclass(frozen=True)
class PPOConfig:
    hidden: int = 128
    buffer_size: int = 20_000
    # Keep buffer_size as the paper replay-buffer capacity. rollout_size controls
    # how often PPO updates during practical tuning; use 20_000 for strict mode.
    rollout_size: int = 20_000
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    batch_size: int = 128
    update_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "KAIST"
    request_distribution: str = "uniform"
    train_steps: int = 80_000
    test_steps: int = 10_000
    seed: int = 42
    users: tuple[int, ...] = (5, 10, 20, 30)
    datasets: tuple[str, ...] = ("KAIST", "NCSU", "NewYork")
    distributions: tuple[str, ...] = ("uniform", "gaussian")
    methods: tuple[str, ...] = (
        "PPO-DSR",
        "PPO-MV",
        "PPO-L",
        "PG",
        "DQN",
        "DRQN",
        "SAC",
    )
    output_dir: Path = field(default_factory=lambda: ROOT / "results_repro")
