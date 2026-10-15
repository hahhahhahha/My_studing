from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAPPOConfig, MAPPOSystemConfig
from .env import PortfolioStats


@dataclass
class GlobalDSR:
    mu: float = 1.5
    eta: float = 0.01
    temperature: float = 5.0
    eps: float = 1e-8
    a: float = 0.0
    b: float = 0.0

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0

    def update_from_x(self, x: float) -> tuple[float, float]:
        return_value = self.mu - x
        prev_a = self.a
        prev_b = self.b
        delta_a = return_value - prev_a
        delta_b = return_value * return_value - prev_b
        denom = max(prev_b - prev_a * prev_a, self.eps) ** 1.5
        dsr_raw = (prev_b * delta_a - 0.5 * prev_a * delta_b) / denom
        self.a = prev_a + self.eta * delta_a
        self.b = prev_b + self.eta * delta_b
        dsr_norm = float(np.tanh(dsr_raw / max(self.temperature, self.eps)))
        return float(dsr_raw), dsr_norm


def topk_mean(values: np.ndarray, k: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    k = int(np.clip(k, 1, values.size))
    return float(np.mean(np.sort(values)[-k:]))


def resolve_tail_k(n_active: int, configured_k: int, fraction: float) -> int:
    if n_active <= 0:
        return 0
    if configured_k > 0:
        return min(configured_k, n_active)
    return max(1, int(np.ceil(fraction * n_active)))


def risk_latency_components(
    per_user_latency: np.ndarray,
    per_user_tasks: np.ndarray,
    system_average_latency: float,
    beta_mean: float,
    beta_tail: float,
    beta_gap: float,
    tail_topk: int = 0,
    tail_fraction: float = 0.6,
) -> dict[str, float]:
    active = np.asarray(per_user_latency, dtype=np.float64)[np.asarray(per_user_tasks) > 0]
    if active.size == 0:
        tail_latency = 0.0
        gap_latency = 0.0
        k = 0
    else:
        k = resolve_tail_k(active.size, tail_topk, tail_fraction)
        tail_latency = topk_mean(active, k)
        gap_latency = float(np.max(active) - np.min(active))
    risk_latency = (
        beta_mean * float(system_average_latency)
        + beta_tail * tail_latency
        + beta_gap * gap_latency
    )
    return {
        "system_risk_latency": float(risk_latency),
        "risk_mean_latency": float(system_average_latency),
        "risk_tail_latency": float(tail_latency),
        "risk_gap_latency": float(gap_latency),
        "risk_tail_topk": int(k),
    }


def compute_q_pressure_vector(
    queues_before_mb: np.ndarray,
    portfolios: np.ndarray,
    per_user_tasks: np.ndarray,
    task_compute_mb: float,
    edge_compute_mb_s: float,
    slot_s: float,
    q_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return server pressure, portfolio-weighted user pressure, and max pressure."""
    c_tau = edge_compute_mb_s * slot_s
    user_loads = np.asarray(per_user_tasks, dtype=np.float64) * task_compute_mb
    incoming_load = np.asarray(portfolios, dtype=np.float64).T @ user_loads
    post_load = np.asarray(queues_before_mb, dtype=np.float64) + incoming_load
    q_serv = post_load / max(c_tau, 1e-8)
    q_serv_norm = np.clip(q_serv, 0.0, q_max) / max(q_max, 1e-8)
    q_port_pressure = np.asarray(portfolios, dtype=np.float64) @ q_serv_norm
    q_pressure_max = float(np.max(q_serv_norm)) if q_serv_norm.size else 0.0
    return q_serv_norm, q_port_pressure, q_pressure_max


def compute_q_pressure(
    queues_before_mb: np.ndarray,
    portfolios: np.ndarray,
    per_user_tasks: np.ndarray,
    task_compute_mb: float,
    edge_compute_mb_s: float,
    slot_s: float,
    q_max: float = 3.0,
) -> float:
    _, _, q_pressure_max = compute_q_pressure_vector(
        queues_before_mb=queues_before_mb,
        portfolios=portfolios,
        per_user_tasks=per_user_tasks,
        task_compute_mb=task_compute_mb,
        edge_compute_mb_s=edge_compute_mb_s,
        slot_s=slot_s,
        q_max=q_max,
    )
    return q_pressure_max


def compute_dsr_input(
    per_user_latency: np.ndarray,
    per_user_tasks: np.ndarray,
    system_average_latency: float,
    portfolios: np.ndarray,
    queues_before_mb: np.ndarray,
    cfg: MAPPOConfig,
    sys_cfg: MAPPOSystemConfig,
    use_q_pressure: bool = True,
) -> dict[str, float]:
    t_norm = max(cfg.latency_norm, 1e-8)
    active = np.asarray(per_user_latency, dtype=np.float64)[np.asarray(per_user_tasks) > 0]
    if active.size == 0:
        tail = 0.0
        gap = 0.0
        k = 0
        t_mean_norm = 0.0
        t_topk_norm = 0.0
        gap_norm = 0.0
    else:
        k = resolve_tail_k(active.size, cfg.dsr_tail_topk, cfg.dsr_tail_fraction)
        tail = topk_mean(active, k)
        gap = float(np.max(active) - np.min(active))
        t_mean_norm = float(system_average_latency) / t_norm
        t_topk_norm = tail / t_norm
        gap_norm = gap / t_norm

    _, q_port_pressure, q_pressure_max = compute_q_pressure_vector(
        queues_before_mb=queues_before_mb,
        portfolios=portfolios,
        per_user_tasks=per_user_tasks,
        task_compute_mb=sys_cfg.task_compute_mb,
        edge_compute_mb_s=sys_cfg.edge_compute_mb_s,
        slot_s=sys_cfg.slot_s,
        q_max=cfg.q_pressure_max,
    )
    q_port_pressure = np.where(np.asarray(per_user_tasks) > 0, q_port_pressure, 0.0)
    q_pressure_for_dsr = q_pressure_max if use_q_pressure else 0.0

    x = (
        cfg.dsr_beta_mean * t_mean_norm
        + cfg.dsr_beta_tail * t_topk_norm
        + cfg.dsr_beta_gap * gap_norm
        + cfg.dsr_beta_qpressure * q_pressure_for_dsr
    )
    return {
        "dsr_input_x": float(x),
        "t_mean_norm": float(t_mean_norm),
        "t_topk_norm": float(t_topk_norm),
        "gap_norm": float(gap_norm),
        "q_pressure": float(q_pressure_max),
        "q_pressure_for_dsr": float(q_pressure_for_dsr),
        "q_port_pressure_mean": float(np.mean(q_port_pressure)) if q_port_pressure.size else 0.0,
        "risk_tail_topk": int(k),
        "risk_mean_latency": float(system_average_latency),
        "risk_tail_latency": float(tail),
        "risk_gap_latency": float(gap),
        # Compatibility alias for older comparison scripts.
        "system_risk_latency": float(x),
    }


class MAPPORewarder:
    """Reward ablations for MAPPO-L, TeamL, and DSR-MTG variants."""

    def __init__(self, cfg: MAPPOConfig, method: str = "MAPPO-DSR") -> None:
        self.cfg = cfg
        self.method = method
        self.dsr = GlobalDSR(mu=cfg.dsr_mu, eta=cfg.dsr_eta, temperature=cfg.dsr_temperature)
        self.last_components: dict[str, float] = {}
        self.z_queue = 0.0
        self.h_queue = 0.0
        self.nu_var = 0.0

    def reset(self) -> None:
        self.dsr.reset()
        self.z_queue = 0.0
        self.h_queue = 0.0
        self.nu_var = 0.0

    @property
    def uses_q_pressure(self) -> bool:
        return self.method in {"DSR-MTG-Q", "DSR-MTG-Q-noConc"}

    @property
    def uses_direct_q_penalty(self) -> bool:
        return self.method in {
            "DSR-MTG-Qpen-v3",
            "DSR-MTG-Qpen-v4",
            "DSR-MTG-Qpen-v4-agegate",
            "DSR-MTG-Qpen-v4-agegate-lyapprior",
            "DSR-MTG-Qpen-v4-agegate-noConc",
        }

    @property
    def logs_direct_q_pressure(self) -> bool:
        return self.method in {
            "DSR-MTG-Qpen-v3",
            "DSR-MTG-noQpen-v3",
            "DSR-MTG-Qpen-v4",
            "DSR-MTG-Qpen-v4-agegate",
            "DSR-MTG-Qpen-v4-agegate-lyapprior",
            "DSR-MTG-Qpen-v4-agegate-noQpen",
            "DSR-MTG-Qpen-v4-agegate-noConc",
        }

    @property
    def uses_conc_penalty(self) -> bool:
        return self.method not in {"DSR-MTG-Q-noConc", "DSR-MTG-Qpen-v4-agegate-noConc", "M2PPO"}

    @property
    def is_dsr_method(self) -> bool:
        return self.method.startswith("DSR-") or self.method in {"MAPPO-DSR", "MAPPO-DSR-risk"}

    @property
    def is_lcc_method(self) -> bool:
        return self.method.startswith("LCC-MAPPO")

    def _compute_lcc_reward(
        self,
        per_user_latency: np.ndarray,
        per_user_tasks: np.ndarray,
        system_average_latency: float,
        portfolio_stats: PortfolioStats,
        portfolios: np.ndarray,
        queues_before_mb: np.ndarray,
        sys_cfg: MAPPOSystemConfig,
        previous_portfolios: np.ndarray | None = None,
        slot_latencies: np.ndarray | None = None,
    ) -> np.ndarray:
        latency_scale = max(float(self.cfg.latency_norm), 1e-8)
        capacity_slot = max(float(sys_cfg.edge_compute_mb_s * sys_cfg.slot_s), 1e-8)
        user_loads = np.asarray(per_user_tasks, dtype=np.float64) * float(sys_cfg.task_compute_mb)
        weights = np.asarray(portfolios, dtype=np.float64)
        arrivals = weights.T @ user_loads if weights.size else np.zeros(sys_cfg.n_edges, dtype=np.float64)
        queue_pressure = np.asarray(queues_before_mb, dtype=np.float64) / capacity_slot
        arrival_pressure = arrivals / capacity_slot

        queue_pressure_norm = np.clip(queue_pressure, 0.0, max(float(self.cfg.q_pressure_max), 1e-8)) / max(
            float(self.cfg.q_pressure_max), 1e-8
        )
        arrival_share = arrivals / max(float(np.sum(arrivals)), 1e-8)
        q_term = float(np.sum(queue_pressure_norm * arrival_share)) if arrival_share.size else 0.0
        task_latencies_norm = (
            np.asarray(slot_latencies, dtype=np.float64) / latency_scale
            if slot_latencies is not None and np.asarray(slot_latencies).size
            else np.asarray(per_user_latency, dtype=np.float64)[np.asarray(per_user_tasks) > 0] / latency_scale
        )
        task_latencies_norm = task_latencies_norm[np.isfinite(task_latencies_norm)]
        alpha = float(np.clip(self.cfg.lcc_cvar_alpha, 0.0, 0.9999))
        if task_latencies_norm.size:
            empirical_var = float(np.quantile(task_latencies_norm, alpha))
            if self.nu_var <= 0.0:
                self.nu_var = empirical_var
            else:
                lr = float(np.clip(self.cfg.lcc_var_lr, 0.0, 1.0))
                self.nu_var = (1.0 - lr) * self.nu_var + lr * empirical_var
            excess = np.maximum(task_latencies_norm - self.nu_var, 0.0)
            tail_latency_norm = float(self.nu_var + np.mean(excess) / max(1.0 - alpha, 1e-8))
            tail_count = max(1, int(np.ceil((1.0 - alpha) * task_latencies_norm.size)))
        else:
            tail_count = 0
            tail_latency_norm = 0.0
        mean_latency_norm = float(system_average_latency) / latency_scale

        concentration = float(portfolio_stats.global_demand_concentration)
        # portfolio_stats.global_demand_concentration is normalized HHI:
        # (sum_j s_j^2 - 1/M) / (1 - 1/M).  If the raw HHI budget is chi/M,
        # the equivalent normalized budget is (chi - 1) / (M - 1).
        concentration_budget = max(
            0.02,
            (float(self.cfg.lcc_concentration_factor) - 1.0) / max(float(sys_cfg.n_edges - 1), 1.0),
        )
        cvar_budget = max(float(self.cfg.lcc_cvar_budget), 1e-8)
        z_violation = tail_latency_norm - cvar_budget
        h_violation = concentration - concentration_budget

        if previous_portfolios is None or np.asarray(previous_portfolios).shape != weights.shape:
            migration = 0.0
        else:
            migration = float(np.mean(np.sum(np.abs(weights - np.asarray(previous_portfolios, dtype=np.float64)), axis=1)))

        z_pressure = 0.0 if (not self.cfg.lcc_use_z_queue or self.cfg.lcc_static_reward) else float(np.tanh(self.z_queue / 10.0))
        h_pressure = 0.0 if (not self.cfg.lcc_use_h_queue or self.cfg.lcc_static_reward) else float(np.tanh(self.h_queue / 5.0))
        drift_part = 0.0 if self.cfg.lcc_static_reward else (
            q_term
            + z_pressure * z_violation
            + h_pressure * h_violation
        )
        penalty_part = float(self.cfg.lcc_v) * (
            mean_latency_norm + float(self.cfg.lcc_migration_weight) * migration
        )
        phi = max(0.0, drift_part + penalty_part)
        team_reward = -float(np.tanh(phi))

        if not self.cfg.lcc_static_reward:
            if self.cfg.lcc_use_z_queue:
                self.z_queue = max(0.0, self.z_queue + 0.05 * z_violation)
            if self.cfg.lcc_use_h_queue:
                self.h_queue = max(0.0, self.h_queue + 0.05 * h_violation)

        self.last_components = {
            "lcc_phi": float(phi),
            "lcc_reward": float(team_reward),
            "lcc_q_term": float(q_term),
            "lcc_mean_latency_norm": float(mean_latency_norm),
            "lcc_tail_latency_norm": float(tail_latency_norm),
            "lcc_tail_topk": int(tail_count),
            "lcc_var_nu": float(self.nu_var),
            "lcc_cvar_violation": float(z_violation),
            "lcc_concentration": float(concentration),
            "lcc_concentration_budget": float(concentration_budget),
            "lcc_concentration_violation": float(h_violation),
            "lcc_migration": float(migration),
            "lcc_z_queue": float(self.z_queue),
            "lcc_h_queue": float(self.h_queue),
            "lcc_v": float(self.cfg.lcc_v),
            "lcc_migration_weight": float(self.cfg.lcc_migration_weight),
        }
        return np.full_like(np.asarray(per_user_latency, dtype=np.float64), team_reward, dtype=np.float32)

    def compute(
        self,
        per_user_latency: np.ndarray,
        per_user_tasks: np.ndarray,
        system_average_latency: float,
        portfolio_stats: PortfolioStats,
        portfolios: np.ndarray | None = None,
        queues_before_mb: np.ndarray | None = None,
        sys_cfg: MAPPOSystemConfig | None = None,
        previous_portfolios: np.ndarray | None = None,
        slot_latencies: np.ndarray | None = None,
    ) -> np.ndarray:
        latency_scale = max(self.cfg.latency_norm, 1e-8)
        individual = -np.asarray(per_user_latency, dtype=np.float64) / latency_scale
        team = -float(system_average_latency) / latency_scale
        if self.uses_conc_penalty:
            penalty = (
                self.cfg.kappa_i * portfolio_stats.concentration.astype(np.float64)
                + self.cfg.kappa_g * portfolio_stats.global_demand_concentration
            )
        else:
            penalty = np.zeros_like(individual)

        if self.is_lcc_method:
            if portfolios is None or queues_before_mb is None or sys_cfg is None:
                raise ValueError("LCC-MAPPO requires portfolios, queues_before_mb, and sys_cfg")
            rewards = self._compute_lcc_reward(
                per_user_latency=per_user_latency,
                per_user_tasks=per_user_tasks,
                system_average_latency=system_average_latency,
                portfolio_stats=portfolio_stats,
                portfolios=portfolios,
                queues_before_mb=queues_before_mb,
                sys_cfg=sys_cfg,
                previous_portfolios=previous_portfolios,
                slot_latencies=slot_latencies,
            )
        elif self.method in {"MAPPO-L", "M2PPO"}:
            rewards = individual - penalty
        elif self.method == "MAPPO-TeamL":
            rewards = np.full_like(individual, team) - penalty
        elif self.is_dsr_method:
            if portfolios is None or queues_before_mb is None or sys_cfg is None:
                raise ValueError("DSR methods require portfolios, queues_before_mb, and sys_cfg")
            components = compute_dsr_input(
                per_user_latency=per_user_latency,
                per_user_tasks=per_user_tasks,
                system_average_latency=system_average_latency,
                portfolios=portfolios,
                queues_before_mb=queues_before_mb,
                cfg=self.cfg,
                sys_cfg=sys_cfg,
                use_q_pressure=self.uses_q_pressure,
            )
            dsr_raw, dsr_norm = self.dsr.update_from_x(components["dsr_input_x"])
            q_penalty = np.zeros_like(individual)
            if self.logs_direct_q_pressure:
                _, q_port_pressure, q_pressure_max = compute_q_pressure_vector(
                    queues_before_mb=queues_before_mb,
                    portfolios=portfolios,
                    per_user_tasks=per_user_tasks,
                    task_compute_mb=sys_cfg.task_compute_mb,
                    edge_compute_mb_s=sys_cfg.edge_compute_mb_s,
                    slot_s=sys_cfg.slot_s,
                    q_max=self.cfg.q_pressure_max,
                )
                active = np.asarray(per_user_tasks) > 0
                q_port_pressure = np.where(active, q_port_pressure, 0.0)
                if self.uses_direct_q_penalty:
                    q_penalty = self.cfg.q_penalty_weight * q_port_pressure
                components = {
                    **components,
                    "q_pressure": float(q_pressure_max),
                    "q_port_pressure_mean": float(np.mean(q_port_pressure)) if q_port_pressure.size else 0.0,
                    "q_penalty_mean": float(np.mean(q_penalty)) if q_penalty.size else 0.0,
                }
            else:
                components = {
                    **components,
                    "q_penalty_mean": 0.0,
                }
            self.last_components = {
                **components,
                "dsr_raw": float(dsr_raw),
                "dsr_norm": float(dsr_norm),
            }
            rewards = (
                self.cfg.dsr_alpha * dsr_norm
                + (1.0 - self.cfg.dsr_alpha) * individual
                - penalty
                - q_penalty
            )
        else:
            raise ValueError(f"Unknown MAPPO reward method: {self.method}")

        # Users with zero tasks still produced a portfolio; keep only the small
        # concentration penalty for MAPPO-L so idle users do not receive a fake
        # latency benefit.
        if self.method in {"MAPPO-L", "M2PPO"}:
            idle = np.asarray(per_user_tasks) <= 0
            rewards[idle] = -penalty[idle]
        return rewards.astype(np.float32)
