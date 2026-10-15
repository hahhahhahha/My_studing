"""Run numerical/structural intervention checks without loading PyTorch."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from mappo_gp_sf_dsr.config import MAPPOConfig, MAPPOSystemConfig
from mappo_gp_sf_dsr.env import PortfolioStats
from mappo_gp_sf_dsr.rewards import MAPPORewarder
from joint_reward_ablation_core import make_no_joint_reward, aggregate_execution


def main():
    original = MAPPORewarder._compute_lcc_reward
    altered, effective = make_no_joint_reward(original)
    cfg = MAPPOConfig(lcc_v=.5, lcc_migration_weight=.05, lcc_use_z_queue=False,
                      lcc_use_h_queue=True, latency_norm=3., q_pressure_max=3.)
    system = MAPPOSystemConfig(n_users=2, n_edges=3, task_compute_mb=15.,
                              edge_compute_mb_s=63., slot_s=1.)
    weights = np.asarray([[.8, .1, .1], [.6, .3, .1]])
    tasks = np.asarray([2, 1])
    q = np.asarray([189., 0., 0.])
    stats = PortfolioStats(np.zeros(2), np.zeros(2), 0., 0.,
                           np.zeros(3), np.zeros(3), .8, .8)

    def run(fn, queues=q, counts=tasks, w=weights, h=0., z=0., use_cfg=cfg):
        rewarder = MAPPORewarder(use_cfg, "LCC-MAPPO")
        rewarder.h_queue, rewarder.z_queue = h, z
        output = fn(rewarder, np.asarray([1., 2.]), counts, 1.5, stats, w, queues,
                    system, previous_portfolios=np.full((2, 3), 1/3),
                    slot_latencies=np.asarray([1., 2., 3.]))
        return output, rewarder

    baseline, b = run(original)
    result, a = run(altered)
    assert np.all(result <= 0) and np.all(result >= -1)
    assert np.all(result == result[0]), "Reward is still shared"
    assert np.isclose(b.last_components["lcc_q_term"], 11/15)
    assert np.isclose(a.last_components["lcc_q_term"], 1/3)
    assert not np.array_equal(baseline, result), "Intervention must not be a no-op"
    assert (a.h_queue, a.z_queue, a.nu_var) == (b.h_queue, b.z_queue, b.nu_var)
    for key in ("lcc_mean_latency_norm", "lcc_migration", "lcc_concentration", "lcc_concentration_budget"):
        assert a.last_components[key] == b.last_components[key]

    for fn in (original, altered):
        _, empty = run(fn, counts=np.zeros(2, dtype=int))
        assert empty.last_components["lcc_q_term"] == 0
    equal_q = np.full(3, 63.)
    assert np.array_equal(run(original, queues=equal_q)[0], run(altered, queues=equal_q)[0])
    perm = [2, 0, 1]
    assert np.array_equal(run(altered)[0], run(altered, queues=q[perm], w=weights[:, perm])[0])
    assert not np.array_equal(run(original, queues=equal_q, h=5.)[0], run(altered, queues=equal_q, h=5.)[0])
    assert run(original, h=5.)[1].h_queue == run(altered, h=5.)[1].h_queue
    static = replace(cfg, lcc_static_reward=True)
    assert np.array_equal(run(original, use_cfg=static)[0], run(altered, use_cfg=static)[0])
    before = np.random.get_state()
    run(altered)
    after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(before, after)), "No RNG consumption"
    assert "self.h_queue =" in effective and "self.nu_var" in effective
    metrics = aggregate_execution([
        {"max_realized_share": .5, "realized_hhi": .5, "max_queue_jump_service_slots": x,
         "summary_receipt": i in (0, 2)} for i, x in enumerate([0., 1.1, 0., 0.])])
    assert metrics["unconditional_queue_spike_rate"] == .25
    assert metrics["broadcast_window_queue_spike_rate"] == .5
    print(json.dumps({"status": "pass", "checks": [
        "non_noop", "uniform_mean_not_sum", "shared_bounded_reward", "unchanged_latency_migration",
        "unchanged_H_Z_nu_updates", "idle_slot_zero", "permutation_invariance", "active_H_removal",
        "static_reward_invariance", "no_rng_consumption", "three_slot_receipt_window"],
        "torch_imported": "torch" in sys.modules}, indent=2))


if __name__ == "__main__":
    main()
