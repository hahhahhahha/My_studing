"""Process-local reward intervention around the immutable R1 training worker."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_reward_ablation_core import (VARIANT, make_no_joint_reward,
                                      aggregate_execution, aggregate_reward)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["Full", VARIANT], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    opts, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    source = opts.source_root.resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "paper_doc_tools"))
    from paired_training_repair_core import source_digest, sha256
    if source_digest(source) != opts.expected_source:
        raise RuntimeError("Frozen execution source changed")
    import numpy as np
    from mappo_gp_sf_dsr import train
    from mappo_gp_sf_dsr.env import JointMAPPOEdgeEnv
    from mappo_gp_sf_dsr.rewards import MAPPORewarder
    role = remaining[remaining.index("--evaluation-split-role") + 1]
    phase = "train"
    execution, reward_rows = [], []
    last_source = {}
    reward_counters = {}
    opts.diagnostic_dir.mkdir(parents=True, exist_ok=True)

    original_reward = MAPPORewarder._compute_lcc_reward
    effective, effective_source = make_no_joint_reward(original_reward)
    chosen = effective if opts.variant == VARIANT else original_reward
    applied_source = effective_source if opts.variant == VARIANT else textwrap.dedent(inspect.getsource(original_reward))
    applied_path = opts.diagnostic_dir / "effective_reward_used.py"
    applied_path.write_text(applied_source, encoding="utf-8")

    def reward(self, per_user_latency, per_user_tasks, system_average_latency,
               portfolio_stats, portfolios, queues_before_mb, sys_cfg,
               previous_portfolios=None, slot_latencies=None):
        before_h = float(np.tanh(self.h_queue / 5.))
        output = chosen(self, per_user_latency, per_user_tasks, system_average_latency,
                        portfolio_stats, portfolios, queues_before_mb, sys_cfg,
                        previous_portfolios=previous_portfolios, slot_latencies=slot_latencies)
        loads = np.asarray(per_user_tasks, dtype=np.float64) * sys_cfg.task_compute_mb
        arrivals = np.asarray(portfolios, dtype=np.float64).T @ loads
        qbar = np.clip(np.asarray(queues_before_mb) / (sys_cfg.edge_compute_mb_s * sys_cfg.slot_s),
                       0., self.cfg.q_pressure_max) / self.cfg.q_pressure_max
        shares = arrivals / max(float(arrivals.sum()), 1e-8)
        components = self.last_components
        reward_slot = reward_counters.get(phase, 0)
        reward_counters[phase] = reward_slot + 1
        reward_rows.append({"phase": phase, "reward_slot": reward_slot,
                            "lcc_reward": float(output[0]), "lcc_phi": components["lcc_phi"],
                            "q_demand_weighted": float(np.sum(qbar * shares)),
                            "q_uniform": float(qbar.mean()) if arrivals.sum() > 0 else 0.,
                            "h_pressure_before": before_h,
                            "h_term_before_intervention": before_h * components["lcc_concentration_violation"],
                            "h_term_applied": 0. if opts.variant == VARIANT else before_h * components["lcc_concentration_violation"],
                            "h_queue_after": float(self.h_queue)})
        return output

    original_step = JointMAPPOEdgeEnv.joint_step
    def step(env, *args, **kwargs):
        feedback = env.feedback_buffer.get(env.feedback_delay)
        source_slot = int(feedback.source_slot)
        key = (phase, id(env))
        receipt = key not in last_source or last_source[key] != source_slot
        last_source[key] = source_slot
        slot = int(env.slot_index)
        result = original_step(env, *args, **kwargs)
        counts = np.asarray(result.server_assignment_counts, dtype=np.float64)
        share = counts / max(float(counts.sum()), 1.)
        jump = np.maximum(result.queue_load_after_mb - result.queue_load_before_mb, 0.)
        jump /= np.maximum(env.edge_capacities_mb_s * env.cfg.slot_s, 1e-8)
        execution.append({"phase": phase, "slot": slot, "source_slot": source_slot,
                          "summary_receipt": int(receipt), "task_count": int(counts.sum()),
                          "max_realized_share": float(share.max()),
                          "realized_hhi": float(np.sum(share ** 2)),
                          "max_queue_jump_service_slots": float(jump.max())})
        return result

    original_evaluate = train.evaluate_policy
    def evaluate(*args, **kwargs):
        nonlocal phase
        phase = role
        return original_evaluate(*args, **kwargs)

    MAPPORewarder._compute_lcc_reward = reward
    JointMAPPOEdgeEnv.joint_step = step
    train.evaluate_policy = evaluate
    path = source / "paper_doc_tools/paired_training_repair_worker.py"
    spec = importlib.util.spec_from_file_location("frozen_r1_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    sys.argv = [str(path), "--repair", "both", "--diagnostic-dir", str(opts.diagnostic_dir),
                "--expected-source", opts.expected_source, "--", *remaining]
    try:
        worker.main()
        evaluation = [r for r in execution if r["phase"] == role]
        rewards_eval = [r for r in reward_rows if r["phase"] == role]
        diagnostics = {"variant": opts.variant, "evaluation_split_role": role,
                       "source_sha256": opts.expected_source,
                       "intervention_worker_sha256": sha256(__file__),
                       "intervention_core_sha256": sha256(HERE / "joint_reward_ablation_core.py"),
                       "effective_reward_file": str(applied_path), "effective_reward_sha256": sha256(applied_path),
                       "reward_range": [-1, 0], "reward_scale_fitted": False,
                       "H_state_update_preserved": True,
                       **aggregate_execution(evaluation), **aggregate_reward(rewards_eval)}
        if phase != role or not evaluation or not rewards_eval:
            raise RuntimeError("Incomplete evaluation diagnostics")
        if not all(np.isfinite(r["lcc_reward"]) and -1. <= r["lcc_reward"] <= 0. for r in reward_rows):
            raise FloatingPointError("Invalid bounded reward")
        training = [r for r in reward_rows if r["phase"] == "train"]
        diagnostics["training_reward_diagnostics"] = aggregate_reward(training) if training else None
        (opts.diagnostic_dir / "intervention_manifest.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        write_csv(opts.diagnostic_dir / "execution_slots.csv", execution)
        write_csv(opts.diagnostic_dir / "reward_slots.csv", reward_rows)


if __name__ == "__main__":
    main()
