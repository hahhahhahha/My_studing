"""Pure reward intervention and diagnostics; no torch/GPU import."""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np

VARIANT = "NoJointDemandReward"


def make_no_joint_reward(original):
    """Change only q aggregation and H*C in the reward, not its state update."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)

    class Intervention(ast.NodeTransformer):
        q_count = 0
        h_count = 0

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "q_term":
                expected = ast.parse("q_term = float(np.sum(queue_pressure_norm * arrival_share)) if arrival_share.size else 0.0").body[0]
                if ast.dump(node, include_attributes=False) != ast.dump(expected, include_attributes=False):
                    raise RuntimeError("Frozen q_term anchor changed")
                self.q_count += 1
                replacement = ast.parse("q_term = float(np.mean(queue_pressure_norm)) if float(np.sum(arrivals)) > 0.0 else 0.0").body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                if node.left.id == "h_pressure" and node.right.id == "h_violation":
                    self.h_count += 1
                    return ast.copy_location(ast.Constant(value=0.0), node)
            return self.generic_visit(node)

    transform = Intervention()
    tree = transform.visit(tree)
    if (transform.q_count, transform.h_count) != (1, 1):
        raise RuntimeError("Reward intervention anchors were not unique")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<NoJointDemandReward>", "exec"), namespace)
    return namespace[original.__name__], ast.unparse(tree) + "\n"


def aggregate_execution(rows):
    if not rows:
        raise ValueError("No execution diagnostics")
    share = np.asarray([r["max_realized_share"] for r in rows], dtype=np.float64)
    jump = np.asarray([r["max_queue_jump_service_slots"] for r in rows], dtype=np.float64)
    spikes = jump > 1.0
    receipts = [i for i, r in enumerate(rows) if r["summary_receipt"]]
    windows = [float(np.any(spikes[i:min(i + 3, len(rows))])) for i in receipts]
    full_windows = [float(np.any(spikes[i:i + 3])) for i in receipts if i > 0 and i + 3 <= len(rows)]
    return {
        "max_realized_share_p99": float(np.quantile(share, .99)),
        "mean_realized_hhi": float(np.mean([r["realized_hhi"] for r in rows])),
        "mean_queue_jump_service_slots": float(jump.mean()),
        "p95_queue_jump_service_slots": float(np.quantile(jump, .95)),
        "p99_queue_jump_service_slots": float(np.quantile(jump, .99)),
        "unconditional_queue_spike_rate": float(spikes.mean()),
        "broadcast_window_queue_spike_rate": float(np.mean(windows)),
        "full_three_slot_update_spike_rate": float(np.mean(full_windows)) if full_windows else None,
        "full_three_slot_update_window_count": len(full_windows),
        "receipt_window_protocol": "F-compatible: initial available summary included; terminal window truncated to remaining slots",
        "summary_receipt_count": len(receipts),
        "evaluation_slots": len(rows),
    }


def aggregate_reward(rows):
    if not rows:
        raise ValueError("No reward diagnostics")
    rewards = np.asarray([r["lcc_reward"] for r in rows], dtype=np.float64)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_saturation_fraction_abs_ge_095": float(np.mean(np.abs(rewards) >= .95)),
        "h_active_slot_fraction": float(np.mean([r["h_pressure_before"] > 0 for r in rows])),
        "h_term_abs_mean": float(np.mean([abs(r["h_term_before_intervention"]) for r in rows])),
        "q_weighted_mean": float(np.mean([r["q_demand_weighted"] for r in rows])),
        "q_uniform_mean": float(np.mean([r["q_uniform"] for r in rows])),
        "reward_slots": len(rows),
    }
