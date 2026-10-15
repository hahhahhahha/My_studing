"""Inference-only worker. Frozen checkpoint files and model weights are audited."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heldout_external_windows import ROOT, save, sha, schedule_arrays


def witness(out):
    expected = json.loads((out / "ENVIRONMENT_SPEC.json").read_text(encoding="utf-8"))
    observed = {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                "torch": importlib.metadata.version("torch")}
    errors = [f"{key}: expected {expected[key]}, found {observed[key]}"
              for key in ("python", "numpy", "torch") if expected[key] != observed[key]]
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available; no automatic CPU fallback")
        else:
            torch.set_num_threads(1)
            torch.manual_seed(123)
            a = torch.randn((64, 64), device="cuda")
            b = a @ a.T
            torch.cuda.synchronize()
            if not torch.isfinite(b).all():
                errors.append("CUDA witness is nonfinite")
            observed.update(device=torch.cuda.get_device_name(), witness_sum=float(b.sum().item()))
    except Exception as exc:
        errors.append(f"PyTorch/CUDA witness failed: {type(exc).__name__}: {exc}")
    save(out / "CURRENT_ENVIRONMENT_METADATA.json", {**observed, "errors": errors, "ready": not errors})
    if errors:
        raise RuntimeError("; ".join(errors))


def tensor_digest(policy):
    digest = hashlib.sha256()
    model = policy.actor if hasattr(policy, "sys_cfg") else policy
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_cell(out, cell):
    import numpy as np
    design = json.loads((out / "DESIGN.json").read_text(encoding="utf-8"))
    if cell not in design["cells"]:
        raise ValueError("Cell is not in the frozen test design")
    source = out / "execution_source"
    for relative, digest in design["execution_source_sha256"].items():
        if sha(source / relative) != digest:
            raise RuntimeError(f"Frozen source changed: {relative}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "paper_doc_tools"))
    import torch
    from repro_risk_averse_edge.data import load_trace_set
    from mappo_gp_sf_dsr.config import MAPPOSystemConfig, ServerScoreConfig
    from mappo_gp_sf_dsr.canonical_protocol import trace_splits
    from mappo_gp_sf_dsr.metrics import finalize_evaluation, phase_diagnostics
    from mappo_gp_sf_dsr.models import MAPPOPolicy
    from mappo_gp_sf_dsr.rewards import MAPPORewarder
    from mappo_gp_sf_dsr.paper_policy_baselines import SharedGaussianPPO
    from mappo_gp_sf_dsr.delayed_information_baselines import JSEQPolicy
    from mappo_gp_sf_dsr.train import augment_lcc_price_actor_observations, normalize_portfolios
    from run_cross_method_sync_diagnostic import qla_config
    from run_real_trace_external_formal import ExternalTraceEnv, ExternalTraceSchedule, _stable_hash

    torch.set_num_threads(1)
    torch.manual_seed(cell["seed"])
    cfg = MAPPOSystemConfig(n_users=30, n_edges=35, slot_s=1., edge_compute_mb_s=63.,
                           queue_upper_mb=630., max_requests_per_user=7,
                           task_package_mb=270. / 1024., task_compute_mb=15.,
                           max_supported_users=50, include_feedback_age=cell["method"] == "QLA-MAPPO",
                           feedback_age_norm=8., extra_actor_obs_dim=36 if cell["method"] == "QLA-MAPPO" else 0)
    traces = load_trace_set(cell["dataset"], 30, 35, 1., cell["seed"],
                            trace_root=ROOT / "Traces_TimeXY_10sec_txt")
    split = trace_splits(traces.trajectories.shape[1], 10)["test"]
    indices = split.start + np.arange(1000) % split.length
    window = cell["window"]
    with np.load(out / "data" / f"{window}_demand.npz", allow_pickle=False) as data:
        lam = data["lambda_per_user"]
    with np.load(out / "data" / f"{window}_capacity.npz", allow_pickle=False) as data:
        capacity = data["capacity_scale"]
    arrays = schedule_arrays(lam, capacity, cell["seed"], indices)
    meta = {"protocol": "unseen_external_windows_v1", "window": window,
            "dataset": cell["dataset"], "seed": cell["seed"], "role": "test",
            "demand_source_sha256": sha(out / "data" / f"{window}_demand.npz"),
            "capacity_source_sha256": sha(out / "data" / f"{window}_capacity.npz")}
    digest = _stable_hash(arrays, meta)
    schedule_path = out / "schedules" / f'{window}_{cell["dataset"]}_seed{cell["seed"]}.npz'
    schedule_path.parent.mkdir(exist_ok=True)
    if schedule_path.exists():
        with np.load(schedule_path, allow_pickle=False) as old:
            if any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError("Schedule changed between paired methods/delays")
    else:
        np.savez_compressed(schedule_path, **arrays)
        save(schedule_path.with_suffix(".json"), {**meta, "content_sha256": digest})
    schedule = ExternalTraceSchedule(schedule_id=schedule_path.stem, role="test", **arrays,
                                     content_sha256=digest, path=schedule_path, protocol_metadata=meta)
    env = ExternalTraceEnv(cfg, traces, request_distribution="poisson_bursty", seed=10000 + cell["seed"],
                           queue_constraint="resample", score_config=ServerScoreConfig(),
                           feedback_delay=cell["nominal_delay"], broadcast_period=2,
                           poisson_lambda=3.5, burst_prob=.1, burst_multiplier=2.,
                           edge_capacities_mb_s=np.full(35, 63., dtype=np.float32), background_load=0.)
    env.install_canonical_schedule(schedule)
    env.reset()
    method, policy, rewarder, previous, state = cell["method"], None, None, None, {}
    entry = None
    if method != "JSEQ-A":
        entry = next(e for e in design["checkpoints"] if e["method"] == method
                     and e["dataset"] == cell["dataset"] and e["seed"] == cell["seed"])
        if sha(entry["checkpoint"]) != entry["checkpoint_sha256"] or sha(entry["config"]) != entry["config_sha256"]:
            raise RuntimeError("Checkpoint/config changed before test")
        raw = json.loads(Path(entry["config"]).read_text(encoding="utf-8"))
        cli = raw.get("cli", raw)
        if method == "QLA-MAPPO":
            args = argparse.Namespace(**{**cli, "device": "cuda"})
            pcfg = qla_config(args)
            policy = MAPPOPolicy(cfg, pcfg, seed=cell["seed"], device="cuda")
            policy.load_checkpoint(entry["checkpoint"], actor_only=True)
            policy.actor.eval()
            rewarder = MAPPORewarder(pcfg, method="LCC-MAPPO")
        else:
            weights = torch.load(entry["checkpoint"], map_location="cuda", weights_only=True)
            obs_dim = int(weights["actor.0.weight"].shape[1])
            if env.actor_observations().shape[1] != obs_dim:
                raise RuntimeError("PPO checkpoint observation shape mismatch")
            policy = SharedGaussianPPO(obs_dim, 35, log_std_min=cli.get("log_std_min", -1.5)).to("cuda")
            policy.load_state_dict(weights)
            policy.eval()
        model = policy.actor if method == "QLA-MAPPO" else policy
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before_weights = tensor_digest(policy)
    else:
        rule = JSEQPolicy(env, design["jseq_temperature"])
    results, slots = [], []
    for slot in range(1000):
        feedback = env.feedback_buffer.get(env.feedback_delay)
        age = int(env.slot_index) - int(feedback.source_slot)
        if method == "QLA-MAPPO":
            obs = augment_lcc_price_actor_observations(env.actor_observations(), env, args, rewarder,
                                                       previous_portfolios=previous, state=state)
            portfolios = normalize_portfolios(policy.act(obs, deterministic=True)[0])
        elif method == "PPO-MV-A":
            portfolios = normalize_portfolios(policy.act(env.actor_observations(), deterministic=True)[3])
        else:
            portfolios = normalize_portfolios(rule.portfolios())
        result = env.joint_step(portfolios, stats=env.portfolio_stats(portfolios), decision_latency_s=0.)
        if method == "QLA-MAPPO":
            rewarder.compute(result.per_user_mean_latency, result.per_user_tasks, result.system_average_latency,
                             result.portfolio_stats, portfolios=result.portfolios,
                             queues_before_mb=result.queue_load_before_mb, sys_cfg=cfg,
                             previous_portfolios=previous, slot_latencies=np.asarray(result.latencies, dtype=np.float32))
        elif method == "JSEQ-A":
            rule.observe(result)
        count = np.asarray(result.server_assignment_counts, dtype=np.float64)
        shares = count / max(float(count.sum()), 1.)
        jump = np.maximum(result.queue_load_after_mb - result.queue_load_before_mb, 0.) / 63.
        slots.append({"slot": slot, "source_age": age, "max_realized_share": float(shares.max()),
                      "max_queue_jump_service_slots": float(jump.max()),
                      "mean_queue_mb": float(np.mean(result.queue_load_after_mb)),
                      "capacity_scale_mean": float(capacity[slot].mean())})
        results.append(result)
        previous = np.asarray(result.portfolios).copy()
    summary, drain = finalize_evaluation(env, results, cfg)
    summary.update(phase_diagnostics(results, cfg))
    queues = np.asarray([r.queue_load_after_mb for r in results], dtype=np.float64)
    summary["queue_oscillation"] = float(np.std(queues / 63., axis=0, ddof=0).mean())
    summary.update({"max_realized_share_p99": float(np.quantile([s["max_realized_share"] for s in slots], .99)),
                    "mean_queue_jump_service_slots": float(np.mean([s["max_queue_jump_service_slots"] for s in slots])),
                    "mass_conservation_error": max(float(r.mass_conservation_error) for r in results),
                    "train_steps": 0, "test_steps": 1000, "checkpoint_training_nominal_delay": 4,
                    "drain_finished": bool(drain.finished) if drain else None})
    if entry and (tensor_digest(policy) != before_weights or sha(entry["checkpoint"]) != entry["checkpoint_sha256"]):
        raise RuntimeError("Frozen model weights changed during test")
    key = f'{window}/{cell["dataset"]}/seed_{cell["seed"]}/delay_{cell["nominal_delay"]}/{method}'
    destination = out / "cells" / key
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "slots.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slots[0]))
        writer.writeheader()
        writer.writerows(slots)
    save(destination / "summary.json", {**cell, **summary, "schedule_sha256": digest,
                                         "checkpoint_sha256": entry["checkpoint_sha256"] if entry else "no-checkpoint",
                                         "frozen_actor_verified": True, "policy_update_calls": 0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--cell-json")
    opts = p.parse_args()
    if opts.preflight:
        witness(opts.output)
    else:
        if not opts.cell_json:
            raise ValueError("One explicit preregistered cell is required")
        run_cell(opts.output, json.loads(opts.cell_json))


if __name__ == "__main__":
    main()
