"""Diagnose whether the 35->4 zero-shot actor transfer preserved QLA's price-aware
decision semantics — to distinguish 'transfer failure' from 'algorithm underperforms
on real tasks'. This is a behavioral probe, NOT a performance result.
"""
import json
import sys
import time
from pathlib import Path

LAB = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013")
sys.path.insert(0, str(LAB))
from systemlab.common import local_windows_runtime, read, write_new, source_receipt  # noqa: E402
local_windows_runtime()
from systemlab.policy import Router, features  # noqa: E402

config = read(LAB / "protocol/frozen_v3/FORMAL_CONFIG.json")
bundle = LAB / "protocol/formal_actor_v3.json"
router = Router("qla", config, str(bundle))

m = len(config["edges"])
edge_ids = [e["id"] for e in config["edges"]]

def make_states(wait_s, service_s, age_s):
    now = time.time_ns()  # real epoch-ns scale, consistent with features()/portfolio()
    return {eid: dict(predicted_wait_s=w, service_estimate_s=s, source_epoch_ns=now - int(a * 1e9))
            for eid, w, s, a in zip(edge_ids, wait_s, service_s, age_s)}

tasks = [dict(payload_bytes=config["feature_scales"]["mean_payload_bytes"])]

def probe(wait_s, service_s, age_s, label):
    states = make_states(wait_s, service_s, age_s)
    obs, costs, ages = features(tasks, states, config, time.time_ns())
    weights, receipt = router.portfolio(tasks, states, time.time_ns(), 0)
    # per-edge price (from features: normalized prices)
    return dict(label=label, weights=[round(w, 4) for w in weights],
                costs=[round(c, 5) for c in costs], age=ages)

probes = []
# 1. uniform: all edges identical -> QLA should not collapse to a single edge.
probes.append(probe([.1, .1, .1, .1], [.03, .03, .03, .03], [.1, .1, .1, .1], "uniform_idle"))
# 2. edge0 heavily loaded (long wait) -> its weight should drop.
probes.append(probe([2.0, .1, .1, .1], [.03, .03, .03, .03], [.1, .1, .1, .1], "edge0_loaded"))
# 3. edge0 fast service -> weight may rise (cost lower).
probes.append(probe([.1, .1, .1, .1], [.01, .03, .03, .03], [.1, .1, .1, .1], "edge0_fast_service"))
# 4. three edges loaded, one idle -> idle should dominate.
probes.append(probe([.1, 2.0, 2.0, 2.0], [.03, .03, .03, .03], [.1, .1, .1, .1], "edge0_idle_others_loaded"))

# summary checks
def spread(weights):
    return max(weights) - min(weights)

results = []
for p in probes:
    w = p["weights"]
    results.append(dict(label=p["label"], weights=w, costs=p["costs"], spread=round(spread(w), 4),
                        argmax=edge_ids[w.index(max(w))], argmin=edge_ids[w.index(min(w))]))

# price-monotonicity check: edge0_loaded should have argmin at edge0
loaded = next(r for r in results if r["label"] == "edge0_loaded")
idle = next(r for r in results if r["label"] == "edge0_idle_others_loaded")
fast = next(r for r in results if r["label"] == "edge0_fast_service")
uniform = next(r for r in results if r["label"] == "uniform_idle")

checks = dict(
    sums_to_one=all(abs(sum(p["weights"]) - 1) < 1e-6 for p in probes),
    price_aware_loaded_drops=loaded["argmin"] == "edge0",
    idle_dominates=idle["argmax"] == "edge0",
    responsive_to_input=max(r["spread"] for r in results) > 0.05,
    not_degenerate_uniform=any(r["spread"] > 0.05 for r in results),
)
value = dict(schema="transfer-behavior-probe-v1", bundle_sha256=str(bundle),
             source_sha256=source_receipt(), probes=results, checks=checks,
             note="Behavioral probe of the frozen 35->4 zero-shot transfer: does the actor still "
                  "price/queue-sensitively allocate? A PASS here means the transfer retained QLA's "
                  "decision semantics; it does NOT prove the algorithm outperforms baselines on real "
                  "tasks (that is the E1 experiment).")
out = LAB / "verification/transfer_behavior_probe_v3.json"
write_new(out, value)
for r in results:
    print(f"{r['label']:24s} weights={r['weights']} spread={r['spread']} argmax={r['argmax']} argmin={r['argmin']}")
print("checks:", json.dumps(checks))
print("wrote", out)
