"""Interference calibration: solo vs concurrent vs affinity-pinned concurrent.

Same deterministic image sequence, warmup and thread count for every node;
the only differences are (a) whether the other three nodes run, and (b) whether
each node is pinned to its own non-overlapping physical cores. This separates
shared-CPU contention from load/arrival/image-pool effects.

Physical topology (queried 2026-10-14): 8 physical cores, SMT siblings are
consecutive pairs (physical core k == logical cores 2k, 2k+1). Affinity maps:
  edge0(1 thr) -> core0 (mask 0x3);   edge1(1 thr) -> core1 (0xc);
  edge2(2 thr) -> cores 2,3 (0xf0);   edge3(2 thr) -> cores 4,5 (0xf00).
"""
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

REPO = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO")
LAB = REPO / "qla_system_prototype_20261013"
WEIGHT = REPO / "dac_visual_calibration_20261009/models/resnet18-f37072fd.pth"
WEIGHT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WARMUP = 20
K_IMAGES = 200
NODES = [("edge0", 1), ("edge1", 1), ("edge2", 2), ("edge3", 2)]
AFFINITY = {"edge0": 0x3, "edge1": 0xC, "edge2": 0xF0, "edge3": 0xF00}
REPEATS = 3
SEED = 20261014


def _worker(node_id, threads, image_ids, image_root, out_path, start_event, affinity_mask):
    import sys
    sys.path.insert(0, str(LAB))
    try:
        if affinity_mask:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.c_int
            if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), affinity_mask):
                raise RuntimeError("SetProcessAffinityMask failed")
        from systemlab.common import local_windows_runtime
        local_windows_runtime()
        from systemlab.edge import VisionEngine
        engine = VisionEngine(str(WEIGHT), WEIGHT_SHA, threads, WARMUP)
        root = Path(image_root)
        start_event.wait()
        wall0, cpu0 = time.perf_counter(), time.process_time()
        rows = []
        for i, image_id in enumerate(image_ids):
            jpeg = (root / image_id).read_bytes()
            result = engine(jpeg)
            rows.append(dict(index=i, image_id=image_id, **result))
        wall_s = time.perf_counter() - wall0
        cpu_s = time.process_time() - cpu0
        value = dict(node_id=node_id, torch_threads=threads, warmup=WARMUP, n=len(rows),
                     affinity_mask=affinity_mask, rows=rows, wall_s=wall_s, cpu_s=cpu_s,
                     cpu_wall_ratio=(cpu_s / wall_s if wall_s else None))
        Path(out_path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        Path(out_path).with_suffix(".error.json").write_text(
            json.dumps(dict(error=repr(error))), encoding="utf-8")


def main():
    manifest = json.load(open(LAB / "data/calibration_images.json"))
    image_ids = [x["image_id"] for x in manifest["images"][:K_IMAGES]]
    image_root = manifest["image_root"]
    assert len(image_ids) == K_IMAGES

    ctx = mp.get_context("spawn")
    out_root = LAB / "interference_calibration"
    out_root.mkdir(exist_ok=True)

    units = [("solo", node_id, threads, None) for (node_id, threads) in NODES for _ in range(REPEATS)]
    units += [("concurrent", None, None, None) for _ in range(REPEATS)]
    units += [("concurrent_affinity", None, None, True) for _ in range(REPEATS)]
    rng = random.Random(SEED)
    rng.shuffle(units)

    plan = []
    for idx, unit in enumerate(units):
        run_dir = out_root / f"run_{idx:02d}"
        run_dir.mkdir(exist_ok=True)
        kind, node_id, threads, affinity = unit
        if kind == "solo":
            start = ctx.Event()
            p = ctx.Process(target=_worker, args=(node_id, threads, image_ids, image_root,
                                                  run_dir / f"{node_id}.json", start, None))
            p.start(); start.set(); p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker exited {p.exitcode}")
            plan.append(dict(run=idx, kind=kind, node_id=node_id, threads=threads))
        else:
            start = ctx.Event()
            procs = []
            for nid, thr in NODES:
                mask = AFFINITY[nid] if affinity else None
                p = ctx.Process(target=_worker, args=(nid, thr, image_ids, image_root,
                                                      run_dir / f"{nid}.json", start, mask))
                p.start(); procs.append((nid, p))
            start.set()
            for _, p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError(f"worker exited {p.exitcode}")
            plan.append(dict(run=idx, kind=kind, affinity=bool(affinity)))

    (out_root / "PLAN.json").write_text(json.dumps(dict(
        seed=SEED, repeats=REPEATS, k_images=K_IMAGES, warmup=WARMUP, weight_sha256=WEIGHT_SHA,
        physical_cores=8, logical_cores=16, smt_siblings="consecutive (2k,2k+1)",
        affinity_masks=AFFINITY, nodes=[dict(id=n, threads=t) for n, t in NODES],
        plan=plan, image_ids=image_ids), indent=2, ensure_ascii=False), encoding="utf-8")
    print("done; runs:", len(units))


if __name__ == "__main__":
    main()
