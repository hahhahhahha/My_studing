"""One independent CPU process. Do not run timing blocks concurrently."""
from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
import platform
import time

from common import BASE, read, save, sha, csv_write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    opts = p.parse_args()
    job = read(opts.job)
    import psutil
    import torch
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights

    versions = read(BASE / "protocol/ENVIRONMENT_SPEC.json")["versions"]
    import importlib.metadata
    for package, expected in versions.items():
        actual = platform.python_version() if package == "python" else importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Measurement environment changed: {package} {actual} != {expected}")
    threads = 1 if job["profile"] == "CPU-1T" else 4
    topology = read(BASE / "protocol/CPU_TOPOLOGY.json")
    affinity = topology["logical_cpu_per_physical_core"][:threads]
    if len(affinity) != threads:
        raise RuntimeError("Insufficient verified physical cores")
    psutil.Process().cpu_affinity(affinity)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261009)
    weights_path = BASE / "models/resnet18-f37072fd.pth"
    weights_hash = sha(weights_path)
    if not weights_hash.startswith("f37072fd"):
        raise RuntimeError("Wrong pretrained weights hash")
    model = resnet18(weights=None).eval().float()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    manifest = read(BASE / "protocol/IMAGE_SPLITS.json")
    entries = {r["image_id"]: r for r in manifest[job["role"]]}
    records, payloads = [], []
    for image_id in job["image_ids"]:
        entry = entries[image_id]
        image_path = BASE / "data/extracted" / image_id
        if sha(image_path) != entry["image_sha256"]:
            raise RuntimeError(f"JPEG changed: {image_id}")
        payloads.append((entry, image_path.read_bytes()))
    if not payloads:
        raise ValueError("No images")

    def one(payload, segmented=True):
        t0 = time.perf_counter_ns()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
        t1 = time.perf_counter_ns() if segmented else 0
        x = preprocess(image).unsqueeze(0)
        t2 = time.perf_counter_ns() if segmented else 0
        logits = model(x)
        t3 = time.perf_counter_ns() if segmented else 0
        response = {"class_index": int(logits.argmax(dim=1).item())}
        t4 = time.perf_counter_ns()
        if not segmented:
            return {"service_ms": (t4-t0)/1e6, "predicted_class": response["class_index"]}
        return {"decode_ms": (t1-t0)/1e6, "preprocess_ms": (t2-t1)/1e6,
                "infer_ms": (t3-t2)/1e6, "postprocess_ms": (t4-t3)/1e6,
                "service_ms": (t4-t0)/1e6, "predicted_class": response["class_index"]}

    before = {"cpu_percent": psutil.cpu_percent(interval=.25),
              "available_memory_bytes": psutil.virtual_memory().available,
              "wall_clock": datetime.now().isoformat()}
    with torch.inference_mode():
        for k in range(200):
            one(payloads[k % len(payloads)][1])
        for order, (entry, payload) in enumerate(payloads, 1):
            record = {"session_id": job["session"], "block_id": job["block"], "run_order": order,
                      "profile_id": job["profile"], "model_id": "ResNet18_IMAGENET1K_V1",
                      "weights_sha256": weights_hash, "image_id": entry["image_id"],
                      "image_sha256": entry["image_sha256"], "image_split": entry["image_split"],
                      "class_id": entry["class_id"], "width": entry["width"], "height": entry["height"],
                      "payload_bytes": len(payload), "cpu_threads": threads, "cpu_affinity": json.dumps(affinity),
                      "device": "cpu", "measurement_timestamp": datetime.now().isoformat(), "error_status": "ok"}
            try:
                record.update(one(payload))
            except Exception as error:
                record.update({"decode_ms": "", "preprocess_ms": "", "infer_ms": "", "postprocess_ms": "",
                               "service_ms": "", "predicted_class": "", "error_status": repr(error)})
            records.append(record)
    stem = f'{job["role"]}_s{job["session"]}_b{job["block"]}_{job["profile"]}'
    target = BASE / "measurements" / job["role"] / f"{stem}.csv"
    if target.exists():
        raise FileExistsError("Refuse to overwrite prior measurement block")
    csv_write(target, records)
    after = {"cpu_percent": psutil.cpu_percent(interval=.25), "wall_clock": datetime.now().isoformat()}
    save(target.with_suffix(".json"), {"job": job, "job_sha256": sha(opts.job),
         "weights_sha256": weights_hash, "versions": versions, "warmup_requests": 200,
         "affinity": affinity, "sensor_before": before, "sensor_after": after,
         "temperature": "unavailable unless provided by supported external sensor",
         "frequency": "not continuously sampled in this timing block",
         "requests": len(records), "errors": sum(r["error_status"] != "ok" for r in records),
         "timing_boundary": "decode+RGB+preprocess+FP32 inference+argmax+dict; no disk/network/queue/controller"})
    if any(r["error_status"] != "ok" for r in records):
        raise RuntimeError("Failed requests retained; block is not successful")
    print(f"Measured {len(records)} requests: {stem}", flush=True)


if __name__ == "__main__":
    main()
