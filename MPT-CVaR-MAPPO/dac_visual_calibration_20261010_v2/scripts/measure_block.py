from __future__ import annotations
import argparse
from datetime import datetime
from io import BytesIO
import importlib.metadata
import json
import platform
import time
from v2_common import BASE, OLD, BOUNDARY, read, save, sha, csv_write, stem

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    opts = p.parse_args()
    job = read(opts.job)
    import psutil
    import torch
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights
    versions = read(BASE / 'protocol/ENVIRONMENT_SPEC.json')['versions']
    for package, expected in versions.items():
        actual = platform.python_version() if package == 'python' else importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f'Environment changed: {package} {actual} != {expected}')
    threads = 1 if job['profile'] == 'CPU-1T' else 4
    affinity = read(BASE / 'protocol/CPU_TOPOLOGY.json')['logical_cpu_per_physical_core'][:threads]
    psutil.Process().cpu_affinity(affinity)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261009)
    weights = OLD / 'models/resnet18-f37072fd.pth'
    weights_sha = sha(weights)
    if weights_sha != read(BASE / 'protocol/PLAN.json')['weights_sha256']:
        raise RuntimeError('Weights changed')
    model = resnet18(weights=None).eval().float()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    entries = {e['image_id']: e for e in read(BASE / 'protocol/SPLITS.json')[job['role']]}
    payloads = []
    for image_id in job['image_ids']:
        entry = entries[image_id]
        path = OLD / 'data/extracted' / image_id
        if sha(path) != entry['image_sha256']:
            raise RuntimeError(f'Image changed: {image_id}')
        payloads.append((entry, path.read_bytes()))
    def one(payload):
        t0 = time.perf_counter_ns()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert('RGB')
        t1 = time.perf_counter_ns()
        x = preprocess(image).unsqueeze(0)
        t2 = time.perf_counter_ns()
        logits = model(x)
        t3 = time.perf_counter_ns()
        response = {'class_index': int(logits.argmax(dim=1).item())}
        t4 = time.perf_counter_ns()
        return dict(decode_ms=(t1-t0)/1e6, preprocess_ms=(t2-t1)/1e6,
                    infer_ms=(t3-t2)/1e6, postprocess_ms=(t4-t3)/1e6,
                    service_ms=(t4-t0)/1e6, predicted_class=response['class_index'])
    before = dict(cpu_percent=psutil.cpu_percent(interval=.25),
                  available_memory_bytes=psutil.virtual_memory().available,
                  wall_clock=datetime.now().astimezone().isoformat())
    rows = []
    with torch.inference_mode():
        for k in range(200):
            one(payloads[k % len(payloads)][1])
        for order, (entry, payload) in enumerate(payloads, 1):
            r = dict(session_id=job['session'], block_id=job['block'], run_order=order,
                     profile_id=job['profile'], model_id='ResNet18_IMAGENET1K_V1',
                     weights_sha256=weights_sha, image_id=entry['image_id'],
                     image_sha256=entry['image_sha256'], image_split=job['role'],
                     class_id=entry['class_id'], width=entry['width'], height=entry['height'],
                     payload_bytes=len(payload), cpu_threads=threads, cpu_affinity=json.dumps(affinity),
                     device='cpu', measurement_timestamp=datetime.now().astimezone().isoformat(),
                     error_status='ok')
            try:
                r.update(one(payload))
            except Exception as error:
                r.update({key: '' for key in ['decode_ms','preprocess_ms','infer_ms','postprocess_ms','service_ms','predicted_class']})
                r['error_status'] = repr(error)
            rows.append(r)
    target = BASE / 'measurements' / job['role'] / (stem(job) + '.csv')
    csv_write(target, rows, exclusive=True)
    save(target.with_suffix('.json'), dict(job=job, job_sha256=sha(opts.job), versions=versions,
         weights_sha256=weights_sha, affinity=affinity, warmup_requests=200,
         requests=len(rows), errors=sum(r['error_status'] != 'ok' for r in rows),
         sensor_before=before, sensor_after=dict(cpu_percent=psutil.cpu_percent(interval=.25),
         wall_clock=datetime.now().astimezone().isoformat()), timing_boundary=BOUNDARY,
         temperature='not available', frequency='not continuously sampled'), exclusive=True)
    if any(r['error_status'] != 'ok' for r in rows):
        raise RuntimeError('Failed requests preserved')
    print('MEASURED', stem(job), len(rows), flush=True)

if __name__ == '__main__':
    main()
