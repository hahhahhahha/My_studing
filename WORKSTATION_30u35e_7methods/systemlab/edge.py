from __future__ import annotations

import hashlib
import io
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .common import SERVER_RESPONSE_TIMEOUT, Journal, encoded, local_windows_runtime, runtime, sha, source_receipt, write_new


class VisionEngine:
    def __init__(self, weight_path, expected_sha, threads=1, warmup=20):
        local_windows_runtime()
        import torch
        import torchvision
        from PIL import Image
        from torchvision.models import ResNet18_Weights, resnet18
        if sha(weight_path) != expected_sha:
            raise ValueError('Registered vision weight SHA mismatch')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261013)
        witness = (torch.arange(16, dtype=torch.float32).reshape(4, 4) @
                   torch.arange(16, dtype=torch.float32).reshape(4, 4).T).sum().item()
        if witness != 3680:
            raise RuntimeError('CPU kernel witness failed')
        self.torch, self.Image = torch, Image
        self.model = resnet18(weights=None).eval().cpu()
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        with torch.inference_mode():
            for _ in range(warmup):
                self.model(torch.zeros(1, 3, 224, 224))
        self.receipt = dict(engine='resnet18_imagenet1k_v1_real_cpu', weight_sha256=expected_sha,
                            torch=torch.__version__, torchvision=torchvision.__version__,
                            torch_path=torch.__file__, torchvision_path=torchvision.__file__,
                            threads=threads, interop_threads=1, batch_size=1, precision='float32',
                            warmup=warmup, seeded_cpu_kernel=witness)

    def __call__(self, jpeg):
        begin = time.perf_counter()
        with self.Image.open(io.BytesIO(jpeg)) as image:
            x = self.transform(image.convert('RGB')).unsqueeze(0)
        ready = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(x)
            if not self.torch.isfinite(output).all():
                raise RuntimeError('Nonfinite vision model output')
            top1 = int(output.argmax(dim=1).item())
        end = time.perf_counter()
        return dict(top1=top1, preprocess_s=ready - begin, compute_s=end - ready)


@dataclass
class Task:
    task_id: str
    user_id: int
    jpeg: bytes
    body_read_s: float
    received_epoch_ns: int
    admitted_mono: float = 0.
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)


class EdgeQueue:
    """One FIFO worker; queue capacity counts waiting requests only."""
    def __init__(self, server_id, capacity, initial_service_s, engine, journal):
        if capacity <= 0 or initial_service_s <= 0:
            raise ValueError('Positive finite-buffer and calibrated service values required')
        self.id, self.capacity, self.engine, self.journal = server_id, capacity, engine, journal
        self.queue = queue.Queue(capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.seen = set()
        self.received = self.admitted = self.completed = self.failed = self.rejected = 0
        self.service_ema_s = initial_service_s
        self.active = None
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def submit(self, task):
        with self.lock:
            if task.task_id in self.seen:
                return 'duplicate'
            self.seen.add(task.task_id)
            self.received += 1
            task.admitted_mono = time.perf_counter()
            try:
                self.queue.put_nowait(task)
            except queue.Full:
                self.rejected += 1
                self.journal.put('rejected', task_id=task.task_id, user_id=task.user_id,
                                 server_id=self.id, payload_bytes=len(task.jpeg))
                return 'queue_rejected'
            self.admitted += 1
            self.journal.put('admitted', task_id=task.task_id, user_id=task.user_id,
                             server_id=self.id, waiting_count=self.queue.qsize(),
                             payload_bytes=len(task.jpeg), received_epoch_ns=task.received_epoch_ns,
                             admitted_epoch_ns=time.time_ns())
        return 'admitted'

    def _work(self):
        while not self.stop.is_set() or not self.queue.empty():
            # Holding this lock across dequeue and active assignment makes the
            # mass-balance snapshot coherent with submit and completion.
            with self.lock:
                try:
                    task = self.queue.get_nowait()
                except queue.Empty:
                    task = None
                if task is not None:
                    start = time.perf_counter()
                    self.active = (task.task_id, start)
            if task is None:
                self.stop.wait(.001)
                continue
            start_epoch = time.time_ns()
            try:
                result = self.engine(task.jpeg)
                service = time.perf_counter() - start
                task.result = dict(status='completed', server_id=self.id, task_id=task.task_id,
                    user_id=task.user_id, queue_s=start - task.admitted_mono,
                    server_service_s=service, body_read_s=task.body_read_s,
                    compute_start_epoch_ns=start_epoch, compute_end_epoch_ns=time.time_ns(), **result)
            except Exception as error:
                task.result = dict(status='inference_error', server_id=self.id, task_id=task.task_id,
                                   error=repr(error))
            # Completion counters and removal of in-service mass are one
            # transition. A telemetry snapshot must never see both copies.
            with self.lock:
                if task.result['status'] == 'completed':
                    self.completed += 1
                    self.service_ema_s = .9 * self.service_ema_s + .1 * service
                else:
                    self.failed += 1
                self.active = None
                self.journal.put('execution_result', **task.result)
                task.done.set()
                self.queue.task_done()

    def snapshot(self):
        with self.lock:
            waiting, active = self.queue.qsize(), int(self.active is not None)
            if self.admitted != self.completed + self.failed + waiting + active:
                raise RuntimeError('Queue mass accounting failed')
            residual = max(0., self.service_ema_s - (time.perf_counter() - self.active[1])) if active else 0.
            return dict(server_id=self.id, source_epoch_ns=time.time_ns(), waiting_count=waiting,
                        capacity=self.capacity, busy=bool(active), service_estimate_s=self.service_ema_s,
                        predicted_wait_s=waiting * self.service_ema_s + residual,
                        received=self.received, admitted=self.admitted, completed=self.completed,
                        inference_errors=self.failed, rejected=self.rejected)


def serve_edge(config, server_id, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    node = next(x for x in config['edges'] if x['id'] == server_id)
    engine = VisionEngine(config['weight_path'], config['weight_sha256'], node['torch_threads'], config['warmup'])
    write_new(output / 'RUNTIME.json', dict(**runtime(), model=engine.receipt, source_sha256=source_receipt()))
    journal = Journal(output / 'edge.jsonl')
    core = EdgeQueue(server_id, config['waiting_capacity'], node['initial_service_s'], engine, journal)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, value):
            body = encoded(value)
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                journal.put('response_disconnected')

        def do_GET(self):
            if self.path == '/health':
                self.send(200, dict(ready=True, pid=runtime()['pid'], server_id=server_id, model=engine.receipt))
            elif self.path == '/snapshot' and self.headers.get('X-Control-Token') == config['control_token']:
                self.send(200, core.snapshot())
            else:
                self.send(403, dict(error='Controller-only telemetry'))

        def do_POST(self):
            if self.path == '/shutdown' and self.headers.get('X-Control-Token') == config['control_token']:
                self.send(200, dict(stopping=True))
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if self.path != '/infer':
                self.send(404, dict(error='Unknown route'))
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= config['max_jpeg_bytes']:
                    self.send(413, dict(status='payload_rejected'))
                    return
                task_id = self.headers['X-Task-Id']
                if not task_id or len(task_id) > 128:
                    raise ValueError('Invalid task id')
                user_id = int(self.headers['X-User-Id'])
                begin, received = time.perf_counter(), time.time_ns()
                jpeg = self.rfile.read(size)
                if len(jpeg) != size or hashlib.sha256(jpeg).hexdigest() != self.headers['X-JPEG-SHA256']:
                    raise ValueError('Truncated or changed JPEG body')
                task = Task(task_id, user_id, jpeg, time.perf_counter() - begin, received)
                status = core.submit(task)
                if status != 'admitted':
                    self.send(409 if status == 'duplicate' else 429, dict(status=status, task_id=task_id, server_id=server_id))
                elif not task.done.wait(config['server_response_timeout_s']):
                    # Execution continues; the final server event remains authoritative.
                    self.send(504, dict(status=SERVER_RESPONSE_TIMEOUT, task_id=task_id, server_id=server_id))
                else:
                    self.send(200 if task.result['status'] == 'completed' else 500, task.result)
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, dict(status='invalid_request', error=str(error)))

    server = ThreadingHTTPServer((config.get('bind_host', '127.0.0.1'), node['port']), Handler)
    server.daemon_threads = True
    journal.put('ready', **core.snapshot())
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        core.stop.set()
        core.worker.join(config['server_response_timeout_s'])
        journal.put('stopped', snapshot=core.snapshot())
