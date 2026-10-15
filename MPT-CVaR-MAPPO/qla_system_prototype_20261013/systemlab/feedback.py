from __future__ import annotations

import hashlib
import heapq
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .common import Journal, encoded, request_json, runtime, source_receipt, write_new


class ByteBudget:
    def __init__(self, rate, burst, now=0.):
        if rate <= 0 or burst <= 0:
            raise ValueError('Positive byte rate and burst required')
        self.rate, self.burst, self.tokens, self.last = rate, burst, float(burst), now
        self.used = 0

    def spend(self, size, now):
        if now < self.last or size <= 0:
            raise ValueError('Monotonic budget clock and positive messages required')
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if size > self.tokens:
            return False
        self.tokens -= size
        self.used += size
        return True


def choose_updates(mode, snapshots, previous, now_ns, rng, cfg):
    ids = sorted(snapshots)
    if not ids:
        return []
    if any(j not in previous for j in ids) or mode == 'periodic':
        return ids
    if mode == 'random':
        return [rng.choice(ids)]
    priorities = {}
    for j in ids:
        current, old = snapshots[j], previous[j]
        age = max(0., (now_ns - old['source_epoch_ns']) / 1e9)
        change = abs(current['waiting_count'] - old['waiting_count']) / current['capacity']
        dt = max(1e-6, (current['source_epoch_ns'] - old['source_epoch_ns']) / 1e9)
        observed_arrival_rate = max(0., current['received'] - old['received']) / dt
        if mode == 'age':
            priorities[j] = age / cfg['age_reference_s']
        elif mode == 'queue_trigger':
            priorities[j] = change
        elif mode == 'coordinated':
            priorities[j] = (cfg['alpha'] * change + cfg['beta'] * age / cfg['age_reference_s'] +
                             cfg['gamma'] * observed_arrival_rate * current['service_estimate_s'])
        else:
            raise ValueError('Unknown feedback mode')
    winner = max(ids, key=lambda j: (priorities[j], j))
    return [winner] if priorities[winner] >= cfg['trigger_threshold'] else []


class FeedbackCache:
    """A user sees only delivered messages, ordered per-server by source time."""
    def __init__(self, session, users, seed=1, loss=0., journal=None):
        self.session, self.users, self.seed, self.loss, self.journal = session, users, seed, loss, journal
        self.lock = threading.Lock()
        self.values = {i: {} for i in range(users)}

    def accept(self, packet, received_epoch_ns):
        if packet.get('session') != self.session or packet.get('user_id') not in self.values:
            return False
        user = packet['user_id']
        # Drop after real UDP reception: bytes were actually transmitted.
        key = f"{self.seed}:{packet['tick']}:{user}".encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        dropped = draw < self.loss
        if self.journal:
            self.journal.put('feedback_received', user_id=user, version=packet['version'],
                tick=packet['tick'], payload_bytes=len(encoded(packet)), injected_drop=dropped,
                received_epoch_ns=received_epoch_ns, send_epoch_ns=packet['send_epoch_ns'])
        if dropped:
            return False
        with self.lock:
            for item in packet['servers']:
                old = self.values[user].get(item['server_id'])
                if old is None or item['source_epoch_ns'] > old['source_epoch_ns']:
                    self.values[user][item['server_id']] = dict(item, version=packet['version'],
                        send_epoch_ns=packet['send_epoch_ns'], received_epoch_ns=received_epoch_ns)
        return True

    def observation(self, user):
        with self.lock:
            return {key: dict(value) for key, value in self.values[user].items()}


def serve_controller(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = config['feedback']
    session = hashlib.sha256(encoded(dict(config=config, start_ns=time.time_ns()))).hexdigest()
    write_new(output / 'RUNTIME.json', dict(**runtime(), session=session, source_sha256=source_receipt()))
    journal = Journal(output / 'feedback_send.jsonl')
    stop = threading.Event()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    begin = time.perf_counter()
    budget = ByteBudget(cfg['budget_bytes_per_s'], cfg['burst_bytes'], begin)
    previous, pending = {}, []
    counters = dict(messages=0, application_bytes=0, budget_deferrals=0, polling_response_bytes=0)
    lock = threading.Lock()

    def loop():
        rng, version, tick, next_period = random.Random(config['seed']), 0, 0, begin
        while not stop.is_set():
            cycle = time.perf_counter()
            snapshots = {}
            for node in config['edges']:
                try:
                    code, value = request_json(node['url'] + '/snapshot',
                        headers={'X-Control-Token': config['control_token']}, timeout=.5)
                    if code == 200:
                        snapshots[node['id']] = value
                        counters['polling_response_bytes'] += len(encoded(value))
                        journal.put('monitor_snapshot', snapshot=value)
                except Exception as error:
                    journal.put('monitor_error', server_id=node['id'], error=repr(error))
            now = time.perf_counter()
            due = cfg['mode'] != 'periodic' or now >= next_period
            selected = choose_updates(cfg['mode'], snapshots, previous, time.time_ns(), rng, cfg) if due else []
            if selected:
                # Use fixed-width nanosecond timestamps when charging payloads.
                packets = [dict(session=session, user_id=user, version=version + 1, tick=tick,
                    send_epoch_ns=time.time_ns(), servers=[snapshots[j] for j in selected])
                    for user in range(config['users'])]
                size = sum(len(encoded(packet)) for packet in packets)
                if any(len(encoded(packet)) > 65000 for packet in packets):
                    raise RuntimeError('Feedback exceeds UDP datagram bound')
                if budget.spend(size, now):
                    version += 1
                    for packet in packets:
                        heapq.heappush(pending, (now + cfg['injected_delay_s'], version, packet['user_id'], packet))
                    for j in selected:
                        previous[j] = snapshots[j]
                    if cfg['mode'] == 'periodic':
                        next_period = now + cfg['period_s']
                else:
                    counters['budget_deferrals'] += 1
                    journal.put('budget_deferred', requested_bytes=size, available_bytes=budget.tokens, tick=tick)
            while pending and pending[0][0] <= time.perf_counter():
                _, _, _, packet = heapq.heappop(pending)
                packet['send_epoch_ns'] = time.time_ns()
                body = encoded(packet)
                sender.sendto(body, (cfg['udp_host'], cfg['udp_port']))
                with lock:
                    counters['messages'] += 1
                    counters['application_bytes'] += len(body)
                journal.put('feedback_sent', user_id=packet['user_id'], version=packet['version'],
                    tick=packet['tick'], payload_bytes=len(body), send_epoch_ns=packet['send_epoch_ns'],
                    server_sources={x['server_id']: x['source_epoch_ns'] for x in packet['servers']})
            tick += 1
            stop.wait(max(0., cfg['monitor_period_s'] - (time.perf_counter() - cycle)))

    worker = threading.Thread(target=loop, daemon=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/health':
                self.send_error(404)
                return
            with lock:
                body = encoded(dict(session=session, pid=runtime()['pid'], ready=worker.is_alive(), elapsed_s=time.perf_counter() - begin,
                                    configured_budget=cfg['budget_bytes_per_s'], **counters))
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == '/shutdown' and self.headers.get('X-Control-Token') == config['control_token']:
                body=encoded(dict(stopping=True))
                self.send_response(200)
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                threading.Thread(target=server.shutdown,daemon=True).start()
            else:
                self.send_error(403)

    server = ThreadingHTTPServer((config.get('bind_host', '127.0.0.1'), cfg['health_port']), Handler)
    worker.start()
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        stop.set()
        worker.join(5)
        server.server_close()
        sender.close()
        journal.put('stopped', **counters, queued_unsent_datagrams=len(pending))
