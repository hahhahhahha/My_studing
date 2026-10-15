"""Continuous-time finite-buffer FIFO foundation; no policy or data fitting.

Durations are seconds. Actual service requirements stay in execution records;
future actor integration must use separate predicted metadata/stale summaries.
The accompanying synthetic fixtures test semantics, not measured calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Request:
    task_id: str
    arrival: float
    upload: float
    server: int
    service: float
    payload_bytes: int
    image_id: str = ''


class FIFOExecution:
    def __init__(self, servers: int, waiting_capacity: int):
        if servers < 1 or waiting_capacity < 0:
            raise ValueError('Invalid server/buffer size')
        self.servers = servers
        self.waiting_capacity = waiting_capacity
        self.time = 0.0
        self.records = {}
        self.waiting = [[] for _ in range(servers)]
        self.running = [None] * servers
        self.events = []

    def submit(self, request: Request):
        if request.task_id in self.records:
            raise ValueError('Duplicate task identity')
        if (not all(math.isfinite(x) for x in [request.arrival, request.upload, request.service])
            or request.arrival < self.time or request.upload < 0 or request.service <= 0
            or not 0 <= request.server < self.servers or request.payload_bytes < 0):
            raise ValueError('Invalid request')
        ready = request.arrival + request.upload
        self.records[request.task_id] = dict(request=request, status='uploading', ready=ready,
                                             admitted=None, started=None, completed=None)
        heapq.heappush(self.events, (ready, 1, request.task_id))

    def _start(self, task_id):
        record = self.records[task_id]
        request = record['request']
        if self.running[request.server] is not None:
            raise RuntimeError('Attempted overlapping service')
        record.update(status='running', started=self.time)
        self.running[request.server] = task_id
        heapq.heappush(self.events, (self.time + request.service, 0, task_id))

    def advance(self, until):
        if not math.isfinite(until) or until < self.time:
            raise ValueError('Time cannot run backwards')
        # Completion precedes new admissions at equal timestamps; upload ties
        # have stable task-ID order, shared by all methods.
        while self.events and self.events[0][0] <= until:
            self.time, kind, task_id = heapq.heappop(self.events)
            record = self.records[task_id]
            server = record['request'].server
            if kind == 0:
                if self.running[server] != task_id:
                    raise RuntimeError('Service identity mismatch')
                record.update(status='completed', completed=self.time)
                self.running[server] = None
                if self.waiting[server]:
                    self._start(self.waiting[server].pop(0))
            elif self.running[server] is None:
                record['admitted'] = self.time
                self._start(task_id)
            elif len(self.waiting[server]) < self.waiting_capacity:
                record.update(status='waiting', admitted=self.time)
                self.waiting[server].append(task_id)
            else:
                record['status'] = 'dropped'
        self.time = until
        self.check_conservation()

    def check_conservation(self):
        states = ['uploading', 'waiting', 'running', 'completed', 'dropped']
        counts = {s: sum(r['status'] == s for r in self.records.values()) for s in states}
        if sum(counts.values()) != len(self.records):
            raise RuntimeError('Task conservation failed')
        if counts['waiting'] != sum(map(len, self.waiting)) or counts['running'] != sum(x is not None for x in self.running):
            raise RuntimeError('Queue/task conservation mismatch')
        if any(len(queue) > self.waiting_capacity for queue in self.waiting):
            raise RuntimeError('Buffer capacity exceeded')
        return counts

    def results(self):
        counts = self.check_conservation()
        latencies = [r['completed'] - r['request'].arrival for r in self.records.values() if r['status'] == 'completed']
        waits = [r['started'] - r['ready'] for r in self.records.values() if r['started'] is not None]
        unfinished = counts['uploading'] + counts['waiting'] + counts['running']
        return dict(generated=len(self.records), accepted=counts['waiting'] + counts['running'] + counts['completed'],
                    completed=counts['completed'], dropped=counts['dropped'], unfinished=unfinished,
                    inflight_upload=counts['uploading'], latency_seconds=latencies, waiting_seconds=waits)

    def work_accounting(self):
        """Per-server service-seconds conservation, including upload and drops.

        This execution diagnostic may read actual durations. It is NOT an actor
        feature or a hardware-independent demand-concentration measure.
        """
        rows = [dict(offered=0., inflight=0., dropped=0., admitted=0.,
                     executed=0., admitted_remaining=0.) for _ in range(self.servers)]
        for record in self.records.values():
            request = record['request']
            row = rows[request.server]
            row['offered'] += request.service
            status = record['status']
            if status == 'uploading':
                row['inflight'] += request.service
            elif status == 'dropped':
                row['dropped'] += request.service
            else:
                row['admitted'] += request.service
                executed = request.service if status == 'completed' else (
                    min(max(self.time - record['started'], 0.), request.service) if status == 'running' else 0.)
                row['executed'] += executed
                row['admitted_remaining'] += request.service - executed
        for row in rows:
            row['offered_error'] = row['offered'] - row['inflight'] - row['dropped'] - row['admitted']
            row['admitted_error'] = row['admitted'] - row['executed'] - row['admitted_remaining']
            if max(abs(row['offered_error']), abs(row['admitted_error'])) > 1e-10 * max(1., row['offered']):
                raise RuntimeError('Service-seconds conservation failed')
        return rows
