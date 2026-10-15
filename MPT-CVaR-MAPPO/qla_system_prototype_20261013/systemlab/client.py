from __future__ import annotations

import concurrent.futures
import json
import math
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path

from .common import Journal, encoded, read, request_json, sha, source_receipt, write_new
from .feedback import FeedbackCache
from .policy import Router, sampled_index
from .admission import validate_test_admission


def replay(config, schedule_path, manifest_path, method, output, bundle=None, entry_review=None, test_registry=None, cell_id=None):
    admission = validate_test_admission(config,schedule_path,manifest_path,method,entry_review,test_registry,cell_id,bundle)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    schedule, manifest = read(schedule_path), read(manifest_path)
    if (schedule['schema'] != 'systemlab-schedule-v1' or schedule['users'] != config['users'] or
            schedule['image_manifest_sha256'] != sha(manifest_path) or schedule['role'] != manifest['role']):
        raise ValueError('Schedule/image/user binding mismatch')
    task_ids = [x['task_id'] for x in schedule['tasks']]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError('Duplicate schedule task ids')
    assets = {}
    root = Path(manifest['image_root']).resolve()
    for item in manifest['images']:
        path = (root / item['image_id']).resolve()
        if not path.is_relative_to(root) or sha(path) != item['image_sha256']:
            raise ValueError('Runtime JPEG changed')
        body = path.read_bytes()
        if len(body) != item['payload_bytes']:
            raise ValueError('JPEG payload length changed')
        assets[item['image_id']] = dict(item, body=body)
    for task in schedule['tasks']:
        if (not 0 <= task['user_id'] < config['users'] or not math.isfinite(task['arrival_s']) or
                not 0 <= task['arrival_s'] < schedule['duration_s'] or task['image_id'] not in assets):
            raise ValueError('Malformed task or role membership')
    # Only policy-relevant configuration reaches the decision module.
    router_cfg = {key: config[key] for key in ('edges', 'feature_scales', 'score_temperature_s', 'seed')}
    router = Router(method, router_cfg, bundle)
    cfg = config['feedback']
    code, health = request_json(config['controller_url'] + '/health')
    if code != 200 or not health['ready']:
        raise RuntimeError('Feedback controller not ready')
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((cfg.get('client_bind_host', '127.0.0.1'), cfg['udp_port']))
    receiver.settimeout(.1)
    feedback_log = Journal(output / 'feedback_receive.jsonl')
    decisions = Journal(output / 'decisions.jsonl')
    task_log = Journal(output / 'tasks.jsonl')
    late_log = Journal(output / 'late_responses.jsonl')
    cache = FeedbackCache(health['session'], config['users'], config['seed'], cfg['injected_loss'], feedback_log)
    stop = threading.Event()

    def receive():
        while not stop.is_set():
            try:
                raw, _ = receiver.recvfrom(65507)
                packet = json.loads(raw)
                cache.accept(packet, time.time_ns())
            except socket.timeout:
                continue
            except Exception as error:
                if not stop.is_set():
                    feedback_log.put('receiver_error', error=repr(error))

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    warmup_deadline = time.perf_counter() + config['feedback_start_timeout_s']
    expected_ids = {edge['id'] for edge in config['edges']}
    while any(set(cache.observation(u)) != expected_ids for u in range(config['users'])):
        if time.perf_counter() >= warmup_deadline:
            stop.set()
            thread.join(1)
            receiver.close()
            write_new(output / 'FAILURE.json', dict(error='No complete initial delivered feedback', actual_tasks=0))
            raise RuntimeError('Initial feedback was not delivered to every user')
        time.sleep(.01)
    initial_sources=source_receipt()
    initial_schedule_sha,initial_manifest_sha=sha(schedule_path),sha(manifest_path)
    write_new(output / 'INPUTS.json', dict(schedule_sha256=initial_schedule_sha, manifest_sha256=initial_manifest_sha,
        config=config, method=method, policy_receipt=router.receipt, source_sha256=source_receipt(),
        role=schedule['role'], schedule_seed=schedule['seed'], controller_session=health['session'],
        entry_review_sha256=sha(entry_review) if entry_review else None,test_admission=admission))
    begin, begin_epoch = time.perf_counter(), time.time_ns()
    cutoff = begin + schedule['duration_s'] + config['fixed_drain_s']
    cutoff_epoch = begin_epoch + int((schedule['duration_s'] + config['fixed_drain_s']) * 1e9)
    client_capacity = threading.BoundedSemaphore(config['max_inflight'])
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=config['max_inflight'])
    outcomes, outcome_lock, futures = {}, threading.Lock(), []

    def send_task(task, asset, edge, weights, decision_start, decision_s):
        sent = time.perf_counter()
        base = dict(task_id=task['task_id'], user_id=task['user_id'], image_id=task['image_id'],
            server_id=edge['id'], planned_release_s=task['arrival_s'], send_elapsed_s=sent-begin,
            decision_s=decision_s, release_to_send_s=sent-begin-task['arrival_s'],
            assignment_uniform=task['assignment_uniform'], payload_bytes=asset['payload_bytes'])
        try:
            code, response = request_json(edge['url'] + '/infer', body=asset['body'], timeout=config['client_timeout_s'],
                headers={'Content-Type': 'image/jpeg', 'X-Task-Id': task['task_id'],
                         'X-User-Id': str(task['user_id']), 'X-JPEG-SHA256': asset['image_sha256']})
            if response.get('status')=='completed':
                if response.get('task_id')!=task['task_id'] or response.get('server_id')!=edge['id']:
                    raise ValueError('Server response task/edge identity mismatch')
                for field in ('queue_s','server_service_s','preprocess_s','compute_s'):
                    if not math.isfinite(response[field]) or response[field]<0:
                        raise ValueError('Invalid server duration')
            end = time.perf_counter()
            value = dict(base, status=response.get('status', 'protocol_error'), http_status=code,
                client_response_elapsed_s=end-begin, e2e_from_send_s=end-sent,
                e2e_from_release_s=end-begin-task['arrival_s'], server_result=response)
        except Exception as error:
            end = time.perf_counter()
            value = dict(base, status='transport_error', error=repr(error), client_response_elapsed_s=end-begin)
        with outcome_lock:
            if end <= cutoff:
                outcomes[task['task_id']] = value
                task_log.put('task_outcome', **value)
            else:
                late_log.put('late_response', **value)
        client_capacity.release()

    bins = defaultdict(lambda: defaultdict(list))
    for task in schedule['tasks']:
        tick = math.ceil(task['arrival_s'] / config['decision_period_s'])
        bins[tick][task['user_id']].append(task)
    error = None
    try:
        for tick in sorted(bins):
            target = begin + tick * config['decision_period_s']
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            for user in sorted(bins[tick]):
                batch = bins[tick][user]
                states = cache.observation(user)
                start = time.perf_counter()
                metadata = [dict(t, payload_bytes=assets[t['image_id']]['payload_bytes']) for t in batch]
                weights, feature_receipt = router.portfolio(metadata, states, time.time_ns(), user)
                decision_s = time.perf_counter() - start
                decisions.put('decision', user_id=user, tick=tick, task_ids=[x['task_id'] for x in batch],
                    method=method, portfolio=weights, decision_s=decision_s,
                    assigned_task_servers={x['task_id']:config['edges'][sampled_index(weights,x['assignment_uniform'])]['id'] for x in batch},
                    source_timestamps={k:v['source_epoch_ns'] for k,v in states.items()},
                    feedback_versions={k:v['version'] for k,v in states.items()}, **feature_receipt)
                for task in batch:
                    index = sampled_index(weights, task['assignment_uniform'])
                    edge = config['edges'][index]
                    if time.perf_counter() >= cutoff or not client_capacity.acquire(blocking=False):
                        value = dict(task_id=task['task_id'], user_id=user, status='client_backpressure',
                                     server_id=edge['id'], planned_release_s=task['arrival_s'])
                        with outcome_lock:
                            outcomes[task['task_id']] = value
                            task_log.put('task_outcome', **value)
                    else:
                        futures.append(pool.submit(send_task, task, assets[task['image_id']], edge, weights, start, decision_s))
        # Preserve the exact registered cutoff even if all requests finish early.
        while time.perf_counter() < cutoff:
            time.sleep(max(0.,min(.05, cutoff-time.perf_counter())))
    except Exception as caught:
        error = repr(caught)
    finally:
        stop.set()
        thread.join(1)
        receiver.close()
        write_new(output / 'CUTOFF.json', dict(begin_epoch_ns=begin_epoch, cutoff_epoch_ns=cutoff_epoch,
            scheduled_tasks=len(schedule['tasks']), error=error, fixed_drain_s=config['fixed_drain_s'],
            protocol_horizon_s=schedule['duration_s']))
        # Await bounded HTTP calls for provenance; late data never improves the cutoff metrics.
        pool.shutdown(wait=True)
        # Only after every in-flight request has joined, finalize tasks that never
        # received a formal outcome. This is the single place an 'unfinished' status
        # is written, so each task has exactly one cutoff outcome (no double-write).
        with outcome_lock:
            for task in schedule['tasks']:
                if task['task_id'] not in outcomes:
                    value = dict(task_id=task['task_id'], user_id=task['user_id'],
                                 status='unfinished_at_cutoff' if error is None else 'aborted_before_cutoff',
                                 planned_release_s=task['arrival_s'])
                    outcomes[task['task_id']] = value
                    task_log.put('task_outcome', **value)
        frozen_receipt=None
        try:
            frozen_receipt=router.verify_frozen()
            if admission != validate_test_admission(config,schedule_path,manifest_path,method,entry_review,test_registry,cell_id,bundle):
                raise RuntimeError('Test admission binding changed during execution')
            if (initial_sources!=source_receipt() or sha(schedule_path)!=initial_schedule_sha or
                    sha(manifest_path)!=initial_manifest_sha):
                raise RuntimeError('Execution source/schedule/manifest changed during this run')
        except Exception as caught:
            error=error or repr(caught)
        write_new(output / 'COMPLETE.json', dict(protocol_complete=error is None, method=method,
            schedule_sha256=sha(schedule_path), actual_outcomes=len(outcomes), error=error,
            inference_only=True, optimizer_calls=0,frozen_actor_verification=frozen_receipt,
            source_continuity_pass=initial_sources==source_receipt()))
        for log in (feedback_log, decisions, task_log, late_log):
            log.close()
    if error:
        raise RuntimeError(error)
    return output
