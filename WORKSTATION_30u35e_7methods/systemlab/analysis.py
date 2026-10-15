from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import encoded, quantile, read, sha, source_receipt, tail_mean, write_new


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(client_dir, output, controller_dir=None):
    root = Path(client_dir)
    complete, inputs, cutoff = read(root/'COMPLETE.json'), read(root/'INPUTS.json'), read(root/'CUTOFF.json')
    tasks = rows(root/'tasks.jsonl')
    if len(tasks) != len({x['task_id'] for x in tasks}) or len(tasks) != cutoff['scheduled_tasks']:
        raise ValueError('All offered tasks must have exactly one cutoff outcome')
    status = Counter(x['status'] for x in tasks)
    completed = [x for x in tasks if x['status'] == 'completed']
    latencies = [x['e2e_from_release_s'] for x in completed]
    if any(not math.isfinite(x) or x < 0 for x in latencies):
        raise ValueError('Invalid latency')
    decisions = rows(root/'decisions.jsonl')
    ages = [age for x in decisions for age in x['feedback_age_s']]
    decision_times = [x['decision_s'] for x in decisions]
    concentration = []
    for record in decisions:
        destinations = Counter(record['assigned_task_servers'].values())
        total = sum(destinations.values())
        if total:
            concentration.append(dict(tick=record['tick'], user_id=record['user_id'],
                max_share=max(destinations.values())/total,
                hhi=sum((n/total)**2 for n in destinations.values())))
    # Across all users in the same decision tick: realized joint assignments.
    by_tick = defaultdict(Counter)
    for record in decisions:
        by_tick[record['tick']].update(record['assigned_task_servers'].values())
    joint = [dict(tick=tick, max_share=max(values.values())/sum(values.values()),
                  hhi=sum((n/sum(values.values()))**2 for n in values.values())) for tick, values in sorted(by_tick.items()) if values]
    value = dict(schema='systemlab-analysis-v1', method=inputs['method'], role=inputs['role'],
        system_id=inputs['method']+'|'+inputs['config']['feedback']['mode'],
        comparison_contract={key:inputs['config'][key] for key in ('users','edges','waiting_capacity','weight_sha256','warmup','fixed_drain_s','decision_period_s','feature_scales')},
        feedback_budget_contract={key:inputs['config']['feedback'][key] for key in ('budget_bytes_per_s','burst_bytes')},
        schedule_sha256=inputs['schedule_sha256'], schedule_seed=inputs['schedule_seed'],
        offered=len(tasks), status_counts=dict(status), completed=len(completed),
        completion_fraction=len(completed)/len(tasks) if tasks else None,
        latency_definition='planned_release_to_client_response; decision_wait_and_generator_lateness_included',
        mean_latency_s=sum(latencies)/len(latencies) if latencies else None,
        p95_latency_s=quantile(latencies,.95), p99_latency_s=quantile(latencies,.99),
        cvar99_s=tail_mean(latencies), cvar_tail_effective_tasks=.01*len(latencies),
        completion_throughput_per_offered_horizon_s=len(completed)/cutoff['protocol_horizon_s'],
        mean_feedback_age_s=sum(ages)/len(ages) if ages else None, p99_feedback_age_s=quantile(ages,.99),
        p99_policy_decision_s=quantile(decision_times,.99), mean_policy_decision_s=sum(decision_times)/len(decision_times) if decision_times else None,
        realized_joint_assignment_windows=joint,
        mean_joint_hhi=sum(x['hhi'] for x in joint)/len(joint) if joint else None,
        p99_joint_max_share=quantile([x['max_share'] for x in joint],.99),
        protocol_complete=complete['protocol_complete'],
        infrastructure_valid=complete['protocol_complete'] and not status['client_backpressure'] and not status['invalid_request'],
        conditional_completed_latency_only=True, caps_or_drop_latency_imputation=False)
    if controller_dir:
        feedback = rows(Path(controller_dir)/'feedback_send.jsonl')
        horizon_end = cutoff['begin_epoch_ns'] + int(cutoff['protocol_horizon_s']*1e9)
        sent = [x for x in feedback if x['kind']=='feedback_sent' and cutoff['begin_epoch_ns']<=x['send_epoch_ns']<horizon_end]
        value['feedback_application_bytes_per_s'] = sum(x['payload_bytes'] for x in sent)/cutoff['protocol_horizon_s']
        value['feedback_messages_per_s'] = len(sent)/cutoff['protocol_horizon_s']
        value['udp_ipv4_header_bytes_estimated'] = 28*len(sent)
        monitor = [x for x in feedback if x['kind']=='monitor_snapshot' and cutoff['begin_epoch_ns']<=x['epoch_ns']<=cutoff['cutoff_epoch_ns']]
        value['maximum_observed_waiting_tasks'] = max((x['snapshot']['waiting_count'] for x in monitor),default=0)
        per_edge = defaultdict(list)
        for x in monitor:
            per_edge[x['snapshot']['server_id']].append(x['snapshot'])
        value['positive_waiting_task_jumps'] = {edge:[max(0,b['waiting_count']-a['waiting_count']) for a,b in zip(items,items[1:])]
                                                for edge,items in per_edge.items()}
    write_new(output, value)
    return value


def paired_compare(files, treatment, baseline, metric, output, seed=20261013):
    data = [read(path) for path in files]
    pairs = defaultdict(dict)
    for item in data:
        if not item['infrastructure_valid']:
            raise ValueError('Incomplete/infrastructure-invalid runs must be repaired as new attempts before comparison')
        label=item.get('system_id',item['method'])
        if label in (treatment,baseline):
            key = (item['schedule_sha256'], item['schedule_seed'])
            if label in pairs[key]:
                raise ValueError('Replicate identity must be explicitly registered; duplicate pair')
            pairs[key][label] = item
    if any(set(v)!={treatment,baseline} for v in pairs.values()) or len(pairs)<3:
        raise ValueError('At least 3 complete paired schedule repetitions required')
    deltas=[]
    for values in pairs.values():
        if (values[treatment]['comparison_contract']!=values[baseline]['comparison_contract'] or
                values[treatment]['feedback_budget_contract']!=values[baseline]['feedback_budget_contract']):
            raise ValueError('Matched resource/feature/horizon and feedback byte budgets required')
        a,b=values[treatment][metric],values[baseline][metric]
        if a is None or b is None:
            raise ValueError('Zero-completion metric is undefined, not zero')
        deltas.append(a-b)
    rng=random.Random(seed)
    resamples=[sum(rng.choice(deltas) for _ in deltas)/len(deltas) for _ in range(10000)]
    value=dict(schema='systemlab-paired-analysis-v1', treatment=treatment,baseline=baseline,metric=metric,
        paired_repetitions=len(deltas), differences=deltas,mean_difference=sum(deltas)/len(deltas),
        descriptive_paired_repetition_bootstrap_95=[quantile(resamples,.025),quantile(resamples,.975)],
        bootstrap_seed=seed,unit='registered_complete_run_pair', no_task_level_iid_CI=True)
    write_new(output,value)
    return value


def calibrate(client_dir, edges_dir, config, manifest_path, output):
    inputs=read(Path(client_dir)/'INPUTS.json')
    if inputs['role']!='calibration' or not read(Path(client_dir)/'COMPLETE.json')['protocol_complete']:
        raise ValueError('Complete calibration-role run required; validation/test cannot fit scales')
    if encoded(config)!=encoded(inputs['config']) or sha(manifest_path)!=inputs['manifest_sha256']:
        raise ValueError('Calibration must consume the exact configuration and image manifest actually executed')
    services={}
    for node in config['edges']:
        items=[x for x in rows(Path(edges_dir)/node['id']/'edge.jsonl')
               if x['kind']=='execution_result' and x['status']=='completed']
        if not items:
            raise ValueError('Every configured edge needs actual calibration execution')
        services[node['id']]=sum(x['server_service_s'] for x in items)/len(items)
    pool=read(manifest_path)
    if pool['role']!='calibration':
        raise ValueError('Calibration image pool required')
    transfers=[]
    tasks=rows(Path(client_dir)/'tasks.jsonl')
    for node in config['edges']:
        values=[]
        for task in tasks:
            if task['status']=='completed' and task['server_id']==node['id']:
                server=task['server_result']
                values.append(max(0.,task['e2e_from_send_s']-server['queue_s']-server['server_service_s']))
        if not values:
            raise ValueError('Client response calibration missing')
        transfers.append(sum(values)/len(values))
    value=dict(config['feature_scales'])
    value.update(fitted_role='calibration',reference_service_s=services[config['edges'][0]['id']],
        status='fitted_calibration_requires_validation_and_registration',
        mean_payload_bytes=sum(x['payload_bytes'] for x in pool['images'])/len(pool['images']),
        transfer_seconds_per_server=transfers,calibration_service_s_by_edge=services,
        calibration_schedule_sha256=inputs['schedule_sha256'],
        calibration_inputs=dict(path=str((Path(client_dir)/'INPUTS.json').resolve()),sha256=sha(Path(client_dir)/'INPUTS.json')),
        calibration_completion=dict(path=str((Path(client_dir)/'COMPLETE.json').resolve()),sha256=sha(Path(client_dir)/'COMPLETE.json')),
        calibration_edge_logs={str((Path(edges_dir)/node['id']/'edge.jsonl').resolve()):sha(Path(edges_dir)/node['id']/'edge.jsonl') for node in config['edges']})
    write_new(output,value)
    return value


def freeze(config, validation_clients, output):
    """Generate pinned deployment files from actual cal/validation evidence."""
    scales=config['feature_scales']
    if scales['status']!='fitted_calibration_requires_validation_and_registration':
        raise ValueError('A fitted calibration contract is required')
    for key in ('calibration_inputs','calibration_completion'):
        binding=scales[key]
        if sha(binding['path'])!=binding['sha256']:
            raise ValueError('Calibration evidence changed')
    if read(scales['calibration_inputs']['path'])['role']!='calibration' or not read(scales['calibration_completion']['path'])['protocol_complete']:
        raise ValueError('Actual complete calibration evidence required')
    for path,fingerprint in scales['calibration_edge_logs'].items():
        if sha(path)!=fingerprint:
            raise ValueError('Calibration execution events changed')
    validations={}
    methods=set()
    for path in validation_clients:
        root=Path(path)
        inputs,complete=read(root/'INPUTS.json'),read(root/'COMPLETE.json')
        if inputs['role']!='validation' or not complete['protocol_complete'] or encoded(inputs['config'])!=encoded(config):
            raise ValueError('Complete validation with the selected exact configuration required')
        if inputs['source_sha256']!=source_receipt():
            raise ValueError('Validation must exercise the current source')
        if any(x['status']=='client_backpressure' for x in rows(root/'tasks.jsonl')):
            raise ValueError('Resolve load-generator bottleneck before freezing')
        methods.add(inputs['method'])
        for name in ('INPUTS.json','COMPLETE.json','CUTOFF.json','tasks.jsonl','decisions.jsonl'):
            validations[str((root/name).resolve())]=sha(root/name)
    required_methods = {'qla', 'ppo_mv', 'ppo_dsr', 'mfc_di', 'qeco', 'jseq', 'lyfoto_lc'}
    if not required_methods.issubset(methods):
        raise ValueError('Validate all seven registered methods before freezing')
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=False)
    formal=json.loads(encoded(config))
    formal['feature_scales']['status']='frozen_after_calibration_validation'
    receipt=dict(schema='systemlab-frozen-protocol-v1',formal_config= formal,
        source_sha256=source_receipt(),validation_artifacts_sha256=validations,
        calibration_contract=scales,independent_entry_review_required=True,
        registration_is_not_independent_approval=True)
    write_new(destination/'FROZEN_SYSTEM_PROTOCOL.json',receipt)
    formal['frozen_protocol_path']=str((destination/'FROZEN_SYSTEM_PROTOCOL.json').resolve())
    formal['frozen_protocol_sha256']=sha(destination/'FROZEN_SYSTEM_PROTOCOL.json')
    write_new(destination/'FORMAL_CONFIG.json',formal)
    write_new(destination/'FEATURE_SCALES.json',formal['feature_scales'])
    return receipt


def register_condition(base_config, changes, output):
    """Register the preregistered feedback scan without changing trained inputs."""
    base=read(base_config)
    if sha(base['frozen_protocol_path'])!=base['frozen_protocol_sha256']:
        raise ValueError('Base frozen protocol changed')
    protocol=read(base['frozen_protocol_path'])
    allowed={'mode','period_s','injected_delay_s','injected_loss','alpha','beta','gamma','trigger_threshold'}
    if not changes or set(changes)-allowed:
        raise ValueError('Only explicit preregistered feedback conditions can be derived here')
    value=json.loads(encoded(base))
    value['feedback'].update(changes)
    if value['feedback']['mode'] not in ('periodic','random','queue_trigger','age','coordinated'):
        raise ValueError('Unknown feedback mode')
    if not 0<=value['feedback']['injected_loss']<=1 or value['feedback']['injected_delay_s']<0:
        raise ValueError('Invalid propagation condition')
    plain={k:v for k,v in value.items() if k not in ('frozen_protocol_path','frozen_protocol_sha256')}
    derived=dict(protocol,formal_config=plain,source_sha256=source_receipt(),
        derived_from_protocol_sha256=base['frozen_protocol_sha256'],preregistered_feedback_changes=changes,
        base_validation_covers_shared_pipeline_only=True,new_condition_functional_check_and_review_required=True)
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=False)
    write_new(destination/'FROZEN_SYSTEM_PROTOCOL.json',derived)
    value['frozen_protocol_path']=str((destination/'FROZEN_SYSTEM_PROTOCOL.json').resolve())
    value['frozen_protocol_sha256']=sha(destination/'FROZEN_SYSTEM_PROTOCOL.json')
    write_new(destination/'FORMAL_CONFIG.json',value)
    return derived
