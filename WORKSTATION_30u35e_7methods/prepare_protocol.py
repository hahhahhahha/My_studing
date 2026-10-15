"""Create a non-frozen 30-user/35-edge proposal. No service/model launch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
from pathlib import Path

from systemlab.common import encoded, sha, source_receipt, write_new


def build_config(base, weight):
    config = copy.deepcopy(base)
    config['users'] = 30
    config['weight_path'] = str(Path(weight).resolve())
    config['edges'] = [dict(id=f'edge{j}', port=8101 + j, url=f'http://127.0.0.1:{8101+j}',
                            torch_threads=1, initial_service_s=.031) for j in range(35)]
    config['feature_scales']['transfer_seconds_per_server'] = [.001] * 35
    config['feature_scales']['status'] = 'provisional_requires_new_concurrent_system_calibration'
    config['feedback']['budget_bytes_per_s'] = 2 * 1024 * 1024
    config['feedback']['burst_bytes'] = 1024 * 1024
    config['feedback']['period_s'] = .2
    return config


def feedback_fixture(config):
    snapshots = [dict(server_id=edge['id'], source_epoch_ns=1792051200000000000, waiting_count=0,
                      capacity=config['waiting_capacity'], busy=False, service_estimate_s=edge['initial_service_s'],
                      predicted_wait_s=0., received=0, admitted=0, completed=0, inference_errors=0, rejected=0)
                 for edge in config['edges']]
    packets = [dict(session='0' * 64, user_id=user, version=1, tick=0,
                    send_epoch_ns=1792051200000000000, servers=snapshots) for user in range(config['users'])]
    sizes = [len(encoded(packet)) for packet in packets]
    result = dict(scope='idle_serialization_fixture_not_measured_feedback', users=config['users'], edges=len(snapshots),
                  broadcast_round_bytes=sum(sizes), largest_user_datagram_bytes=max(sizes),
                  nominal_round_bytes_per_s=sum(sizes) / config['feedback']['period_s'])
    if max(sizes) > 65000 or sum(sizes) > config['feedback']['burst_bytes']:
        raise ValueError('Feedback fixture cannot fit datagram/burst limits')
    if result['nominal_round_bytes_per_s'] > config['feedback']['budget_bytes_per_s']:
        raise ValueError('Budget insufficient even for the idle nominal-period fixture')
    return result


def check_ports(config):
    for port, kind in [(8100, socket.SOCK_STREAM)] + [(x['port'], socket.SOCK_STREAM) for x in config['edges']] + [(9100, socket.SOCK_DGRAM)]:
        with socket.socket(socket.AF_INET, kind) as sock:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(('127.0.0.1', port))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path.home() / 'qla_assets_20261015')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-ports', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        base_path = root / 'configs/BASE_LOCAL_4EDGE_HISTORY.json'
        base = json.loads(base_path.read_text(encoding='utf-8-sig'))
        weight = args.assets.resolve() / 'models/resnet18-f37072fd.pth'
        config = build_config(base, weight)
        if sha(weight) != config['weight_sha256']:
            raise ValueError('Vision weight identity mismatch')
        fixture = feedback_fixture(config)
        if args.check_ports:
            check_ports(config)
        config_path = out / 'CANDIDATE_CONFIG.json'
        write_new(config_path, config)
        write_new(out / 'REGISTRATION.json', dict(
            schema='workstation-30u35e-proposal-v1', status='registered_source_preparation_not_runtime',
            user_revision='30 users / 35 logical servers; 2MiB/s total feedback; 1MiB burst; nominal 200ms',
            users=30, logical_edge_processes=35, torch_threads_per_edge=1,
            resource_policy='provisional homogeneous one-thread CPU processes; shared physical host, no hard isolation',
            provisional_service_values_not_measured=True, config_sha256=sha(config_path),
            generator_sha256=sha(__file__), base_config_sha256=sha(base_path), source_sha256=source_receipt(),
            old_results_unchanged=True, ports_probed=args.check_ports, port_probe_is_not_reservation=True,
            actual_models_launched=0, actual_services_launched=0, training_calls=0,
            six_baselines_deployed=False, settings_frozen=False, formal_ready=False,
            pending=['all six real-system baseline adapters and source/checkpoint binding',
                     'target-resource capacity/feedback timing evidence', 'full-horizon validation and independent entry review'],
            feedback_fixture=fixture))
        print(json.dumps(dict(output=str(out), users=30, logical_servers=35,
                             budget_bytes_per_s=2097152, burst_bytes=1048576, period_s=.2,
                             ports_probed=args.check_ports, services_launched=0,
                             six_baselines_deployed=False, formal_ready=False), indent=2))
    except Exception as error:
        write_new(out / 'PREPARATION_FAILURE.json', dict(error=repr(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
