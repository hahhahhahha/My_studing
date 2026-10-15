"""Analytic cal/validation payload-load diagnosis; no simulation/model launch."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility, TrainingWindow
from visual_method_features import calibration_scales
from visual_upload_link import UploadLinkConfig
from visual_learned_bindings import source_bindings

BASE = Path(__file__).resolve().parents[1]
PREVIOUS = BASE / 'visual_calibrated_policy/engineering_attempts/20261011_091538_576632_episode_collector/RESULT.json'
PREVIOUS_SHA = 'd932db9bab612daf66556eaec83280f99c8c0e99ef7376150568b034bd815ebf'
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')
PLAN = {'status': 'engineering_diagnostic_rules_not_formal_protocol', 'users': 30,
    'virtual_profiles': PROFILES, 'geometry_seed': 123, 'candidate_training_mobility_window_seconds': [0., 500.],
    'decision_seconds': .05, 'decision_points': 10000, 'rho': [.60, .75, .85],
    'load_model': 'Uniform per-user unconditional rate; JPEG/position independence',
    'capacity_source': 'Calibration mean only; validation JPEG payloads diagnostic only',
    'default_network_unchanged': asdict(UploadLinkConfig()), 'upload_headroom_diagnostic_target': .8,
    'training_budget_candidate': {'total_decisions': 10000, 'episodes': 10, 'decisions_per_episode': 1000,
        'generation_seconds_per_episode': 50., 'fixed_drain_seconds_per_episode': 30.,
        'updates_after_complete_episodes_only': True, 'gamma_per_second': .99, 'lambda_per_second': .95,
        'MV_reset_rule_not_frozen_or_changed': True},
    'no_policy_performance_or_test_outcomes_for_parameter_selection': True,
    'no_network_or_rho_modification_in_this_diagnostic': True}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def payload_statistics(catalog):
    values = np.asarray([catalog.observable_metadata(image).payload_bytes for image in catalog.image_ids()], dtype=np.float64)
    return {'images': len(values), 'mean_bytes': float(values.mean()), 'min_bytes': int(values.min()),
            'median_bytes': float(np.quantile(values, .5)), 'p95_bytes': float(np.quantile(values, .95)),
            'p99_bytes': float(np.quantile(values, .99)), 'max_bytes': int(values.max())}


def rate_grid(mobility, edges, link):
    times = [float(i * Fraction(str(PLAN['decision_seconds']))) for i in range(1, PLAN['decision_points']+1)]
    positions = np.asarray([mobility.positions(time) for time in times])
    difference = positions[:, :, None, :] - np.asarray(edges)[None, None, :, :]
    distances = np.maximum(np.linalg.norm(difference, axis=-1), link.min_distance_m)
    gain = link.antenna_gain * (link.light_speed_m_s / (4*math.pi*link.carrier_hz*distances)) ** 2
    rates = link.bandwidth_hz * np.log1p(link.tx_power_w*gain/link.noise_power_w)/math.log(2.)
    assert rates.shape == (10000, 30, 4) and np.isfinite(rates).all() and np.all(rates > 0)
    # Verify the vectorization against the actual scalar link, no alternative
    # noise/bandwidth/distance interpretation silently introduced.
    error = 0.
    for time_index in (0, 4999, 9999):
        for user in (0, 14, 29):
            for server in range(4):
                expected = link.bits_per_second(positions[time_index, user], edges[server])
                error = max(error, abs(rates[time_index, user, server]-expected)/expected)
    assert error < 1e-12
    return rates, error


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_feasibility')
    attempt.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), BASE / 'README_VISUAL_FEASIBILITY_中文.md'] + [BASE / 'scripts' / name for name in
        ('visual_upload_link.py', 'visual_method_features.py', 'physical_mobility.py', 'execution_service_catalog.py')]
    source_hashes = {str(path): sha(path) for path in sources}
    save_json(attempt / 'ANALYTIC_PLAN.json', {'plan': PLAN, 'source_sha256_before_analysis': source_hashes})
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        assert sha(PREVIOUS) == PREVIOUS_SHA
        prior = json.loads(PREVIOUS.read_text(encoding='utf-8'))
        protected = dict(prior['read_only_prior_evidence_sha256']) | prior['implementation_sha256'] | {str(PREVIOUS): PREVIOUS_SHA}
        assert all(sha(Path(path)) == value for path, value in protected.items())
        calibration, validation = ExecutionServiceCatalog('calibration'), ExecutionServiceCatalog('validation')
        mobility = PhysicalMobility(users=30)
        window = TrainingWindow(*PLAN['candidate_training_mobility_window_seconds'])
        edges = mobility.server_positions(window, PLAN['geometry_seed'], 4)
        scales = calibration_scales(calibration, mobility, window, edges, PROFILES)
        capacity = math.fsum(1/service for service in scales.mean_profile_service_seconds)
        payloads = {role: payload_statistics(catalog) for role, catalog in
                    (('calibration', calibration), ('validation', validation))}
        link = UploadLinkConfig()
        rates, vectorization_error = rate_grid(mobility, edges, link)
        inverse_rate = {'fastest_link_lower_bound': (1/rates.max(axis=2)).mean(axis=0),
                        'uniform_routing_diagnostic': (1/rates).mean(axis=(0, 2)),
                        'slowest_link_upper_bound': (1/rates.min(axis=2)).mean(axis=0)}
        rows, conditions = [], []
        for role, payload in payloads.items():
            bits = payload['mean_bytes'] * 8
            for rho in PLAN['rho']:
                total_rate = rho * capacity
                by_policy = {}
                for kind, inverse in inverse_rate.items():
                    loads = total_rate/30 * bits * inverse
                    assert np.isfinite(loads).all() and np.all(loads >= 0)
                    by_policy[kind] = {'mean_upload_work_ratio': float(loads.mean()),
                        'min_user_upload_work_ratio': float(loads.min()), 'max_user_upload_work_ratio': float(loads.max()),
                        'users_expected_work_at_least_available_time': int(np.sum(loads >= 1)),
                        'linear_bandwidth_scale_for_all_users_at_most_0p8': float(loads.max()/.8)}
                    rows.extend({'payload_role': role, 'rho': rho, 'routing_bound': kind, 'user': user,
                        'mean_upload_seconds_per_request': float(bits*inverse[user]),
                        'unconditional_requests_per_second': total_rate/30,
                        'expected_upload_work_ratio': float(loads[user]),
                        'expected_work_at_least_available_time': bool(loads[user] >= 1)} for user in range(30))
                assert (by_policy['fastest_link_lower_bound']['max_user_upload_work_ratio'] <=
                        by_policy['uniform_routing_diagnostic']['max_user_upload_work_ratio'] <=
                        by_policy['slowest_link_upper_bound']['max_user_upload_work_ratio'])
                conditions.append({'payload_role': role, 'rho': rho, 'total_unconditional_requests_per_second': total_rate,
                    'unconditional_frames_per_user_second': total_rate/30,
                    'mean_payload_traffic_per_user_megabits_second': total_rate/30*bits/1e6, 'upload_work_bounds': by_policy})
        with (attempt / 'PER_USER_UPLOAD_DIAGNOSTICS.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        candidate = PLAN['training_budget_candidate']
        budget = {'total_decisions': candidate['episodes'] * candidate['decisions_per_episode'],
            'decision_generation_seconds': candidate['episodes'] * candidate['generation_seconds_per_episode'],
            'independent_episode_execution_seconds_including_fixed_drains': candidate['episodes'] *
                (candidate['generation_seconds_per_episode'] + candidate['fixed_drain_seconds_per_episode']),
            'updates_candidate_not_executed': candidate['episodes'],
            'gamma_per_50ms': .99 ** .05, 'lambda_per_50ms': .95 ** .05,
            'gamma_half_life_seconds': math.log(.5)/math.log(.99),
            'lambda_half_life_seconds': math.log(.5)/math.log(.95),
            'gamma_over_50_second_episode': .99 ** 50, 'lambda_over_50_second_episode': .95 ** 50,
            'full_episode_float32_critic_storage_bytes': 1000 * 30 * 705 * 4}
        assert budget['total_decisions'] == 10000 and budget['decision_generation_seconds'] == 500
        assert budget['independent_episode_execution_seconds_including_fixed_drains'] == 800
        binding = source_bindings()
        # Read-only provenance for the MV continuity design question, not a
        # claim that the new candidate already matches old full training.
        historical = next(Path(path) for path in binding if Path(path).name == 'paper_policy_baselines.py')
        text = historical.read_text(encoding='utf-8')
        assert 'else MVReward(args.dsr_mu)' in text and 'rewarder(float(x)) for x in result.per_user_mean_latency' in text
        historical_mv = {'source_sha256': binding, 'statistic_created_before_continuous_training_loop': True,
            'new_closed_episode_reset_candidate_not_yet_frozen_or_fixed': True}
        assert 'torch' not in sys.modules
        evidence = protected | calibration.source_sha256 | validation.source_sha256 | mobility.source_hashes | source_hashes
        assert all(sha(Path(path)) == value for path, value in evidence.items())
        report = {'at': datetime.now().astimezone().isoformat(), 'diagnostic_completed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'plan': PLAN, 'capacity_marker_requests_per_second': capacity,
            'calibration_mean_profile_service_seconds': dict(zip(PROFILES, scales.mean_profile_service_seconds)),
            'edge_positions_meters': edges, 'payload_statistics': payloads,
            'decision_grid_all_links_megabits_per_second_min_median_max': [float(x)/1e6 for x in
                (rates.min(), np.median(rates), rates.max())], 'scalar_vectorization_max_relative_error': vectorization_error,
            'conditions': conditions, 'candidate_budget_and_physical_discount': budget,
            'historical_MV_statistic_continuity_design_question': historical_mv,
            'protected_input_and_source_sha256': evidence, 'per_user_table_sha256': sha(attempt / 'PER_USER_UPLOAD_DIAGNOSTICS.csv'),
            'test_measurements_accessed': False, 'network_parameters_changed': False, 'rho_changed': False,
            'method_performance_measured': False, 'FIFO_or_upload_queue_replay_executed': False,
            'policy_training_cells': 0, 'formal_policy_test_cells': 0, 'formal_protocol_frozen': False,
            'independent_full_entry_review': False,
            'scope': 'Analytic workload/serialized-upload work bounds and candidate budget units only; not queue stability, learned policy, wireless measurement, service remeasurement or deployment throughput'}
        save_json(attempt / 'RESULT.json', report)
        lines = ['# 视觉负载与上传可行性诊断（非正式结果）', '',
            f'校准均值的4虚拟服务器能力标尺：{capacity:.4f}请求/秒；不是四设备吞吐实测。',
            f"JPEG均值：校准{payloads['calibration']['mean_bytes']:.1f}字节，验证{payloads['validation']['mean_bytes']:.1f}字节。", '',
            '| 载荷池 | rho | 总请求/秒 | 帧/用户/秒 | 最快链路最大工作比 | 均匀路由最大工作比 | 最慢链路最大工作比 | 最快下界≥1的用户数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in conditions:
            b = row['upload_work_bounds']
            lines.append(f"| {row['payload_role']} | {row['rho']:.2f} | {row['total_unconditional_requests_per_second']:.3f} | "
                f"{row['unconditional_frames_per_user_second']:.3f} | {b['fastest_link_lower_bound']['max_user_upload_work_ratio']:.3f} | "
                f"{b['uniform_routing_diagnostic']['max_user_upload_work_ratio']:.3f} | {b['slowest_link_upper_bound']['max_user_upload_work_ratio']:.3f} | "
                f"{b['fastest_link_lower_bound']['users_expected_work_at_least_available_time']} |")
        unavoidable = max(row['upload_work_bounds']['fastest_link_lower_bound']['users_expected_work_at_least_available_time'] for row in conditions)
        lines += ['', f'最快链路下界≥1用户数最高为{unavoidable}。如非零，存在该窗期望上传工作缺口；不通过改标签、删图或按算法成绩调参隐藏。',
            '均匀／最慢链路负荷大于1不是所有策略必然失败的证明；这些是有限窗口期望工作边界，不是实际队列稳定性或尾时延验收。', '',
            '候选共同预算：10×1000决策，500秒动作到达窗；加10×30秒固定drain后共800秒独立episode执行。尚未冻结或运行；不把800秒说成连续部署轨迹。',
            f"物理折扣gamma(.05s)={budget['gamma_per_50ms']:.8f}、lambda(.05s)={budget['lambda_per_50ms']:.8f}；半衰期分别{budget['gamma_half_life_seconds']:.3f}/{budget['lambda_half_life_seconds']:.3f}秒。", '',
            '历史MV统计器在连续训练循环外创建；新闭合episode重置候选不等于历史完整目标，必须在正式训练前明确／实现跨episode连续规则或披露重置适配。',
            '多帧／用户／秒只能作为高频视觉帧压力场景，不代表校园拍照典型业务。2MHz×30正交／无干扰网络是假设，不是实测。',
            '本次未修改网络、rho、模型、服务测量或测试协议；没有Torch、策略训练、算法性能、FIFO重放、独立完整入口审查。下一步按工程原则做上传／队列预实验与共同设置冻结，不能按QLA优势选择。']
        (attempt / '负载与上传可行性诊断_中文.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('VISUAL_ANALYTIC_FEASIBILITY_DIAGNOSTIC_COMPLETE', attempt)
        print(json.dumps({'capacity_marker': capacity, 'conditions': conditions}, ensure_ascii=False))
    except Exception as error:
        save_json(attempt / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(), 'error': repr(error),
            'policy_training_cells': 0, 'formal_policy_test_cells': 0})
        raise


if __name__ == '__main__':
    main()
