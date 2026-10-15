"""Read-only workflow checks and bounded official-endpoint probe; no RL launch.

Provisioning/measurement/training are deliberately not hidden in this monitor.
Each future launch must have its own immutable plan and reviewed entry.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from windows_runtime import initialize
initialize()

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
V2 = ROOT / 'dac_visual_calibration_20261010_v2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def state():
    service = read(V2 / 'protocol/RETAINED_TEST_STATE.json')
    stage_paths = {
        'visual_calibrated_policy': BASE / 'visual_calibrated_policy/STATUS.json',
        'joint_reward': ROOT / 'results_qla_joint_reward_ablation_20261010_repaired/STATUS.json',
        'unseen_windows': ROOT / 'results_qla_unseen_external_windows_20261010_repaired/STATUS.json'}
    stages = {name: read(path) if path.exists() else {'status': 'not_started'} for name, path in stage_paths.items()}
    return dict(at=datetime.now().astimezone().isoformat(), service=service,
                service_measurement_ended=service['status'] in ['complete_pending_independent_audit', 'complete_audited'],
                visual_policy_stage_allowed=service['status'] == 'complete_audited',
                qla_environment_exists=(BASE / 'environment/qla_np226/Scripts/python.exe').exists(),
                alibaba2017_exists=(ROOT / 'datasets/external_traces/extracted/alibaba2017/server_usage.csv').is_file(),
                stages=stages,
                new_policy_or_external_experiments_started=any(
                    x.get('training_started') or x.get('model_tests_started') or x.get('completed_model_cells', 0)
                    or x.get('status') not in ['not_started', 'prepared_not_started', 'blocked_preflight', 'blocked_preflight_or_data']
                    for x in stages.values()))


def check():
    for relative in [
        'paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md',
        'results_qla_joint_reward_ablation_20261009/DESIGN.json',
        'results_qla_unseen_external_windows_20261009/DESIGN.json',
        'results_qla_unseen_external_windows_20261009/FROZEN_BEFORE_TEST.json']:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    joint = read(ROOT / 'results_qla_joint_reward_ablation_20261009/DESIGN.json')
    external = read(ROOT / 'results_qla_unseen_external_windows_20261009/DESIGN.json')
    assert len(joint['Full_references']) == 25
    assert external['expected_cells'] == 243 and len(external['checkpoints']) == 18
    env = read(BASE / 'protocol/QLA_ENVIRONMENT_SPEC.json')
    canonical = json.dumps(env, sort_keys=True, separators=(',', ':')).encode()
    result = subprocess.run([sys.executable, '-X', 'utf8', str(BASE / 'scripts/test_fifo_execution.py')], check=False)
    if result.returncode:
        raise RuntimeError('Synthetic FIFO semantics test failed')
    print('WORKFLOW_ENTRY_CHECK_OK ' + json.dumps(dict(environment_spec_hash=hashlib.sha256(canonical).hexdigest(),
          readiness='planning/semantic checks only; CUDA/data/policy entries pending', **state()), ensure_ascii=False))


def probe():
    import urllib.request
    folder = BASE / 'data_acquisition'
    folder.mkdir(exist_ok=True)
    # Original failed probes remain untouched. Unique evidence per new probe.
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    urls = ['https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz',
            'http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz']
    rows = []
    for url in urls:
        row = dict(url=url, at=datetime.now().astimezone().isoformat(), method='HEAD', archive_downloaded=False)
        try:
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request, timeout=15) as response:
                row.update(status=response.status, final_url=response.url, headers=dict(response.headers.items()))
        except Exception as error:
            row['error'] = repr(error)
        rows.append(row)
    evidence = dict(probes=rows, primary_source='https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/README2017.md',
                    checksum_source='https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/SHA256SUM',
                    official_checksum_name='trace_201708.tgz',
                    expected_archive_sha256='02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df',
                    bytes_downloaded=0, required_mapping_still_to_verify=True)
    with (folder / (stamp + '_ALIBABA_HEAD.json')).open('x', encoding='utf-8') as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['check', 'status', 'probe'])
    action = p.parse_args().action
    if action == 'check':
        check()
    elif action == 'probe':
        probe()
    else:
        print(json.dumps(state(), ensure_ascii=False, indent=2))
