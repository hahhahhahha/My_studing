"""Real frozen checkpoint forwards on artificial cached inputs, not a system run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemlab.baselines import METHODS, FrozenBaseline
from systemlab.common import read, source_receipt, write_new


def check_one(method, config, bundle):
    actor = FrozenBaseline(method, config, bundle)
    now = 1792051202000000000
    states = {e['id']: dict(predicted_wait_s=j * .003, service_estimate_s=.03+j*.0001,
        waiting_count=j % 4, busy=bool(j % 2), source_epoch_ns=now-100000000,
        received_epoch_ns=now-50000000) for j, e in enumerate(config['edges'])}
    class Cache:
        def observation(self, user):
            return states
    actor.observe_cache(Cache(), now)
    actor.observe_released({u: 1 for u in range(30)})
    portfolios = []
    for user in range(30):
        weights, details = actor.portfolio([dict(payload_bytes=10000)], states, now, user)
        portfolios.append(dict(user=user, weights=weights, details=details))
    return dict(scope='synthetic_cached_input_actual_frozen_inference_not_system_performance',
        method=method, users_checked=30, portfolio_dimensions=35, receipt=actor.receipt,
        verification=actor.verify_frozen(), portfolios=portfolios, training_calls=0,
        six_baselines_deployed=False, formal_ready=False, source_sha256=source_receipt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bundles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=list(METHODS))
    args = parser.parse_args()
    if args.method:
        write_new(args.output, check_one(args.method, read(args.config), args.bundles/(args.method+'.json')))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for method in METHODS:
        command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config),
            '--bundles', str(args.bundles), '--output', str(args.output/(method+'.json')), '--method', method]
        result = subprocess.run(command, capture_output=True, text=True)
        write_new(args.output/(method+'_PROCESS.json'), dict(command=command, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        results.append(dict(method=method, passed=result.returncode == 0))
        if result.returncode:
            write_new(args.output/'FAILURE.json', dict(results=results, automatic_retry=False))
            raise RuntimeError('Frozen inference failed; preserve this attempt: ' + method)
    summary = dict(results=results, actual_checkpoint_methods=4, rule_methods=2, training_calls=0,
        scope='artificial_input_model_functionality_only', six_baselines_deployed=False, formal_ready=False)
    write_new(args.output/'SUMMARY.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
