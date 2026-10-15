"""Bind the six original baseline assets to a target config; no model launch."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemlab.baselines import METHODS, register_baseline
from systemlab.common import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for method in METHODS:
            register_baseline(method, args.config, args.assets, args.output / (method + '.json'))
        write_new(args.output / 'REGISTRATION.json', dict(methods=list(METHODS),
            actual_inference_calls=0, six_baselines_deployed=False, formal_ready=False))
    except Exception as error:
        write_new(args.output / 'FAILURE.json', dict(error=repr(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
