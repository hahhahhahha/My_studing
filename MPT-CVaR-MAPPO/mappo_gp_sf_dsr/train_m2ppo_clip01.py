"""Run the existing M2PPO adaptation with the paper's PPO clip epsilon.

This wrapper intentionally leaves the shared environment and model code
untouched.  It only changes clip_eps from the repository default 0.2 to the
0.1 stated in Yao et al.'s mixed-MAPPO paper.
"""

from __future__ import annotations

import json
from dataclasses import replace

from . import train as base_train


_OriginalMAPPOConfig = base_train.MAPPOConfig


def _paper_clip_config(*args, **kwargs):
    return replace(_OriginalMAPPOConfig(*args, **kwargs), clip_eps=0.1)


def main() -> None:
    args = base_train.parse_args()
    if args.method != "M2PPO":
        raise ValueError("train_m2ppo_clip01 only supports --method M2PPO")
    base_train.MAPPOConfig = _paper_clip_config
    summary = base_train.train_once(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
