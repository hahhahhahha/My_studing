"""Observable-policy -> execute -> real receipts -> shared reward engineering loop.

No neural model construction, optimizer, training or retained-test entry here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import numpy as np

from visual_method_features import pack_features
from visual_receipt_environment import VisualReceiptEnvironment
from visual_shared_reward import VisualSharedReward


def rule_engineering_episode(environment, scales, policy, fixed_drain_seconds):
    if not isinstance(environment, VisualReceiptEnvironment):
        raise TypeError('Strict receipt environment required')
    rewarder = VisualSharedReward(scales)
    observations, transitions = [], []
    for _ in range(environment._schedule.decisions):
        pre = environment.observe()
        observations.append(asdict(pre))
        packed = pack_features(pre, scales, prior_h_queue=rewarder.h_queue)
        critic_digest = hashlib.sha256(packed.critic.tobytes()).hexdigest()
        portfolios = policy.portfolios(pre, packed)  # NO environment argument
        environment.apply(portfolios)
        outcomes = environment.settle_action()
        reward = rewarder.compute(pre, portfolios, outcomes)
        if critic_digest != hashlib.sha256(packed.critic.tobytes()).hexdigest():
            raise RuntimeError('Post-action reward mutated the current pre-action critic')
        if not np.all(reward['per_user_rewards'] == reward['per_user_rewards'][0]):
            raise RuntimeError('Shared reward mismatch')
        transitions.append({'index': pre.index, 'time_seconds': pre.physical_seconds,
            'end_seconds': outcomes.end_seconds, 'pre_critic_sha256': critic_digest,
            'portfolios': portfolios.tolist(), 'outcome_batch': asdict(outcomes),
            'reward': {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in reward.items()}})
    final = environment.finish(fixed_drain_seconds)
    accounting = final['controller_accounting']
    if (accounting['settled_completed'] + accounting['drain_only_completed'] != final['tasks']['completed']
            or accounting['settled_dropped'] + accounting['drain_only_dropped'] != final['tasks']['dropped']
            or sum(row['reward']['completed_count'] for row in transitions) != accounting['settled_completed']
            or sum(row['reward']['observed_drop_count'] for row in transitions) != accounting['settled_dropped']):
        raise RuntimeError('Reward/controller/final task accounting mismatch')
    # Saved past receipts, not predicted completion times, determine this sum.
    reward_latency_sum = math.fsum(row['completed_seconds'] - row['actual_arrival_seconds']
        for transition in transitions for row in transition['outcome_batch']['completed'])
    return {'observations': observations, 'transitions': transitions, 'final': final,
            'completed_receipt_latency_sum_seconds': reward_latency_sum,
            'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Rule-adapter engineering only; no independent audit or QLA/PPO outcome'}
