"""No-Torch feature, source-binding and synthetic shared-reward fixtures."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from execution_service_catalog import ExecutionServiceCatalog
from physical_mobility import PhysicalMobility, TrainingWindow
from visual_environment_adapter import VisualEnvironmentAdapter
from visual_learned_bindings import source_bindings, VisualPolicyDimensions
from visual_method_features import calibration_scales, pack_features
from visual_request_schedule import generate_schedule
from visual_shared_reward import CompletedReceipt, RewardOutcomeBatch, VisualSharedReward

BASE = Path(__file__).resolve().parents[1]
PROFILES = ('CPU-1T', 'CPU-1T', 'CPU-4T', 'CPU-4T')


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError):
        return
    raise AssertionError('Scientific/feature boundary was not enforced')


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_method_features')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        binding = source_bindings()
        class_names = {}
        for path in binding:
            tree = ast.parse(Path(path).read_text(encoding='utf-8'))
            class_names[path] = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        assert any('EdgeWisePriceGaussianActor' in names and 'MAPPOPolicy' in names for names in class_names.values())
        assert any('SharedGaussianPPO' in names for names in class_names.values())
        calibration = ExecutionServiceCatalog('calibration')
        validation = ExecutionServiceCatalog('validation')
        mobility = PhysicalMobility(users=30)
        window = TrainingWindow(0., 500.)  # engineering geometry, not final split
        edges = mobility.server_positions(window, 123, 4)
        scales = calibration_scales(calibration, mobility, window, edges, PROFILES)
        refused(lambda: calibration_scales(validation, mobility, window, edges, PROFILES))
        schedule = generate_schedule(validation, 30, 80, .05, 123, 30.)
        env = VisualEnvironmentAdapter(schedule, validation, mobility, edges, PROFILES, 0.)
        view = env.observe()
        packed = pack_features(view, scales)
        assert packed.actor.shape == (30, 19) and packed.critic.shape == (30, 705)
        assert packed.actor_for('QLA-MAPPO') is packed.actor_for('PPO-MV-A')
        assert not packed.actor.flags.writeable and not packed.critic.flags.writeable
        assert np.isfinite(packed.actor).all() and np.isfinite(packed.critic).all()
        assert np.all(packed.actor[:, 13] == -1.)  # explicit startup-missing age sentinel
        critic_before = packed.critic.tobytes()
        # The pure packing API has no current action, actual-service catalog,
        # current completion receipt or future schedule argument.
        refused(lambda: pack_features(env, scales))
        permutation = (2, 0, 3, 1)
        feedback = replace(view.feedback,
            predicted_backlog_seconds=tuple(view.feedback.predicted_backlog_seconds[j] for j in permutation),
            waiting_counts=tuple(view.feedback.waiting_counts[j] for j in permutation),
            busy=tuple(view.feedback.busy[j] for j in permutation))
        users = tuple(replace(user,
            previous_portfolio=tuple(user.previous_portfolio[j] for j in permutation),
            requests=tuple(replace(request,
                predicted_service_seconds=tuple(request.predicted_service_seconds[j] for j in permutation),
                transmission_seconds=tuple(request.transmission_seconds[j] for j in permutation))
                for request in user.requests)) for user in view.users)
        permuted_view = replace(view, feedback=feedback, users=users)
        permuted_scales = replace(scales, profiles=tuple(PROFILES[j] for j in permutation),
            edge_positions=tuple(edges[j] for j in permutation),
            mean_profile_service_seconds=tuple(scales.mean_profile_service_seconds[j] for j in permutation))
        reordered = pack_features(permuted_view, permuted_scales)
        expected = packed.actor.copy()
        for start in (2, 9, 14):
            expected[:, start:start+4] = packed.actor[:, start: start+4][:, permutation]
        assert np.allclose(reordered.actor, expected, atol=1e-6, rtol=0.)
        uniform = ((.25,) * 4,) * 30
        concentrated = ((1., 0., 0., 0.),) * 30
        batch = RewardOutcomeBatch(.05, .1, (.03, 0., .02, 0.),
                (CompletedReceipt('synthetic-receipt', 0, 0., .08),))
        rewarder = VisualSharedReward(scales)
        reward = rewarder.compute(view, uniform, batch)
        q_manual = (.03 / .15 + .02 / .15) / 4
        expected_reward = -math.tanh(q_manual + .5 * .08)
        assert math.isclose(reward['team_reward'], expected_reward, abs_tol=1e-12, rel_tol=0.)
        assert np.all(reward['per_user_rewards'] == reward['per_user_rewards'][0])
        assert -1 <= reward['team_reward'] <= 0 and reward['h_after'] == 0.
        other = VisualSharedReward(scales)
        concentration = other.compute(view, concentrated, batch)
        assert math.isclose(concentration['expected_concentration'], 1., abs_tol=1e-12)
        assert math.isclose(concentration['h_after'], .05 * .05 * (1 - 1/3), abs_tol=1e-12)
        assert packed.critic.tobytes() == critic_before  # post-action state cannot mutate t-critic
        env.apply(uniform)
        next_view = env.observe()
        h_before = rewarder.h_queue
        refused(lambda: rewarder.compute(next_view, uniform,
            RewardOutcomeBatch(.1, .15, (0.,) * 4, (CompletedReceipt('future', 0, .01, .2),))))
        refused(lambda: rewarder.compute(next_view, uniform,
            RewardOutcomeBatch(.1, .15, (0.,) * 4, (CompletedReceipt('synthetic-receipt', 0, .01, .12),))))
        assert rewarder.h_queue == h_before
        assert 'torch' not in sys.modules  # no model construction or CPU surrogate
        dimensions = VisualPolicyDimensions(30, 4, packed.actor.shape[1], packed.critic.shape[1], 30)
        implementation = [Path(__file__), BASE / 'README_VISUAL_METHODS_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_method_features.py', 'visual_shared_reward.py', 'visual_learned_bindings.py',
            'visual_environment_adapter.py', 'metadata_service_predictor.py', 'execution_service_catalog.py')]
        report = dict(at=datetime.now().astimezone().isoformat(), passed=True,
            runtime={'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            actor_shape=list(packed.actor.shape), critic_shape=list(packed.critic.shape),
            learning_inputs_identical=True, current_action_not_critic_argument=True,
            permutation_feature_check=True, calibration_only_scales=asdict(scales),
            prepared_dimensions=asdict(dimensions), historical_model_sources=binding,
            historical_class_names_read_not_executed=class_names,
            synthetic_reward_fixture={'team_reward': reward['team_reward'],
                    'manual_formula_reward': expected_reward, 'concentrated_h_after': concentration['h_after']},
            calibration_sources=calibration.source_sha256,
            implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation},
            real_receipt_hook_connected=False, model_factory_executed=False,
            actual_model_smoke=False, independent_complete_entry_review=False,
            formal_protocol_frozen=False, policy_training_cells=0, formal_policy_test_cells=0,
            scope='Feature/source-binding and synthetic receipt-reward engineering checks only; no learned policy, performance or final audit')
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_METHOD_FEATURES_ENGINEERING_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'error': repr(error), 'at': datetime.now().astimezone().isoformat(),
             'formal_policy_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
