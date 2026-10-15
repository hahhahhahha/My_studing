"""Static historical bindings plus EXPLICIT SYNTHETIC action/buffer fixtures."""
from __future__ import annotations
from windows_runtime import initialize
initialize()

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

from visual_learned_bindings import source_bindings
from visual_learning_buffer import (map_historical_action_output, VisualEpisodeBuffer,
                                    HistoricalMAPPOBatchFacade)
from visual_neural_controller import VisualOptimizerSettings

BASE = Path(__file__).resolve().parents[1]


def refused(callback):
    try:
        callback()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError('Action/buffer boundary was not enforced')


def fixture_data():
    n, m = 30, 4
    latent = np.tile(np.array([-.3, .2, .4, -.1]), (n, 1))
    ports = np.exp(latent - latent.max(axis=1, keepdims=True))
    ports /= ports.sum(axis=1, keepdims=True)
    # logp/value here are deliberately artificial interface fixture values.
    logp, values = np.full(n, -2.), np.arange(n) / 100.
    qla = map_historical_action_output('QLA-MAPPO', (ports, latent, logp), n, m, values)
    ppo = map_historical_action_output('PPO-MV-A', (latent, logp, values, ports), n, m)
    assert all(np.array_equal(getattr(qla, name), getattr(ppo, name))
               for name in ('portfolios', 'latent_actions', 'log_probabilities', 'values'))
    assert not qla.portfolios.flags.writeable and not qla.values.flags.writeable
    refused(lambda: map_historical_action_output('QLA-MAPPO', (ports, latent, logp), n, m))
    refused(lambda: map_historical_action_output('PPO-MV-A', (latent, logp, values, ports), n, m, values))
    refused(lambda: map_historical_action_output('PPO-MV-A', (latent, logp[:-1], values, ports), n, m))
    wrong = ports[:, ::-1]
    refused(lambda: map_historical_action_output('QLA-MAPPO', (wrong, latent, logp), n, m, values))
    packed = SimpleNamespace(actor=np.ones((n, 19), dtype=np.float32),
                             critic=np.arange(n*705, dtype=np.float32).reshape(n, 705) / 1000)
    sha = hashlib.sha256(packed.critic.tobytes()).hexdigest()
    transitions = []
    for index in (1, 2, 3):
        transitions.append({'index': index, 'time_seconds': index * .05,
            'end_seconds': (index + 1) * .05, 'pre_critic_sha256': sha, 'portfolios': qla.portfolios.tolist(),
            'reward': {'team_reward': -.1 * index, 'per_user_rewards': [-.1 * index] * n}})
    transitions[-1].update(terminal_closing_applied=True, terminal_done=[True] * n,
                            terminal_bootstrap_value=[0.] * n)
    buffer = VisualEpisodeBuffer(n, m, 19, 705, 3)
    refused(lambda: buffer.add_action(2, .05, .1, packed, qla))
    for row in transitions:
        buffer.add_action(row['index'], row['time_seconds'], row['end_seconds'], packed, qla)
    refused(lambda: buffer.add_action(4, .2, .25, packed, qla))  # no hidden overwrite/truncation
    broken = deepcopy(transitions)
    broken[0]['portfolios'][0] = [1., 0., 0., 0.]
    refused(lambda: buffer.prepare_closed_episode(broken))
    broken = deepcopy(transitions)
    broken[0]['pre_critic_sha256'] = 'not-the-stored-critic'
    refused(lambda: buffer.prepare_closed_episode(broken))
    broken = deepcopy(transitions)
    broken[-1]['terminal_closing_applied'] = False
    refused(lambda: buffer.prepare_closed_episode(broken))
    # Input mutation cannot change stored behavior features/critic.
    packed.actor[:] = 9.
    packed.critic[:] = 9.
    prepared = buffer.prepare_closed_episode(transitions)
    assert prepared.reward_semantics == 'QLA-shared-receipt'
    assert np.all(prepared.time_user_fields['actor_obs'] == 1.)
    assert np.array_equal(prepared.time_user_fields['value'][0], qla.values)
    assert prepared.advantages.shape == (3, 30) and prepared.returns.shape == (3, 30)
    flat = prepared.flat()
    assert flat['actor_obs'].shape == (90, 19) and flat['critic_obs'].shape == (90, 705)
    assert np.array_equal(flat['user_id'], np.tile(np.arange(n), 3))
    assert np.array_equal(flat['slot'], np.repeat([1, 2, 3], n))
    refused(lambda: buffer.prepare_closed_episode(transitions))
    facade = HistoricalMAPPOBatchFacade(prepared)
    arrays = facade.arrays()
    adv, returns = facade.supplied_gae(arrays)
    assert len(facade) == 90 and arrays['aux_target'].shape == (90, 3)
    assert np.allclose(adv, prepared.advantages.reshape(-1), atol=1e-7, rtol=0.)
    assert np.allclose(returns, prepared.returns.reshape(-1), atol=1e-7, rtol=0.)
    refused(lambda: facade.supplied_gae(dict(arrays)))
    facade.clear()
    assert len(facade) == 0
    refused(facade.arrays)
    refused(facade.clear)
    return {'synthetic_users': n, 'synthetic_steps': 3, 'synthetic_action_dimensions': m,
            'time_user_GAE_shape': [3, 30], 'flattened_samples_after_GAE': 90,
            'source_tuple_layouts_checked': True, 'copied_pre_action_data': True,
            'capacity_order_critic_hash_and_terminal_guards': True,
            'one_shot_prepared_batch_guard': True,
            'model_act_value_update_called': False, 'artificial_logp_and_values': True}


def main():
    attempt = BASE / 'visual_calibrated_policy/engineering_attempts' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_learning_buffer')
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        assert sys.version.split()[0] == '3.13.5' and np.__version__ == '2.2.6'
        binding = source_bindings()
        static_classes = {}
        for path in binding:
            tree = ast.parse(Path(path).read_text(encoding='utf-8'))
            static_classes[path] = {node.name: [child.name for child in node.body
                if isinstance(child, ast.FunctionDef)] for node in tree.body if isinstance(node, ast.ClassDef)}
        assert any(all(name in classes.get('MAPPOPolicy', []) for name in
            ('act', 'value', 'update', '_compute_gae')) for classes in static_classes.values())
        assert any(all(name in classes.get('SharedGaussianPPO', []) for name in ('act', 'distribution'))
                   for classes in static_classes.values())
        tree = ast.parse((BASE / 'scripts/visual_neural_controller.py').read_text(encoding='utf-8'))
        top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert not any(isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names)
                       for node in top_level_imports)
        checks = fixture_data()
        assert 'torch' not in sys.modules
        sources = [Path(__file__), BASE / 'README_VISUAL_LEARNING_CONTROLLER_中文.md'] + [BASE / 'scripts' / name for name in (
            'visual_learning_buffer.py', 'visual_neural_controller.py', 'visual_learned_bindings.py',
            'visual_temporal_gae.py', 'visual_terminal_accounting.py')]
        report = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch_imported': False},
            'synthetic_interface_fixture': checks, 'static_historical_class_methods': static_classes,
            'historical_source_sha256': binding, 'candidate_optimizer_settings': asdict(VisualOptimizerSettings()),
            'implementation_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'actual_model_controller_constructed': False, 'actual_actor_or_critic_forward': False,
            'actual_optimizer_update': False, 'episode_length_frozen': False, 'full_training_collector_connected': False,
            'PPO_MV_reward_adapter_connected': False, 'reward_semantics_guard_prepared': True,
            'formal_protocol_frozen': False, 'independent_complete_entry_review': False,
            'policy_training_cells': 0, 'formal_policy_test_cells': 0,
            'scope': 'Static real-source bindings and explicitly artificial NumPy action/buffer fixtures only, not neural model smoke, update, rollout, training or performance'}
        (attempt / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('VISUAL_LEARNING_BUFFER_PREPARATION_OK', attempt)
    except Exception as error:
        (attempt / 'FAILURE.json').write_text(json.dumps({'at': datetime.now().astimezone().isoformat(),
            'error': repr(error), 'policy_training_cells': 0, 'formal_policy_test_cells': 0}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
