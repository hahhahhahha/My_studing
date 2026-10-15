"""CPU-only numerical/source checks; full PyTorch smoke is a separate gate."""
import ast
from pathlib import Path
import unittest

import numpy as np

from paired_training_repair_core import gae_with_bootstrap, instrument_train_source, mask_action_features


class RepairTests(unittest.TestCase):
    def arrays(self, done=0.):
        return {"reward": np.array([1., 2.], dtype=np.float32),
                "value": np.array([.5, .8], dtype=np.float32),
                "done": np.array([0., done], dtype=np.float32),
                "user_id": np.array([0, 0]), "slot": np.array([1, 2])}

    def test_continuing_bootstrap(self):
        adv, ret = gae_with_bootstrap(self.arrays(), .9, .95, np.array([3.]))
        last = 2 + .9 * 3 - .8
        np.testing.assert_allclose(adv, [1 + .9 * .8 - .5 + .9 * .95 * last, last], rtol=1e-6)
        np.testing.assert_allclose(ret, adv + [.5, .8])

    def test_true_terminal_masks_bootstrap(self):
        a, _ = gae_with_bootstrap(self.arrays(1.), .9, .95, np.array([999.]))
        b, _ = gae_with_bootstrap(self.arrays(1.), .9, .95, np.array([0.]))
        np.testing.assert_array_equal(a, b)

    def test_zero_bootstrap_matches_original_formula(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "mappo_gp_sf_dsr/models.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MAPPOPolicy")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_gae")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "original_gae", "exec"), namespace)
        from types import SimpleNamespace
        policy = SimpleNamespace(cfg=SimpleNamespace(gamma=.9, gae_lambda=.95))
        expected = namespace["_compute_gae"](policy, self.arrays())
        actual = gae_with_bootstrap(self.arrays(), .9, .95, np.array([0.]))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_interleaved_users(self):
        data = {"reward": np.ones(4, dtype=np.float32), "value": np.zeros(4, dtype=np.float32),
                "done": np.zeros(4, dtype=np.float32), "user_id": np.array([0, 1, 0, 1]), "slot": np.array([1, 1, 2, 2])}
        a, _ = gae_with_bootstrap(data, 1., 1., np.array([10., 20.]))
        np.testing.assert_array_equal(a, [12., 22., 11., 21.])

    def test_action_features_only(self):
        obs = np.arange(2 * 40, dtype=np.float32).reshape(2, 40)
        result = mask_action_features(obs, n_edges=4, system_dim=8)
        np.testing.assert_array_equal(result[:, :28], obs[:, :28])
        np.testing.assert_array_equal(result[:, 28:34], 0.)
        np.testing.assert_array_equal(result[:, 34:], obs[:, 34:])
        self.assertFalse(np.shares_memory(obs, result))

    def test_critic_action_invariance_and_source_layout(self):
        source = (Path(__file__).resolve().parents[1] / "mappo_gp_sf_dsr/env.py").read_text(encoding="utf-8")
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "JointMAPPOEdgeEnv")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "critic_observations")
        namespace = {"np": np}
        exec(compile("from __future__ import annotations\n" + ast.unparse(fn), "critic_layout", "exec"), namespace)
        from types import SimpleNamespace
        feedback = SimpleNamespace(**{k: np.arange(4, dtype=np.float32) for k in
                                       ("score", "queue_task_count_norm", "available_resource_norm", "capacity_norm", "wait_norm")})
        def pstats(w):
            return SimpleNamespace(mean_entropy=w.mean(), mean_concentration=w.max(),
                                   global_demand_concentration=w.sum(), max_server_expected_demand_share=w.min(),
                                   entropy=w[:, 0], concentration=w[:, 1])
        env = SimpleNamespace(cfg=SimpleNamespace(n_users=2, n_edges=4), coord_min=0., coord_span=1.,
                              _system_summary=lambda: np.arange(8), _locations=lambda: np.ones((2, 2)),
                              _request_summary=lambda u: np.ones(3), server_feedback=lambda: feedback,
                              portfolio_stats=pstats)
        obs1 = namespace["critic_observations"](env, np.ones((2, 4)))
        obs2 = namespace["critic_observations"](env, np.full((2, 4), .25))
        self.assertFalse(np.array_equal(obs1, obs2))
        np.testing.assert_array_equal(mask_action_features(obs1, 4, 8), mask_action_features(obs2, 4, 8))

    def test_training_hook_exactly_two_boundaries(self):
        path = Path(__file__).resolve().parents[1] / "mappo_gp_sf_dsr/train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_once")
        source = "\n" + ast.get_source_segment(path.read_text(encoding="utf-8"), fn) + "\n"
        changed = instrument_train_source(source)
        self.assertEqual(changed.count("repair_update(policy, buffer"), 2)
        self.assertNotIn("policy.update(buffer)", changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
