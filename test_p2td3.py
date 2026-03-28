import unittest

import numpy as np

from p2td3 import P2TD3Agent, ReplayBuffer


class P2TD3Tests(unittest.TestCase):
    def test_replay_buffer_add_and_sample(self) -> None:
        buf = ReplayBuffer(capacity=16)
        for i in range(8):
            s = np.array([i, i + 1], dtype=np.float32)
            a = np.array([0.7, 0.3], dtype=np.float32)
            r = float(i)
            ns = s + 1
            d = float(i % 2)
            buf.add(s, a, r, ns, d)

        self.assertEqual(len(buf), 8)
        states, actions, rewards, next_states, dones = buf.sample(4)
        self.assertEqual(states.shape, (4, 2))
        self.assertEqual(actions.shape, (4, 2))
        self.assertEqual(rewards.shape, (4, 1))
        self.assertEqual(next_states.shape, (4, 2))
        self.assertEqual(dones.shape, (4, 1))

    def test_agent_select_action_and_train_step(self) -> None:
        np.random.seed(0)
        agent = P2TD3Agent(state_dim=3, action_dim=2, hidden_dim=32)
        state = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        action = agent.select_action(state)
        self.assertEqual(action.shape, (2,))
        self.assertAlmostEqual(float(np.sum(action)), 1.0, places=5)
        self.assertTrue(np.all(action >= 0.0))

        buf = ReplayBuffer(capacity=128)
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.array([0.6, 0.4], dtype=np.float32)
            r = float(np.random.randn())
            ns = np.random.randn(3).astype(np.float32)
            d = 0.0
            buf.add(s, a, r, ns, d)

        stats = agent.train_step(buf, batch_size=16)
        self.assertIsNotNone(stats)
        self.assertIn("critic1_loss", stats)
        self.assertIn("critic2_loss", stats)
        self.assertIn("actor_loss", stats)


if __name__ == "__main__":
    unittest.main()
