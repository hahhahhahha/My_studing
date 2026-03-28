import unittest

from train import TrainingConfig, state_to_vector, train


class _FakeEnv:
    def __init__(self, done_after_slot: int = 3) -> None:
        self.num_nodes = 2
        self._done_after_slot = done_after_slot
        self._slot = 0

    def reset(self):
        self._slot = 0
        return self._state(), {}

    def step(self, action):
        self._slot += 1
        done = self._slot >= self._done_after_slot
        reward = 1.0
        return self._state(), reward, done, {"action_seen": action}

    def _state(self):
        return {
            "time_slot": self._slot,
            "bandwidth": [[0.0, 1.0], [1.0, 0.0]],
            "nodes": [
                {"AQ": ["a"], "CQ": [], "PZ": None, "SZ": None},
                {"AQ": [], "CQ": ["b"], "PZ": "pz", "SZ": None},
            ],
            "links": [
                {"queue_task_ids": ["t1"], "queue_remaining_sizes": [0.5]},
            ],
        }


class _FakeAgent:
    def __init__(self) -> None:
        self.select_calls = 0
        self.train_calls = 0

    def select_action(self, state, noise_std=0.0):
        self.select_calls += 1
        return [0.8, 0.2]

    def train_step(self, replay_buffer, batch_size=1):
        self.train_calls += 1
        sampled = replay_buffer.sample(batch_size)
        assert len(sampled) == 5
        assert len(sampled[0]) == batch_size
        return {"critic1_loss": 1.0, "critic2_loss": 1.0, "actor_loss": 0.5}


class TrainLoopTests(unittest.TestCase):
    def test_state_to_vector_contains_expected_parts(self) -> None:
        state = _FakeEnv()._state()
        vec = state_to_vector(state)
        self.assertGreater(len(vec), 0)
        self.assertEqual(vec[0], 0.0)

    def test_train_runs_episode_time_slot_loop_and_trains(self) -> None:
        env = _FakeEnv(done_after_slot=3)
        agent = _FakeAgent()
        cfg = TrainingConfig(
            episodes=2,
            max_time_slots=10,
            replay_capacity=100,
            batch_size=2,
            warmup_steps=2,
            exploration_noise=0.0,
        )
        result = train(env, agent, cfg)

        # Each episode stops after 3 slots due to done signal.
        self.assertEqual(result["total_steps"], 6)
        self.assertEqual(len(result["episode_rewards"]), 2)
        self.assertGreaterEqual(agent.select_calls, 6)
        self.assertGreater(result["train_steps"], 0)
        self.assertIsNotNone(result["last_train_stats"])


if __name__ == "__main__":
    unittest.main()
