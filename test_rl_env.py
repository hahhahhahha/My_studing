import unittest

from p2ts import PriorityWeights, QueueTask
from rl_env import P2MEnvironment, RewardWeights


class RLEnvironmentTests(unittest.TestCase):
    def test_reset_and_state_shape(self) -> None:
        env = P2MEnvironment(
            num_nodes=2,
            bandwidth_matrix=[[1.0, 0.5], [0.5, 1.0]],
            priority_weights=PriorityWeights(0.4, 0.2, 0.2, 0.2),
            reward_weights=RewardWeights(1.0, 0.0, 0.0),
            arrivals_by_slot=[],
        )
        state, info = env.reset()
        self.assertEqual(state["time_slot"], 0)
        self.assertEqual(len(state["nodes"]), 2)
        self.assertEqual(info, {})

    def test_step_computes_reward_components(self) -> None:
        task = QueueTask(
            task_id="t1",
            task_type_score=0.8,
            tolerable_delay=5.0,
            compute_delay=1.0,
            arrival_slot=0,
            left_delay=5.0,
        )
        env = P2MEnvironment(
            num_nodes=1,
            bandwidth_matrix=[[1.0]],
            priority_weights=PriorityWeights(0.4, 0.2, 0.2, 0.2),
            reward_weights=RewardWeights(1.0, 1.0, 1.0),
            arrivals_by_slot=[[task]],
        )

        env.reset()
        _, reward, done, info = env.step([0])

        self.assertIn("E", info)
        self.assertIn("D", info)
        self.assertIn("W", info)
        self.assertEqual(info["completed_task_ids"], ["t1"])
        self.assertEqual(info["dropped_task_ids"], [])
        self.assertAlmostEqual(info["E"], 1.0)
        self.assertAlmostEqual(info["D"], 1.0)
        self.assertAlmostEqual(info["W"], 0.0)
        self.assertAlmostEqual(reward, 0.0)  # 1*1 -1*1 -1*0
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
