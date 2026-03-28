import unittest
from statistics import pvariance

from p2ts import PriorityWeights, QueueTask
from rl_env import P2MEnvironment, RewardWeights


class RLEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _node_task_ids(node_state: dict) -> list[str]:
        ids = []
        ids.extend(node_state["AQ"])
        ids.extend(node_state["CQ"])
        if node_state["PZ"] is not None:
            ids.append(node_state["PZ"])
        if node_state["SZ"] is not None:
            ids.append(node_state["SZ"])
        return ids

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

    def test_cross_node_transfer_respects_bandwidth_queue(self) -> None:
        task = QueueTask(
            task_id="tx_task",
            task_type_score=0.6,
            tolerable_delay=10.0,
            compute_delay=1.0,
            arrival_slot=0,
            left_delay=10.0,
        )
        env = P2MEnvironment(
            num_nodes=2,
            bandwidth_matrix=[[1.0, 0.5], [0.5, 1.0]],
            priority_weights=PriorityWeights(0.4, 0.2, 0.2, 0.2),
            reward_weights=RewardWeights(1.0, 0.0, 0.0),
            arrivals_by_slot=[[task]],
            arrivals_source_by_slot=[[0]],
            transfer_size_by_task={"tx_task": 1.0},
        )

        env.reset()
        state1, _, done1, info1 = env.step([1])  # schedule to node 1, requires 2 slots for transfer with bw=0.5
        self.assertFalse(done1)
        self.assertEqual(info1["transferred_task_ids"], [])
        link01 = next(link for link in state1["links"] if link["src"] == 0 and link["dst"] == 1)
        self.assertEqual(link01["queue_task_ids"], ["tx_task"])
        self.assertAlmostEqual(link01["queue_remaining_sizes"][0], 0.5)

        state2, _, done2, info2 = env.step([])  # no new tasks, finish transfer then execute
        self.assertEqual(info2["transferred_task_ids"], ["tx_task"])
        self.assertFalse(done2)
        node1_ids_step2 = self._node_task_ids(state2["nodes"][1])
        self.assertIn("tx_task", node1_ids_step2)

        _, _, done3, info3 = env.step([])
        self.assertTrue(done3)
        self.assertEqual(info3["completed_task_ids"], ["tx_task"])

    def test_action_batch_mapping_repeats_when_shorter_than_tasks(self) -> None:
        t1 = QueueTask("t1", 0.7, 10.0, 1.0, 0, 10.0)
        t2 = QueueTask("t2", 0.5, 10.0, 1.0, 0, 10.0)
        t3 = QueueTask("t3", 0.4, 10.0, 1.0, 0, 10.0)
        env = P2MEnvironment(
            num_nodes=2,
            bandwidth_matrix=[[1.0, 1.0], [1.0, 1.0]],
            priority_weights=PriorityWeights(0.4, 0.2, 0.2, 0.2),
            reward_weights=RewardWeights(1.0, 0.0, 0.0),
            arrivals_by_slot=[[t1, t2, t3]],
            arrivals_source_by_slot=[[0, 0, 0]],
        )
        env.reset()
        state, _, _, _ = env.step([1])  # single action is broadcast to all incoming tasks

        node1_ids = self._node_task_ids(state["nodes"][1])
        self.assertTrue(any(tid in node1_ids for tid in ["t1", "t2", "t3"]))

    def test_load_variance_uses_remaining_compute_workload_not_task_count(self) -> None:
        heavy = QueueTask("heavy", 0.6, 20.0, 5.0, 0, 20.0)
        env = P2MEnvironment(
            num_nodes=2,
            bandwidth_matrix=[[1.0, 1.0], [1.0, 1.0]],
            priority_weights=PriorityWeights(0.4, 0.2, 0.2, 0.2),
            reward_weights=RewardWeights(0.0, 0.0, 1.0),
            arrivals_by_slot=[[heavy]],
            arrivals_source_by_slot=[[0]],
        )
        env.reset()
        _, _, _, info1 = env.step([0])

        # After one slot:
        # node0 runs the only task, remaining compute workload = 4.0
        # node1 has no workload = 0.0
        # W should be variance of workloads, not variance of task counts.
        expected_w = pvariance([4.0, 0.0])
        self.assertAlmostEqual(info1["W"], expected_w)


if __name__ == "__main__":
    unittest.main()
