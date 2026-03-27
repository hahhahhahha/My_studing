import unittest

from p2ts import (
    PreemptionResult,
    PriorityWeights,
    QueueTask,
    RunningTask,
    execution_order_after_evaluation,
    execution_priority,
    preemptive_evaluation,
    reorder_cq_by_priority,
)


class P2TSTests(unittest.TestCase):
    def test_priority_formula_eq7(self) -> None:
        weights = PriorityWeights(0.4, 0.2, 0.2, 0.2)
        task = QueueTask(
            task_id="t1",
            task_type_score=0.5,
            tolerable_delay=10.0,
            compute_delay=5.0,
            arrival_slot=2,
            left_delay=9.0,
        )
        # e=0.5, s=0.1, g=0.2, u=(7-2)/10=0.5 => 0.4*0.5+0.2*0.1+0.2*0.2+0.2*0.5=0.36
        self.assertAlmostEqual(execution_priority(task, current_slot=7, weights=weights), 0.36)

    def test_reorder_cq_by_priority(self) -> None:
        weights = PriorityWeights(0.7, 0.1, 0.1, 0.1)
        tasks = [
            QueueTask("low", 0.1, 10.0, 5.0, 0, 8.0),
            QueueTask("high", 0.9, 10.0, 5.0, 0, 8.0),
            QueueTask("mid", 0.5, 10.0, 5.0, 0, 8.0),
        ]
        ordered = reorder_cq_by_priority(tasks, current_slot=5, weights=weights)
        self.assertEqual([t.task_id for t in ordered], ["high", "mid", "low"])

    def test_preemptive_evaluation_situation_a_preempt_first(self) -> None:
        cq = [
            QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0),
            QueueTask("q2", 0.8, 10.0, 1.0, 0, 10.0),
        ]
        pz = RunningTask("pz", remaining_compute_delay=5.0, left_delay=8.0)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=None)

        self.assertEqual(result.preemptive_task_ids, ["q1"])
        self.assertEqual(result.regular_task_ids, ["q2"])
        self.assertFalse(result.discarded_target)
        self.assertEqual(result.situation, "A")

        order = execution_order_after_evaluation(result, cq, pz_task=pz)
        self.assertEqual(order, ("q1", "pz", "q2"))

    def test_preemptive_evaluation_situation_a_discard_target(self) -> None:
        cq = [QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0)]
        pz = RunningTask("pz", remaining_compute_delay=5.0, left_delay=4.0)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=None)

        self.assertTrue(result.discarded_target)
        self.assertEqual(result.preemptive_task_ids, [])
        self.assertEqual(result.regular_task_ids, ["q1"])
        self.assertEqual(execution_order_after_evaluation(result, cq, pz_task=pz), ("q1",))

    def test_preemptive_evaluation_situation_b(self) -> None:
        cq = [
            QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0),
            QueueTask("q2", 0.8, 10.0, 1.0, 0, 20.0),
        ]
        pz = RunningTask("pz", remaining_compute_delay=1.0, left_delay=30.0)
        sz = RunningTask("sz", remaining_compute_delay=2.0, left_delay=5.0)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=sz)

        self.assertEqual(result.preemptive_task_ids, ["q1"])
        self.assertEqual(result.regular_task_ids, ["q2"])
        self.assertFalse(result.discarded_target)
        self.assertEqual(result.situation, "B")
        order = execution_order_after_evaluation(result, cq, pz_task=pz, sz_task=sz)
        self.assertEqual(order, ("pz", "q1", "sz", "q2"))

    def test_no_trigger_when_no_pz(self) -> None:
        cq = [QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0)]
        result = preemptive_evaluation(cq, pz_task=None, sz_task=None)
        self.assertEqual(
            result,
            PreemptionResult(
                preemptive_task_ids=[],
                regular_task_ids=["q1"],
                discarded_target=False,
                target_task_id=None,
                situation="none",
            ),
        )


if __name__ == "__main__":
    unittest.main()
