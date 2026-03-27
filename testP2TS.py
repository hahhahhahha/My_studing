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

# Shared weights used across most tests.
_W = PriorityWeights(0.25, 0.25, 0.25, 0.25)


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
        # pz priority_score is set lower than both CQ tasks so both qualify as
        # uncertain tasks.  q1 and q2 priorities ≈ 0.5 / 0.475; pz set to 0.3.
        pz = RunningTask("pz", remaining_compute_delay=5.0, left_delay=8.0, priority_score=0.3)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=None, current_slot=0, weights=_W)

        self.assertEqual(result.preemptive_task_ids, ["q1"])
        self.assertEqual(result.regular_task_ids, ["q2"])
        self.assertFalse(result.discarded_target)
        self.assertEqual(result.situation, "A")

        order = execution_order_after_evaluation(result, cq, pz_task=pz)
        self.assertEqual(order, ("q1", "pz", "q2"))

    def test_preemptive_evaluation_situation_a_discard_target(self) -> None:
        cq = [QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0)]
        # q1 priority ≈ 0.5; pz priority_score is below that → q1 is uncertain.
        pz = RunningTask("pz", remaining_compute_delay=5.0, left_delay=4.0, priority_score=0.3)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=None, current_slot=0, weights=_W)

        self.assertTrue(result.discarded_target)
        self.assertEqual(result.preemptive_task_ids, [])
        self.assertEqual(result.regular_task_ids, ["q1"])
        self.assertEqual(execution_order_after_evaluation(result, cq, pz_task=pz), ("q1",))

    def test_preemptive_evaluation_situation_b(self) -> None:
        cq = [
            QueueTask("q1", 0.9, 10.0, 1.0, 0, 1.0),
            QueueTask("q2", 0.8, 10.0, 1.0, 0, 20.0),
        ]
        pz = RunningTask("pz", remaining_compute_delay=1.0, left_delay=30.0, priority_score=0.3)
        # sz is the target in Situation B; its priority_score must be below both
        # CQ tasks' priorities so they both qualify as uncertain tasks.
        sz = RunningTask("sz", remaining_compute_delay=2.0, left_delay=5.0, priority_score=0.3)
        result = preemptive_evaluation(cq, pz_task=pz, sz_task=sz, current_slot=0, weights=_W)

        self.assertEqual(result.preemptive_task_ids, ["q1"])
        self.assertEqual(result.regular_task_ids, ["q2"])
        self.assertFalse(result.discarded_target)
        self.assertEqual(result.situation, "B")
        order = execution_order_after_evaluation(result, cq, pz_task=pz, sz_task=sz)
        self.assertEqual(order, ("pz", "q1", "sz", "q2"))

    def test_low_priority_cq_tasks_not_uncertain(self) -> None:
        """CQ tasks with priority <= target's priority_score must not become
        preemptive tasks, even if they would satisfy the slack condition."""
        # q_high: priority ≈ 0.5, tight left_delay so it cannot wait for PZ.
        # q_low:  priority ≈ 0.1, equal to the target's score → excluded from
        #         uncertain-task evaluation and must stay as a regular task.
        q_high = QueueTask("q_high", 0.9, 10.0, 1.0, 0, 2.0)
        q_low  = QueueTask("q_low",  0.1, 10.0, 5.0, 0, 2.0)

        p_high = execution_priority(q_high, current_slot=0, weights=_W)  # ≈ 0.5
        p_low  = execution_priority(q_low,  current_slot=0, weights=_W)  # ≈ 0.1
        self.assertGreater(p_high, p_low)

        # Target (PZ) priority_score equals q_low's priority, so only q_high
        # qualifies as an uncertain task (strictly greater).
        # pz remaining=3, left_delay=10 → q_high (compute_delay=1) can fit as
        # preemptive: upsilon_A(q_high.left_delay=2)=2-3-1=-2<0 (can't wait),
        # upsilon_A(pz.left_delay=10)=10-3-1=6>=0 (target survives).
        pz = RunningTask("pz", remaining_compute_delay=3.0, left_delay=10.0,
                         priority_score=p_low)

        result = preemptive_evaluation(
            [q_high, q_low], pz_task=pz, current_slot=0, weights=_W
        )

        # q_high should be preemptive (high priority, tight left_delay).
        self.assertIn("q_high", result.preemptive_task_ids)
        # q_low must never appear in preemptive_task_ids.
        self.assertNotIn("q_low", result.preemptive_task_ids)
        # q_low must remain a regular task.
        self.assertIn("q_low", result.regular_task_ids)

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