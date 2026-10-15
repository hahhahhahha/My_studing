import copy
import json
import unittest
from pathlib import Path

from prepare_protocol import build_config, feedback_fixture
from systemlab.policy import Router


class GeometryRevisionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'configs/BASE_LOCAL_4EDGE_HISTORY.json'
        self.base = json.loads(path.read_text(encoding='utf-8-sig'))
        self.config = build_config(self.base, Path('fixture-weight-not-loaded.pth'))

    def test_counts_budget_and_original_preserved(self):
        before = copy.deepcopy(self.base)
        config = build_config(self.base, Path('fixture-weight-not-loaded.pth'))
        self.assertEqual(self.base, before)
        self.assertEqual((config['users'], len(config['edges'])), (30, 35))
        self.assertEqual({x['torch_threads'] for x in config['edges']}, {1})
        self.assertEqual(config['feedback']['budget_bytes_per_s'], 2097152)
        self.assertEqual(config['feedback']['burst_bytes'], 1048576)
        self.assertEqual(config['feedback']['period_s'], .2)
        for key in ('weight_sha256', 'waiting_capacity', 'fixed_drain_s', 'decision_period_s'):
            self.assertEqual(config[key], before[key])

    def test_new_feedback_fits_old_burst_rejects(self):
        estimate = feedback_fixture(self.config)
        self.assertGreater(estimate['broadcast_round_bytes'], 65536)
        self.config['feedback']['burst_bytes'] = 65536
        with self.assertRaises(ValueError):
            feedback_fixture(self.config)

    def test_thirty_users_can_exercise_all_thirty_five_edges(self):
        router = Router('round_robin', self.config)
        states = {edge['id']: dict(predicted_wait_s=0., service_estimate_s=.031,
                                  source_epoch_ns=1000000000) for edge in self.config['edges']}
        selected = []
        for i in range(70):
            weights, _ = router.portfolio([dict(payload_bytes=10000)], states, 1100000000, i % 30)
            selected.append(weights.index(1.))
        self.assertEqual(selected[:35], list(range(35)))
        self.assertEqual(selected[35:], list(range(35)))
        self.assertEqual(router.receipt['assignment_rule'], 'global_sequential_decision_batch_rotation_v2')

    def test_same_user_also_rotates(self):
        router = Router('round_robin', self.config)
        states = {edge['id']: dict(predicted_wait_s=0., service_estimate_s=.031,
                                  source_epoch_ns=1000000000) for edge in self.config['edges']}
        picks = [router.portfolio([dict(payload_bytes=10000)], states, 1100000000, 0)[0].index(1.) for _ in range(35)]
        self.assertEqual(picks, list(range(35)))


if __name__ == '__main__':
    unittest.main()
