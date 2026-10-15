import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_protocol import build_config
from systemlab.baselines import FrozenBaseline, cached_features, METHODS, validate_bundle
from systemlab.admission import validate_test_admission
from systemlab.common import SERVER_RESPONSE_TIMEOUT, write_new


class BaselineTests(unittest.TestCase):
    def setUp(self):
        base = json.loads((Path(__file__).resolve().parents[1]/'configs/BASE_LOCAL_4EDGE_HISTORY.json').read_text())
        self.config = build_config(base, Path('not-loaded.pth'))
        self.cli = dict(slot_s=1, score_resource_weight=1, score_capacity_weight=.5,
            score_task_count_weight=.5, score_wait_weight=1, max_requests_per_user=7,
            history_steps=10, jseq_temperature=.25)
        self.states = {e['id']: dict(predicted_wait_s=.05, service_estimate_s=.03,
            waiting_count=1, busy=True, source_epoch_ns=1000000000) for e in self.config['edges']}

    def stub(self, method):
        obj = FrozenBaseline.__new__(FrozenBaseline)
        obj.method, obj.config, obj.cli = method, self.config, self.cli
        obj.lock, obj.histories, obj.history_times = threading.RLock(), {}, {}
        obj.max_history_gap_s, obj.released_counts = 0., {}
        obj.arrivals = {e['id']: [] for e in self.config['edges']}
        return obj

    def test_original_observation_dimension_and_cached_only(self):
        obs, wait, service, count, wait_norm = cached_features([dict(payload_bytes=10000)], self.states, self.config, self.cli)
        self.assertEqual(obs.shape, (75,))
        self.assertEqual(wait.shape, (35,))
        modified = copy.deepcopy(self.states)
        modified['edge0']['compute_s_FUTURE_PRIVATE'] = 99999
        self.assertTrue((obs == cached_features([dict(payload_bytes=10000)], modified, self.config, self.cli)[0]).all())
        modified['edge0']['service_estimate_s'] = 0
        with self.assertRaises(ValueError):
            cached_features([], modified, self.config, self.cli)

    def test_qeco_samples_real_times_without_backfill(self):
        actor = self.stub('qeco')
        states = self.states
        class Cache:
            def observation(self, user):
                return states
        actor.observe_cache(Cache(), 1000000000)
        self.assertEqual(len(actor.histories), 30)
        self.assertEqual(list(actor.history_times[29]), [1000000000]*10)
        actor.observe_cache(Cache(), 1100000000)
        self.assertEqual(list(actor.history_times[0]), [1000000000]*10)
        actor.observe_cache(Cache(), 3500000000)
        self.assertEqual(list(actor.history_times[0])[-2:], [1000000000,3500000000])
        self.assertEqual(actor.max_history_gap_s, 2.5)

    def test_jseq_only_acknowledged_admission_not_rejections(self):
        actor = self.stub('jseq')
        item = dict(server_id='edge0', send_epoch_ns=1100000000, predicted_service_s=.03)
        for status in ('queue_rejected', 'transport_error', 'client_backpressure'):
            actor.observe_outcome(dict(item, status=status))
        self.assertEqual(actor.arrivals['edge0'], [])
        actor.observe_outcome(dict(item, status='completed'))
        self.assertEqual(actor.arrivals['edge0'], [(1100000000,.03)])
        weights, info = actor.portfolio([dict(payload_bytes=10000)], self.states, 1200000000, 0)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(len(info['jseq_estimated_wait_s']), 35)

    def test_future_feedback_and_wrong_user_rejected(self):
        actor = self.stub('jseq')
        for user, now in ((30,1200000000),(0,999999999)):
            with self.assertRaises(ValueError):
                actor.portfolio([dict(payload_bytes=1)], self.states, now, user)

    def test_real_server_timeout_contract_reaches_jseq(self):
        import ast
        import inspect
        from systemlab.edge import serve_edge
        # No HTTP/service launch: assert the actual server emission references
        # the shared contract, then exercise that exact value at the observer.
        tree = ast.parse(inspect.getsource(serve_edge))
        statuses = [node.value.id for node in ast.walk(tree) if isinstance(node, ast.keyword)
                    and node.arg == 'status' and isinstance(node.value, ast.Name)]
        self.assertIn('SERVER_RESPONSE_TIMEOUT', statuses)
        self.assertEqual(SERVER_RESPONSE_TIMEOUT, 'server_response_timeout')
        actor = self.stub('jseq')
        item = dict(server_id='edge0', send_epoch_ns=1100000000, predicted_service_s=.03)
        actor.observe_outcome(dict(item, status=SERVER_RESPONSE_TIMEOUT))
        self.assertEqual(actor.arrivals['edge0'], [(1100000000,.03)])
        for status in ('server_timeout','queue_rejected','transport_error','client_backpressure'):
            actor.observe_outcome(dict(item, status=status))
        self.assertEqual(len(actor.arrivals['edge0']), 1)

    def test_new_test_methods_fail_closed_without_services(self):
        with tempfile.TemporaryDirectory() as temp:
            schedule = Path(temp)/'schedule.json'
            write_new(schedule, dict(role='test'))
            for method in METHODS:
                with self.assertRaisesRegex(ValueError, 'Calibration/validation freeze required'):
                    validate_test_admission(self.config,schedule,'unused',method,None,None,None,None)

    def test_source_mismatch_rejects_before_asset_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)/'bundle.json'
            write_new(bundle, dict(schema='systemlab-frozen-baseline-v1', method='qeco', evaluation_only=True,
                adapter_protocol='real_cached_baselines_v1', target_users=30, target_edges=35,
                feature_contract=self.config['feature_scales'],source_sha256={}))
            with patch('systemlab.baselines.asset_identity') as identity:
                with self.assertRaisesRegex(ValueError,'source/geometry/features'):
                    validate_bundle('qeco',self.config,bundle)
                identity.assert_not_called()


if __name__ == '__main__':
    unittest.main()
