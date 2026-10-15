import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from systemlab.common import Journal, tail_mean, quantile
from systemlab.data import import_images, schedule
from systemlab.edge import EdgeQueue, Task
from systemlab.feedback import ByteBudget, FeedbackCache, choose_updates
from systemlab.policy import sampled_index, features


class CoreTests(unittest.TestCase):
    def test_finite_fifo_and_mass(self):
        with tempfile.TemporaryDirectory() as d:
            gate,started=threading.Event(),threading.Event()
            def fixture_engine(body):
                started.set()
                gate.wait(2)
                return dict(preprocess_s=0.,compute_s=0.,top1=0)
            journal=Journal(Path(d)/'fixture.jsonl')
            core=EdgeQueue('test',1,.01,fixture_engine,journal)
            first=Task('a',0,b'x',0.,0)
            self.assertEqual(core.submit(first),'admitted')
            self.assertTrue(started.wait(1))
            second=Task('b',0,b'x',0.,0)
            self.assertEqual(core.submit(second),'admitted')
            self.assertEqual(core.submit(Task('c',0,b'x',0.,0)),'queue_rejected')
            self.assertEqual(core.submit(second),'duplicate')
            snap=core.snapshot()
            self.assertEqual(snap['admitted'],2)
            self.assertEqual(snap['waiting_count'],1)
            gate.set()
            self.assertTrue(second.done.wait(2))
            self.assertEqual(core.snapshot()['completed'],2)
            core.stop.set()
            core.worker.join(1)
            journal.close()

    def test_budget_burst_and_rate(self):
        budget=ByteBudget(100,50)
        self.assertTrue(budget.spend(50,0))
        self.assertFalse(budget.spend(1,0))
        self.assertTrue(budget.spend(50,.5))
        self.assertEqual(budget.used,100)

    def test_sparse_versions_and_actual_delivery(self):
        cache=FeedbackCache('session',1,loss=0.)
        def packet(version,source):
            return dict(session='session',user_id=0,version=version,tick=version,send_epoch_ns=0,
                        servers=[dict(server_id='e',source_epoch_ns=source,waiting_count=1)])
        self.assertTrue(cache.accept(packet(2,200),300))
        cache.accept(packet(3,100),400)
        self.assertEqual(cache.observation(0)['e']['source_epoch_ns'],200)
        self.assertFalse(FeedbackCache('session',1,loss=1.).accept(packet(4,500),600))
        self.assertFalse(cache.accept(dict(packet(5,500),session='wrong'),600))

    def test_tail_fractional_boundary(self):
        self.assertAlmostEqual(tail_mean([1.,2.,3.,4.],.5),3.5)
        self.assertEqual(tail_mean([1.,100.],.99),100.)
        self.assertIsNone(tail_mean([]))
        self.assertEqual(quantile([0.,10.],.95),9.5)

    def test_registered_image_role_and_common_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'x.jpeg').write_bytes(b'fixture')
            item=dict(image_id='x.jpeg',image_sha256=hashlib.sha256(b'fixture').hexdigest(),payload_bytes=7,width=1,height=1)
            split=root/'split.json'
            split.write_text(json.dumps(dict(pilot=[item],test=[item])))
            manifest=root/'images.json'
            import_images(split,root,'pilot',manifest,1)
            a=schedule(manifest,root/'a.json',duration=1.,rate=10.,role='pilot')
            b=schedule(manifest,root/'b.json',duration=1.,rate=10.,role='pilot')
            self.assertEqual(a,b)
            self.assertTrue(all(0<=x['assignment_uniform']<1 for x in a['tasks']))
            with self.assertRaises(ValueError):
                schedule(manifest,root/'bad.json',role='test')

    def test_azure2021_start_and_explicit_unit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest=root/'images.json'
            manifest.write_text(json.dumps(dict(role='pilot',images=[dict(image_id='fixture.jpeg')])) )
            csv_file=root/'trace.csv'
            csv_file.write_text('app,func,end_timestamp,duration\na,f,12000,2000\na,f,15500,500\n')
            result=schedule(manifest,root/'out.json',users=1,role='pilot',azure=csv_file,
                trace_start=9,trace_window=6,time_scale=2,time_unit='milliseconds')
            self.assertEqual(len(result['tasks']),1)
            self.assertEqual(result['tasks'][0]['arrival_s'],2.)
            self.assertEqual(result['duration_s'],12.)

    def test_age_and_demand_pressure_are_past_observations(self):
        import random
        cfg=dict(age_reference_s=.5,alpha=1.,beta=.5,gamma=.5,trigger_threshold=0.)
        old={j:dict(source_epoch_ns=0,waiting_count=0,capacity=32,received=0) for j in ('a','b')}
        current={j:dict(source_epoch_ns=1000000000,waiting_count=0,capacity=32,received=n,
                       service_estimate_s=.1) for j,n in [('a',1),('b',10)]}
        self.assertEqual(choose_updates('coordinated',current,old,1000000000,random.Random(1),cfg),['b'])

    def test_stale_only_features_and_sampling(self):
        cfg=dict(edges=[dict(id='e')],feature_scales=dict(fitted_role='calibration',request_scale=7.,
            mean_payload_bytes=100.,reference_service_s=.01,transfer_seconds_per_server=[.001],
            queue_reference_s=1.,latency_scale_s=1.,age_scale_s=.5))
        obs,cost,age=features([dict(payload_bytes=100)],{'e':dict(predicted_wait_s=.2,
            service_estimate_s=.01,source_epoch_ns=1_000_000_000)},cfg,2_000_000_000)
        self.assertEqual(len(obs),10)
        self.assertAlmostEqual(cost[0],.211)
        self.assertEqual(age,[1.])
        self.assertEqual(sampled_index([0.,1.],0.),1)


    def test_runtime_pid_liveness_not_shim_pid(self):
        # The venv launcher shim PID differs from the real service PID on
        # Windows; liveness must be judged by the runtime PID only.
        import subprocess, sys
        from systemlab.launcher import _wait_runtime_pid
        child = subprocess.Popen([sys.executable, '-c',
            'import os,sys,time;print(os.getpid());sys.stdout.flush();time.sleep(2)'],
            stdout=subprocess.PIPE)
        runtime_pid = int(child.stdout.readline().decode().strip())
        self.assertNotEqual(runtime_pid, 0)
        self.assertFalse(_wait_runtime_pid(runtime_pid, .2))  # still running
        child.terminate()
        child.wait(timeout=5)
        self.assertTrue(_wait_runtime_pid(runtime_pid, 2.))
        self.assertFalse(_wait_runtime_pid(None, 2.))  # unknown PID is never assumed exited
        self.assertTrue(_wait_runtime_pid(runtime_pid + 10_000_000, .2))  # gone PID


if __name__=='__main__':
    unittest.main()
