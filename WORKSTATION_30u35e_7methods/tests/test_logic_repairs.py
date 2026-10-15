"""Synthetic admission/concurrency fixtures only; not formal entry evidence."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from systemlab.common import local_windows_runtime,read,sha,source_receipt,write_new
local_windows_runtime()
from unittest.mock import Mock,patch
from systemlab.edge import EdgeQueue,Task
from systemlab.launcher import _wait_windows_runtime_pid,local_run
from systemlab.admission import register_test_registry,validate_test_admission
from systemlab.common import Journal


class RepairTests(unittest.TestCase):
    def test_completion_and_failure_snapshot_are_atomic(self):
        for fails in (False,True):
            with self.subTest(inference_error=fails),tempfile.TemporaryDirectory() as folder:
                gate,started,window,resume=(threading.Event() for _ in range(4))
                def engine(body):
                    started.set(); gate.wait(1)
                    if fails:
                        raise ValueError('injected fixture inference failure')
                    return dict(preprocess_s=0.,compute_s=0.,top1=0)
                journal=Journal(Path(folder)/'fixture.jsonl')
                core=EdgeQueue('fixture',1,.01,engine,journal)
                class InterleavingLock:
                    def __init__(self,real): self.real=real
                    def __enter__(self): self.real.acquire(); return self
                    def __exit__(self,*args):
                        inconsistent=(threading.current_thread() is core.worker and
                            core.completed+core.failed==1 and core.active is not None)
                        self.real.release()
                        if inconsistent: window.set(); resume.wait(1)
                core.lock=InterleavingLock(core.lock)
                task=Task('x',0,b'fixture',0.,0)
                try:
                    core.submit(task); self.assertTrue(started.wait(1)); gate.set()
                    window.wait(.05)
                    core.snapshot()  # Old split-lock transition deterministically raises here.
                    self.assertFalse(window.is_set())
                    self.assertTrue(task.done.wait(1))
                    snapshot=core.snapshot()
                    self.assertFalse(snapshot['busy'])
                    self.assertEqual(snapshot['completed'],int(not fails))
                    self.assertEqual(snapshot['inference_errors'],int(fails))
                finally:
                    gate.set(); resume.set(); core.stop.set(); core.worker.join(2); journal.close()

    def test_windows_open_failure_is_not_necessarily_exit(self):
        api=Mock()
        api.OpenProcess.return_value=0
        for error in (5,0,6,999):
            self.assertFalse(_wait_windows_runtime_pid(123,.2,api,lambda:error))
        self.assertTrue(_wait_windows_runtime_pid(123,.2,api,lambda:87))
        api.WaitForSingleObject.assert_not_called()
        api.CloseHandle.assert_not_called()

    def test_windows_wait_and_close_fail_closed_with_pointer_sized_handle(self):
        api=Mock()
        handle=0x123456789abc
        api.OpenProcess.return_value=handle
        api.CloseHandle.return_value=1
        for result in (258,0xffffffff,128):
            api.WaitForSingleObject.return_value=result
            self.assertFalse(_wait_windows_runtime_pid(123,.2,api,lambda:0))
        api.WaitForSingleObject.return_value=0
        self.assertTrue(_wait_windows_runtime_pid(123,.2,api,lambda:0))
        api.CloseHandle.assert_called_with(handle)
        api.WaitForSingleObject.assert_called_with(handle,200)
        api.CloseHandle.return_value=0
        self.assertFalse(_wait_windows_runtime_pid(123,.2,api,lambda:0))

    def fixture(self,root):
        self.root=root
        self.paths={key:root/(key+'.json') for key in
            ('cutoff','protocol','config','grid','registration','manifest','scales','schedule','registry','review','actor')}
        p=self.paths
        scales=dict(status='frozen_after_calibration_validation')
        config=dict(users=16,fixed_drain_s=30,feature_scales=scales,
            feedback=dict(mode='periodic',injected_delay_s=0.,injected_loss=0.,budget_bytes_per_s=65536))
        cutoff=root/'CUTOFF.json'
        write_new(cutoff,dict(protocol_horizon_s=120,fixed_drain_s=30))
        self.checkpoint=root/'FIXTURE_NOT_A_MODEL.txt'
        self.checkpoint.write_text('fixture weights only; never model-loaded',encoding='utf-8')
        source=root/'actor_source'; source.mkdir()
        self.actor_source=source/'models.py'
        self.actor_source.write_text('# fixture, not executable model implementation\n',encoding='utf-8')
        write_new(p['actor'],dict(checkpoint=str(self.checkpoint),checkpoint_sha256=sha(self.checkpoint),
            source_root=str(source),source_sha256={'models.py':sha(self.actor_source)},feature_contract=scales))
        val_inputs=root/'INPUTS.json'
        write_new(val_inputs,dict(method='qla',policy_receipt=dict(checkpoint_sha256=sha(self.checkpoint))))
        write_new(p['protocol'],dict(formal_config=config,source_sha256=source_receipt(),
            validation_artifacts_sha256={str(cutoff):sha(cutoff),str(val_inputs):sha(val_inputs)}))
        config.update(frozen_protocol_path=str(p['protocol']),frozen_protocol_sha256=sha(p['protocol']))
        write_new(p['config'],config); write_new(p['scales'],scales)
        write_new(p['registration'],dict(role='validation',horizon_s=120,users=16,fixed_drain_s=30))
        write_new(p['manifest'],dict(role='test',images=[]))
        cell=dict(id='E1_001',seed=1,method='qla',common_schedule_key='0.6_1',aggregate_rate_per_s=105.23721539481573,
            feedback_mode='periodic',injected_feedback_delay_s=0.,injected_feedback_loss=0.,feedback_budget_bytes_per_s=65536)
        write_new(p['grid'],dict(schema='systemlab-grid-v1',block='E1',config_sha256=sha(p['config']),
            feature_scales_sha256=sha(p['scales']),feature_scales_path=str(p['scales']),cells=[cell]))
        write_new(p['schedule'],dict(schema='systemlab-schedule-v1',role='test',seed=1,users=16,duration_s=120,
            image_manifest_sha256=sha(p['manifest']),provenance=dict(source='synthetic_poisson',
            aggregate_rate_per_s=cell['aggregate_rate_per_s']),tasks=[]))
        self.generator=root/'orchestrator.py'
        self.generator.write_text('# temporary non-executable test fixture\n',encoding='utf-8')
        register_test_registry(p['config'],p['grid'],p['registration'],p['manifest'],[p['schedule']],p['registry'],
                               generators=[self.generator],actor_bundles={'qla':p['actor']})
        report=root/'TEST_FIXTURE_NOT_AN_APPROVAL.md'
        report.write_text('Synthetic gate test, not an independent review or runtime approval.',encoding='utf-8')
        write_new(p['review'],dict(schema='systemlab-independent-entry-v1',scope='complete_system_entry',status='PASS',
            blockers=[],complete_report=True,reviewer_identity='SYNTHETIC_UNIT_TEST_ONLY',report_path=str(report),
            report_sha256=sha(report),source_sha256=source_receipt(),protocol_sha256=config['frozen_protocol_sha256'],
            test_registry_sha256=sha(p['registry'])))
        self.config=config
        return p

    def admit(self,**changes):
        p=self.paths
        arguments=dict(config=self.config,schedule_path=p['schedule'],manifest_path=p['manifest'],method='qla',
            entry_review=p['review'],test_registry=p['registry'],cell_id='E1_001',bundle=p['actor'])
        arguments.update(changes)
        return validate_test_admission(**arguments)

    def test_registered_cell_admits_and_records_exact_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.fixture(Path(folder))
            result=self.admit()
            self.assertEqual(result['cell_id'],'E1_001')
            self.assertEqual(result['test_registry_sha256'],sha(p['registry']))

    def test_wrong_cell_method_or_unregistered_schedule_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.fixture(Path(folder))
            for changes in (dict(cell_id='E1_999'),dict(method='score_softmax'),dict(test_registry=None)):
                with self.subTest(changes=changes),self.assertRaises(ValueError): self.admit(**changes)
            alternative=self.root/'alternative.json'
            modified=read(p['schedule']); modified['seed']=2
            write_new(alternative,modified)
            with self.assertRaises(ValueError): self.admit(schedule_path=alternative)

    def test_registered_artifact_mutations_reject(self):
        for name in ('grid','registration','manifest','scales','schedule','config','generator','actor','checkpoint','actor_source'):
            with self.subTest(artifact=name),tempfile.TemporaryDirectory() as folder:
                p=self.fixture(Path(folder))
                target=getattr(self,name) if name in ('generator','checkpoint','actor_source') else p[name]
                with target.open('ab') as stream: stream.write(b'\n')  # JSON remains valid; bytes changed.
                with self.assertRaises(ValueError): self.admit()

    def test_review_must_bind_registry_and_not_be_self_audit(self):
        for change in (dict(test_registry_sha256='wrong'),dict(blockers=['missing independent reviewer']),dict(self_audit=True)):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                p=self.fixture(Path(folder))
                review=read(p['review']); review.update(change)
                alternative=self.root/'different_review.json'; write_new(alternative,review)
                with self.assertRaises(ValueError): self.admit(entry_review=alternative)

    def test_registration_rejects_wrong_horizon_users_rate_seed_or_drain(self):
        for change in (dict(duration_s=60),dict(users=8),dict(seed=2),dict(rate=105.237215)):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                p=self.fixture(Path(folder))
                value=read(p['schedule'])
                for k,v in change.items():
                    if k=='rate': value['provenance']['aggregate_rate_per_s']=v
                    else: value[k]=v
                other=self.root/'different_schedule.json'; write_new(other,value)
                with self.assertRaises(ValueError):
                    register_test_registry(p['config'],p['grid'],p['registration'],p['manifest'],[other],
                        self.root/'not_written.json',generators=[self.generator],actor_bundles={'qla':p['actor']})
                self.assertFalse((self.root/'not_written.json').exists())

    def test_missing_admission_rejects_before_any_service_or_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.fixture(Path(folder))
            output=self.root/'must_not_start'
            with patch('systemlab.launcher.start_child') as launch:
                with self.assertRaises(ValueError):
                    local_run(p['config'],self.config,p['schedule'],p['manifest'],'qla',output)
                launch.assert_not_called()
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
