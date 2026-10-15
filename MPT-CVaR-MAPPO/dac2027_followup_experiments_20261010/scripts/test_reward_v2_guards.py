"""Synthetic control-guard fixtures only; no CUDA, policy or experimental cells."""
from windows_runtime import initialize
initialize()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reward_followup_v2 as entry


class GuardFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.binding = self.out / 'ADDITIVE_V2_BINDING.json'
        self.binding.write_text('{}', encoding='utf-8')
        self.bind_sha = entry.original.module.sha(self.binding)
        self.stack = [patch.object(entry, 'OUT', self.out), patch.object(entry, 'BINDING', self.binding),
                      patch.object(entry, 'check', return_value={})]
        for item in self.stack:
            item.start()

    def tearDown(self):
        for item in reversed(self.stack):
            item.stop()
        self.temp.cleanup()

    def test_partial_smoke_directory_rejected_before_witness(self):
        (self.out / 'smoke').mkdir()
        with patch.object(entry.original, 'smoke') as work:
            with self.assertRaises(RuntimeError):
                entry.smoke()
            work.assert_not_called()
        self.assertFalse((self.out / 'SMOKE_V2_STARTED.json').exists())

    def test_failed_first_smoke_keeps_lock_and_no_second_attempt(self):
        with patch.object(entry.original, 'smoke', side_effect=RuntimeError('synthetic failure')) as work:
            with self.assertRaises(RuntimeError):
                entry.smoke()
            first_lock = (self.out / 'SMOKE_V2_STARTED.json').read_bytes()
            self.assertTrue((self.out / 'SMOKE_V2_FAILURE.json').is_file())
            with self.assertRaises(RuntimeError):
                entry.smoke()
            self.assertEqual(work.call_count, 1)
            self.assertEqual((self.out / 'SMOKE_V2_STARTED.json').read_bytes(), first_lock)

    def test_old_formal_lock_rejected_before_work(self):
        (self.out / 'RUN_STARTED.json').write_text('{}', encoding='utf-8')
        (self.out / 'ENTRY_REVIEW.json').write_text(json.dumps({'passed': True, 'v2_binding_sha256': self.bind_sha}), encoding='utf-8')
        with patch.object(entry, 'entry_check'), patch.object(entry.original, 'formal') as work:
            with self.assertRaises(RuntimeError):
                entry.run()
            work.assert_not_called()
        self.assertFalse((self.out / 'RUN_V2_STARTED.json').exists())

    def test_missing_independent_review_refuses_launch(self):
        (self.out / 'ENTRY_REVIEW.json').write_text(json.dumps({'passed': False, 'v2_binding_sha256': self.bind_sha}), encoding='utf-8')
        with patch.object(entry, 'entry_check'), patch.object(entry.original, 'formal') as work:
            with self.assertRaises(RuntimeError):
                entry.run()
            work.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
