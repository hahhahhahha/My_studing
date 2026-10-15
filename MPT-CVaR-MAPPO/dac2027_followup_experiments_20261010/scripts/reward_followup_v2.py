"""Additive, zero-run revision: reject partial smoke reuse and duplicate starts.

The original prepared design/binding and controller remain unchanged. This
wrapper does not change actions, rewards, seeds, budgets, schedules or metrics.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import reward_followup as original

OUT = original.OUT
DOC = original.BASE / 'README_REWARD_EXECUTION_V2_中文.md'
BINDING = OUT / 'ADDITIVE_V2_BINDING.json'


def v2_sources():
    return {str(path): original.module.sha(path) for path in (Path(__file__), DOC)}


def prepare():
    original.check()
    if original.read(OUT / 'STATUS.json')['status'] != 'prepared_not_started':
        raise RuntimeError('V2 preparation permitted only before any smoke/formal execution')
    for name in ('smoke', 'RUN_STARTED.json', 'SMOKE_V2_STARTED.json', 'TWO_ARM_SMOKE_COMPLETE.json'):
        if (OUT / name).exists():
            raise RuntimeError('Execution evidence exists; no rebinding or migration of partial work')
    original.save(BINDING, dict(at=original.stamp(), files=v2_sources(),
         original_binding_sha256=original.module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'),
         reason='Add exclusive smoke/formal starts before execution; no partial-cell automatic reuse',
         formal_or_smoke_started=False, old_binding_or_design_modified=False))
    print('REWARD_V2_ADDITIVE_PREPARED_NOT_STARTED', flush=True)


def check():
    design = original.check()
    binding = original.read(BINDING)
    if binding['files'] != v2_sources() or binding['original_binding_sha256'] != original.module.sha(OUT / 'ADDITIVE_ENTRY_BINDING.json'):
        raise RuntimeError('V2 source/doc/original binding changed')
    print('REWARD_V2_READONLY_CHECK_OK', flush=True)
    return design


def smoke():
    check()
    for name in ('smoke', 'SMOKE_V2_STARTED.json', 'TWO_ARM_SMOKE_COMPLETE.json', 'RUN_STARTED.json'):
        if (OUT / name).exists():
            raise RuntimeError('Smoke/run evidence already exists; preserve and diagnose, never append or auto-resume')
    # Exclusive before witness/allocation as well as before run_cell. A resource
    # failure preserves this attempt; it cannot be invisibly retried in place.
    original.save(OUT / 'SMOKE_V2_STARTED.json', dict(at=original.stamp(), pid=os.getpid(),
                  v2_binding_sha256=original.module.sha(BINDING)))
    try:
        original.smoke()
        original.save(OUT / 'SMOKE_V2_COMPLETE.json', dict(at=original.stamp(),
             v2_binding_sha256=original.module.sha(BINDING),
             smoke_complete_sha256=original.module.sha(OUT / 'TWO_ARM_SMOKE_COMPLETE.json')))
    except BaseException as exc:
        original.save(OUT / 'SMOKE_V2_FAILURE.json', dict(at=original.stamp(), error=repr(exc),
             partial_evidence_preserved=True, automatic_resume=False, formal_training_started=False))
        raise


def entry_check():
    check()
    if original.read(OUT / 'SMOKE_V2_COMPLETE.json')['v2_binding_sha256'] != original.module.sha(BINDING):
        raise RuntimeError('Actual V2 smoke completion required')
    original.entry_check()
    print('REWARD_V2_ACTUAL_ENTRY_CHECK_OK', flush=True)


def run():
    entry_check()
    review = original.read(OUT / 'ENTRY_REVIEW.json')
    if review.get('passed') is not True or review.get('v2_binding_sha256') != original.module.sha(BINDING):
        raise RuntimeError('Fresh V2 source/doc/actual-smoke review required')
    for name in ('RUN_STARTED.json', 'RUN_V2_STARTED.json'):
        if (OUT / name).exists():
            raise RuntimeError('Formal execution already started; no automatic duplicate or resume')
    original.save(OUT / 'RUN_V2_STARTED.json', dict(at=original.stamp(), pid=os.getpid(),
         v2_binding_sha256=original.module.sha(BINDING), entry_review_sha256=original.module.sha(OUT / 'ENTRY_REVIEW.json')))
    try:
        original.formal()
    except BaseException as exc:
        original.save(OUT / 'RUN_V2_FAILURE.json', dict(at=original.stamp(), error=repr(exc), automatic_resume=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'smoke', 'entry-check', 'run'])
    {'prepare': prepare, 'check': check, 'smoke': smoke, 'entry-check': entry_check, 'run': run}[parser.parse_args().action]()
