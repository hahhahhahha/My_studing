"""Author functional verification receipt; not independent experiment approval."""
import argparse
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from systemlab.common import local_windows_runtime,runtime,source_receipt,write_new

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
local_windows_runtime()
log=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
write_new(a.output,dict(schema='systemlab-author-unit-verification-v1',tests_run=result.testsRun,
    failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
    runtime=runtime(),source_sha256=source_receipt(),log=log.getvalue(),
    independent_review=False,performance_evidence=False))
print(log.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)
