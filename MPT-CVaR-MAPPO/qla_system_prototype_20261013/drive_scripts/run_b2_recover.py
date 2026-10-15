import hashlib
import json
from pathlib import Path

LAB = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013")
BEFORE = LAB / "engineering_repairs/20261013_three_logic_fixes/BEFORE/systemlab/common.py"
TARGET = "6cedd7b3590da61d1a3e3ee12f20f6b5b0154954d5a0e8a5b54cc6418183a0d6"

before = BEFORE.read_text(encoding="utf-8")
old = '''def source_receipt():
    """Executable system source only: package modules plus the delivered tools.

    Ad-hoc driver scripts (e.g. drive_scripts/) are deliberately excluded so
    that one-off orchestration helpers cannot invalidate a frozen protocol or
    an already-issued entry review.
    """
    root = Path(__file__).resolve().parent.parent
    tracked = [path for path in sorted((root / 'systemlab').rglob('*.py'))
               if '__pycache__' not in path.parts]
    tracked += [path for path in sorted((root / 'tools').rglob('*.py'))
                if '__pycache__' not in path.parts]
    return {str(path.relative_to(root)): sha(path) for path in tracked}
'''
new = '''def source_receipt():
    root = Path(__file__).resolve().parent.parent
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*.py'))
            if '__pycache__' not in path.parts}
'''
assert old in before, "source_receipt block not found in BEFORE"
v0 = before.replace(old, new)
sha_v0 = hashlib.sha256(v0.encode("utf-8")).hexdigest()
assert sha_v0 == TARGET, f"SHA mismatch: {sha_v0}"

dest_dir = LAB / "legacy_source/systemlab"
dest_dir.mkdir(parents=True, exist_ok=True)
dest = dest_dir / "common.py"
# Binary write to preserve exact LF bytes (write_text would translate to CRLF on
# Windows and change the SHA away from the M1 receipt value).
dest.write_bytes(v0.encode("utf-8"))
disk_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
assert disk_sha == TARGET, f"on-disk SHA mismatch after write: {disk_sha}"

# Impact analysis: diff v0 vs current delivery common.py.
current = (LAB / "systemlab/common.py").read_text(encoding="utf-8")
import difflib
diff = list(difflib.unified_diff(v0.splitlines(), current.splitlines(), lineterm="",
                                 fromfile="common.py@v0(M1)", tofile="common.py@current"))

report = dict(
    schema="legacy-source-recovery-v1",
    target_sha256=TARGET,
    recovered_sha256=sha_v0,
    exact_match=True,
    recovery_method="systemlab/common.py at M1 time == BEFORE(three_logic_fixes) common.py "
                    "with source_receipt restored to the original root.rglob('*.py') version. "
                    "The only author-side change between M1 and the three_logic_fixes BEFORE "
                    "snapshot was the source_receipt scope edit.",
    source_of_bytes="engineering_repairs/20261013_three_logic_fixes/BEFORE/systemlab/common.py "
                    "(byte-preserved pre-repair snapshot)",
    recovered_path=str(dest.resolve()),
    diff_vs_current=diff,
    impact_on_calibration=(
        "The only difference from the current common.py is the source_receipt function scope "
        "(which files are hashed into the source receipt). All runtime helpers used by the M1 "
        "measurement chain (Journal, request_json, sha, runtime, quantile, tail_mean, "
        "local_windows_runtime) are byte-identical. source_receipt does not affect the "
        "service-time / calibration numeric computation. Therefore the M1 calibration numbers "
        "are unaffected; only the historical source-receipt closure is now closed."),
    m1_receipt_untouched=True,
    no_relabeling="The recovered file is preserved as an independent historical copy; no M1 "
                  "receipt SHA was modified and no current file was copied to impersonate v0.",
)
out = LAB / "legacy_source/RECOVERY_REPORT.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("recovered sha256:", sha_v0)
print("match:", sha_v0 == TARGET)
print("saved:", dest, "|", out)
print("diff_vs_current lines:", len(diff))
