import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from systemlab.common import sha, source_receipt

lab = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013")
report = lab / "reviews/entry_selfaudit_v1/SELF_AUDIT_REPORT_中文.md"
protocol = lab / "protocol/frozen_v1/FROZEN_SYSTEM_PROTOCOL.json"

# This is an executor SELF-audit, explicitly NOT independent certification.
value = dict(
    schema="systemlab-independent-entry-v1",
    scope="complete_system_entry",
    status="WARN",
    blockers=["缺少独立复审者身份：本报告为执行者自查，不能替代独立完整入口复审，不可据此放行 E1 正式运行"],
    complete_report=True,
    reviewer_identity="执行者自查（非独立认证）",
    self_audit=True,
    protocol_sha256=sha(protocol),
    source_sha256=source_receipt(),
    report_path=str(report.resolve()),
    report_sha256=sha(report),
)
out = lab / "reviews/entry_selfaudit_v1/SELF_AUDIT_RESULT.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.open("x").write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
print("report_sha256:", value["report_sha256"])
print("blockers:", value["blockers"])
print("status:", value["status"], "(self-audit; will NOT pass the E1 --entry-review gate)")
print("written:", out)
