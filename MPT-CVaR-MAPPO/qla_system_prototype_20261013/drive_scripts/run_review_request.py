import json
from pathlib import Path
import sys

LAB = Path("X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013")
sys.path.insert(0, str(LAB))
from systemlab.common import sha  # noqa: E402

OUT = LAB / "reviews/entry_v3_independent_20261014"
OUT.mkdir(exist_ok=True)

def s(p):
    return sha(LAB / p)

# ---- key artifacts and their actual SHAs ----
artifacts = {
    "frozen_v3/FROZEN_SYSTEM_PROTOCOL.json": s("protocol/frozen_v3/FROZEN_SYSTEM_PROTOCOL.json"),
    "frozen_v3/FORMAL_CONFIG.json": s("protocol/frozen_v3/FORMAL_CONFIG.json"),
    "frozen_v3/FEATURE_SCALES.json": s("protocol/frozen_v3/FEATURE_SCALES.json"),
    "formal_actor_v3.json": s("protocol/formal_actor_v3.json"),
    "E1_grid_v3.json": s("protocol/E1_grid_v3.json"),
    "E1_test_registry_v3.json": s("protocol/E1_test_registry_v3.json"),
    "validation_registration_v1.json": s("protocol/validation_registration_v1.json"),
    "calibration_scales_v1.json": s("protocol/calibration_scales_v1.json"),
    "data/calibration_images.json": s("data/calibration_images.json"),
    "data/validation_images.json": s("data/validation_images.json"),
    "data/test_images.json": s("data/test_images.json"),
    "vision weight": s("X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009/models/resnet18-f37072fd.pth"),
    "actor checkpoint": s("X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"),
    "legacy_source/systemlab/common.py": s("legacy_source/systemlab/common.py"),
    "legacy_source/RECOVERY_REPORT.json": s("legacy_source/RECOVERY_REPORT.json"),
    "tests/test_cutoff_race.py": s("tests/test_cutoff_race.py"),
}

# common schedules
for name in sorted((LAB / "data/E1_common_v3").glob("*.json")):
    artifacts[f"data/E1_common_v3/{name.name}"] = s(f"data/E1_common_v3/{name.name}")

# M2d run receipts
for method in ("qla", "score_softmax", "stale_shortest"):
    base = f"runs/M2d_{method}_s101_attempt_01/client"
    for f in ("INPUTS.json", "COMPLETE.json", "CUTOFF.json", "tasks.jsonl", "decisions.jsonl"):
        artifacts[f"{base}/{f}"] = s(f"{base}/{f}")

# M1 run receipts (source-closure now closed via B2)
for run in ("M1_rr_s1_attempt_01", "M1_stress_s1_attempt_01"):
    base = f"runs/{run}/client"
    for f in ("INPUTS.json", "COMPLETE.json", "tasks.jsonl"):
        artifacts[f"{base}/{f}"] = s(f"{base}/{f}")

# source receipt (current 19-file dictionary)
from systemlab.common import source_receipt  # noqa: E402
source_dict = source_receipt()

# ---- generate REVIEW_REQUEST.md ----
md = f"""你是一个全新上下文、未参与该系统实现的 Type-A 独立完整入口 reviewer。用户授权对正式 E1 前置入口做真正独立复审。GPT 同家族审查只能称 Type-A，不能称 Type-B 或跨家族认证。主执行者曾参与实现，所以你的判断必须直接来自文件，不能依赖执行者总结。请审查下列路径，默认所有作者主张都需要验证，不带结论预期。不要修改源码、科学设置、旧计划/结果或状态；只在指定新 review 目录写完整报告、machine JSON、有限功能验证输出及审查 trace。禁止启动 E1 正式矩阵、训练、重下载数据、杀未知进程、全局安装/重置。若需要有限功能驱动，使用现有功能配置在新路径做一次真实运行并正常关闭本次已核 PID；预先查看当前端口/源身份，若占用保留阻断，不能关闭他人服务。不得自动追加/合并部分失败重试。若需新的 authority 或来源不完整，报告具体阻断。

根目录：{LAB}

## 本次复审的背景（上一轮 FAIL 的两个阻断已由执行者处理，需你独立验证）

上一轮独立复审 `reviews/entry_v2_independent_20261014/REVIEW_RESPONSE.md` 结论 FAIL，两项阻断：

- **B1 截止双结局竞态**：执行者已修改 `systemlab/client.py`（把"写 unfinished_at_cutoff"的循环移到 `pool.shutdown(wait=True)` 之后，使每任务恰好一条正式 outcome），并新增 `tests/test_cutoff_race.py`（3 个回归：锁前暂停/正常完成/截止后响应）。你需要独立验证该修复正确、回归真实覆盖竞态窗口、且未引入新的结局语义变化。
- **B2 历史 common.py 缺口**：执行者已从 `engineering_repairs/20261013_three_logic_fixes/BEFORE/systemlab/common.py`（SHA `d72aca9e…`）把 `source_receipt` 恢复为原始 `rglob('*.py')` 版本，重建后 SHA 精确等于 M1 收据绑定的 `6cedd7b3…`，保存为 `legacy_source/systemlab/common.py`（恢复报告 `legacy_source/RECOVERY_REPORT.json`）。你需要独立验证该恢复的 SHA 匹配、与当前 `systemlab/common.py` 的逐行 diff、以及"恢复仅影响 source_receipt 范围、不影响 M1 校准数值链"这一结论。

由于 B1 改变了 `systemlab/client.py`，当前待审身份是 **M2d / frozen_v3 / E1 v3**（不是上一轮的 M2c / frozen_v2 / E1 v2）。旧的 M1/M2/M2b/M2c、frozen_v1/v2、E1_grid/registry v1/v2、entry_v2 复审及其失败记录均原样保留，不作为新源码的批准依据。

## 请直接完整读取

- {LAB}/README.md
- {LAB}/docs/AGENT_HANDOFF_中文.md
- {LAB}/docs/EXPERIMENT_PLAN_中文.md
- {LAB}/docs/DATA_PLAN_中文.md
- {LAB}/docs/EXPERIMENT_TRACKER_中文.md
- {LAB}/docs/SECON_BASELINE_SEARCH_20261014_中文.md
- {LAB}/systemlab/ 全部 .py（含 client.py、admission.py、common.py 当前版本）
- {LAB}/tools/ 全部 .py
- {LAB}/tests/ 全部 .py（含 test_cutoff_race.py）
- {LAB}/drive_scripts/run_m2d_validation.py、run_freeze_v3.py、run_e1_prep.py、run_interference_calibration.py、run_b2_recover.py
- {LAB}/configs/ 全部 JSON
- {LAB}/protocol/frozen_v3/{{FROZEN_SYSTEM_PROTOCOL,FORMAL_CONFIG,FEATURE_SCALES}}.json
- {LAB}/protocol/formal_actor_v3.json
- {LAB}/protocol/E1_grid_v3.json
- {LAB}/protocol/E1_test_registry_v3.json
- {LAB}/protocol/validation_registration_v1.json
- {LAB}/protocol/calibration_scales_v1.json
- {LAB}/legacy_source/systemlab/common.py 与 {LAB}/legacy_source/RECOVERY_REPORT.json（B2 恢复）
- {LAB}/runs/M1_rr_s1_attempt_01/ 全部实际输出
- {LAB}/runs/M1_stress_s1_attempt_01/ 全部实际输出
- {LAB}/runs/M2d_qla_s101_attempt_01/ 全部实际输出
- {LAB}/runs/M2d_score_softmax_s101_attempt_01/ 全部实际输出
- {LAB}/runs/M2d_stale_shortest_s101_attempt_01/ 全部实际输出
- {LAB}/data/calibration_images.json、validation_images.json、test_images.json、E1_common_v3/ 全部日程
- {LAB}/interference_calibration/PLAN.json、REPORT_中文.md、全部 run_*/edge*.json
- {LAB}/interference_calibration_affinity_attempt_01_broken/ 全部计划/错误及部分输出
- {LAB}/engineering_repairs/20261013_three_logic_fixes/{{REPAIR_REPORT_中文.md,RESULT.json,UNIT_RESULT_attempt_02.json,functional_4edge_attempt_02/,BEFORE/}}
- 从上述绑定直接定位并核对源图片、实际权重、checkpoint、历史可执行源、隔离环境依赖；不能只检查 MD 声称的 SHA。

## 审查清单（A–I）

A ground-truth/proxy 来源及 service 真实执行，不以模型输出制造 reference。
B 自归一化/截断/失败样本选择、真实 metric 定义、CVaR 分数质量及配对 run 统计。
C 所有主张对应实际文件/字段/数值，源码/计划/角色/原始 JPEG bytes、checkpoint 及 helper 闭包启动前/结束连续性，重算 M2d 结果。
D 三修复回归覆盖、B1 截止竞态修复是否真正消除双结局、每任务唯一次结局、FIFO 等待/active 与守恒并发、dead code。
E 实际 scope vs 声称，零样本 35→4 迁移映射/归一化/softmax/域迁移；fair actor/baseline inputs、current cache 而非 fresh 私有 queue、common schedule/uniform、同 resource/drain/初态、未用 test 挑设置。
F 真实本地 HTTP/UDP、应用层 delay/loss 注入 vs 真实无线网络范围；budget per recipient、监测开销、驱动干扰、负载发生器 backpressure。
G Windows wrapper/runtime PID、句柄错误 fail-closed、自有进程 shutdown、不杀未知进程；单机共享物理资源、线程≠quota、affinity≠完全隔离。
H 干扰标定严格配对性与因果边界：相同图像顺序/预热/线程、并发启动屏障、时间重叠、样本 CPU 与墙钟定义、三重复/顺序登记、拓扑获取与 SMT 归属、成功与失败尝试隔离、是否真正测 CPU 时间/CPU 墙钟比；"唯一因果/缓存睿频/保守估计"主张是否得到直接证据，不能从相关性替代测量。
I frozen_v3 当前身份与实际 M2d、v3 registry/grid/scales/全部日程/actor/generator binding、具体 cell 准入、review 校验/缺项拒绝、启动前/末尾 fresh 检查；B2 恢复的 common.py 与 M1 收据 SHA 匹配及影响分析。

## 关键 artifact 实际 SHA（供核对，勿改）

| 文件 | SHA-256 |
|---|---|
"""
for k, v in sorted(artifacts.items()):
    md += f"| {k} | `{v}` |\n"

md += f"""
当前 source_receipt 共 {len(source_dict)} 个交付源码文件，其字典如下（machine JSON 须原样引用 `source_sha256`）：

```json
{json.dumps(source_dict, indent=2, ensure_ascii=False, sort_keys=True)}
```

## 输出到新目录：{OUT}

1. `REVIEW_RESPONSE.md`：完整中文报告，各项 PASS/WARN/FAIL、确切 file:line/数值/证据、blockers、warnings、允许/不允许主张、有限复现的完整 command/路径、来源读一致性限制，不声称 Type-B。
2. `REVIEW_RESULT.json`：入口可读 schema `systemlab-independent-entry-v1`、scope `complete_system_entry`、status PASS/WARN/FAIL、blockers 数组真实完整、complete_report、reviewer_identity、`protocol_sha256`（当前 frozen_v3 actual）、`test_registry_sha256`（当前 E1_test_registry_v3 actual）、`source_sha256` 完整实际字典、report_path 绝对、report_sha256 完整报告字节；补 audit_checks、warnings、independence_type='Type-A'、independent_context=true、cross_family=false、verified_execution_receipts。若 FAIL 或未完成同样保存真实状态，禁止填空 blockers 冒充审批。
3. `EXPERIMENT_AUDIT.md`/`.json`（可由同一完整报告派生但不能掩盖不同 scope）；保存本 review 完整 prompt 及 response 用于 .aris forensic tracing。报告文件使用 apply_patch；历史来源只读。不要新增 Python 到 systemlab/tools 或改 source_receipt，因为那会改变待审身份。有限验证用现有脚本输出到 review 目录。你不需要寻论文；只审这个入口。

结束给主 agent 完整结论与所有产物路径、SHA、执行 receipt、未解决 blockers，不只提供摘要。
"""

(OUT / "REVIEW_REQUEST.md").write_text(md, encoding="utf-8")

# machine-readable index
index = dict(
    schema="systemlab-review-request-index-v1",
    target_identity="M2d / frozen_v3 / E1 v3",
    artifacts=artifacts,
    source_sha256=source_dict,
    source_file_count=len(source_dict),
    prior_review="reviews/entry_v2_independent_20261014/REVIEW_RESPONSE.md (FAIL, B1/B2 resolved by executor)",
)
(OUT / "REVIEW_REQUEST_INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

print("wrote REVIEW_REQUEST.md and INDEX.json to", OUT)
print("artifacts:", len(artifacts), "| source files:", len(source_dict))
