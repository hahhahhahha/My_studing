# Revision 5 — 独立 Type-A 限范围源码复审

## 结论

**FAIL：1 个 blocker。** R4_B1 目录路由已修，revision-4 夹具的前置条件污染已修；preparation 中的 helper SHA 仍旧，导致完整六项绑定不一致。

Requested reviewer configuration：gpt-6-astra / xhigh。日期：2026-10-13，Asia/Shanghai。

范围仅为 caller-directory 修复、preparation pins、revision-5 fixture 隔离改进和完整 209/210/9 文件身份。没有执行任何项目代码、作者夹具、Torch/GPU、plan/smoke/pilot/formal/test 或模型更新；仅保存本目录两份报告。旧 revision 1–4 报告未覆盖。本报告不提供运行或 Type-B 授权。

## Blocker R5-B1：preparation 仍引用旧 helper SHA

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:44](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:44) 将 reward_input_integrity.py 固定为 revision-4 SHA：

`0d46288c0019a8d5c6597b6052d010d099c90aeb50da1cd9620c67b7c9de44ae`

而当前 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199) 已把 helper allowlist 改成 revision 5，其实际 SHA 为：

`693bde8850879fe861e8bd13a387ef9cb8fd96a924148b3e02a30cf1ab0d8c39`

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:61](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:61) 至第 73 行的 _small_smoke_source_lock() 会逐项核对六项 additions 的真实 SHA；第 71–72 行必然拒绝上述差异。静态预期错误为：

`A4 small-smoke source input changed; no old result may be relabeled`

这一结论来自实际字节哈希与源常量比较，未调用该函数。

caller 引脚已经正确更新为 `40709cd0fe6485373da4ec1f189503dd30e804c2190b08cc97d107b923c69a23`；问题是另一个随目录改动而变化的 helper pin 没有同步。因此不能把“caller pin 正确”扩大为“完整 preparation map 可接收当前 revision”。

应让所有受影响 pin 对应同一实际源码版本，再取得新完整身份审查；不得放宽校验或改写旧报告。本 reviewer 不改源码。

## 本次已确认修复

### R4_B1：真实 caller 与 helper 目录一致

- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:42](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:42) 当前为 independent_repair_rereview_revision_5。
- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217) 和 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373) 分别在 worker/run 传该常量，并现场调用 source_lock()、protected_bindings()；不是由计划或作者输出提供 map。
- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199) 当前 allowlist 也是 independent_repair_rereview_revision_5。
- preparation 第 42 行 caller pin 与当前真实 caller 字节相同。

因此 revision-4 的 caller-directory mismatch 已消除。helper 的完整 209/210 字典比较、独立九项 own/resource 图、真实 8+1 分布、资源 canonical digest 与 strict 1800 检查没有在本轮放宽；helper diff 仅改变允许目录。

### fixture：先确认 caller，恢复合法资源前置条件

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_5.py:88](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_5.py:88) 至第 96 行从真实 smoke 源码 AST 读取 INDEPENDENT_REVIEW_DIR，并检查 revision5 后缀。这是源码后缀检查，不是运行真实 caller，也不是完整 integration test。

第 113–118 行测试旧 500 policy 后，[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_5.py:120](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_5.py:120) 至第 126 行已恢复：

- 1800 spec 文件；
- baseline canonical digest；
- own resource SHA；
- allowed review 的 own map；
- legacy review。

随后第 139–140 行才执行 source-only / protected-only mutation negatives。revision-4 中资源前置条件已坏、提前触发拒绝而掩盖 map 分支的具体问题已修。函数仍在 synthetic/temp 环境中测试，不创建实验计划，不启动模型。

保存的 AUTHOR_CHECKS_REVISION_5.json SHA 为 `0779618f2d6831a3e9f80c6a8eee37c4ed41bad13f4d26855d5dab45fab54ea9`。本 reviewer 只读其内容及对应脚本，未重新执行；saved PASS 仍是作者合成证据，没有覆盖当前 stale preparation helper pin。

## 独立完整清单

已重新枚举 scripts/*.py、历史 Python source tree 与六项 additions，并按实际 protected_bindings 的历史映射分离 source 与 immutable inputs。所有 protected 字节匹配。

| 清单 | 数量 | SHA256 |
|---|---:|---|
| source | 209（203+6） | 548f329dc4c8ca4b4da35c0e3bc8d2cdb09256165f414937c64747e5b0cde1fc |
| protected | 210 | 29945dbe9ca758058b6f255f993c8f7e084705d33ce87258fd8ba8d8c45abbb5 |
| own/resource | 9（8+1） | 87d6f9e9e01e646f8bb8c0f1792c27714bfcf273985c842de208a28f2c65714d |

算法：精确 Windows 绝对路径按 ordinal 排序，逐项连接 path + NUL + lowercase SHA256 + LF，对 UTF-8 字节计算 SHA256。

完整三图直接写入 JSON 的 reviewed_sources_sha256、protected_input_and_source_sha256、independent_own_resource_pins_sha256。另写入 preparation_declared_addition_pins_sha256，保留“声明值”和“实际值”的区别，没有伪造图来满足接收器。

| 当前文件 | 实际 SHA256 |
|---|---|
| visual_model_smoke_small.py | 40709cd0fe6485373da4ec1f189503dd30e804c2190b08cc97d107b923c69a23 |
| reward_input_integrity.py | 693bde8850879fe861e8bd13a387ef9cb8fd96a924148b3e02a30cf1ab0d8c39 |
| visual_pilot_preparation.py | e2d59d925c2b1da81ec67359ed9fa50418a118684274b15a33a074995f94c10b |

全图相对 revision 4 仅上述三文件改变；其 before snapshots 均匹配旧报告。资源规格与之前相同，独立 own/resource 字段仍完整保留其 SHA；未实施任何实时资源核查。

## 保留 W1–W6 边界

- W1：paired-design 尚未按内容公式重算；checkpoint/witness 声明不能替代实际 payload 与执行来源审计。
- W2：作者夹具只证明其合成范围。本轮特定隔离错误已修，但 caller 检查仅为 AST 后缀，未测试 preparation pins，saved PASS 不是真实接收或运行证明。
- W3：reward 默认目录保持不变；共享 helper 的新字节没有自动重新认证旧 reward review。
- W4：后续 retained metadata-generation 的 203-source 与此 small-smoke 的 209-source 边界仍未打通，本轮不扩大范围。
- W5：preparation 的 baseline 数量检查不等于独立路径清单锁；完整 review-map 比较仍需保留。当前 preparation 还受 R5-B1 阻断。
- W6：本地方向记录不等于人类授权。本次没有额外用户方向核验，source review 不代替运行或 Type-B 权限。

旧 W1–W6 的完整历史证据仍保留在原报告；本轮未修改科学参数、20 MHz link、8-decision/8×30 语义、旧 2 MHz 证据、预算或资源规则。

## 旧报告与只读说明

revision 1–4 的八份 MD/JSON SHA 均重新读取核对，并记录于新 JSON 的 reviewed_artifacts_sha256。特别是 revision-4：

- MD：`8e981e94139ced2bde8bce55d542f0ca991d8bfd4b4f435fd43bb240c04ba40c`。
- JSON：`6f9f2f8b2f8a099274ab69425153eec7c6156529ebe63ef69cd656cee623d032`。

可选范围文件 REVISION_5_SCOPE.md 在本次读取时不存在，因此本轮范围以委派请求与实际 diff 为准。一次只读哈希辅助命名与 PowerShell H alias 冲突，随后改用无歧义名称完成；未写文件、执行项目代码或触发 quota retry。

## 最终边界

本次完成限范围静态审查和完整身份核对。R4_B1 与夹具隔离修复成立；总体 FAIL 来自 stale helper pin。没有实际 smoke、pilot、formal/test、性能或运行授权结论。

