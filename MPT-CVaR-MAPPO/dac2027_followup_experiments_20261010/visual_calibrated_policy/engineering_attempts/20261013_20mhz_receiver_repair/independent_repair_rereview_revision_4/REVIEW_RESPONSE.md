# Revision 4 — Independent Type-A source/runtime review

## 结论与边界

**FAIL：1 个 blocker，保留 W1–W6。**

Requested reviewer configuration：gpt-6-astra / xhigh。日期：2026-10-13，Asia/Shanghai。

本轮仅审 revision-4 gate、真实调用点和当前三类绑定，不重新扩大到实验设计或正式入口。对完整 209-source、210-protected 与 9-own/resource 图进行了独立枚举和 SHA256 核验。没有执行项目代码、作者夹具、计划、Torch、GPU/CUDA witness、模型、optimizer、smoke、pilot、formal 或 test；只新增本目录两份报告。未遇 quota error，没有错误重试。

revision 3 的资源成员归属缺陷已在 helper 内修复，但实际调用者仍指向 revision 3，不能消费 revision 4 的报告。因此当前没有无阻断的 source/runtime 接收结论，更不构成 experiment/run/Type-B authorization。

## Blocker R4-B1 — 实际入口仍传 revision 3 目录

证据：

- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:42](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:42)：INDEPENDENT_REVIEW_DIR 仍为 `independent_repair_rereview_revision_3`。
- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217)：worker() 把该常量与现场 source_lock()/protected_bindings() 传给 require_review。
- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373)：run() 也传同一常量。
- [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:199) 至第 201 行：新 gate 只接受 `independent_repair_rereview_revision_4`，不匹配时立即抛 ValueError。

当前 small-smoke 文件 SHA 仍为 `1e5c6a112caed73bc2235969fef3badfdb2af23f5dcce9fb28deee502dd115fa`，与 SOURCE_BEFORE_REVISION_4 和 revision-3 报告完全相同。独立全图比较确认本轮只改变 helper 和 preparation 两个文件，没有改变 smoke 调用者。

因此，真实 run/worker 会在读取任何审查报告之前，被错误目录检查拒绝。即使本目录含合法完整的 209/210/9 图，实际入口也无法消费；新计划不能改变该调用常量。

静态预期异常文本：

`Only the separately designated 20 MHz repair review may authorize this visual entry`

本 reviewer 未执行该调用。修复需一致地绑定真实调用目录及相应源 SHA，再取得新的精确来源审查；不能放宽 allowlist 或改写旧报告。本报告不修改源码。

## revision-4 helper 的正确改善

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:214](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:214) 至第 235 行已将三类身份分开：

1. source map 必须是完整 209 项 dict。
2. protected map 必须是完整 210 项 dict。
3. own/resource map 由 actual own_source_map() 重算，必须完整等于报告中的 independent_own_resource_pins_sha256。
4. 八项 own-source 必须属于 source map 且 SHA 一致；唯一独立项必须为 exact active QLA_ENVIRONMENT_SPEC.json 路径。
5. 该规格文件再次核对 SHA、FROZEN_BASELINE 的 canonical digest 及 gpu_occupancy_max_mib_exclusive == 1800。
6. 第 232–235 行仍比较完整 source/protected 字典，不以成员数量代替完整路径和 SHA 比较。

这修正了 revision-3 把资源 JSON 强塞进 source map 的问题。资源 JSON 仍在两个大图之外，没有删源或扩图凑数。

独立核对的真实数据：

| 身份 | 数量 | 清单 SHA256 |
|---|---:|---|
| source | 209（203+6） | af433fa87ff51251a8ce86b393880b38de2e6a88fe3a53b3618bc22e5c3ca2b1 |
| protected | 210 | 29945dbe9ca758058b6f255f993c8f7e084705d33ce87258fd8ba8d8c45abbb5 |
| own/resource | 9（8+1） | 4cffb1344f3f54c7a5ef112fe77299949660412807539d1baa1e084fd8f09df1 |

清单算法：按精确 Windows 绝对路径 ordinal 排序，连接 path + NUL + lowercase SHA256 + LF，对 UTF-8 字节计算 SHA256。所有 protected 实际字节匹配，零不一致。完整三图直接嵌入 REVIEW_RESULT.json，不仅提供摘要。

资源规格实际文件 SHA：`bd5442151a52e6e2fc9ab1033d3d9104ad744c6308d7d84bb1289475bd0dacb3`。独立对当前 JSON canonical 序列化计算的 SHA 与 FROZEN_BASELINE 对应字段均为 `8deaf4b6fbbf27332c795dcb5f08febde21269417f7879348e032643a9fe46f3`；exclusive 阈值仍为 1800。已知资源 guard 的 >=1800 拒绝规则没有改变。未核查实时设备占用、可用性或运行时环境。

## 真实调用及 preparation 绑定

run/worker 仍各自调用 source_lock()/protected_bindings()，没有从新计划或作者检查文件拿 map 代替现场值；但目录不一致构成 R4-B1。

preparation 只将 helper 引脚更新为 `0d46288c0019a8d5c6597b6052d010d099c90aeb50da1cd9620c67b7c9de44ae`，small-smoke 引脚仍正确匹配当前未变文件。preparation 当前 SHA 为 `109b7f0b2bf0ec18d04db7560e222c67e9ec9401ad4bce19aca310af7f76ec9a`。与发审锚点独立匹配。

20 MHz link、方向记录、8-decision / 双臂 8×30 receiver、持续 plan 边界和科学参数均没有本轮语义改动；本次按限定范围不重做已完成的旧审查，也不将其转成新运行结论。

reward 无参数 require_review() 仍走原 independent_command_rereview；helper 第 236–239 行保留旧 own-map 契约。路径隔离不等于旧 reward review 对新 helper 已重新认证。

## 作者夹具及证据限制

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_4.py:65](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_4.py:65) 至第 87 行确实采用 8-source + 1-resource 的 synthetic map 形状，并检查 exact maps 正例；missing/changed resource-review field 和 500 阈值负例也直接可见。它仅从 AST 提取 helper 函数，不读取真实 smoke 入口的目录常量，因此遗漏 R4-B1。

此外，[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_4.py:104](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_4.py:104) 至第 109 行把资源规格改为 500 并同步 baseline、own map、review。随后没有恢复 1800；第 122–123 行才执行 changed_source/changed_protected 负例。此时合法资源前置条件已经不成立，这两个负例可以先因 500-policy 被拒绝，不能隔离证明“对应 map 变化导致拒绝”。完整字典比较在 helper 源码中确实存在，但 saved PASS 不能扩大为独立测试证明。

AUTHOR_CHECKS_REVISION_4.json 实际 SHA 为 `4bd12375a74d066a8d094a6666f4a7a2fba349352be31d62b4c287b046f3bf2d`，与发审锚点匹配。本 reviewer 没有运行或修正该夹具。

## W1–W6 保留

- **W1：paired-design / 实际 payload。** receiver 仍仅验证 paired-design SHA 语法与两臂相同，未按内容公式重算；checkpoint/witness 声明不替代真实 archive/NPZ 字节和执行来源审计。
- **W2：作者夹具。** synthetic PASS 不是实际接收、smoke 或执行证明。本轮新增限制为真实 caller 未覆盖及两个 map 负例受 500-policy 前置失败干扰。旧报告的历史夹具局限继续保留。
- **W3：reward 未重新认证。** 默认路径保持不变，共享 helper 字节变化不能由 visual review 自动升级旧 reward review。
- **W4：后续 209-vs-203。** retained metadata-generation 仍使用普通 203-source map，本轮不修复或授权 pilot→formal 接收边界。
- **W5：baseline membership。** preparation 自身仍只检查 203 数量加六项 pins；完整 review-map equality 可以约束固定来源，但真实调用仍受 R4-B1 阻断。本报告只绑定当前精确图。
- **W6：人类授权。** 本地方向 JSON 不是人类授权证明，本次没有追加原始用户消息核验。source/runtime review 不代替运行权限或 Type-B 审查。

## 当前来源与旧证据

| 受审入口 | 当前 SHA256 |
|---|---|
| reward_input_integrity.py | 0d46288c0019a8d5c6597b6052d010d099c90aeb50da1cd9620c67b7c9de44ae |
| visual_model_smoke_small.py | 1e5c6a112caed73bc2235969fef3badfdb2af23f5dcce9fb28deee502dd115fa |
| visual_pilot_preparation.py | 109b7f0b2bf0ec18d04db7560e222c67e9ec9401ad4bce19aca310af7f76ec9a |

SOURCE_BEFORE_REVISION_4 三份快照分别匹配 revision-3 原 SHA。旧 revision-1/2/3 的 MD/JSON 均未改写，其 SHA 记录在新 JSON 的 reviewed_artifacts_sha256：

- revision-3 MD：`6622877a2fda7bce8f4822af374710e1b904c52acbb3036da55b8b03f4a686c2`。
- revision-3 JSON：`a1762cc116e8639f0c355ef5720c3c85c3ee13b25821aca6d3e4f00c0e66ec96`。
- revision-2 MD：`0559d5beb0849ab6443d1f4250a2b35aa5a237075b7c94d9cec68ad2afd9e30d`。
- revision-2 JSON：`1b8b1875f6c6c4f1b22aaf447ad6df34e569482b2d61f44b063bbde26eebe56f`。

## 最终边界

三类完整图与资源 canonical 身份已独立核对；revision-3 的 map 分类 bug 已修。当前总体 FAIL 来自真实调用者与新 gate 的目录不一致。报告不启动任何后续工作，也不授予 experiment/run、smoke、pilot、formal/test、性能或 Type-B 结论。

