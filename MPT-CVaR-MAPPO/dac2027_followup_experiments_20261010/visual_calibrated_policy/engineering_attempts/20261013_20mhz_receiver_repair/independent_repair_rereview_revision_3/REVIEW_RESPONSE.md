# Revision 3 — 独立 Type-A source/runtime review

## 结论与范围

- Verdict：**FAIL**，1 个当前 blocker；保留 W1–W6 边界。
- Requested reviewer configuration：gpt-6-astra / xhigh。
- 日期：2026-10-13，Asia/Shanghai。
- 范围仅为 revision-3 的 20 MHz source/runtime review gate、当前完整 source/protected 文件身份及未改变的 8-decision 双臂接收边界。
- 只读阅读源码、差异、清单与保存证据；仅新增本目录两份报告。未执行项目代码或作者检查，没有生成计划、导入 Torch、运行 CUDA/GPU witness、模型、optimizer、smoke、pilot、formal 或 test。
- 这不是 experiment/run/Type-B authorization，也不是任何执行、性能、pilot 或正式就绪证明。
- 本次没有遇到 quota error；没有执行任何配额错误重试。

Revision 3 确实让实际 run()/worker() 传入现场重算的完整 maps，并增加了完整 map 精确比较。但是新 gate 对 own/resource 文件的归属要求与实际 209-source map 矛盾：资源规格 JSON 不属于该源码图，合法输入会在比较完整审查图之前被拒绝。作者的 synthetic PASS 未覆盖这一真实情况。

## Blocker R3-B1：资源规格被错误要求属于 209-source 图

### 直接证据

1. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:187](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:187) 至第 190 行：own_source_map() 包含六项 S4 文件、两项资源相关 Python，以及 `protocol/QLA_ENVIRONMENT_SPEC.json`，共 9 项。
2. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:215](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:215) 至第 221 行：visual 分支先要求 reviewed_source_map 长度 209、protected_map 长度 210，然后要求 `set(own).issubset(reviewed_source_map)`，并匹配全部 own SHA。
3. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:45](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:45) 至第 53 行：实际 source_lock() 来源是 scripts/*.py、历史 Python tree 和六项固定 additions；它不包含资源规格 JSON。
4. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217) 与 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373)：worker 与 run 实际传的是上述 source_lock() 的完整结果，另传 protected_bindings()。
5. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_3.py:54](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_complete_review_gate_revision_3.py:54) 至第 58 行：作者夹具把 own_map 定义为 synthetic source_map 的前九项，因而人为保证 own 是 source 子集，未模拟真实资源文件归属。

独立读取、枚举、SHA256 重算得到：

| 项目 | 实际结果 |
|---|---|
| source_lock 成员数 | 209 |
| protected_bindings 成员数 | 210 |
| own/resource 成员数 | 9 |
| own 中确实属于 source 的成员 | 8 |
| QLA_ENVIRONMENT_SPEC.json 属于 source | 否 |
| QLA_ENVIRONMENT_SPEC.json 属于 protected | 否 |
| 该资源规格文件 SHA256 | bd5442151a52e6e2fc9ab1033d3d9104ad744c6308d7d84bb1289475bd0dacb3 |

缺失路径：[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json)。

因此，即使在指定目录提供无 blocker 的完整报告，且报告内的 209/210 图都与当前真实字节完全一致，实际 gate 仍会在第 221 行抛出：

`Complete source map does not include the current own/resource bytes`

这是一条静态必然失败路径的分析，不是一次运行记录。我们没有执行 require_review、run 或 worker 来触发它。

### 影响与修复边界

第 222–225 行的完整 map 比较已存在，但合法实际输入不能到达该比较。重新生成计划不能修复资源文件归属矛盾；把本报告的完整 maps 落盘也不是执行通行证。

需要在不丢失资源 pin 的前提下明确独立 own/resource 身份与 source/protected 身份的验证契约。不能为了凑合通过而把资源 JSON 静默塞入原 209-source 图、删去某个源文件、放松资源 pin，或将 synthetic PASS 作为真实入口接受证明。本报告不实施任何修复。

## Revision 3 各项核对

### 实际调用点、完整图与旧审查复用

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:217) 和 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:373) 两处均直接调用：

- source_lock()；
- protected_bindings()；
- require_review(INDEPENDENT_REVIEW_DIR, reviewed_source_map=..., protected_map=...)。

没有从计划或作者输出提取这些 map 代替现场重算。第 42 行指定的新目录是 independent_repair_rereview_revision_3。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:197](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:197) 至第 202 行拒绝其他显式目录；第 215–217 行拒绝缺 map、非 dict 和旧九项图；第 222–225 行比较完整 path→SHA 字典，因此来源缺失、额外项、路径替换或任一值变化不能仅凭同数量过关。报告 status 必须 PASS/WARN、blockers 必须为空、review_type 必须 Type-A。

这些拒绝逻辑在源码中可见，未由本 reviewer 执行。由于 R3-B1，不能把它们描述为“实际完整 maps 已通过运行门禁”。

新计划仍按当前 source/protected/settings 构造，见 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:83](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:83)；但新 gate 的完整审查 map 比较已独立于计划存在，原则上会拒绝用新计划复用旧 source review。当前实现仍须先解决 R3-B1 才能获得有效的实际正向接收路径。

worker 在建模前和完成时、parent 在每臂前与最终汇总前继续调用 assert_runtime_bindings；这些检查绑定计划所列源和输入字节，并不赋予运行权限。

### reward 默认分支隔离

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193) 至第 196 行仍将无参数 require_review() 指向原 independent_command_rereview，并设置 visual_review=False。第 226–229 行保留原 own_source_map 比较。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_smoke_small.py:34](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_smoke_small.py:34) 和第 53 行没有改成 visual 新目录。

路径契约未被改写，但共享 helper 新 SHA 不意味着旧 reward review 自动更新；本报告没有重新认证 reward smoke，见 W3。

### preparation 引脚与 8-decision/two-8×30 receiver

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:41](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:41) 至第 47 行的六项 additions 引脚均与当前实际文件一致；其中改动的两项为：

- visual_model_smoke_small.py：`1e5c6a112caed73bc2235969fef3badfdb2af23f5dcce9fb28deee502dd115fa`。
- reward_input_integrity.py：`06a4b49ef92ee744664ac8277732adf7b727f9ea52621f3cc4397789f9e7a7c7`。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:263](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:263) 至第 293 行仍要求完整精确 20 MHz、8-decision、calibration-only plan，匹配 byte/semantic/source/protected 身份。第 294–421 行仍检查两个有序方法回执、共同合法 schedule SHA、重算 input-condition digest、精确原 calibration/mobility map、唯一完成 episode、8×30=240 counts、有限 update-statistics digest、checkpoint descriptor 和五事件链。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:503](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:503) 仍把 plan path/SHA 加入持续 inputs；第 534–535 行保存 plan 身份；第 540–561 行仍重读计划与 smoke 文件并重新校验。以上 receiver 逻辑未在 revision 3 改动：preparation 文件 diff 仅改变两项源 SHA 引脚。W1 的 paired-design 重算和真实 payload 审计限制保持。

### 20 MHz 方向及精确链路

方向文件 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/human_direction/20261013_visual_network_choice.json:1](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/human_direction/20261013_visual_network_choice.json:1) 当前 SHA：

`b0635013bd3d07dc330847c03baca313612813c836dbca174f266c050e270b32`。

其 upload-link 语义 SHA 为 `2870f4b3ef40d37e81dc21fcb65d64c3a3161e4324379c832ab3f0e3b3051d46`，两者未混用。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:424](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:424) 至第 443 行继续绑定固定路径、文件 SHA 参数、完整 schema、network choice 和 exact config。第 483–497 行的后续 entry review 仍要求外部可信用户方向核验，不将本地 JSON 自认为授权。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_upload_link.py:8](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_upload_link.py:8) 至第 39 行显示：20 MHz 条件仅改变 bandwidth_hz；noise_power_w=1e-10、antenna_gain=4.11、carrier_hz=915000000、tx_power_w=0.25、light_speed_m_s=300000000、min_distance_m=1 保持原值。2 MHz 条件和历史不利证据保留，不是第二套默认 full grid。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:266](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:266) / 第 273 行同时将同一 SMOKE_UPLOAD_LINK 传给 calibration feature scales 和实际环境；actor 维度第 276 行仍为 19。该链路实现及必要依赖当前 SHA 与 revision 2 相同，未重做已完成的无关语义审查。

### 资源身份

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json:12](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json:12) 仍定义 exclusive 1800 MiB；[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py:104](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py:104) 仍以 `>=` 拒绝达到阈值的 occupied memory。隔离 Python 3.13.5、NumPy 2.2.6、Torch 2.11.0+cu128、RTX 4060 Ti、单线程和无 CPU fallback 未变。

资源规格 file SHA 为 `bd5442151a52e6e2fc9ab1033d3d9104ad744c6308d7d84bb1289475bd0dacb3`；guard 中固定 canonical SHA 为 `8deaf4b6fbbf27332c795dcb5f08febde21269417f7879348e032643a9fe46f3`。本次仅审源码和文件身份，没有验证当前设备/占用或运行 CUDA witness。R3-B1 是 map 分类错误，不是放宽这些资源规则的理由。

## W1–W6 保留边界

### W1 — paired-design 和真实 payload

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:352](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:352) 仍只要求 paired_design_sha256 为合法 SHA，并在第 417 行比较两臂一致；未按 visual_training_transactions.py:114 的公式重算该内容身份。checkpoint/witness 字段也只做声明 SHA、descriptor 和 ledger 的一致性校验，不在 receiver 中直接重读 archive/NPZ 实际字节。真实执行来源与 payload audit 仍是单独要求；本地 True/计数字段不是独立执行证明。

### W2 — 作者夹具的有限证明力

保存的 AUTHOR_CHECKS_REVISION_3.json SHA 为 `b97ca17f3a86f9bdcd0ded58c60828f6dffc02416e47954acfc78060b64349b8`，与范围说明引用相符。脚本只从 AST 提取 require_review，在临时目录操作 synthetic maps/reviews；本 reviewer 未运行它。

它证明的范围由夹具决定：own_map 被设置为 source 子集，遗漏真实资源 JSON，正是 R3-B1 未被发现的原因。保存输出也没有绑定当前真实源 SHA。revision-2 旧作者证据的局限仍保留在旧报告中：七条 assert 语句报为八项、live import 并非纯标准库等；未覆盖或改写这些记录。

### W3 — reward 分支未重新认证

默认目录与旧契约保持，但共享 helper 的字节已变。旧 reward review 不能因 visual source review 完成而升级，本报告不授权 reward smoke。

### W4 — 后续 209-vs-203 边界仍在

retained_test_schedule_generation.py:148–159 继续使用普通 203-source map；当前 small smoke/preparation 使用 209。该后续接收器不能直接把 209-map receipt 当作自己的 exact 203-map receipt。本轮不扩大到 pilot→formal 打通。

### W5 — baseline 数量不是独立 membership 锁

preparation:61–73 自身仍只检查 baseline 数量 203，加六项精确 additions。revision 3 增加的完整 review-map equality 理论上可对固定已审图拒绝同数量路径替换，但当前真实正向路径受 R3-B1 阻断。本报告直接嵌入并绑定当前精确 209 图，不接受任意路径替换。

### W6 — 本地方向记录不等于人类授权

direction JSON 显式声明 local_record_alone_is_not_authorization。除委派上下文外，本次未额外检索原人类消息；后续独立 entry review 的外部 provenance 与允许动作仍需单独核验。本 source/runtime review 不代替 experiment/run/Type-B authorization。

## 完整身份与旧报告保全

JSON 中直接包含完整 reviewed_sources_sha256（209 项）与 protected_input_and_source_sha256（210 项），不是只存 digest。另单列 independent_own_resource_pins_sha256，明确资源 spec 不在这两个图内；未为通过 gate 改变 map 成员或数量。

清单算法：将精确 Windows 绝对路径按 ordinal 排序，逐项连接 path + NUL + lowercase SHA256 + LF，对 UTF-8 字节计算 SHA256。

- 当前 source inventory：`cfe6306e18e16d31204d4cec9402c90f30fe50167e3127ed0616425bb3c94692`。
- 当前 protected inventory：`29945dbe9ca758058b6f255f993c8f7e084705d33ce87258fd8ba8d8c45abbb5`。
- protected 实际字节不匹配：0。
- current source 相对 revision 2 仅三文件 SHA 变化；路径成员数仍 209。
- SOURCE_BEFORE_REVISION_3 三份字节与 revision-2 报告中的当前 SHA 一致。
- 旧 A4 smoke SHA 仍为 `207742f1663774237a71f8e1ec170cad0ed206255da68c562ca703a1002899db`，仅属于旧来源身份。

| 当前变更文件 | SHA256 |
|---|---|
| visual_model_smoke_small.py | 1e5c6a112caed73bc2235969fef3badfdb2af23f5dcce9fb28deee502dd115fa |
| reward_input_integrity.py | 06a4b49ef92ee744664ac8277732adf7b727f9ea52621f3cc4397789f9e7a7c7 |
| visual_pilot_preparation.py | 050459e8fac43e491b446ef6e759bda92def5eb5bb8735a517f930337efdf3fc |

旧报告保持：

- revision-2 REVIEW_RESPONSE.md：`0559d5beb0849ab6443d1f4250a2b35aa5a237075b7c94d9cec68ad2afd9e30d`。
- revision-2 REVIEW_RESULT.json：`1b8b1875f6c6c4f1b22aaf447ad6df34e569482b2d61f44b063bbde26eebe56f`。
- 更早完整 FAIL MD：`efa1bb893e5275aee62ecb9501af3d3f0d53752dae4c40959e5995502fe6daa9`。
- 更早完整 FAIL JSON：`c04a0243b486faf549ac4bdfe9782ebf63f5c9be9c1d2d2e27368848d2f0504c`。

## 最终边界

此次完成 source/runtime review 与文件身份核对，不是执行。FAIL 来自合法实际 map 无法通过 gate 的确定性源码矛盾。没有启动或完成任何 20 MHz smoke、pilot、formal/test，也没有授予任何运行、性能或 Type-B 结论。

