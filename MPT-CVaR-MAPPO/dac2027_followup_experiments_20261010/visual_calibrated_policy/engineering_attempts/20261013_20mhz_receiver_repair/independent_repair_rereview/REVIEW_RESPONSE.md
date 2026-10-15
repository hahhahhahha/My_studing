# Revision-2 独立 Type-A 源码复审

## 结论

- 日期：2026-10-13，Asia/Shanghai。
- Requested reviewer configuration：gpt-6-astra / xhigh。
- Verdict：**FAIL**。1 个当前阻断项，6 项边界警告。
- 本次只审 revision-2 改动和必要依赖；对完整 209 项 source / 210 项 protected 做了独立文件身份核验，但不声称逐份语义审查全部 209 个文件。
- 旧 B1、B2、B3 的具体缺陷已有源码层修复；实际 smoke 的新审查入口仍没有强制绑定完整已审来源身份，因此不能给出无阻断的源码接收结论。
- 未运行任何项目代码、自检脚本、模型、Torch、GPU 探测、计划生成、smoke、pilot 或正式实验。仅按授权保存本目录的两份报告；源码和旧报告未改。
- 本报告不证明实际执行、性能、pilot 完成或正式就绪，也不提供运行授权。

## 当前阻断项

### R2-B1 — 完整审查来源没有进入实际 smoke 的强制接收边界

证据：

1. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:187](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:187) 的 own_source_map() 只有六项 S4 additions，加 qla_environment.py、windows_runtime.py、QLA_ENVIRONMENT_SPEC.json，共 9 项。
2. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193) 至第 209 行虽然选择新的独立目录，但只校验 status、blockers、review_type 和这 9 项 reviewed_sources_sha256。它没有读取本报告的完整 209-source / 210-protected 清单或其清单身份，也没有检查完整审查 scope/completeness。
3. [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:45](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:45) 的 source_lock() 动态枚举当前 scripts 与历史 Python 树；[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:83](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:83) 的 expected_plan() 按当前源重新构造计划；第 90–102 行仅把传入计划与当前 expected_plan 比较。
4. 实际 worker 在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:215](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:215)、parent run 在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:369](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:369) 调用这个 9 项接收器后再读计划。

因此，如果九项之外的源码（例如 visual_method_features.py 或 visual_pilot_preparation.py）改变，并生成符合当前文件的计划，九项 own/resource map 不变，旧完整来源的审查仍可能满足运行门槛。当前计划与当前文件一致，不等于当前文件经过该份审查。此结论来自静态调用链；没有进行变更、夹具执行或启动试验。

修复边界应让 visual entry 在接受计划前将完整当前 source/protected 清单与指定新审查精确比较，同时保持 own/resource pins。仅换目录或重新生成计划不能代替新来源审查。本报告不修改该实现。

## 三项旧阻断的复核

| 旧项 | 本次源码结论 | 核对内容 |
|---|---|---|
| B1：旧 232 项 stale protected map | PASS_SOURCE | 当前源码与历史不可变输入分离；实际小入口和普通入口都得出同一 210 项 protected。 |
| B2：预算计数冒充实际完成计数 | PASS_SOURCE，保留 W1 | 检查唯一完成 episode、实际 8×30 counts、有限更新统计 SHA、checkpoint descriptor 和完整有序事件链。 |
| B3：计划未入持续边界 | PASS_SOURCE | 计划绝对路径/文件 SHA 持久保存，每次 boundary 重读并再次校验。 |

### B1 详细证据

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:56](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:56) 至第 80 行固定 PRIOR 字节身份，确认原始两份 map 为 dict；从完整当前 source map 中排除旧实现 SHA 的重复约束，所有其余历史输入逐项复核原 SHA；历史 new_implementation 中的 Python 文件若不在当前来源锁中会失败。非 Python 的原 README 保留为受保护证据，未因“实现”标签被误排除。

普通入口对应实现位于 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_model_smoke.py:44](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_model_smoke.py:44)。独立只读重算确认：

- baseline 203 + exact six additions = 209。
- actual-small 与 ordinary 的 protected map 同为 210，清单身份一致，210 项实际字节全部匹配。
- 历史 Python implementation 无缺失。
- 固定 PRIOR 实际 SHA：`4ed5a34d5b55b8465813a162c87e947eef27d73041e3b42706998cbe65359b5a`。

### B2 详细证据

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:294](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:294) 至第 330 行要求按方法顺序排列的两个回执、完整 input-condition schema、合法小写 SHA，重算 input_condition digest，检查 exact calibration/mobility map、共用 settings、20 MHz upload link。缺失共同 SHA 不再通过 None == None。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:331](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:331) 至第 398 行要求：

- 完成 phase 为 budget_complete_pending_validation；无 failure/current_episode。
- 正确 method、seed、一个 8-decision budget。
- 唯一 committed episode，8 decisions、30 users、240 samples、0.4 秒及对应 reward semantics。
- 非空有限更新统计的 digest 等于提交记录。
- counts 与 episode token 一致，checkpoint SHA 和 descriptor 与已提交计数一致。
- begin_collect → collection_closed → begin_update → update_committed → checkpoint_attached 五事件精确匹配，禁止缺失或乱序。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:399](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:399) 至第 421 行还要求每臂同一 source/protected/plan 身份、参数 SHA 实际不同的回执字段、两份 roundtrip witness SHA、相同 schedule/input-condition/paired-design，并重算汇总 input-condition SHA。

生产端对应 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:283](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:283) 至第 354 行；ledger 对应 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_transactions.py:123](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_transactions.py:123) 至第 221 行。实际计数生成在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_coordinator.py:91](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_coordinator.py:91)，检查完整 T×N 数据后生成 ClosedEpisodeCounts。这些字段结构相容；旧保存的 A4 回执也显示相同 8×30/0.4 秒事件结构，但其旧来源身份绝不能升级成当前 20 MHz 结果。

上述是接收 schema 与生产源码一致性的结论，不是本次已完成模型更新或 checkpoint roundtrip 的证明。W1 中说明 paired-design 重算和实际 payload 审计的边界。

### B3 详细证据

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:469](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:469) 初次读 smoke 及其计划；[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:503](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:503) 将解析后的计划绝对路径与文件 SHA 加入 files；[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:532](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:532) 至第 535 行保存对应 source/protected/path/hash/semantic identity。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:540](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:540) 至第 561 行重新核对完整当前来源锁、所有持久 inputs、candidate、smoke 文件和同一个计划文件，再调用相同严格 receiver；异常将 _failed 置为 True。_read_actual 在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_grid_registration.py:45](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_grid_registration.py:45) 至第 64 行检查根目录、路径重定向、失败标志、文件 SHA 并严格读取 JSON。

因此计划后续被删除或改变不再被该边界漏掉。作者对这一点只提供源断言，而本次也未运行对象级删除/变更试验。

## 方向、链路与资源

方向文件 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/human_direction/20261013_visual_network_choice.json:1](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/protocol/human_direction/20261013_visual_network_choice.json:1) 实际 SHA 为 `b0635013bd3d07dc330847c03baca313612813c836dbca174f266c050e270b32`；其中 `2870f4b3ef40d37e81dc21fcb65d64c3a3161e4324379c832ab3f0e3b3051d46` 是 upload-link 语义 SHA，不是文件 SHA。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:424](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:424) 至第 443 行区分这两者，绑定固定路径、完整 schema、network choice 和 exact UploadLinkConfig。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_upload_link.py:8](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_upload_link.py:8) 至第 39 行保留 2 MHz 默认，注册 20 MHz 条件仅改变 bandwidth_hz；noise/power/carrier 等常量未变。实际小入口 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:265](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:265) / [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:272](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py:272) 将同一个 SMOKE_UPLOAD_LINK 传给 calibration scales 和环境。空请求 fallback [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_method_features.py:100](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_method_features.py:100) / 第 112 行、有请求 JPEG 传输 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_environment_adapter.py:100](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_environment_adapter.py:100)、串行入队第 140 行仍共用相同链路语义。两方法仍用相同输入布局；小入口第 275 行的 actor dimension 为 19。训练/验证环境匹配在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_coordinator.py:175](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_coordinator.py:175)，保留测试匹配在 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_evaluator.py:175](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_evaluator.py:175)。

本次三文件源码差异没有修改模型、算法、reward、horizon、H、split、服务输入或预算；当前全来源清单将三项 SHA 替换为 before-snapshot 后，恰好恢复旧完整 FAIL 的来源清单身份，进一步确认该来源集合内仅三项文件发生 revision-2 变化。

资源规格文件 SHA 仍为 `bd5442151a52e6e2fc9ab1033d3d9104ad744c6308d7d84bb1289475bd0dacb3`。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py:104](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py:104) 仍在 occupied memory >=1800 MiB 时拒绝，随后才可进入 seeded CUDA witness；[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_learned_bindings.py:50](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_learned_bindings.py:50) 至第 68 行仍要求隔离环境、gate 和 CUDA，无 CPU fallback。Python/NumPy/Torch/设备/单线程规范未变。本次没有运行 gate，不报告当前资源可用。

## 警告及不包含的结论

### W1 — paired-design 和实际 payload 仍需独立核对

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:352](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:352) 只验证 paired_design_sha256 的语法，并在第 417 行比较两臂相同；没有按 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_transactions.py:114](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_training_transactions.py:114) 的公式重算 seed + budget + schedules + common_condition。作者正夹具 [X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_receiver_repair.py:134](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_receiver_repair.py:134) 使用 `'b'*64`。因此不能声称该字段已验证为真实 paired-design 内容身份。

checkpoint/witness 在 receiver 中也只是声明 SHA 与 ledger/descriptor 的内部一致性；它未直接重读 checkpoint archive 或两份 NPZ 字节。实际执行来源、模型更新和 payload 审计仍是单独边界，不能由本地 JSON 中的 True 字段代替。

### W2 — 作者检查是有限夹具证据

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/AUTHOR_CHECKS_REVISION_2.json:1](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/AUTHOR_CHECKS_REVISION_2.json:1) 保存的三项 current source SHA 与本次实际字节一致，209/210 身份也一致。其 12 项负例和 1 项正例逻辑已直接阅读，但本次未执行。

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_receiver_repair.py:94](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261013_20mhz_receiver_repair/check_20mhz_receiver_repair.py:94) 的 review-directory fixture 把 own_source_map 换成一项 mock，没有测试完整 source closure 变更。第 256–270 行有七条 assert 语句，却返回常量 8；不能把保存的 8 解释为实际执行的 assert 语句计数。第 273–280 行 live import 会导入 NumPy 并进行 Windows process-local 初始化，因此文件首行 standard-library-only 描述不准确。预期检查路径未调用模型/GPU 或测试角色；保存输出仍只是作者声称的结果，不是独立执行证明。

### W3 — reward 默认目录保持不变，但没有重新认证

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_input_integrity.py:193) 的无参数默认仍指向 independent_command_rereview；visual worker/run 两处才显式指定新 independent_repair_rereview。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_smoke_small.py:32](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4/reward_smoke_small.py:32) 和第 53 行仍使用无参数默认。

共享 helper 字节变化使旧 reward review 不再天然匹配；本报告不重标旧审查，也不提供 reward smoke 运行资格。

### W4 — 正式下游仍是单独边界

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_schedule_generation.py:148](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/retained_test_schedule_generation.py:148) 至第 159 行仍从普通入口取得 203-source map，不能把当前 209-map small smoke 直接当成自己的 exact source receipt。此后续边界不是本轮修复内容；没有完成 pilot→formal 贯通的结论。

### W5 — baseline membership 的内部检查仍仅为数量

[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:61](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:61) 至第 73 行精确固定六项 additions，但 baseline 自身只检查 len==203。同数量路径替换不能仅靠数量检查拒绝。当前 209 路径集合已独立核对，与旧 A4 209-map membership 相同；后续变更仍需完整已审 map，见 R2-B1。

### W6 — 本地 human-direction JSON 不是授权来源

该文件第 19 行明确声明 local_record_alone_is_not_authorization。本次未额外检索原人类消息，只按委派上下文审查源码。[X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:483](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_pilot_preparation.py:483) 至第 497 行仍要求另一个独立 entry review 验证实际用户方向，并限定工程动作，明确不授予正式训练/测试和冻结权限。本报告不代替这份外部核验。

## 来源与旧证据保全

完整 path→SHA 清单见本目录 REVIEW_RESULT.json 的 reviewed_sources_sha256（209）与 protected_input_and_source_sha256（210）。其 semantic review 范围明确有限；directly_inspected_sources_and_dependencies_sha256 单列直接阅读依赖。

清单身份算法：按精确 Windows 绝对路径 ordinal 排序，拼接 path + NUL + lowercase SHA256 + LF，对 UTF-8 字节计算 SHA256。

- 当前来源身份：`5661d250203daa4902dcc0c3e52dc49e2617541fdfd3afe619c587fc0590da79`。
- 当前 protected 身份：`29945dbe9ca758058b6f255f993c8f7e084705d33ce87258fd8ba8d8c45abbb5`。
- 三项 before SHA 还原后的旧来源身份：`b22e30dfeae5e8974f8ed1a2539e90b6da5242895932ed86f5815886e13ce363`。
- 旧 A4 smoke 文件 SHA 仍是 `207742f1663774237a71f8e1ec170cad0ed206255da68c562ca703a1002899db`；只保留其旧来源身份。

| 文件 | 当前 SHA256 |
|---|---|
| visual_model_smoke_small.py | 3f67de57302ed19e22fd70bb6a189b698bde40652fdd9166fa1a25a2a23e327b |
| reward_input_integrity.py | 3d92695b38a4b4576f3976c4796525f7384714bca299ff134b4bd4369a1fb37f |
| visual_pilot_preparation.py | 5c6bfa1e89f508fa61843f8d62d80a27a8a1cd88cf1b90139760360a8fe489fe |
| 原完整 FAIL：REVIEW_RESPONSE.md | efa1bb893e5275aee62ecb9501af3d3f0d53752dae4c40959e5995502fe6daa9 |
| 原完整 FAIL：REVIEW_RESULT.json | c04a0243b486faf549ac4bdfe9782ebf63f5c9be9c1d2d2e27368848d2f0504c |

SOURCE_BEFORE_REVISION_2 中三项 SHA 与旧完整 FAIL 对应项一一匹配：small smoke `cc2b14cafde350511322f6111b0fdaf442bf1d40d0554a94df76909448b26bdd`，reward helper `1f774d8322a72b8fdb82fb3f5c2a49353e22d4fb8f45cf5864ec872926837cf3`，preparation `ae934106b9ae438874b7e4dfdd28a46c3d6c1d79950802336a7ac55fbf2576e2`。旧报告和快照保留，不覆盖。

## 最终边界

本轮完成的是有限源码复审及身份核对。总体 FAIL 由 R2-B1 决定；三项旧缺陷的 PASS_SOURCE 不覆盖这一阻断。没有生成新计划或实际 20 MHz 回执，没有实际 smoke/性能/pilot/正式就绪结论，没有执行授权。

