# REVIEW_RESPONSE.md

## 审查结论

- Review type：Type-A
- Requested reviewer configuration：gpt-6-astra / xhigh
- Date：2026-10-13，Asia/Shanghai
- Verdict：FAIL
- Scope：当前 20 MHz 每用户正交链路适配、小规模双臂 8×30 烟测入口，以及其 pilot 接收边界的有限只读源码复审。
- 本报告不证明实际烟测成功、性能提升、pilot 完成或正式实验就绪，不提供执行授权。
- 审查方法：直接阅读当前源码及保存的检查记录；只读 SHA256、源码差异和清单比较。未执行项目代码或检查脚本，未导入模型，未探测/分配 GPU，未写入文件。
- 对完整当前源集合进行了文件身份核对；语义复审集中于相关入口、变更文件及必要调用依赖，不声称逐份审查全部 209 个文件的全部语义。

## 阻断项

### B1 — FAIL：实际 20 MHz 小烟测仍使用旧受保护清单

普通 scripts/visual_model_smoke.py 已将当前源码与不可变输入分开，得到 210 项受保护输入；但实际 20 MHz 入口仍独立构造旧 232 项清单，并要求其中全部旧 SHA 匹配活跃文件。

证据：

- 实际小烟测 protected_bindings 在 visual_calibrated_policy/engineering_attempts/20261012_s4/visual_model_smoke_small.py 第 55–62 行仍读取旧清单并逐项比较旧 SHA。
- 实际 check 在第 130 行、生成计划前调用该旧函数。
- 实际计划校验调用链为 expected_plan → protected_bindings，run/worker → read_plan → semantic_plan_identity → expected_plan。
- 普通入口的新实现位于 scripts/visual_model_smoke.py 第 44–67 行，与小烟测入口是不同实现。
- 新增自检 check_high_bandwidth_adapter.py 第 118 行仅检查普通入口的新实现，未检查实际小烟测入口。

已独立核对的三个冲突：

| 活跃文件 | 旧受保护 SHA256 | 当前 SHA256 |
|---|---|---|
| visual_method_features.py | b74ed6336c6dc44c0041d1427f904a22b022cc4fd6a57dfc086b226aa58bd722 | 440c41b771e311bc2b1f0e8d03e944408940711d2f45c229407125faa8602b91 |
| visual_upload_link.py | 4c3f5c41a5d7554cfc4bac8e759eb96b560efdf731dd3bb74d12209d76949534 | f9f75f9a1f6dd5340d0872b2555039b546df31c09748bc607b25f44388c549d5 |
| visual_training_coordinator.py | bcda7f4dadba74246ab563d85a8b116fe5b2e25bc08cfe6c4c17a9a7660df3bf | e2d792431bb7d518b7f2879c186449bfff11615999fe5f1cc45f5ceaac7436f6 |

因此，当前保存的 ADAPTER_CHECKS_PASS 不能证明实际小烟测入口可准备或运行。

### B2 — FAIL：8×30 检查读取预算计数，不是实际完成计数

接收器检查 ledger.accounting.environment_decisions == 8 和 user_behavior_samples == 240。该 accounting 来自 self.budget.accounting()；即使 ledger 尚未采集或提交更新，也具有相同预算数值。

证据：

- 接收器计数检查在 scripts/visual_pilot_preparation.py 第 295–307 行，只读取预算计数。
- 预算计数生成在 scripts/visual_training_transactions.py 第 64–71 行。
- ledger 快照在第 215–221 行将预算 accounting 与实际 committed_episodes 分别保存。
- 人工接收器夹具 check_high_bandwidth_adapter.py 第 151 行没有实际 counts、completed phase 或 committed episode，却作为正例被接受。

当前校验未核对每臂实际 counts、完成 phase、唯一已提交 episode 的 8 decisions / 30 users / 240 samples、失败状态及实际更新提交记录。第 308–310 行也仅比较两个 schedule/input-condition 字段是否彼此相同；缺失字段可形成共同的 None，未独立验证为合法 SHA 或重新计算其内容身份。

真实 producer 的采集和更新路径仍有实际闭合检查，但这不等于新 receiver 已实现所声称的“精确实际 8×30 回执校验”。

### B3 — FAIL：烟测计划文件未纳入持续边界保护

构造 preparation 时读取并校验了烟测计划文件，但随后没有把 smoke_plan_path → plan_sha256 加入 self._inputs。

证据：

- 初次计划读取在 scripts/visual_pilot_preparation.py 第 359–363 行。
- 文件绑定集合构造在第 388–392 行，仅加入 candidate、actual_smoke、两份 review 和 human_direction_record。
- 后续边界检查在第 426–440 行，只重新检查 _inputs 和 candidate，没有重新读取烟测计划。

因此，preparation 成功构造后，单独修改或删除计划文件不会被该持续边界检查发现。父烟测 JSON 的 plan SHA 声明不变，不能代替重新验证计划文件的实际字节。

## 已确认改善及通过项

### PASS：方向记录文件身份与 schema 已修复

- 当前方向记录文件 SHA256 为 b0635013bd3d07dc330847c03baca313612813c836dbca174f266c050e270b32。
- 2870f4b3ef40d37e81dc21fcb65d64c3a3161e4324379c832ab3f0e3b3051d46 是记录中的 upload-link 语义身份字段，不再被混作文件 SHA。
- scripts/visual_pilot_preparation.py 当前核对固定路径、文件 SHA 参数、完整十字段 schema、网络选择和精确配置。
- 实际读取与独立方向核验要求保留外部可信用户消息核验；本报告不将本地 JSON 本身认定为人类授权证明。

### PASS：20 MHz 已传入实际小烟测的两条链路

- 小烟测定义唯一 SMOKE_UPLOAD_LINK。
- 实际特征和环境构造把同一个配置同时传给 calibration_scales 和 VisualReceiptEnvironment。
- 空请求特征使用 scales.upload_link；有请求 JPEG 传输使用环境链路，串行上传消费该传输时间。
- 2 MHz 默认仍保留；20 MHz 仅改变 bandwidth_hz。噪声、功率、载频等常数未变。
- matched-input 和 actor 维度仍为两方法共用 actor；四 server 时为每用户 19 维。
- 训练/验证链路匹配、保留测试链路匹配和协议跨角色一致性保持一致。

### PASS：资源限制源码未放宽

资源规格 SHA256 为 bd5442151a52e6e2fc9ab1033d3d9104ad744c6308d7d84bb1289475bd0dacb3。

资源规格继续规定隔离 Python 3.13.5、NumPy 2.2.6、Torch 2.11.0+cu128、RTX 4060 Ti、单线程及占用严格小于 1800 MiB。资源检查继续拒绝占用达到阈值，之后执行有种子的 CUDA kernel witness；无 CPU fallback。本次没有执行该检查，因此不报告当前 GPU 空闲或运行时就绪。

### PASS：旧证据没有被本次复审升级

- 旧 2 MHz actual smoke SHA 仍为 207742f1663774237a71f8e1ec170cad0ed206255da68c562ca703a1002899db。
- 原 A4 小烟测源码保存在 source_before_quota_block_repair/visual_model_smoke_small.py，SHA 为 ec0a7535d3c74f758bad6c0ba3928cf6f4f88d5ce246e0e85674056fd442e406。
- 当前小烟测源码 SHA 为 cc2b14cafde350511322f6111b0fdaf442bf1d40d0554a94df76909448b26bdd。
- 旧计划、旧 source map 和旧烟测结果不能作为当前 20 MHz 烟测结果。

## 保留警告与后续边界

1. 当前 reward_input_integrity.require_review() 仍固定读取 S4 的旧 independent_command_rereview 路径；其中当前保存的 review 绑定旧小烟测 SHA，不能满足现在的源码。需要保留旧审查证据，不能把旧 PASS/WARN 重标为当前审查。
2. 后续 metadata-generation 接收器仍使用 203 项普通 source_lock()，而新 preparation 使用 209 项；因此本次不能声称完整 pilot→formal 链路已经打通。
3. 新自检只保存四项 source SHA，没有绑定其直接依赖的新 preparation receiver、普通 protected_bindings 实现或实际小烟测入口；其输出是有限的作者自检，不是实际烟测或独立执行证明。
4. 203 项 baseline 的内部 membership 检查仍仅为长度；外部 review/map 精确相等可拒绝旧 map，但该函数自身没有独立固定 203 个路径集合。同数量的源码路径替换不由长度检查单独拒绝。
5. REPAIR_SCOPE 第 15 行仍把旧 protocol-assembly SHA 写为 5e57566220ef7002145dceb9b049aec7caafc9c87ee7703710714c91e70f64388；实际快照及旧计划一致为 5e57566220ef7002145dceb9d049aec7cafc9c87ee7703710714c91e70f64388。原 upload-link 字节已通过 visual_upload_link_a4_baseline.py 保留，但同名 source_before/visual_upload_link.py 多两个空行，SHA 不等于原 A4 字节。

## 当前源集合身份

- 当前 baseline：203 文件。
- 当前已绑定 S4 additions：6 文件。
- 总计：209 文件。
- 完整源清单身份：b22e30dfeae5e8974f8ed1a2539e90b6da5242895932ed86f5815886e13ce363。
- 算法：对精确 Windows 绝对路径按 ordinal 排序，逐项连接 path + NUL + lowercase SHA256 + LF，对 UTF-8 字节计算 SHA256。
- 普通入口当前 protected 清单：210 项，所有实际字节匹配。
- 其清单身份：29945dbe9ca758058b6f255f993c8f7e084705d33ce87258fd8ba8d8c45abbb5，使用同一算法。
- 实际小烟测当前仍要求旧 232 项清单，导致 B1。

## 最终边界

当前源码不能获得无阻断结论。完成本报告不启动任何新计划、烟测、pilot 或正式实验；本报告中的 PASS 项仅表示相应源码检查通过，不能覆盖三项 FAIL，也不能升级为实际执行或性能证据。
