# 三项逻辑缺陷修复与交接

日期：2026-10-13。授权范围：修复上一轮代码审查指出的三项缺陷、做比例适当的回归/功能验证。没有运行E1，没有改模型、奖励、图片角色、负载规则、CPU资源或正式预算，没有新增监控自动化。

## 修复内容

1. `systemlab/edge.py`：成功计数/失败计数、service EMA、active清除、完成通知在同一个锁区间提交。旧代码在计数增加后、active清除前允许snapshot重复计数；受控线程交错测试覆盖成功和失败两条路径。
2. `systemlab/launcher.py`：显式声明OpenProcess/WaitForSingleObject/CloseHandle的DWORD/BOOL/HANDLE签名并使用线程本地Win32 error。仅有效非零PID对应的ERROR_INVALID_PARAMETER(87)作为不存在；权限不足、未知error、WAIT_TIMEOUT、WAIT_FAILED、CloseHandle失败都不放行“已退出”。新增测试使用高位handle、错误码和实际自然退出的临时解释器。测试不再终止venv shim。
3. `systemlab/admission.py`、`tools/register_test.py`、`drive_scripts/run_e1_prep.py`：新增v2 test登记，绑定具体config、冻结protocol、grid、scales、validation预登记、test manifest、完整共同schedule和其生成器源码，同时固定formal actor bundle/checkpoint/历史模型源。该checkpoint必须与实际冻结validation的policy_receipt一致，不能换一份结构相同的模型混入正式臂。正式入口必须提供`--test-registry`和`--cell-id`，独立review还必须绑定`test_registry_sha256`。本地launcher在启动任何服务前检查；client在模型加载前检查，结束时再检查连续性。self_audit标记明确拒绝。

辅助变更：make_grid记录实际scales路径；E1 prep不再重复硬编码120秒/16用户/drain30，从validation登记读设置，从矩阵读seed/rate；传完整浮点rate，不再截为6位小数。run_m2_prep按原登记的窗口公式计算H，读取config的users/drain，并明确原round负载规则。当前C_ref=175.395...时公式仍为120秒，没有改现有登记数值。新E1工具需显式新版本路径，不会覆盖旧输出。

在线source_receipt继续只跟踪systemlab/tools，避免无关临时脚本使服务身份漂移；涉及科学设置的实际生成器另作为v2 registry的逐文件SHA绑定。其修改不再能悄悄使用旧test批准。其他新到达源（Azure等）需要对应的单独登记契约，当前v2 E1契约明确为原Poisson矩阵，不混入不同工作负载。

## 实际验证

- `UNIT_RESULT_attempt_02.json`：最终版本18项测试，0 failure、0 error；Windows真实进程自然退出测试也已执行。artifact变更测试还覆盖actor bundle、checkpoint与历史模型源。attempt_01为增加actor准入绑定前的通过记录，保留原字节。
- `functional_4edge_attempt_02/`：最终版本实际4节点CPU ResNet、16逻辑用户配置、4个pilot请求、固定1秒功能drain；QLA真实加载和forward，4/4完成；前后actor参数SHA相同，optimizer调用0。attempt_01为中间版本完整功能检查，保留不拼接。
- 5个本次服务的真实runtime PID与venv shim均退出；全部shim退出码0，unknown/lingering列表为空，forced_termination=false。
- `FUNCTIONAL_ANALYSIS_final.json`：仅作者功能检查，不是容量pilot、正式性能结果或独立认证。功能配置的1秒drain不替代正式30秒。
- 原`protocol/`与`reviews/`逐文件对比BEFORE：历史文件变化0。旧M1/M2/M2b、原test日程、自审及失败记录没有覆盖、删除、重新盖章。

仅少量真实请求的验证不证明高负载稳态或新代码下完整validation通过。UNIT测试中的review/协议都是自动清理的临时fixture，明确不是实际独立审批。

## 历史保护

`BEFORE/`保存修复前systemlab/tools/tests/drive_scripts/docs/protocol/reviews字节。当前`protocol/frozen_v1`、旧E1_grid/registry和entry_selfaudit_v1仍属于原代码身份。源码已改变，它们对当前正式入口是过期证据，不得修改旧SHA或重新封装旧运行冒充新验证。

## 下一个agent的有限步骤

1. 复核本修复及记录；不自动重跑整个服务测量或旧experiment branch。旧calibration原始数据保留。复用旧CPU服务校准是否足够由新完整复审明确判断；若有影响校准的实际证据，再新增calibration版本，而非覆盖旧尺度。
2. 按不变的`protocol/validation_config_v1.json`、`data/validation_s101.json`与`protocol/validation_actor_v1.json`，依原登记顺序在新`runs/M2c_{method}_s101_attempt_01`目录完成三方法validation及分析。不能使用M2b旧source收据替代；拒绝/负收益仍完整保留。
3. 使用新源码的实际三臂验证冻结到`protocol/frozen_v2`，再以其FEATURE_SCALES重新登记`protocol/formal_actor_v2.json`。freeze仍强制实际当前源码validation，因此本轮没有制造frozen_v2。
4. 重新登记grid到`protocol/E1_grid_v2.json`，依据v2 prep生成全新的共同日程及`protocol/E1_test_registry_v2.json`。复用现有test manifest原字节，不读性能挑成员。
5. 新鲜完整的独立入口复审，报告/JSON绑定当前源、frozen_v2 protocol SHA及E1_test_registry_v2 SHA；范围包括实际M1/M2c、修复、矩阵/角色、计时/FIFO/cache/预算和正常关闭。自审不是批准。
6. 只在完整无阻断复审后，按grid实际顺序逐格进入E1。此次修复本身没有启动E1权限记录或正式结果。

### v2准备和正式参数示例

```powershell
Set-Location 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013'
$py = 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe'
# 前提：三臂新源码validation完整，frozen_v2和formal_actor_v2实际存在。
& $py tools/make_grid.py --feature-scales protocol/frozen_v2/FEATURE_SCALES.json --config protocol/frozen_v2/FORMAL_CONFIG.json --block E1 --output protocol/E1_grid_v2.json
& $py drive_scripts/run_e1_prep.py --config protocol/frozen_v2/FORMAL_CONFIG.json --grid protocol/E1_grid_v2.json --validation-registration protocol/validation_registration_v1.json --manifest data/test_images.json --actor-bundle protocol/formal_actor_v2.json --schedule-output-dir data/E1_common_v2 --registry-output protocol/E1_test_registry_v2.json
# 由独立review实际产生review JSON，并读取grid确定cell ID及对应schedule路径。
& $py -m systemlab run-local --config protocol/frozen_v2/FORMAL_CONFIG.json --manifest data/test_images.json --schedule '<该cell已登记schedule的绝对路径>' --method qla --actor-bundle protocol/formal_actor_v2.json --entry-review reviews/entry_v2/REVIEW_RESULT.json --test-registry protocol/E1_test_registry_v2.json --cell-id '<grid中的实际QLA格子ID>' --output '<新的运行目录>'
```

复审JSON继续使用`schema=systemlab-independent-entry-v1`，但新增强制字段`test_registry_sha256`。报告必须完整，PASS/WARN且blockers=[]，不含self_audit=true，绑定当前protocol/source。文件校验无法认证填写者身份，仍须由真实独立reviewer完成，不能手工清空旧自审阻断。

如果既有共同schedule精确满足新的grid/scales/registration，可用`tools/register_test.py --config ... --grid ... --validation-registration ... --manifest ... --qla-actor <实际formal bundle> --schedules <全部六份> --generator <实际原生成器快照路径> --output <新registry>`绑定原字节，不重采样。旧v1 rate曾格式化为6位小数，与grid全精度值不完全一致，当前严格契约会拒绝；若沿用这些日程需事先单独登记明确舍入规则并复审，不能隐式放宽本门槛。

E2/E4将来每个具体feedback config需要匹配自己的matrix/registry/review，不能把E1 registry当传播条件批准。本修复不提前执行或批准那些分支。
