# 完整保留测试网格登记（仅源码与人工描述覆盖检查）

`scripts/retained_test_grid_registration.py` 不运行模型、策略、队列或日程生成器，不做统计、图表、拟合或checkpoint选择。`RealRetainedGridRegistrar` 实际构造／register／seal均为0；源码准备不是120项真实结果、启动权限、环境READY或独立审计PASS。

纯覆盖函数要求四方法×两负载×三延迟×五种子正好120项；全部字段对应冻结候选，每个负载／种子的12项共享同一外生输入，共10个绑定。缺失、重复、更换方法／种子／负载／日程／horizon／drain、按完成数早停、附加隐藏字段均拒绝；行顺序不影响身份。描述检查返回真实结果证明数0，不能拿人工描述代替执行。

真实登记源要求已有完整pilot冻结、真实CUDA两方法smoke、fresh gpt-6-astra xhigh完整Type-A入口和十份固定最终权重注册；当前源必须在完整审查源指纹表内。只消费同一合同下实际成功结束的单cell evaluator与外部保存的完成标志SHA。核对原单cell科学语义锁／输出目录、四份完整输出字节、权重／RNG／零更新、共同日程／环境、完整有序决策／动作前观测以及从uncapped时延与全部计数重建的单cell摘要。零完成、drop、unfinished和不利时延仍登记，不按数值筛选或补造完成。

原实测配对／物理移动文件、日程、审查／smoke／源码、权重及全部完成输出在登记和最终seal前再次按字节SHA核对；相同路径不能换绑定。不会重测服务、修改测试文件、把服务真值送给actor／QL／critic或恢复训练H／MV／RNG。真实执行报告仍供以后独立原始事件／输入边界重建，作者登记不能替代该重建或完整公平性证明。

网格科学身份由全部120项科学cell身份组成，不含输出名称／源审查装饰路径／JSON字节编码；永久独占锁及新目录防止重复登记覆盖。失败保留已登记原cell的引用、完整输入表和失败现场，不自动append／resume／retry／删锁／合并部分失败；后续恢复须另行新增审查。完整120项且输入未变才可产出 `ALL_120_ACTUAL_CELLS_PENDING_FINAL_AUDIT.json`，状态只能pending独立最终审计，不能自动标complete_audited或发布统计图表。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_retained_test_grid_registration.py'
```

此命令只复用既有明确标注的人工120描述，检查覆盖／字段变更／人工独占文件锁和真实类缺少权限时的拒绝。不会构造假的actual-complete／checkpoint／审计文件，真实register／seal不调用、测试计时CSV禁止打开、Torch不导入。完整网格启动／正确新模型加载、共同真实日程生成登记、实际smoke／pilot冻结与独立修复复审仍未完成；原源FAIL、GPU门槛及网络／Alibaba协议的用户方向不改。新增源使旧smoke准备计划过期，不重封或运行旧计划。
