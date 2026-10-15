# 四方法固定权重单cell只读评估循环：仅源码准备

`retained_test_evaluator.py`将完整合同、受登记不可变日程、执行私有实测表、动作前相同观测／特征、两真实学习模型或两规则适配器接入同一个单cell循环源码。没有launcher／模型工厂／加载器，也没有实际构造或执行。当前完整pilot、实际CUDA冒烟、fresh Type-A完整入口、全部10权重和日程都不存在；不能运行此类、构造伪证书或把纯描述当实际结果。

真实学习cell须传入同种子／方法、正确加载的最终checkpoint新控制器；静态历史类／CUDA检查、存档外部SHA／实际参数字节SHA、无更新token共同核对，参数及Torch RNG在确定性测试前后不变。读取归一化／闭合设置须与校准pilot逐项一致；环境全新，日程和服务表为精确受保护类，共同物理移动／原始数据／网络／delay／容量／资源绑定一致，cell前后实际调用日程及服务表边界保护。旧checkpoint/validation加载API未改；未来总调度须先完成真实正确权重加载再调用此入口，不用初始化模型代替。

四方法均仅接收不可变动作前观测及公平特征；执行表、测量会话、分配uniform和未来实测服务不传策略。current-action-free critic的字节SHA在动作／执行／过去回执后复核。独立在线H从0开始，训练H／MV／RNG不恢复；共享QL诊断日志适用于解释，不冒充PPO的均值—方差训练奖励，也没有任何更新。PPO既有控制器仍构造未使用的optimizer，明确披露；评估源码没有step／backward／update调用。

当前`pack_features`的无请求用户仍用默认2MHz链路计算载荷代理，因此本入口明确拒绝非默认链路，防止在未独立修复／审查该适配前接受更高带宽。原2MHz不悄悄改变；若用户授权另列高带宽仿真，须保留弱链路证据并先做新增的公平特征适配及cal/validation-only pilot／fresh审查。这是已识别的源码适用范围，不是2MHz负载可行性通过。

固定循环消费完整决策窗，随后按统一H+drain结束，不按完成数量早停或延长drain。完整uncapped真实到达时延、接纳／drop／unfinished／长上传、事件闭合和各虚拟服务器服务秒守恒必须对应；零完成保持统计None、空到达不删除，尾部样本不足在最终审计／统计阶段标明，不制造低时延成功。闭合helper产生的训练罚项只作过去任务身份核对，结果丢弃，不插入任务时延或更新。最后导出全部执行任务记录、广播预测历史、共同观测／决策／回执以及参数／RNG／输入输出SHA，供真正独立重建；实际未来服务值只出现在执行及privileged报告，不反馈拟合／选参。

模型调用前后CUDA同步记录policy wall time；规则在CPU，时间不加到仿真任务时延，也不是视觉服务／无线硬件实测。承认此实现差异，不能称控制器部署时限通过。累计接纳历史以一次archive＋每观测prefix计数保存，未丢事件；仍没有实际速度或峰值内存证据。

正式单cell锁采用科学条件／固定权重／共同外生SHA的语义身份，不包含输出目录或修复源／审查路径的装饰字段；选择另一目录、JSON编码或源修订不能自动重跑同一个已完成／失败cell。新输出目录不能覆盖旧／部分证据。失败保留部分观测、动作、执行记录、时刻和锁，不自动append／resume／retry；需单独新增审查恢复，不自动删锁。独占文件保护只是作者的人工I/O检查，实际完整入口仍待独立审查。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_retained_test_evaluator.py'
```

命令只检查3组**人工计数／服务质量守恒／时延**报告、科学身份及独占文件保护，不产生actual-policy结果，也不读取真实test计时CSV、运行环境／队列／Torch／模型。没有成功实际evaluator构造；不伪造已完成权重／smoke／PASS。单cell成功输出也只能pending独立结果审计，不称120cell完成。完整120网格总调度、共同日程生成登记、实际正确模型加载和全部覆盖／最终审计仍待接通；原GPU门槛、网络／pilot选择及修复独立复审未满足，旧源FAIL保留，旧smoke计划过期不重封／重跑。原服务测量／导出／论文／旧RL输出不改。
