# 学习 episode 采集器：源码接通与非模型工程检查

本阶段没有构造、调用或训练QLA/PPO模型，没有保留测试访问，也不是完整入口独立审计。`RealLearningEpisodeCollector`只接受真实历史CUDA控制器，拒绝人工样例、stub、未初始化对象和CPU替代；因原GPU门槛未通过，真实采集器的构造／collect从未执行。

## 1. 源码中接通的链路

真实入口要求新的**校准角色**环境，不能用确认／测试图片训练。每次：观察动作前不可变摘要→用过去H状态组装公共特征→真实控制器只接收该特征→保存当时的latent／logp／value及critic哈希→执行portfolio→实际区间结算→真实过去回执计算奖励。PPO仍用其明确的MV奖励；QLA仍用共享QL奖励，不混用目标。

执行桥将模型float32 portfolio用float64做单位和归一，保存归一前／后的两份动作；只是处理浮点表示，buffer仍要求与采样latent的softmax在原1e-6范围内一致，不调整路由目标或科学验收门槛。当前critic不能由执行后信息重新计算。

完整episode和预先声明的固定drain结束后，核对所有任务的完成／drop／unfinished，闭合方法对应奖励，再做每用户时间GAE；先闭合后更新。采集器**不调用优化器**。未来训练器还需按共同预算分段采集、更新并保存checkpoint，不能拿一次长随机rollout当作充分训练。

每个装配器绑定同一个环境实例以及对应服务器位置／配置，拒绝中途切换环境或借另一环境的终末结果闭合。首次091418成功及源码快照保留后新增这一身份边界检查；没有覆盖原报告或测量记录。

真实入口使用独占新目录；保留逐步状态、闭合记录、行为／GAE数组。任何部分失败均保存证据，不自动追加、重跑或延长drain。当前没有正式运行CLI、训练启动或resume入口。

## 2. 本次工程检查确实做了什么

低层`EpisodeAssembly`不含模型，只把已验证样例动作与真实FIFO回执连接。检查以既有共同102任务的验证角色日程，驱动**明确人工的有限latent、logp、value**，运行新装配链路，并与单独串行oracle比对。它们不是神经策略输出、真实概率密度或学习成绩；所有model act/value/update调用数为0，不能称“两模型smoke”。

两种奖励语义用完全相同人工动作、外生日程和配对服务记录，检验原始执行结果一致而目标不同；另外保留集中路由的drop和短固定drain的unfinished样例。核对80×30完整缓冲、动作前特征／critic与实际行为记录、真实到达时延、服务守恒、终末零bootstrap、独占阶段／方法／信息边界。实际模型入口仅做静态调用布局检查与错误对象拒绝检查。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_episode_collector.py'
```

## 3. 尚未冻结的关键项

物理折扣、每episode时长、10,000决策步的分段及更新频率、load、网络可行性、drain／代理罚项仍需校准／验证pilot。此源码候选每闭合episode重置MV统计器；旧连续训练未必这样重置，因此是额外视觉适配，**尚未作为正式多episode规则采用**。必须在真实训练前明确统计器重置或跨episode连续策略，不能暗称完整忠实复现。

约96%的用户—动作区间在前次4秒样例中没有回执输入，需检查该稀疏性、用户调用顺序、异步信用分配和终末聚合，不按测试成绩或QLA优势调参。10个同预算训练与120个固定checkpoint测试均未启动；仍需GPU资源／种子kernel、实际模型smoke及fresh完整独立入口审查。
