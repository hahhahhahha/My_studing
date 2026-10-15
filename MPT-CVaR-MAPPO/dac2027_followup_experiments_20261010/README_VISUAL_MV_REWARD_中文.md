# PPO-MV-A 均值—方差奖励：真实回执适配候选

仅为视觉实验的奖励／缓冲工程接入，不是策略训练或严格复现旧逐槽结果。旧历史模型与MVReward源码不改，原记录只读；没有Torch、真实actor/critic、优化器或保留测试执行。当前资源门槛仍阻止模型工作。

## 1. 保留真实历史递推，而不偷换成QLA目标

冻结R1 `paper_policy_baselines.py` 的 `MVReward`：输入代价L，令return=`mu−L`，用Welford递推样本均值和population variance，统计比值=`mean/max(sqrt(var),1e-8)`；首次奖励0，之后比值不低于前值时奖励1，否则0。这里候选mu沿用1.5秒，不按测试结果重拟合。

历史训练代码对每一时刻按用户0..N−1调用**同一个统计器**，因此奖励并非共享team reward，且依赖固定用户调用顺序。无任务用户输入为0。这两点保留并披露，不偷偷改成每用户独立统计器或QLA共享目标。该比值只是训练信号；评估mean/CVaR/drop/unfinished仍是原始指标，不能把比值当归一化性能分数。

## 2. 必须披露的视觉回执适配

旧逐槽环境可得到当槽提交任务的training latency；新连续FIFO不能提前读取未来实测服务时长，因此输入改为**本物理区间内已发生的完成回执**及明确的丢弃代理代价，按原任务用户求均值。无回执用户仍输入0；这不是声称这些用户的实际任务已经0秒完成。

终末固定drain结束后，把drain中真实完成／丢弃以及unfinished代理条目合并到最后一个动作区间的对应用户代价列表，然后重新按整个episode的固定用户顺序递推MV。最后一步只计算N次统计更新，不在drain里增加动作或额外统计步；done为真、bootstrap为0。普通区间drop罚项固定1秒，unfinished代理=`max(1秒, 已经过时间下界)`，完成时延字段与unfinished严格分开。原最终任务计数、真实完成时延和旧QLA诊断不修改。

该候选不是精确连续时间折扣目标或旧逐槽输入的逐值复现。空回执／零输入占比、统计顺序依赖、末区间合并及代理罚项都必须在校准／验证pilot检查和冻结，完整披露；不按QLA优势调整。PPO仍是MV目标，QLA仍是共享QL目标；同预算不意味着相同奖励，比较不能解释为只隔离actor结构。

## 3. 缓冲保护与检查

新缓冲显式区分`QLA-shared-receipt`（共享[-1,0]）和`PPO-MV-A`（每用户0/1），禁止混用或把共享QLA奖励PPO沿用MV名称。动作／logp/value／特征测试均为明确人工接口夹具，不是PPO模型输出。缓冲样例使用自己生成的有限latent及其softmax动作；不把规则轨迹的路由动作冒充神经采样动作。实际回执、MV代价／奖励核对仍只读原轨迹，人工动作仅用于独立的缓冲接口检查。

首次`20261011_084211_175874_mv_reward`因人工缓冲样例对集中路由的零权重取对数而失败，有限值保护正确拒绝了无穷latent；失败、部分输出和源快照保留。修复仅更换该人工样例的动作，不放宽有限值保护、不改变原路由／原始指标／奖励，不重跑服务或原工程episode。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_mv_reward.py'
```

检查把原MVReward单个类从已锁定源码AST隔离执行（只用NumPy，不导入原模块/Torch），与新增递推逐值比对；这只是奖励算术的原定义基准，不是由模型输出伪造的数据真值。再只读复用5份验证来源工程轨迹，核对任务完整闭合、输入／输出范围、原记录不变、buffer拒绝方法混用以及每用户GAE接入。所有正式训练／测试仍0。正式学习入口还需完整采集器、pilot与独立审查，以及真实模型smoke。
