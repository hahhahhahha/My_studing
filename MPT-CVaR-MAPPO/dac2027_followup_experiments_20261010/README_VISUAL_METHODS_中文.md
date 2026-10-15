# 视觉方法输入、结构绑定与共享奖励（工程准备，不是策略结果）

## 共同行为输入及动作前critic

`visual_method_features.py` 只接收不可变的动作前DecisionObservation、校准来源的固定尺度及上一步H虚拟队列。没有当前portfolio、实际服务、当前完成结果或未来日程参数。

四服务器actor布局为19维：位置2、服务器距离4、请求摘要3、共享陈旧完成成本score4、反馈年龄1、动作对齐QL价格4、价格绝对尺度1。它与冻结历史EdgeWisePriceGaussianActor的索引一致；QLA和视觉适配PPO-MV-A使用**完全相同的19维数值输入**。PPO在本补充实验也获得确定性的QL特征，是显式的matched-input适配，不宣称它与旧canonical PPO输入未变；其flat Gaussian actor/local critic和MV目标仍须使用真实历史类接入。未来结果应说明这一更强信息匹配对照，不能将对它的差值单独归因于QL信号。

QL价格是当前用户参考预测需求×陈旧预测队列压力，加V×预测完成成本；完成成本包括既有用户上传等待、当前批次候选串行上传、预测服务和陈旧预测积压。价格按服务器方向min–max归一化，另保留log绝对尺度，沿用QL的内部特征处理；这不是性能指标归一化。所有论文时延指标仍取原始未截断任务时延。

用户/任务尺度只从20000条校准测量和校准图片元数据得到；位置尺度只来自显式训练时间窗。请求scale7不截断请求，队列压力尺度为3个决策周期的服务秒，延迟scale1秒、age scale0.5秒目前是工程候选，须统一pilot冻结。缺反馈年龄编码-1，不把missing当零年龄。若验证位置超出训练区域，不根据验证/测试重拟合位置尺度。

critic为当前全部用户actor特征、先前portfolio、陈旧队列/count/busy和先前H/联合当前预测需求/missing的705维状态；不含当前动作或动作后统计。QLA保留centralized agent critic；PPO的历史SharedGaussianPPO保留local critic，输入相同19维。共同可观测数据不意味着两者训练结构/奖励相同。

## 历史类绑定，不以未执行模型宣称完成

`visual_learned_bindings.py` 按SHA绑定正式R1修复快照中的models.py、config.py和paper_policy_baselines.py：真实MAPPOPolicy/EdgeWisePriceGaussianActor、AgentCentralizedCritic及SharedGaussianPPO，不新造名称相同的替代网络。维度适配器只替换环境输入尺寸，不改这些历史网络/优化类源码。QLA保持edge-wise actor和显式bias0.5。

工厂函数仅在声明隔离解释器中、先通过原GPU资源门槛和实际种子CUDA witness之后才导入Torch并构造CUDA模型，拒绝CPU回退或混入已加载的其他源码包。当前此工厂**没有实际调用**，模型smoke/训练/完整入口独立审查均未完成。模型结构绑定不等于模型链路可运行。

## 共享奖励与异步完成回执

`visual_shared_reward.py` 准备QLA队列需求加权项、H集中度反馈、完成时延与迁移项。联合需求用当前任务的参考CPU1T预测服务秒，不用JPEG字节；按当前portfolio形成**预期**份额和normalized HHI，不冒充实际分配集中度。当前动作产生的份额只能进入动作后的奖励，不进入当前critic。

连续执行的时延项只使用该回执区间内**已经完成**的任务：完成秒减原始实际到达秒；拒绝未来、重复、冲突完成/丢弃回执。延迟到达的完成回执可能来自较早动作，这是从旧同步环境到连续系统的显式适配，实际回执钩子/终末奖励和未完成处理尚未接入。训练中丢弃可用固定1秒工程惩罚，与未截断完成时延指标分开；没有丢弃/unfinished指标替换或隐去。

H更新先计算奖励再更新状态，工程候选为每物理秒0.05的旧更新率；50ms决策步乘实际区间时长，避免仅因时隙变短而放大20倍。V=.5、迁移=.05、H尺度5和浓度factor2暂沿用结构设置；它们仍须写入完整pilot/final协议。Z保持关闭。H不活跃时不能据此声称其集中反馈贡献。

## 工程检查与下一步

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_method_features.py'
```

检查只运行NumPy/标准库：校准尺度访问保护、19/705维布局、两学习输入完全相同、服务器特征/动作维度共同置换、不可变动作前critic、共享奖励手算和重复/未来回执拒绝、历史模型源码/类名静态绑定。合成奖励回执明确标注，不能当实测服务/性能真值。报告记录源码/数据SHA、model_factory_executed=false和正式cell数0。

下一步仍需把真实完成/丢弃/接纳回执接到统一控制循环，接入JSEQ的历史估计和score-softmax，写正确按用户/时间分离的GAE及两模型同预算训练流程，再进行校准/验证pilot、冻结全协议、实际CUDA两模型smoke和fresh独立完整入口审查。不能直接使用旧PPOBuffer把不同用户串成一条时间轨迹。旧论文、源码快照、结果与冻结服务测量/模型保持不变。
