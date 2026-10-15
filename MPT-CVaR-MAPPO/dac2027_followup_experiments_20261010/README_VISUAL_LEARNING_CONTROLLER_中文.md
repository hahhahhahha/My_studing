# 视觉学习控制器与完整 episode 缓冲：源码／接口准备

本次准备的是调用真实历史模型的控制器源码，以及只用 NumPy 检查的数据缓冲。`NeuralVisualController` 从未构造，真实模型 act/value/update、Torch、GPU、优化器均未执行；正式训练／测试仍0。旧模型包、已保存轨迹、测量及论文不改。

## 1. 正确绑定真实调用

历史 QLA `MAPPOPolicy.act` 返回 `(portfolio, latent, logp)`，value 单独读取动作前集中式 critic 和用户ID；历史 `SharedGaussianPPO.act` 返回 `(latent, logp, value, portfolio)`，PPO critic 仍是本地19维输入，QLA仍是集中式705维输入。这些差别是模型结构，不是偷偷给予未来服务真值。双方 actor 都用同一19维可观察特征。

入口核验真实历史类身份、源码SHA、声明主进程环境及CUDA参数设备，不接受模拟／替代模型或CPU。采样出的 latent 必须与实际执行 portfolio 的 softmax 对应，不得对执行动作作不记录的额外变换。

## 2. 缓冲与 GAE

按完整用户批次保存每个动作时刻的 actor／动作前critic／latent／logp／value，不使用旧20,000条自动截断缓冲（30用户×10,000步会超过它）。容量单位明确为**episode 决策步**；超限直接拒绝，不丢弃旧样本。结束后只接受与行为动作／时间逐项匹配且完成终末闭合的共享奖励。

先按 `[time,user]` 计算物理时间 GAE，再展平优化器样本。历史 PPOBuffer 的先展平再递推不用于新视觉入口；旧源码不改。QLA保留历史更新函数，在调用期间暂时给**单个模型实例**提供预先计算的 GAE，finally恢复原方法；没有修改整个类／源码，也不改变其PPO损失与参数更新结构。

候选更新设置统一为lr0.001、4 epochs、batch128、clip0.2、entropy0.01、value0.5、grad-norm0.5；更新后的优势标准化仅供优化器，不是性能指标归一化。QLA沿用自己的优化器；PPO使用相同候选更新设置的独立Adam。两模型log-std结构及critic不同，保留且须完整披露，不能把“同预算”误说成所有架构细节都相同。

## 3. PPO-MV-A 的奖励语义还未接入

历史 `SharedGaussianPPO` 是网络结构，`PPO-MV-A` 还包含原 mean-variance 奖励逻辑；不能仅调用该网络、改用QLA共享奖励，就宣称复现了PPO-MV-A。当前闭合缓冲只提供 `QLA-shared-receipt` 奖励标记，控制器在更新前拒绝把它用于PPO-MV-A。均值—方差奖励的真实回执与终末适配仍未完成；它需要单独接入并明确其奖励范围和历史定义。若将来采用共享QLA目标的matched PPO，必须另命名、明确作为另一个对照设计，而不是悄悄改变当前预定方法。

## 4. 终末回报与 on-policy 时序尚待 pilot

当前末步闭合代理要求在固定 drain 完成后，才能计算完整 episode GAE并更新。当一个episode仍未闭合时不提前更新后续又回填早期奖励。**episode长度、分段数量与10,000步总预算如何组合尚未冻结**；必须在校准／验证预实验核定可行的训练更新频率、内存和终末信用分配。不能把一个500秒的初始随机策略轨迹只更新一次而冒充充分训练，也不能偷偷延长QLA预算。

训练控制器源码尚未与完整采集／checkpoint／验证流水线接通，部分更新异常必须保存新attempt及模型状态，不重用失败实例。实际学习模型两臂smoke和fresh独立完整入口审查仍不可省略。当前GPU资源门槛没有通过，没有真实学习调用。

## 5. 无模型工程检查

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_learning_buffer.py'
```

检查只解析真实历史／新控制器源码，并检查人工构造的、明确标注的 action/logp/value 接口夹具及NumPy缓冲；不把规则动作转换出的人工 latent 称为学习模型输出。正式性能、训练完成或更新成功的证据必须等实际模型入口执行后产生。
