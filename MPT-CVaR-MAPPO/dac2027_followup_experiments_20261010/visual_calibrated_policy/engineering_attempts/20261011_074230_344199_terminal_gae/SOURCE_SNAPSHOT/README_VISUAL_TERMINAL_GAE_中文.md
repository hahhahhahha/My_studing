# 视觉训练终末记账与每用户时间 GAE：工程候选

这是正式训练前的代码／记账准备，不是已冻结的训练协议，也不是学习实验结果。原服务测量、原规则工程轨迹及旧奖励消融协议不改；新入口只读已保存的验证来源工程轨迹，输出到独立 attempt。没有调用模型工厂、Torch、优化器或保留测试。

## 1. 终末闭合候选

请求生成止于H，最后动作在H执行，最后普通奖励结算到H+Δ；共同 drain 的绝对终点始终是H+D，不追加一个Δ。等到该固定终点实际结束后，才能读取已经发生的完成／丢弃与仍未完成身份。

普通动作区间内的完成／drop回执沿用原工程共享奖励。其余每个任务只能有一个终末条目：

- drain 内真正完成：使用实际完成时刻减真实外生到达时刻，不截断、不以预测代替。
- drain 内实际丢弃：固定训练罚项1秒；不混入 completed latency。
- 到固定终点仍未完成：保持 unfinished 标签，完成时延为空。训练代理罚项为 `max(1秒, 固定终点−真实到达时刻)`，后者只是已知经过时间的下界，绝不是未来完成时间或实测剩余工作量。

终末条目的训练代价均值只在最后一条 transition 增加一个非负 phi 项，再使用相同 tanh 边界获得共享奖励。此前 transition、动作前观测／critic、H 状态以及原始最终指标不修改；不会在无新动作的 drain 内额外累计 H。终末 done 为真、bootstrap 为0，额外 drain 不消耗策略决策预算。

这是明确披露的**末步汇聚代理目标**，不是精确的连续时间折扣逐请求目标。长 drain 延迟被汇聚，且均值受 tail 数量影响；固定 unfinished/drop 罚项也可能带来取舍。必须在校准／验证预实验检查饱和、信用分配、可靠性与负载量级，冻结所有方法共用规则并独立审查后才能正式训练；不得按照 QLA 相对优势选择规则或用保留测试调它。

## 2. 时间×用户 GAE

`visual_temporal_gae.py` 接收 `[T,N]` rewards、values、next_values、显式布尔 terminal，以及每一步的物理秒数。逆序递推只沿时间，N 个用户分别递推，不能先把 T×N 展成一条长轨迹。可以在完成 GAE 后展平为优化器样本。

候选物理时间换算为 `gamma(dt)=0.99**dt`、`lambda(dt)=0.95**dt`；50ms不是一个1秒折扣步。两参数仍待预实验核定，不是正式选定值。真实终止使 bootstrap 与后续递推为0；仅 rollout 截断不应伪装成终止，必须显式提供下一状态 value。优势／回报不作性能指标归一化或裁剪。

本检查使用**明确标注的全零合成 value 夹具**验证接口；没有真实神经 critic forward，不能宣称 PPO/MAPPO rollout 或训练成功。

## 3. 检查入口

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_terminal_gae.py'
```

入口验证五份既有工程轨迹和共同日程的哈希，用真实已发生回执核对闭合计数、时延总和与原结果不变，并覆盖真实 drop／unfinished 夹具。合成解析测试覆盖跨用户隔离、内部 episode 边界、终止与截断区别、物理折扣、展平时机和非法输入。输出闭合账本及新奖励副本，原轨迹仍只读。所有正式策略训练／测试 cell 数仍为0。

下一步：把闭合候选和 GAE 接入实际学习控制器，进行校准／验证-only pilot；真实两模型冒烟受原 GPU 门槛控制，完整入口需要 fresh 独立审查。不能仅凭本工程检查启动正式10个训练／120个测试。
