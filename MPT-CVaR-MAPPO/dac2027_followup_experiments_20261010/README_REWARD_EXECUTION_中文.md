# 奖励独立消融：隔离执行入口

范围仅为原5种子×5延迟canonical奖励消融，不是视觉4服务器实验。原协议、原源码、原输出不修改。唯一变体仍为NoJointDemandReward；输入、QL、actor、critic/GAE修复、奖励尺度与所有预算不变。

新环境使用`environment/qla_np226/Scripts/python.exe`，声明规格见`protocol/QLA_ENVIRONMENT_SPEC.json`（canonical SHA75aa3aced99d49fa90a7318cd2b5dc8b6d8d320cd89c7ae740aff5f50cacbc88）。只在venv安装NumPy2.2.6，复用只读Torch2.11+cu128，不修改全局Anaconda。构建/metadata成功并非ready；GPU必须空闲<500MiB，CUDA kernel、实际两臂smoke和fresh文档审查都通过才可正式运行。不自动CPU回退或终止其他程序。

输出目录为`X:/Yi-one/workspace/MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261010_repaired`。prepare只能执行一次；check只读验证入口、文档、设计和25个旧Full checkpoint指纹。文档或绑定源码变化不能静默重封存。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py' prepare
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py' check
```

以上命令不做CUDA、训练或测试。准备者在确认GPU资源后执行smoke：两臂各72训练/32验证步，仍不是正式结果。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py' smoke
```

smoke完整后，fresh独立审查代理只原样执行下面命令并静态核对数据/动作/奖励边界；不得执行prepare/smoke/run、修文件或重新测量。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py' entry-check
```

预期退出0并显示`REWARD_FOLLOWUP_ACTUAL_ENTRY_CHECK_OK`。审查报告保存到新输出`ENTRY_REVIEW.json`，包含`passed`和`binding_sha256`（ADDITIVE_ENTRY_BINDING.json文件SHA）。作者不能自填通过。

审查后才能由准备者隐藏启动run一次：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup.py' run
```

顺序：25个冻结Full test重放先严格复现原四指标（abs1e-8及相同日程），再25个新变体训练/验证，全部冻结后25个变体test。Full先行仅作历史实现一致性核查，不用于新变体选参。失败保存现场，无自动resume/门槛放宽/旧结果补空。只有50个正式结果完整且独立结果审计后才可作科学结论；H不活跃时不声称集中度反馈项被证实。每个正式训练仍10000步、验证/测试各1000步。
