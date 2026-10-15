# 奖励消融执行入口 V2：一次启动与部分记录保护

这是零smoke/零正式cell状态下的新增控制包装层，不替换原设计、绑定、源码或准备输出。原`reward_followup.py`及其文档保持不变；后续不直接调用它的smoke/run，仅用本V2入口。新增理由：原单元控制器能在COMPLETE缺失时再次写入一个已部分执行的目录；V2在任何witness/训练前创建一次性锁，禁止这种隐式续跑。

唯一科学干预、5种子×5延迟、25个Full预重放、25个NoJointDemandReward训练、全部冻结后25个变体测试及所有数值门槛均不变。这里不改变任务/奖励/随机数/历史Full比较规则；没有通过删慢记录、重测或放宽一致性标准来修复程序。

准备者只执行一次prepare，然后check；已存在的原绑定只读校验，V2绑定另存ADDITIVE_V2_BINDING.json：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py' prepare
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py' check
```

上述命令不分配CUDA、不做训练或模型测试。smoke前须先确认GPU已满足原<500MiB空闲门槛，不能终止未知进程、回退CPU或放宽门槛。准备者执行smoke仅一次，两臂各72训练/32验证步：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py' smoke
```

SMOKE_V2_STARTED独占锁创建在witness之前；即使资源或程序失败也保留证据，不能自动重试同一目录。任何smoke目录/锁/完成标记已存在均拒绝重复。需要修复时先归档失败、使用新attempt或新的已复核续接入口，绝不将旧partial目录当作空白目录。

完整smoke后，fresh独立审查代理只原样执行以下命令并静态审查原协议、两层控制器、奖励干预、实际日志/完成记录、Full重放顺序和共同预算：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py' entry-check
```

预期退出0并输出REWARD_V2_ACTUAL_ENTRY_CHECK_OK。审查代理不得调用prepare/smoke/run或修源文件。独立ENTRY_REVIEW.json须包含passed、原ADDITIVE_ENTRY_BINDING的文件SHA作为binding_sha256、ADDITIVE_V2_BINDING的文件SHA作为v2_binding_sha256；作者不自行填写通过。V2没有实际smoke证据时不能获得入口通过。

此后准备者隐藏启动run一次：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/reward_followup_v2.py' run
```

RUN_V2_STARTED与原RUN_STARTED禁止重复启动。任何失败现场保持不变，没有自动resume。最终50个正式结果需新的独立审计，smoke/入口成功不代表实验结论。原输出20261009及准备版V1控制器/绑定保持只读。
