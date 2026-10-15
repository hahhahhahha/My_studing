# 共同预算与一次性训练／存档事务：源级准备

新增保护将环境决策数、每用户行为样本数、动作窗物理秒和固定drain分开记账。10×1000决策的**候选**预算为10,000环境步、300,000用户行为样本、500秒生成窗及300秒独立drain；每段最后动作的50ms结算在既定drain内，不额外加到800秒执行窗。该分段、更新频率、虚拟H／终末规则仍未正式冻结，不能称共同预算已通过模型预实验。

`TrainingTransactionLedger`按采集→完整闭合→开始更新→更新提交→新checkpoint的顺序推进。同种子两方法必须有同一候选预算、共同条件指纹和逐段日程指纹；方法名不进入配对设计指纹。训练角色只准校准池，不能将验证／测试段加到训练预算。每段T×N样本及T×50ms动作区间完整，不接受静默截断、重复token、错奖励语义、乱序更新或只给一个方法增加预算。开始更新在调用优化器之前消耗一次性事务；失败现场永久锁定，无自动回滚、追加、resume或继续部分参数状态。

真实checkpoint导出源要求真实历史CUDA控制器，先完成并匹配自己的更新事务；PPO还须同一个训练用途MV统计器，其调用数等于累计T×N。QLA使用历史原save_checkpoint接口；PPO保存实际state_dict，另存优化器／Torch随机状态／训练记账。新目录独占、三份文件字节SHA和完成标志绑定方法／种子／预算／共同日程；部分、失败、篡改或错误方法的目录不能复用。**真实导出函数尚未调用，没有任何真实新权重。** 当前快照不提供resume；NumPy外生流、完整训练器和虚拟H状态尚未综合保存，不能将此快照称为可续训checkpoint。

验证参考只准validation角色，不恢复训练状态或允许优化器更新；这里只检查字节完整性，不反序列化、执行或验证模型权重，更不选择checkpoint。正式完整验证器、最终checkpoint选择规则及完整训练调度仍须cal/validation-only pilot与独立入口审查后冻结。

此次自检仅用明确合成计数／优化器统计数值和**纯文本占位字节**测试保护。占位文件即使使用约定文件名也不是Torch权重，不反序列化，不称真实checkpoint、学习行为、模型smoke或训练结果。另只读绑定已保存连续MV工程结果及其保护文件，不重跑服务、原FIFO、原轨迹或任何策略。所有结果均存新attempt；检查失败也保留。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_training_transactions.py'
```

这是预算／文件保护工程核对，不是实际优化器、正式10项训练／120cells或独立完整入口认证。仍等待GPU门槛、网络方向及完整协议；不因此更改2MHz、服务池、原测试条件或科学验收线。
