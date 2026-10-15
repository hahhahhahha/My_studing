# 完整有序训练调度与只读验证：源码准备，尚未运行模型

新调度源连接真实控制器的动作前act→完整episode采集与固定drain→连续MV／共享QL奖励→每用户时间GAE→唯一优化器更新→真实权重存档。输入为事先绑定的共同校准日程及相同预算分段，不允许验证／测试角色训练，不复用已消耗环境、部分更新或失败目录；模型参数在采集前后核对不变，更新后记录完整参数指纹并与存档关联。旧工程入口及旧测量均保留。

两个学习方法的输入、预算、归一化及时间尺度仍须共同冻结；10×1000只是未冻结候选。H边界规则必须在独立校准／验证pilot后明确登记“跨训练段延续”或“每独立episode重置”，源码支持明确规则但当前没有替用户选择。PPO的MV统计器仍只创建一次并跨校准episode连续。每段环境队列／上传／上一portfolio是新初态，即使辅助统计延续，也不是单次连续部署；固定drain不增加决策、H或MV统计调用。

新低层装配把累计接纳回执仅保存一份append-only档案，各动作记录前缀长度，可精确重建每个动作的完整历史；不改变actor／critic／基线可读输入。旧父装配仍临时构造一份完整观测，这只降低**留存**历史重复内存，不是测得的速度／峰值内存优化成果。行为／真实事件／drop／unfinished不删减。H边界与回执压缩的检查使用明确人工动作／logp／value，不能算学习行为或模型输出。

真实调度和真实验证的构造均要求外部既有的冻结pilot协议、实际CUDA两方法smoke和fresh gpt-6-astra xhigh Type-A完整入口审查文件及其精确SHA。要求的注册JSON状态分别为`frozen_after_cal_validation_only_pilot`、`actual_cuda_two_arm_smoke_pass`，审查scope为`complete_visual_entry_after_actual_model_smoke`；必须锁定全部完整入口源，并核对原服务状态complete_audited。旧foundation、dry-entry或合成检查不能充当这些文件，当前尚无此类新合格授权文件，不生成假PASS记录。此注册结构是待后续审查的源级接口约定，不是已冻结的实验协议。

验证源只接受已加载、与外部存档及参数SHA匹配且从未在本控制器更新过的真实控制器，验证池日程事先登记。只读act采用确定性latent，任务目的地仍按共同日程uniform执行；在线H从0独立递推，不恢复训练H／MV状态，不调用优化器、不改参数或Torch RNG。此函数自身不构造优化器，但外部NeuralVisualController当前构造PPO时会创建一个未使用优化器；没有优化器更新，不能误称其对象不存在。完整模型加载／roundtrip及实际验证尚待smoke与独立审查。

checkpoint存档仍是只读快照，没有resume接口；调度另保存H／MV／共同日程／状态指纹以便审计，但不声称完整随机状态可续训。最终checkpoint选择及测试入口未实现／未冻结，当前训练／验证／120正式测试cell均0。网络方向与GPU门槛未改变，绝不以该源码接入越过门槛。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_coordinator.py'
```

上述命令仅检查新增低层H／存档格式保护与真实入口拒绝／静态布局，不运行模型／优化器／真实新权重，也不冒充独立完整审计。
