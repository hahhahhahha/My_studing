# 用户授权的显存占用门槛修订

2026-10-12，用户明确要求将原 `memory.used <500MiB` 改为 **`memory.used <1800MiB`**。1800MiB本身仍不通过；读数缺失、设备身份不符或查询失败仍停止。此值限制的是启动前已占用显存，不是GPU总容量、实验显存预算或成功保证。

当前有效规格是 `protocol/QLA_ENVIRONMENT_SPEC.json` 的 `qla-followup-environment-v2-user-gpu1800`；`scripts/qla_environment.py` 固定其新语义SHA，仍在Torch/CUDA分配前检查占用。视觉模型工厂及未来经重新绑定的入口沿用此共同检查。包版本、隔离环境路径、线程、种子CUDA kernel检查、禁止CPU回退和禁止重复OpenMP覆盖均不变；不需要重装软件。

旧500规格、入口源码和变更前状态按原字节保存在 `environment/attempts/20261012_1940_user_authorized_gpu1800/BEFORE`。旧500门槛失败记录仍为失败，不把它们改判为1800通过。原服务测量、数据、模型、论文和正式预算不修改。

旧043949视觉计划、旧奖励V1/V2绑定及旧小规模v1源审查都绑定旧字节。它们保留但不再是当前环境的可运行身份；不得覆盖或重封旧计划，不能直接沿用旧WARN启动模型。旧 `paper_doc_tools/run_joint_reward_ablation.py` 的500预检属于冻结历史入口，不修改或绕过它；它不用于当前修订启动。旧小规模v1也仅作保存的审查输入，不运行。后续缩小版入口须在新修订中使用当前共同门槛、完整来源绑定及所需独立复核。

缩小版原范围审查已完整交付FAIL，奖励B1/B2运行前原始轨迹／日程与完整执行源码绑定缺陷仍未修复。19:43的MD缺失及最终额度错误已原样保存；同一审查者19:49补齐MD（SHA91bdbafd5118b6faa50eb7d7b2cfc52a9e1cbcc93bf984c3d1de46debd814549）和DELIVERY_COMPLETION.json，原JSON未改，明确新1800修订不在原范围认证内。此资源修订不消除源码缺陷，也不允许正式训练、OOD协议变更或更改2MHz网络条件。

一次资源／种子kernel见证可用下列命令；这不是模型冒烟或完整正式入口通过：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/qla_environment.py' witness
```

每次见证另存目录，资源或kernel失败证据保留；不自动重试已部分执行的模型双臂。新的小规模实际证据和独立完整入口仍须满足，才可推进下游。
