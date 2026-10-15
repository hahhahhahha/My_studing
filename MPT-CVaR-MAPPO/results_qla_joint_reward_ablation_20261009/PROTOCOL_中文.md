# 联合需求感知共享奖励：独立消融协议

本实验沿用正式 R1 消融的固定设置，不采用上一份“真实视觉服务校准”方案的四服务器或新时间尺度。它回答的是：显式联合需求感知的奖励塑形，相对于均匀全局队列惩罚，是否带来额外收益？不是取消合作训练或取消共享目标。

## 1. 两个对照

Full 保持冻结的正式实现：

\[
r_t=-\tanh\left(\max\{0,q_t+\bar H_t(C_t-\bar C)+0.5[\widetilde T_t+0.05\operatorname{Mig}_t]\}\right),
\quad q_t=\sum_j\bar q_j(t)s_j^A(t).
\]

唯一新增变体 `NoJointDemandReward`：

\[
r_t^{\rm NJ}=-\tanh\left(\max\{0,q_t^{\rm uniform}+0.5[\widetilde T_t+0.05\operatorname{Mig}_t]\}\right),
\]

有任务时 `q_uniform=mean_j(qbar_j)`，无任务时保持为 0。只改变奖励中的需求加权队列项和集中度反馈乘积；继续计算并记录需求摘要、C、H 递推，不改变 actor 输入、QL 路径、参数共享、critic 或 bootstrap。

不使用 `--lcc-no-h-queue`，因为它还会改变状态递推。两项 q 都在 [0,1]、系数都为 1；相同归一化、max 与 tanh 保持奖励在 [-1,0]，不对新变体单独拟合缩放。该控制不意味着两种奖励分布的均值与方差完全相同，因此还保存逐槽 reward/phi、饱和比例和两种 q 的反事实数值。

## 2. 固定设置，与旧消融一致

| 项目 | 值 |
| --- | --- |
| 移动数据、用户、服务器 | KAIST、30 用户、35 服务器 |
| 到达 | Poisson-bursty；lambda=3.5，burst prob=.1，multiplier=2，至多7任务/用户/槽 |
| 种子 | 1、2、3、4、5 |
| 名义反馈延迟 | 0、1、2、4、8 槽；实际 source age 仍单独记录 |
| 时间、广播 | 1 s/槽；每2槽广播 |
| 计算与缓冲 | 63 MB/s；任务计算工作量15 MB；缓冲630 MB；整任务接纳与完整drain |
| 通信载荷 | 原配置 270/1024 MB；不在此消融引入真实图片任务 |
| QL、actor | QT、edge-wise sharing、aligned、bias=.5，关闭 stale margin 和 Z |
| 学习 | lr=.001、rollout1024、batch128；其余正式配置全量等价校验 |
| 预算 | 每条件训练10,000槽；验证1,000槽；最终测试1,000槽 |
| checkpoint | 最终checkpoint；全部25个新训练完成验证后冻结，再进入test |
| 外生日程 | 沿用 `results_qla_canonical_v1/shared_schedules`，各角色hash逐条件核对 |
| 正式运行环境 | Python3.13.5、NumPy2.2.6、PyTorch2.11.0+cu128、RTX4060Ti、CUDA；拒绝CPU回退 |

旧源码包已发生与本实验无关的变化，所以执行使用正式 R1/pilot 保存的完整源码快照，源锁为 `028c265e80d6b2f868a3e62179bb0cfd93c4ee78c581d76e6ef9bfca2872c4d9`。新的奖励干预仅在独立进程内安装，原代码包、旧结果、正文和附录不覆盖。

## 3. 数量与 Full 复用

- 25 个 NoJointDemandReward 新训练条件，验证后统一冻结。
- 25 个 NoJointDemandReward test-only 评估。
- 25 个旧 Full checkpoint 的 test-only 重放，用于补录原消融未保存的实际集中度指标。
- 不额外重训 Full 整个网格；须核对旧训练修复证据、配置/权重hash、运行环境和共同日程。
- Full 重放须在 average latency、CVaR99、QOsc、Mig 上复现历史值，绝对容差1e-8；不通过时停止，不能把不同执行语义的结果混入比较。

正式运行前另做 Full/NoJoint 两臂各72步的冒烟测试。冒烟结果不是论文证据。

## 4. 评估输出与指标

1. 逐槽 `server_assignment_counts` 得到实际最大服务器份额、P99、实际HHI，不能用 nominal expected share 代替。
2. `max_j [(Q_after-Q_before)_+/(mu_j*slot)]` 的均值/P95/P99；严格大于1定义spike。
3. 全槽spike率。
4. 与附录F一致的摘要receipt窗口spike率：source timestamp变化时锚定，首槽现有摘要也计入，末尾窗口截断至最多三槽。这是 **F兼容的最多三槽窗口**，不要误称所有窗口都有完整三槽。
5. 另外保存严格三槽诊断：排除首槽初始化、排除尾部不足三槽，仅记录真实source更新后的完整三槽窗口。延迟>0时这是摘要接收窗口，不等于服务器原始广播时刻。
6. 平均uncapped accepted-completed latency、CVaR99、QOsc、Mig，以及generated/accepted/dropped/unfinished和队列守恒。
7. 逐槽reward、q_weighted/q_uniform、H-active、H项大小、饱和比例；新变体训练槽和两方法评估槽都记录。

历史Full训练日志只每100步保存一次H，采样点为0不能证明所有训练槽H均为0。本实验重放能得到Full评估期的逐槽H，但不能补造历史训练逐槽记录。

如果 H 在观察到的评估中不活跃，不能把收益解释为集中度反馈项的贡献；最有依据的结论应限定为需求加权队列塑形。即使结果有利，也不声称已经隔离“联合摘要全部信息”的作用。

## 5. 统计与解释

用五个种子为配对块，每个种子保留完整五延迟profile；沿用R1的3,125个exact bootstrap重采样，给出`NoJointDemandReward−Full`差值与95%区间。按延迟结果同样保留，不把25个cell当作25个独立训练重复。

没有任何性能优劣门槛用于选择checkpoint。变体改善、下降、无清晰差异或发生迁移代价权衡，都保存。是否支持第三机制的创新贡献，必须等正式结果完成后判断。

## 6. 执行入口

在项目根目录的 PowerShell/Anaconda Prompt 中：

```powershell
& "X:/anaconda/ANACONDA/python.exe" -B "X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/test_joint_reward_ablation.py"
& "X:/anaconda/ANACONDA/python.exe" -B "X:/Yi-one/workspace/MPT-CVaR-MAPPO/paper_doc_tools/run_joint_reward_ablation.py"
```

第二条依次执行：GPU/runtime预检 → 两臂冒烟 → 25个新训练/验证 → 统一冻结 → 50个测试/重放 → 配对结果汇总。串行使用同一GPU，日志持续保存在本目录，支持通过完成清单恢复；发生错误即停止，不用旧数字补空cell。

只准备设计而不启动训练：在第二条末尾加 `--prepare-only`。

此入口不会自动修改安装环境。若检测到NumPy/Python/Torch版本不一致或GPU不可用，写入 `STATUS.json` 和 `preflight.log`，不会运行训练。不要绕过检查直接把当前2.1.3环境与历史2.2.6结果混用。可以在隔离环境恢复匹配的软件版本；若选择新环境，则需要另行重训Full配对对照并采用新输出目录。

## 7. 当前状态

实现与纯数值检查已完成，GPU/PyTorch正式预检尚未通过。读取机器元数据发现当前NumPy2.1.3与历史2.2.6不同，且当前工具执行环境的PyTorch/GPU初始化报错。这不等于已诊断用户主机驱动损坏。本次没有安装/升级全局软件或更改系统网络/驱动设置。

以本目录 `STATUS.json` 为准。`prepared_not_started` 或 `blocked_preflight` 都不代表实验正在运行；只有完整50个测试cell通过后才会生成 `RESULTS.md`。
