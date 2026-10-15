# 冻结策略的未见外生需求／容量压力测试

日期：2026-10-09。状态以同目录 `STATUS.json` 为准。此文件是预先规定的实验协议，不是结果报告。

## 1. 要回答的问题与证据边界

主问题：已经训练好的 QLA-MAPPO，在没有参与原训练与选参的外生需求和容量序列上，是否仍能同时控制平均任务时延和任务级 CVaR99？

辅助问题：在相同未见序列、移动轨迹、任务抽样、分类任务分配随机数和反馈机制下，与 PPO-MV-A、JSEQ-A 相比，队列波动、集中需求和相邻时隙动作变化有什么差异？

不重训，不更新神经网络参数，不根据新测试结果选 checkpoint 或超参数。JSEQ 保留固定在线队列估计规则；这不是学习权重更新。QLA 保留原算法的辅助状态更新；这也不是重训。

这是一项**冻结策略的外部 trace-shaped 压力测试**。不能称为实体部署、视觉模型完成时延测量、已证明任意外部环境泛化，或只改变时间窗口的因果干预。

## 2. 为什么不能直接沿用原读取方法换 CSV 行

检查原 `run_real_trace_external_comparison.py` / `run_real_trace_external_formal.py`，原通用读取器对每行所有可转换为数值的单元取平均：

- Azure invocation 表的时间轴在列 `1`–`1440`，每行是一个函数的一天。行平均不是按分钟的调用序列。
- Azure duration 表包含均值、次数、极值与分位数，不能混合成服务时间；其中分位数还是短区间平均时长的分位数，而非逐请求时长。
- Alibaba 2018 的行包含时间戳、CPU、内存、网络等字段，混合平均不等于 CPU 利用率。文件没有表头，原读取器还跳过了首条数据。

已审计 189 份 `results*/external_schedules` 清单，原来源为 Azure 2019 d01 与 Alibaba 2018 `machine_usage.csv`。旧生成器实际使用的最长前缀为 10,000 行；该 Alibaba 前缀覆盖 140–691,110 秒，涉及两台机器。因此“换到后面的行”不能保证时间隔离。

具体证据：`SOURCE_USAGE_AUDIT.json`。审计范围是可保存的历史清单与对应生成器，不把“仓库内没有所有使用记录”解释成绝对未使用证明。

新实验按官方字段读取。因为转换语义与原 R6 不同，不能把新结果减去旧 R6 结果归因为“时间窗口变了”；新实验内的各策略比较使用完全相同的新转换。

## 3. 数据选择与获取

### Azure 2019：需求强度

官方说明：[Azure Functions 2019 数据说明](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)。

官方下载：[azurefunctions_dataset2019 数据包](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz)。本地已缓存并解压，不需要重复下载。

- 校准：d01，1440 分钟，仅拟合需求强度的 P05、P95。
- 测试：d11、d12、d13；每个日期取分钟列 1–200。
- 原始日文件按字段逐列求和，得到每分钟全部函数的调用数，不混入 Hash、Trigger 等元数据。
- 官方调用数是执行后记录的计数，因此这里只作为**外生需求强度代理**，不声称是客户端逐请求到达时间。
- 固定换算：令 `v=clip((count-P05_d01)/(P95_d01-P05_d01),0,1)`；用户均值请求率 `lambda=0.65+4.2*(0.20+0.80*v)`。
- 保留原 trace 实验的用户 lognormal 异质性、Poisson 抽样与每用户 7 请求上限；所有策略用相同实际抽样数组。
- 测试不重新归一化。必须报告每窗的上下界截断比例。截断会压缩原始变化，不能据此声称模型适应了无界需求。

目前这三个需求窗口已提取，具体源文件哈希和映射分布保存在 `data/`。

### Alibaba 2017：独立容量源

选择未参与已有实验的 **v2017 发布版**，而不是强行从已横跨整个日期范围的 v2018 前缀中寻找“新日期”。这是独立发布版外推，不是同一批服务器未来时刻预测。

- [官方介绍及下载入口](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/README2017.md)
- [官方字段与采样说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md)
- [官方 schema.csv](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/schema.csv)
- [官方数据包下载](http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2017Traces/alibaba-trace-2017.tar.gz)
- [官方 SHA256SUM](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/SHA256SUM)

只需要提取 `server_usage.csv`，放到：

`X:/Yi-one/workspace/MPT-CVaR-MAPPO/datasets/external_traces/extracted/alibaba2017/server_usage.csv`

官方 checksum 列出的文件名是 `trace_201708.tgz`，哈希为 `02c1fbc412fccc03b748c05ec18fc30fd93c0bc383eea47ac21146c4aa2235df`。下载完成后须核对实际归档与校验文件是否对应；若命名或哈希不匹配，暂停核实，不假定改名后一定是同一归档。本次网络下载未成功，尚未完成归档校验。

读取无表头 CSV 的第 1 列时间戳、第 2 列机器 ID、第 3 列 CPU 百分比。有效 CPU 值限定 0–100。重复分钟的有效测量取平均；不混入内存等字段。

时间原点取最早有效测量所在的分钟。预定三个窗口：相对分钟 `[0,200)`、`[220,420)`、`[440,640)`，中间各留 20 分钟隔离。按数值机器 ID 排序，选择最先满足三个完整窗口的 35 台机器。不按照 QLA 的性能、平均 CPU 利用率或拥塞强弱挑机器。不补齐缺失分钟；若不足 35 台，记录数据可用性失败并重新讨论协议，不自动放宽标准。

容量系数 `clip(1-CPU/100,0.35,1.0)`，乘原基础服务率 63 MB/s。它是**可用服务能力代理**，不是测得的视觉识别服务率，不是机房原有任务被卸载算法重新执行。

2017 利用率采样约每 60 秒、平均约 300 秒，不能据此测试毫秒级真实资源突变。

## 4. 窗口长度与时间尺度

| 窗口 | Azure 日期与分钟 | Alibaba 2017 相对分钟 | 仿真时隙数 |
|---|---|---|---:|
| W1 | d11，1–200 | 0–199 | 1000 |
| W2 | d12，1–200 | 220–419 | 1000 |
| W3 | d13，1–200 | 440–639 | 1000 |

每个原始分钟的需求强度与容量值保持 5 个仿真时隙，仿真时隙仍为 1 秒。即原始 60 秒映射为 5 仿真秒，**12 倍时间压缩**。这不是把真实一分钟解释成实际 1 秒，也不是无提示地循环复用有限外部数据。不同地区/发布版数据只用于合成外生压力，Azure 与 Alibaba 的时间戳不表示现实世界同步观测。

1000 时隙保持原正式评估长度。移动轨迹继续使用原带隔离的测试划分；若移动片段不足 1000 时隙，沿用原循环规则并在清单中记录，不循环外部窗口。

任务工作量固定 15 MB、上传大小 270/1024 MB。本实验只改变需求／可用容量来源，不把 Azure 日分位数误用作逐任务服务时间分布。硬件校准实验是另一个实验块。

## 5. 冻结模型、基线和固定设置

- 使用 R6 已归档、在名义反馈延迟 4 下训练的 checkpoint：QLA-MAPPO 与 PPO-MV-A；KAIST/NCSU/NewYork 各三个训练种子 1–3，共 18 个。
- 测试名义反馈延迟为 0、4、8。不是每个测试延迟重新训练，也不是按新测试结果选择最优训练延迟。
- 加 JSEQ-A，temperature=0.25，规则固定，无新调参。
- 30 用户、35 服务器、基础服务率 63 MB/s、630 MB 队列缓冲、1 秒时隙、2 时隙广播；接纳、服务后残余队列、分类抽样、有限缓冲和终末 drain 沿用原协议。
- 保留 R6 的 3% 摘要丢失和每服务器 0–8 额外源年龄。额外年龄、loss 与名义 retrieval delay 区分记录，不能称实际源年龄就是 0/4/8。
- QLA：QT 信号、显式偏置 0.5、V=0.5、迁移权重 0.05、H 保留、Z 关闭、stale margin 关闭，加载原 config。
- PPO 使用真实加载的 SharedGaussianPPO，不调用训练入口或训练零步后误用初始化模型。
- 神经 actor 使用确定性 latent mean；任务目的地仍按组合概率分类抽样并经过可行性筛选。

共有 `3方法 × 3移动数据集 × 3训练种子 × 3测试延迟 × 3外部窗口 = 243` 条测试轨迹，**0 条训练任务**。固定随机顺序；先用 W1/KAIST/seed1/delay4 三方法验证整个执行链。不能只保留正向窗口。

同一窗口、数据集、种子的各方法和延迟共享日程，分别锁定请求、任务分配 uniform、反馈 loss、额外年龄的随机流；Poisson 请求强度变化不会推进其余随机流。每个模型加载前后检查 checkpoint 哈希，actor 参数运行前后再作比特级哈希；任何参数变化都失败。

## 6. 指标与分析

首要指标：未截断 accepted/completed 任务平均时延和 CVaR99，沿用原尾部取样规则；dropped/unfinished 单列，不把删除这些任务后的低平均值当成功。

其次：QOsc（评价时隙结束队列 / 固定基础服务率 63 MB/s，population Std，不含 drain）、portfolio migration、每槽最大实际服务器份额 P99、最大正向队列跳变的时隙均值、质量守恒误差、源年龄分布、drain 状态。不要把容量下降引起的分母变化误判为队列增长。

先画每个窗口、移动数据集和延迟的完整结果；再给等权汇总。paired difference 以同一个窗口／数据集／种子／延迟比较 QLA 与两个基线。三个窗口来自固定外部源，不是三个独立训练样本；同一 checkpoint 重复评估也不是新增训练种子。

若需要 95% CI，按训练种子重采样整个窗口×数据集×延迟 profile；只有三个训练种子，CI 应解释为描述性不确定性，不能把 81 条重复轨迹当 n=81 的独立训练实验。同步报告每种子差异和各窗差异。不为了显著性临时增加随机种子或删窗口。

结果判据：同时查看 mean/CVaR、可靠性、队列与迁移，完整报告 trade-off。若只在低负载窗获益，应限定为所测需求/容量范围；若在窗口下降，不以“版本差异”消解负向结果。若某窗严重过载，原样保留，区分协调失效和总容量不足，不自动减小到达率。

计划图表：一个窗口分解 mean/CVaR 的多面板图；一张平均／尾部／QOsc／迁移／drop 汇总表；原始各 cell 结果一并保存。模型测试完成并通过审计前不生成性能图，不写性能结论。

## 7. 执行入口与当前阻碍

准备并检查：

```powershell
Set-Location X:/Yi-one/workspace/MPT-CVaR-MAPPO
& X:/anaconda/ANACONDA/python.exe paper_doc_tools/run_unseen_external_windows.py --prepare-data
```

待获得可用、匹配的运行环境，并完成 Alibaba 2017 数据获取、归档对应关系校验后：

```powershell
& X:/anaconda/ANACONDA/python.exe paper_doc_tools/run_unseen_external_windows.py --run
```

声明环境沿用正式结果：Python 3.13.5、NumPy 2.2.6、PyTorch 2.11.0+cu128、CUDA；线程 1。运行前须通过版本核对、种子 123 CUDA kernel witness 和实际模型 smoke，不自动转 CPU，不升级全局环境。

2026-10-09 当前环境 NumPy 为 2.1.3，且 PyTorch 导入出现 WinError10106；下载端点在当前工具执行环境无法解析。无法确定该问题是否存在于用户正常终端或只存在于当前执行环境。模型测试未开始。

已完成：原数据用途审计、Azure 四个日期真实分钟读取、三个未见需求窗、6 项纯数据单元测试、checkpoint/source 冻结准备、隔离推理入口。未完成：Alibaba2017 下载及校验、实际服务能力窗生成、CUDA preflight、243 cell 推理、结果审计/统计/可视化。

源码没有修改原训练算法，正文/附录没有修改。原结果图也没有覆盖。
