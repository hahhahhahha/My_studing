# V2 确认记录与隔离后处理独立审计

日期：2026-10-10。审计者：gpt-6-astra，xhigh，独立代理。审计级别：Type-A（第二位 GPT 家族审计者）；不是 Type-B 跨模型家族认证，也不是 GPT-5.5 审计。

## 结论

总体：**WARN（结论边界与运行时诊断限制）**。记录完整性检查：**PASS**。六项预登记数值验收：**PASS**。现有确认记录可复用于后处理；没有发现需要否定这些计时的记录级证据。此判断不授权启动保留测试，不把原运行状态改写成完整实验结束，也不证明视觉分类精度、队列模型或 QLA 收益。

本审计独立读取原始文件，使用仅标准库的 [independent_audit.py](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/independent_audit.py:1)，没有导入原实验模块、调用原 weighted_stats/evaluate、执行测量、重新封存、安装软件或设置不安全 OpenMP 环境变量。冻结文件与旧文件均未修改。所有审计输出仅位于当前 reviewer 目录。

## 独立重算的六项结果

| 配置 | 指标 | 预测 ms | 实测 ms | 相对误差 | 冻结容差 | 判定 |
|---|---|---:|---:|---:|---:|---|
| CPU-1T | mean | 28.2265995161 | 30.9050409400 | 8.666681% | 10% | PASS |
| CPU-1T | P95 | 33.8728 | 35.9973 | 5.901832% | 15% | PASS |
| CPU-1T | P99 | 40.1974 | 45.6524 | 11.948988% | 20% | PASS |
| CPU-4T | mean | 12.2080735995 | 13.1995167852 | 7.511208% | 10% | PASS |
| CPU-4T | P95 | 15.9146 | 18.5304 | 14.116263% | 15% | PASS |
| CPU-4T | P99 | 28.5398 | 31.6225 | 9.748439% | 20% | PASS |

权重从重建的 6,867 张合格库存及“尺寸×类别”计数重新计算为精确分数 `N_cell/(6867*n_cell)`；验证每图三次会话等权；预测对同尺寸层最近邻及五次原校准会话等权。重建模型使用原 V1 的全部 20,000 行，而非复制冻结模型的服务值作答案。分位数采用 `inf{x:F(x)>=q}` 的经验左 CDF 逆，无插值；误差分母是实测值，不是预测最大值、预测均值或自归一化分数。

与新报告相比，CPU-1T 预测均值仅有 `3.552713678800501e-15 ms` 的浮点舍入差，其余五项预测/观察值完全一致；六项判定相同。证据：[独立结果及逐项差值](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:85)、[分位数 CDF 见证](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:169)、[新报告](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/VALIDATION_REPORT.json:4)、[原公式与分母](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:391)、[冻结容差](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/protocol/PLAN.json:24622)。

## A—F 完整性检查

### A. 测量与目标来源：PASS

实测目标是主机运行 JPEG 解码/RGB、官方预处理、FP32 ResNet-18 推理、argmax 与响应字典所得的计时，不是用服务模型输出制造的目标。磁盘、网络、排队、控制器不在边界内。模型预测只使用 V1 校准图计时；V1 旧验证图用于方案诊断后被排除，200 张参考图仅用于漂移诊断，不拟合修正因子。新验证请求的权重来自输入库存，不来自时延标签。

证据：[实际计时段及调用](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/measure_block.py:46)、[预热和逐请求执行](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/measure_block.py:64)、[原 V1 同边界计时](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009/scripts/measure_block.py:58)、[模型仅由校准行构成](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:194)、[预登记角色边界](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/README_中文.md:11)。

### B. 分数归一化：PASS

均值/P95/P99均保留原始毫秒单位。概率质量归一化用于定义目标混合分布，不是把模型自己的最大输出当分母伪造高分。六个误差均为 `abs(predicted-observed)/observed`。没有发现参考图漂移归一化或按结果删去慢请求。

证据：[人口权重来源](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:67)、[预测与实测权重](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:411)、[误差计算](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:427)、[独立原始单位输出](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_METRICS.csv:2)。

### C. 结果文件与证据存在性：PASS（仅确认后处理）

120 块、240 个原始 CSV/JSON、120 个任务文件、120 个块日志一致；共有 12,000 行新验证和 1,200 行参考计时。独立核验了每行身份、顺序、三会话×两配置覆盖、无重复键、有限非负阶段耗时、正服务总耗时与阶段和、时间戳、配对及跨会话预测类别一致性；未发现错误行。合并表的 13,200 行逐字段、逐顺序等于原始行，不含测试。PNG 已目视检查；PDF 内六个实测数值标签及三份导出文件哈希均核对一致。

冻结的 3 个源文件、7 个材料、262 个 V1 保护项均匹配；中断归档的 240 份计时及 5 份运行证据均匹配；恢复前后 517 个保护项匹配；校准与新验证的 4,000 张 JPEG 的完整哈希和字节数匹配。V1 失败仍是失败（CPU-4T P99 相对误差 39.1797%）。没有保留测试记录、测试报告或自动测试放行文件。

证据：[检查计数与空异常列表](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:4)、[原完成日志最后一块](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/logs/runner_stdout.log:124)、[V1 失败保留](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009/calibration/VALIDATION_REPORT.json:55)、[原状态并非 complete](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/protocol/STATE.json:2)、[导出指纹](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/EXPORT.json:13)。

### D. 评价代码是否实际调用：PASS；原中断路径需保留

原运行器在全部确认块结束后调用 `evaluate(False)`，但 OpenMP 原生错误使其在写出确认报告前终止。恢复入口把原冻结检查和原确认统计分在两个进程，调用 `audit(False)`、`evaluate(False)`；统计日志输出的数值、保存报告、CSV与独立重算互相一致，非只有未调用的函数。原 summarize_v2.py 不是本次实际导出路径；导出由新增入口的 `export_worker()` 执行。

证据：[原调用路径](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:507)、[原致命错误](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/logs/runner_stderr.log:1)、[恢复统计调用](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/analyze_confirmation_isolated_analysis_run_source.py:115)、[实际数值输出](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/isolated_statistics_stdout.log:2)。

### E. 结论范围：WARN

本次证据是单机器、同一天、两种 CPU 配置、2,000 张新确认图片×3 会话的边际服务分布检查；200 张参考图片不是额外验证图。目标是去重未使用的 Imagenette 官方 train 库存，不是校园地标用户请求分布，也不是 ImageNet 全数据覆盖。重复计时不等于新增独立图像。

49 个非空“尺寸×类别”格全部有样本，但其中 9 格仅一张验证图；相应分层重采样无法反映未观察到的格内图像异质性。>2MP 校准图仅16张，因此这些请求用最多16张而非20张邻居。所有验证请求均满足冻结的“最近邻距离≤2”门槛，最大最近邻距离为1.53170763；这不等于所有邻居均在该距离内，更不证明每个条件格都准确。没有目标库存中的非空未覆盖格；库存之外或未支持的新输入不能套用本通过结论。

CPU-4T P95误差14.1163%，距离15%容差仅约0.8837个百分点。点估计通过不是误差置信界保证。原代码的500次交叉重采样仅给实测分布区间；本审计核对重采样代码与解释，未独立重现其随机区间端点。区间不涵盖模型估计/选择不确定性，也不是预测误差区间。

证据：[限定任务说明](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/README_中文.md:14)、[稀有格计数](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/protocol/POPULATION_PROFILE.json:75)、[支持判据](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:106)、[交叉重采样](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:431)、[重算支持与格数结果](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:289)。

### F. 评价类型：PASS（明确分类）

分类为 **real_measured_runtime / empirical service-distribution calibration**：真实主机运行时间作为经验参考；不是 synthetic_proxy、自监督代理、simulation_only 或 human_eval。技能的 `real_gt` 若作为宽泛上位类，只能表示独立实测参考，不能声称“数据集提供了真实服务时间”或“官方分类准确率评价”。数据集给的是图像和类别信息；本验收不使用类别真值计算识别准确率。

证据：[计时参考生成位置](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/measure_block.py:47)、[评价仅取 service_ms](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/runner.py:420)、[无视觉训练/QLA主张](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/V2_确认验证结果_中文.md:14)。

## 运行时恢复与测量有效性

原块日志全部完成，最后计时结束于17:50:46.292201；原父进程状态停于 auditing_confirmation，随后 stderr 记录 OpenMP Error #15。记录与程序结构支持“测量已结束、后处理崩溃”的解释，不支持把原流程标为完成。源码、原计时、故障证据与前后快照一致，且六项结果可在完全不加载 PyTorch/NumPy 的独立实现中重算，因此复用记录做后处理有依据。

恢复短探针的 torch、numpy、mixed 三种模式均 exit 0；mixed 探针**没有复现**原错误。独立检查中可见 PyTorch 自带 libiomp 路径，统计进程可见 Anaconda Library/bin/libiomp 路径；这些只是运行时加载观察，不是对原崩溃根因的完全证明。恢复 README 已正确保留这个限制。不能宣称“所有混合进程必崩溃”或“整个环境已永久修复”。证据：[探针三退出码](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/RUNTIME_PROBE.json:3)、[mixed探针确实完成](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/probe_mixed_stdout.log:6)、[统计进程库路径](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/isolated_statistics_stdout.log:1)、[恢复说明限定](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/README_中文.md:18)。

分析运行时源码的存档 SHA256 为 `3a78680328857c51ae879a62bd022ac4d907fb01821e5358d5f6f26d3cc6190e`，与 ANALYSIS_STARTED 匹配。当前入口的唯一新增内容是导出函数内的本进程 Matplotlib 配置路径，分析/探针函数文本不变；不能把当前文件哈希冒充历史运行版本。证据：[历史入口锁](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/ANALYSIS_STARTED.json:5)、[导出局部改动](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/analyze_confirmation_isolated.py:200)。

三会话首块时间约为17:14:03、17:29:08、17:44:13；空闲门记录间隔905.008428与905.007172秒，均超过900秒。所有块串行且实际顺序与冻结随机/反平衡顺序一致。空闲门在会话前通过；它不是整段运行持续空闲保证。元数据包版本、200次预热、CPU0/CPU0,2,4,6亲和性、错误计数均一致，但温度不可用、频率未连续采样；线程/亲和性记录主要由设置成功的代码与元数据佐证，不是独立硬件遥测证明。证据：[空闲门1](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/protocol/idle_confirmation_s1.json:3)、[会话间隔重算](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:224)、[运行设置](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/measure_block.py:20)、[传感限制](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/measure_block.py:89)。

## 可支持与不可支持的主张

- 支持：在冻结目标权重和本次测量条件下，既有确认记录完整，冻结条件经验模型的六项边际服务时间点估计误差满足预登记容差。
- 支持：分析隔离恢复成功，没有重新测量、修改模型/邻居/阈值或覆盖 V1 失败；合并表和图来自同一套确认数据。
- 需要限定：外推到未来同机器运行；每个输入条件格的准确性；稀有格及P99的稳定性；根因诊断和环境永久修复。
- 不支持：保留测试通过、整个V2实验完成、多设备/跨天/真实校园流量泛化、视觉识别精度或视觉模型训练、卸载队列有效性、QLA收益、Type-B跨家族认证。

## 留存与审计限制

无需为本次确认统计重测或改门槛。继续保留原状态、故障日志、V1失败以及Type-A标签；后续若考虑测试、集成或训练，应另行明确授权和方案，不可由本审计自动放行。

本审计没有访问保留测试图像或测量标签；读取冻结划分中的测试身份元数据仅用于排除重叠与检验未变。图片完整哈希已重新计算；像素级dHash未重新解码计算，近重复筛查重建依据冻结manifest中的dHash。文件哈希/代码/时间戳的相互一致不是历史OS执行的密码学认证。本次独立数值核验不等于独立物理复测。

机器可读明细、所有读入哈希与量化检查计数见 [INDEPENDENT_RECOMPUTATION.json](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_RECOMPUTATION.json:1)，简表见 [INDEPENDENT_METRICS.csv](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/recovery_20261010/reviewer/INDEPENDENT_METRICS.csv:1)。
