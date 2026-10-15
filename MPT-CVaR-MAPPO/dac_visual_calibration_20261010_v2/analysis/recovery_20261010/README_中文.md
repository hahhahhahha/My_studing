# V2 确认记录的隔离分析入口

授权范围：修复分析进程并严格审计已经产生的确认记录。不重测、不改模型或验收规则、不启动保留测试、不训练视觉网络或 QLA，也不修改论文。

## 修复方式

新增 `scripts/analyze_confirmation_isolated.py`，原冻结的三个源码文件及 PLAN 均不修改、不重新封存。

1. 独立进程运行原 `runner.py check`，完成冻结指纹、版本及 PyTorch CPU 核心计算见证检查。
2. 另一个进程直接调用原 `runner.audit(False)` 与 `runner.evaluate(False)`，且确认该统计进程没有加载 `torch`。统计公式、权重、重采样和六项门槛仍是原冻结实现。
3. 分析前后逐项核对冻结源码／材料、受保护 V1 文件、全部 V2 原始 CSV／JSON及故障时的状态、锁和日志指纹，要求完全一致。
4. 如果数值验收未通过，保留失败报告；不重跑、不改门槛。本次入口无测量与保留测试启动代码，不生成 `protocol/VALIDATION_ACCEPTED.json`。

没有安装／升级包、修改系统环境或使用 `KMP_DUPLICATE_LIB_OK`。统计进程实际加载的 Intel OpenMP 库路径记录在 `isolated_statistics_stdout.log`，与独立 PyTorch 检查分别存档。

## 诊断证据的边界

原工作进程的日志记录 OpenMP Error #15，但这次三种短诊断（单独 PyTorch、单独 NumPy、短小的混合计算）均正常退出，**没有完整复现原长时间工作进程的故障**。不能据此声称已证明所有混合进程必然失败或整个环境已永久修复。可核实的是：采用分离进程的新分析入口完成了本次完整确认统计，且其原始记录与冻结材料未改变。

## 已执行的命令

使用现有 Python，不安装软件：

```powershell
& 'X:\anaconda\ANACONDA\python.exe' -X utf8 'X:\Yi-one\workspace\MPT-CVaR-MAPPO\dac_visual_calibration_20261010_v2\scripts\analyze_confirmation_isolated.py' probe
& 'X:\anaconda\ANACONDA\python.exe' -X utf8 'X:\Yi-one\workspace\MPT-CVaR-MAPPO\dac_visual_calibration_20261010_v2\scripts\analyze_confirmation_isolated.py' analyze
```

上述命令已运行并使用排他写入锁，不能原样再运行以覆盖日志／报告。原 `runner.py run` 也不能重启。后续只读审计可检查现有报告和原始记录。

## 主要证据

- `RUNTIME_PROBE.json`：短诊断退出码、日志指纹与受保护材料快照。
- `ANALYSIS_STARTED.json`：新分析入口源码指纹、授权范围和开始快照。
- `STRICT_CONFIRMATION_AUDIT.json`：原冻结审计函数对13200条记录的检查结果。
- `AFTER_ANALYSIS_SNAPSHOT.json`：分析后的受保护材料快照。
- `RECOVERY_STATE.json`：仅代表分析恢复状态，不覆写原测量工作进程的历史 STATE。
- 上级目录 `VALIDATION_REPORT.json`／`validation_errors.csv`／`V2_确认验证结果_中文.md`：原冻结统计实现的确认报告。
- `reviewer/`：第二位审计代理直接从原始材料复算的结果；它是独立代理审查，不是跨模型家族认证。

图表导出第一次遇到工具启动环境缺少用户主目录的配置错误，未产生图或汇总 CSV。只在新导出入口中加入进程局部 `MPLCONFIGDIR`，将绘图库缓存放在本实验 `.runtime_cache/matplotlib` 后，导出正常完成。分析阶段实际执行的入口源码已原样保存在 `analyze_confirmation_isolated_analysis_run_source.py`，其指纹与 `ANALYSIS_STARTED.json` 一致；导出入口的这一独立修改记录在 `PLOT_ENTRY_PATCH.json`。没有重做数值分析或改动原冻结入口。

结论只覆盖去重后未使用 Imagenette 官方 train 库存目标权重下、单机器两种 CPU 配置的边际服务分布。通过六项点估计门槛不证明每个条件格均准确、保留测试通过、模拟器队列有效或 QLA 收益。实测分布的图片×会话交叉重采样区间也不是模型预测误差的置信区间。
