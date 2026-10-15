# V2 保留测试：原冻结方案的隔离续接入口

范围：完成固定预训练 ResNet-18 的服务时间测量与条件经验分布验收。没有视觉模型训练、QLA 训练、队列模拟器实验或论文修改。确认阶段及 V1 的所有原始记录、失败证据、冻结源文件保持不变；原 `runner.py run` 不得重启。

环境沿用 `.aris/compute/local-visual-calibration.md` 的 `f3609a91`，解释器为 `X:/anaconda/ANACONDA/python.exe`。父进程与统计进程不导入 PyTorch；原冻结 PyTorch 检查和每个真实测量块都在独立子进程运行。无需安装包或设置 `KMP_DUPLICATE_LIB_OK`。GPU 不参与测量。

先决证据：确认验证六项原门槛通过，严格独立审计的记录完整性及六项数值验收为 PASS（总体 WARN 是适用范围与不确定性限定）。历史审计中的“未授权测试”描述的是当时的状态；本次用户同意继续完成原服务测量方案，不扩大到政策实验。

## 冻结的实验设置

原 `PLAN.json` SHA256 为 `bbddc7b6e15e68625deb41992707d64bf32faa184910de8d8c01fbab1a64f282`。保留原 V1 的 1,000 张测试图片（与校准、确认图片分离），原五次会话与每次十个块的顺序；两种 CPU 配置配对测量，每块 200 张、200 次预热，合计新增 10,000 条真实请求记录。加上确认阶段 13,200 条，完整 V2 共 23,200 条新记录。服务边界为 JPEG 解码/RGB、官方预处理、FP32 batch-1 推理与后处理；不含磁盘读取、网络、排队或控制器开销。

会话开始至少间隔 900 秒；原空闲门槛、亲和性、线程配置、固定权重、测量顺序和全部慢请求均保留。不改变均值 10%、P95 15%、P99 20% 的六项验收，不重新拟合、不依据测试结果调参。统计完全调用原冻结 `evaluate(True)`，测试每张图片等权，五会话等权。500 次“尺寸×类别内图片与整会话”交叉 bootstrap 仅是实测分布区间，不是预测误差的区间。若失败，保存报告与原始记录后停止，没有自动重试。

## 准备与只读入口检查

工作目录：`X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2`。

准备者先执行 `resources`、`prepare` 各一次。新的 `RETAINED_TEST_CONTINUATION.json` 会冻结此入口、本文档、原会话及全部确认/故障证据。独立审查代理只执行下面的原样命令，不执行 prepare/run、不修文件、不测测试集：

```powershell
& 'X:/anaconda/ANACONDA/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/run_retained_test_isolated.py' check
```

预期退出 0，并显示 `RETAINED_TEST_READONLY_CHECK_OK` 与独立子进程中的原 CPU kernel witness。该命令检查冻结指纹、确认原始完整性、未启动测试状态及原测试元数据，不测测试图片的服务时间。审查代理还需核对源代码的数据隔离、无覆盖/重启、原计划一致性、统计隔离，并将结论绑定到 continuation 与入口源码 SHA256。入口通过不代表未来测试通过。

## 唯一正式启动与状态

仅在独立入口报告 `protocol/RETAINED_TEST_ENTRY_REVIEW.json` 通过后，启动 `run` 一次。使用隐藏 Windows 进程，输出分别保存至新 `logs/retained_test_stdout.log` 与 `logs/retained_test_stderr.log`。唯一锁为 `protocol/RETAINED_TEST_RUN_STARTED.json`，新状态为 `protocol/RETAINED_TEST_STATE.json`。这些都与原故障状态分离。运行不提供 resume、重测或重新封存；异常或进程消失需保留现场并停止。同日协议不允许静默跨日。

完成时原始测试在 `measurements/test/`，验收在 `analysis/TEST_REPORT.json` 与 `analysis/V2_保留测试结果_中文.md`。即便测量与点估计门槛通过，状态仍为 `complete_pending_independent_audit`，需从原始行独立重算后才能报告完整验收。

完整导出使用新名字 `analysis/v2_complete_service_requests.csv`、`figures/fig_visual_service_v2_complete.png/.pdf`，原确认图和 13,200 行 CSV 不覆盖。若数值验收失败，不生成带成功含义的完整输出，可另外生成如实的失败比较图。

完成后只读核查：

```powershell
& 'X:/anaconda/ANACONDA/python.exe' -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/scripts/run_retained_test_isolated.py' audit-all
```

独立结果审计应核查 23,200 条新记录、原测试池、V1 与确认指纹未变、实测阶段和时间戳、串行会话间隔、冻结模型预测及六项误差。结论限于同日单机的两种 CPU 资源配置；不证明逐条件准确性、真实部署、视觉精度、模拟队列有效性或 QLA 收益。
