# QLA-MAPPO 真实系统原型

本目录实现用户提供的两份新方案中的单机真实执行、周期反馈及固定策略下的预算反馈对照。进入点为 `python -m systemlab`。先读 [其他 agent 交接步骤](docs/AGENT_HANDOFF_中文.md)，再读 [实验规划](docs/EXPERIMENT_PLAN_中文.md) 和 [数据复用与获取](docs/DATA_PLAN_中文.md)。

已实现真实 JPEG→HTTP→有限 FIFO→ResNet-18 FP32 CPU 推理→响应、单 worker/节点、独立进程、实际 UDP 反馈、每服务器源时间戳、用户缓存、乱序处理、显式注入反馈延迟/丢包、序列化字节 token bucket、五种反馈机制、共同请求日程及分配 uniform、固定 drain、全部任务状态和实测阶段计时、冻结 actor 加载、分数/陈旧最短队列基线、配对重复分析。CPU 是这项新原型的显式部署配置。原 RL 训练目录和 CUDA 规则保持原协议。

2026-10-13 作者验证：6 项核心测试通过；一次 1-edge / 2-user / 0.4 秒请求窗中 4 个真实 ResNet-18 HTTP 请求全部完成，两个服务进程退出码 0；历史 Full actor 35→4 服务器前向验证通过，参数哈希未变。记录见 `verification/`。这些记录只验证软件功能，不能作为方法优劣、正式 pilot 或独立审查结论。

当前系统配置 `configs/local_4edge.json` 含 provisional 尺度。正式 test 调用要求尺度状态为 `frozen_after_calibration_validation`；该状态只能依据实际 pilot 与复审登记，不能凭空改标签。示例 `verification/actor_fixture_bundle.json` 仅用于 synthetic input 的 actor 检查，正式实验重新登记自己的经过 pilot 验证的 bundle。

主要源码：

| 文件 | 职责 |
|---|---|
| `systemlab/edge.py` | 模型加载、FIFO、接纳/拒绝、真实执行、私有监测接口 |
| `systemlab/feedback.py` | 反馈采样、预算、发送、缓存及丢包 |
| `systemlab/policy.py` | 仅用已送达摘要的特征、基线、冻结历史 actor |
| `systemlab/client.py` | 共同日程实时重放、用户分批决策、结果截止窗 |
| `systemlab/data.py` | 原图片角色导入、Poisson/Azure 到达日程 |
| `systemlab/analysis.py` | 全任务状态、完成任务未截断时延、CVaR99、配对分析、校准 |
| `systemlab/launcher.py` | 本次进程启动、就绪检查及正常关闭 |
| `systemlab/__main__.py` | 命令行接口 |
| `tools/check_actor_transfer.py` | 真正加载 actor 的前向/参数检查 |
| `tools/make_grid.py` | E1/E2/E4 随机化配对矩阵登记，生成后人工/agent执行 |

用现有隔离 Python 执行，无需安装 HTTP 框架：

```powershell
$py = 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe'
Set-Location 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013'
& $py -m unittest discover -s tests -v
& $py -m systemlab --help
```

Linux 容器入口另见 `Dockerfile` / `compose.yaml`。Docker/WSL 未在本机验证；已完成的 HTTP 功能验证是在 Windows 多进程中进行。启动容器前配置路径和资源配额，并记录实际 image digest/cgroups。

所有输出命令要求新文件/新目录；每个运行格子重启它自己的服务，使队列、EMA 和缓存从登记初态开始。Ctrl-C 或 `/shutdown` 关闭本次服务；失败记录和已经运行的部分完整保留。没有自动化监控、循环重试或正式训练入口。
