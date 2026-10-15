# QLA-MAPPO：远程实验源码同步包

本包包含系统原型和三个补充实验所需的代码、配置、协议、历史源码快照、小型分析结果与复审材料。它不是完整数据备份，也不是远程实验已获批准的证明。原始文件逐字节复制，不改写绝对路径、模型/输入 SHA、冻结协议、失败标记或审查结论。

## 目录导航

| 工作包 | 入口 |
|---|---|
| 单机真实执行系统原型 | `MPT-CVaR-MAPPO/qla_system_prototype_20261013/README.md`、`docs/AGENT_HANDOFF_中文.md`、`systemlab/`、`tests/` |
| 视觉任务服务校准与后续视觉策略实验 | `MPT-CVaR-MAPPO/dac_visual_calibration_20261009/`、`dac_visual_calibration_20261010_v2/`、`dac2027_followup_experiments_20261010/scripts/` |
| 联合需求奖励消融 | `MPT-CVaR-MAPPO/results_qla_joint_reward_ablation_20261009/PROTOCOL_中文.md`、`results_qla_joint_reward_ablation_20261010_repaired/`、followup 的 reward 脚本 |
| 外部 OOD 压力实验 | `MPT-CVaR-MAPPO/results_qla_unseen_external_windows_20261009/EXPERIMENT_PLAN_中文.md`、`paper_doc_tools/run_unseen_external_windows.py` |
| 共享实现与冻结历史执行源码 | `mappo_gp_sf_dsr/`、`repro_risk_averse_edge/`、`results_qla_training_repair_pilot_20260930/source_snapshot/` |

`SYNC_MANIFEST.csv` 记录复制文件的相对路径、字节数和 SHA256。`EXTERNAL_ASSETS.csv` 记录扫描范围内因类型/大小而未进入 Git 的文件；它不是所有外部依赖的完备清单。虚拟环境、缓存、`.aris`、原始图像解压树未遍历打包。原始图片仍需根据 image manifest 单独同步。

## 先校验源码包

```bash
python3 remote_tools/verify_snapshot.py
python3 remote_tools/verify_snapshot.py --absolute-refs
```

第一条必须没有 missing/mismatch。第二条只列 Windows/绝对路径引用分布，不修改任何文件。

## 环境与数据恢复

1. 在服务器新建独立环境。系统原型的原依赖见其 `requirements.txt`；原本机是 Python 3.13.5、NumPy 2.2.6、Torch 2.11.0、TorchVision 0.26.0、Pillow 11.1.0。版本可获得性、GPU/驱动与服务器兼容性需要在服务器实际验证，不能在安装失败时悄悄换版本继续正式实验。
2. 不上传或复用本机 `qla_np226` 虚拟环境。补充实验的 Windows/CUDA/设备规则以其当前环境协议为准；Linux/新 GPU 不会自动通过本机门禁。系统原型明确使用 CPU，与原 RL 禁止 CPU 回退不是同一工作包。
3. 图像、视觉模型权重、冻结 actor checkpoint、KAIST 移动轨迹、Azure/Alibaba 原始数据、大型 JSONL/运行日志单独传输。使用既有协议中的 SHA 校验，不重下已保存的 Alibaba 原始包，不补造缺失数据或日志。
4. 本机路径通常以 `X:/Yi-one/workspace/MPT-CVaR-MAPPO` 或 `X:\\Yi-one\\workspace\\MPT-CVaR-MAPPO` 开头；相对位置虽保留，Linux 无法原样解析。**不要全局替换冻结 JSON/源码中的路径后继续声称原 SHA 和入口批准有效。** 建立单独版本的服务器路径映射/派生配置，登记新运行身份，再做受影响的功能验证、校准/验证和独立入口审查。
5. 原型 `Dockerfile` / `compose.yaml` 是原作者未验证模板，不应直接视为可运行的正式容器实验。CPU 亲和性/配额/主机变更会改变服务性能，不能把本机 M1 容量标定直接认作服务器容量。

可在 Windows 单独传输依赖目录（将示例服务器地址与实际目标替换）：

```powershell
scp -r <原始图片目录> user@server:/srv/qla-assets/images/
scp <ResNet权重文件> user@server:/srv/qla-assets/models/
scp <冻结actor checkpoint> user@server:/srv/qla-assets/checkpoints/
```

这只是传输示例，不会自动执行、重下载数据或改写协议。完整运行溯源还需依协议传输相关原始日志与历史 source，而不是仅传模型权重。

## 当前审查边界

- 系统原型 v2 入口是 FAIL；保留其两个阻断与完整报告。
- v3 复审请求及索引已经准备；在本会话的独立 reviewer 因额度限制中断时，尚未交付完整 v3 批准。以包内实际 `REVIEW_RESULT.json`、完整报告及绑定为准，不能把请求文件或执行者的“已修复”当作批准。
- v3 本机审查即使后来通过，也不会自动批准另一主机、系统、路径和硬件配置的运行。
- 视觉补充实验的服务校准已完成不代表策略训练/20MHz pilot 已完成；历史 handoff 含不同时期陈述，需结合对应版本和真实输出判断。OOD 的原 Alibaba 分钟协议有 native 300 秒采样不兼容问题，不直接启动原 243 格。
- 本次同步没有启动训练、测试网格或定时监控，没有删除旧失败记录。

## 给远程执行 agent

> 先阅读本文件并校验 SYNC_MANIFEST.csv。按四个工作包分别读取权威协议、实际状态和完整复审证据，列出缺失外部资产与机器相关路径。先完成服务器环境/数据身份登记和功能验证，不修改或冒用本机冻结身份；必要时用新版本做校准/验证及独立入口复审。所有运行使用新目录，失败完整保留，不自动重试/追加/合并，不按结果选设置。未经新的有效入口批准不要启动正式矩阵，也不要恢复已关闭监控。
