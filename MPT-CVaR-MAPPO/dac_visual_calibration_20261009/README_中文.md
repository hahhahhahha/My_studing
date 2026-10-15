# 视觉工作负载实测与校准实验

本目录独立保存下载数据、模型、协议与结果，不覆盖原论文实验。

原设计：[完整中文方案](../paper_aamas2027_submission_v2/dac2027_visual_workload_calibration_plan_20261009.md)。当前状态看 `STATUS.json`，不能把入口准备完成理解为实测完成。

## 目录

- `data/raw/imagenette2.tgz`：官方原尺寸数据包。
- `data/extracted/imagenette2/`：保留原 JPEG 的解压数据。
- `models/resnet18-f37072fd.pth`：官方 IMAGENET1K_V1 权重。
- `protocol/`：环境规格、图片清单、划分、会话/测量块清单与哈希。
- `measurements/`：逐请求服务计时，pilot 和正式样本分开。
- `calibration/`：校准和验收文件；未经独立验收不得默认把平均服务时间替代完整分布。
- `schedules/`、`policies/`、`evaluations/`：后续校准模拟与策略比较，当前入口不伪造这些结果。
- `analysis/`、`figures/`：统计结果与最终图表。
- `logs/`：下载、预检和各块执行日志。
- `scripts/`：本实验隔离执行入口。

## 数据与模型来源

- [Imagenette 官方仓库](https://github.com/fastai/imagenette)
- [原尺寸下载](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)
- [ResNet-18 官方说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [IMAGENET1K_V1 权重下载](https://download.pytorch.org/models/resnet18-f37072fd.pth)

数据包下载后记录完整 SHA256，并验证归档和图像。权重必须匹配官方文件名的 SHA256 前缀 `f37072fd`，同时记录完整哈希。Imagenette 没有在本入口预设未经核对的“官方完整校验值”；自行记录的哈希是可复现标识，不是独立来源证明。

只作视觉分类服务负载测量，不声称预训练 ResNet 已识别校园建筑，也不以 Imagenette 图像划分评估视觉模型的泛化能力。视觉模型冻结，不训练。

## 如何执行

PyTorch/GPU 初始化已修复。本实验 Python 入口自动补齐缺失的进程环境变量。
若要在同一 PowerShell 中启动其他 Python 脚本，可先执行（不安装软件，也不修改持久环境）：

```powershell
. X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009/Initialize-PyTorch.ps1
```

该入口验证 CPU/CUDA 运算、反向传播、TorchVision 前向和 nvidia-smi，结果存入
`protocol/PYTORCH_GPU_INITIALIZATION.json`。只初始化而不复测可加 `-SkipCheck`。
GPU 可用不改变原定 CPU-1T/CPU-4T 的服务测量协议。

准备目录、下载、环境预检：

```powershell
Set-Location X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py prepare
```

若当前自动下载不可用，可在正常浏览器中下载上述两个官方文件，分别放入 `data/raw/` 与 `models/`，保留文件名，然后重跑 prepare。不要放 160/320px 版本冒充原尺寸包。

查看状态（不启动下载、推理或训练）：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py inspect
```

单独重做 CPU 推理环境检查（不会下载或计时）：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py check
```

数据和环境均通过后，生成图片清单、冻结划分，运行小规模实测：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py pilot
```

正式校准及验证测量分开执行：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py calibration --session 1
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py validation --session 1
```

最终 test 测量入口受校准冻结门控制，不能在看过测试结果后再选校准规则：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/pipeline.py test --session 1
```

`test` 需要后续校准验收产生的 `calibration/CALIBRATION_FREEZE.json`；它必须包含合法 schema、accepted、环境/图片哈希、全部5个校准与3个验证会话的数据指纹，以及独立 `VALIDATION_REPORT.json` 的哈希与验收通过标志，不能仅放一个空标记。正式测量还要求真实模型 pilot 通过，图片清单缓存每次核对冻结哈希与角色隔离。脚本不自动假定原模拟器能够正确复现实测服务分布，也不会擅自训练新策略。完成测量后按原方案的路线 A/B 做接入与验收，再执行策略比较。

## 已固定的测量规则

模型：ResNet18_Weights.IMAGENET1K_V1，FP32，batch=1，eval，inference_mode；官方 resize/crop/normalize；不量化、不 compile、不训练。

两资源配置：CPU-1T / CPU-4T，interop=1。用 Windows 原生处理器拓扑选择一个／四个物理核心上的各一个逻辑执行单元；如果不能识别拓扑则停止，不把连续逻辑编号猜成物理核心。只修改测量子进程的 affinity，不修改系统设置或电源计划。

每个块独立进程、模型常驻、完整流水线预热 200 次。JPEG 字节在计时前读入内存；计时包括解码、RGB转换、预处理、模型推理、argmax与结果字典生成；不包括读盘、上传、排队、日志写盘、QLA 控制器计算。各阶段与总时长均记录，异常慢请求不裁剪。CPU 同步执行，不借用 GPU 时间。

图片角色：train 中预实验 100、校准 2000、验证 500；官方 val 中测试 1000。十类等额采样，seed=20261009。先按文件 SHA 去除重复；同时记录 dHash，采用 Hamming<=2 的近重复检查，跨角色冲突停止核实，不声称该筛选穷尽一切语义近重复。

会话内两配置使用同一图片清单和顺序；会话之间按固定种子重新洗牌。A/B按最多200张的块交替、首块方向按会话反平衡。校准5会话、验证3新会话、test5保留会话，共33000条正式请求（两个profile总计），不含pilot/预热。会话分散到不同时段，需要人工确认空闲时段，不能连续启动后声称覆盖了独立日期。每个块生成后台负载、频率/温度可用性和版本记录；传感器不可读时标记 unavailable，不编造温度。

正式阶段使用 `--session` 显式指定一次会话。例如 `scripts/pipeline.py calibration --session 1`；缺少 `--session` 时会停止并提示补充参数，不是交互式选择。这样避免一次性连续跑完五会话并冒充跨时间测量。

上述示例仅启动各阶段的第1会话。校准需要完成1–5会话，验证完成1–3会话，测试完成1–5会话；依次安排在不同空闲时段。示例不是跳过剩余会话的许可。

本测量是后续“实测校准模拟”的数据来源，不是35服务器实体部署。策略测试和部署结论必须另行验证。

## 运行环境与状态

本新实验可以冻结当前现有软件版本，不要求把全局 NumPy升级为历史强化学习实验版本。`protocol/ENVIRONMENT_SPEC.json` 记录选定版本；import成功不是ready，必须通过固定种子 CPU kernel witness 和真实预训练模型流水线pilot。

2026-10-10首次检查时，工具执行环境缺少关键Windows环境变量，PyTorch导入报WinError10106，NVML也不能初始化。已在本实验启动入口按需恢复缺失的SystemRoot、windir、ProgramFiles、ProgramW6432、USERNAME、TEMP/TMP；CPU、CUDA及TorchVision运行检查通过。持久环境未变。官方数据和权重现已下载，真实模型pilot的200条记录已通过门控。

## 本次正式测量执行安排（2026-10-10）

本次安排13个同日会话，首块预热前时间至少相隔15分钟；首个校准会话已完成。它不是跨日或多设备测量。启动前检查机器和所选核心负载，繁忙时延后，不修改已测慢样本；温度不可读、频率没有连续采样会明确保留。

`scripts/formal_runner.py init`在验证测量前冻结执行顺序、间隔、空闲检查和源文件哈希。`SERVICE_DISTRIBUTION_VALIDATION_RULE.json`预先指定经验服务表对保留验证池的分布验收：均值/P95/P99相对误差分别不超过10%/15%/20%，两配置均需通过。它只验收服务分布，不验收模拟器队列或策略收益；不会把平均值当完整服务模型。

独立复核入口（只读，不启动计时、下载或训练）：

```powershell
Set-Location X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261009
& X:/anaconda/ANACONDA/python.exe scripts/formal_runner.py check --pre-review
```

独立源文件/协议复核通过并存档后，后台执行入口为：

```powershell
& X:/anaconda/ANACONDA/python.exe scripts/formal_runner.py run
```

入口自动采用已完整审计的第1校准会话，然后顺序完成其余校准、验证和保留测试测量；不并行计时、不覆盖既有块。冻结验证容差失败或出现不完整记录时立即存档并停止，不偷偷改变验收线。全部通过后输出严格计时审计、服务分位数、阶段比例、会话块区间和PDF/PNG图到`analysis/`与`figures/`。这些输出不代表模拟器接入或QLA策略比较已经完成。

没有改全局软件、系统网络、驱动、电源设置或论文文件。没有用模拟/随机数字代替实测服务时间。
