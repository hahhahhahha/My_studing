# 数据复用与新增数据获取方案

日期：2026-10-13。建议第一阶段零新增数据下载，复用现有 Imagenette 与 ResNet-18；先完成真实执行链路。新增 Azure 2021 逐调用数据作为第二个到达工作负载，视觉任务扩展放到主结果之后。

## 现有数据能否继续使用

| 已有资产 | 新系统中的用途 | 使用方法 |
|---|---|---|
| `dac_visual_calibration_20261009/data/extracted/imagenette2/` | 实际上传 JPEG 与真实分类推理 | 原样发送字节；每次测量解码/预处理/forward；可直接复用 |
| `dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json` | 冻结图片成员及角色 | 原 pilot 100、calibration 2000、validation 500、test 1000，沿用角色，导入时重核 SHA |
| `dac_visual_calibration_20261009/models/resnet18-f37072fd.pth` | 所有节点同一个视觉执行器 | 已核 SHA `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`；batch1、FP32、eval、inference_mode |
| `dac_visual_calibration_20261010_v2/measurements/` 和 analysis | 初始容量/服务量级参考，已审计历史证据 | 保留其 23200 条新记录；多服务并发原型另做校准，真实执行时不抽样这些 service 时间 |
| `datasets/external_traces/extracted/azure/` | 到达强度形状 | 2019 每分钟计数必须按列解析；分钟内时间为明确的合成重构 |
| `results_qla_unseen_external_windows_20261009/data/` | 已抽取 d01/d11/d12/d13 强度窗和原转换证据 | 可用于系统输入，但不将一分钟数据称为逐请求真实时间戳；与旧 OOD 的测试窗使用历史一并披露 |
| `Traces_TimeXY_10sec_txt/KAIST/` | 以后做移动场景时的真实位置输入 | 10 秒物理时间插值；当前 localhost 节点共址，特征位置/距离为显式0；未接入无线物理模型 |
| Alibaba 2017 / 2018 | 可选外部容量压力形状 | 不用于修改物理节点 CPU 实测值；2017 原 minute-window 准入失败仍保留，另行登记 native300秒方案后才做相关实验 |
| `results_qla_formal_bothfix_20260930/validation/Full/...` | 冻结 QLA actor 的零样本域迁移 | actor-only 加载并绑定完整源码/权重；35→4 的共享编码器可结构加载，真实性能靠 pilot；旧 flat NoQL 不能缩成4输出 |

因此答案是：**现有 JPEG、模型、轨迹可以继续使用，旧服务分布不能代替新系统的实际执行时间。** 现有图像角色已经在历史服务研究中使用过；新论文须说“同一工作负载的真实系统重放”，不包装成此前完全未见的视觉泛化集。

## 推荐新增数据：Azure Functions 2021

官方逐调用记录字段是 `app, func, end_timestamp, duration`。使用 `start=end_timestamp-duration` 获取记录中的开始时间，再选预登记半开时间窗。图片由共同随机日程独立绑定，原 duration 只用于重建开始时间；新 ResNet 服务时间实际测量。

来源：[数据专页与下载](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md)。发布方说明时间戳经过修改；引用 SOSP 2021 *Faster and Cheaper Serverless Computing on Harvested Resources*，遵守其 CC-BY 许可。

需要特别核对单位：该数据专页写 seconds，而仓库总 README 写 millisecond。不能自动猜单位。按专页 schema 与原始样本核对后，在记录中明确 `seconds` 或 `milliseconds`；CLI 强制 Azure2021 提供 `--time-unit`。保留采用的文档版本、内容 SHA、原始 schema/样本和选择理由。

执行方案：

1. 在新目录 `datasets/system_prototype/azure2021/acquisition_attempt_01/` 下载官方 RAR，保留归档。
2. `systemlab fetch` 输出实际归档 SHA 和下载 URL。若官方未给 checksum，明确 `upstream_checksum_verified=false`；本地 SHA 固定后用于后续成员与字节审计，不能称作官方 checksum 通过。
3. 用已安装且可审计的 7-Zip/bsdtar 在新目录提取 CSV；拒绝路径穿越、链接或覆盖旧输入，记录工具版本与提取成员 SHA。
4. 检查列名、单位、非负 duration、`start=end-duration`、时间范围和重复调用；不按 QLA 效果选窗。
5. 预先指定不重叠的 calibration / validation / test 时间窗和间隔。窗口内取全部 invocation；若需要降负载，登记共同 thinning 或显式时间缩放，报告变换。
6. `systemlab schedule --azure2021-csv ... --trace-start ... --trace-window ... --time-scale ... --time-unit seconds` 生成一份固定任务表；所有方法共享图像/到达/user_id/assignment_uniform。

下载命令（只有其他 agent 开始数据阶段时才执行）：

```powershell
& $py -m systemlab fetch --source azure2021 --output data/acquisition_attempt_01/AzureFunctionsInvocationTrace2021.rar
# 安装好的 7-Zip：先列成员并审查，再解压到新目录。
7z l data/acquisition_attempt_01/AzureFunctionsInvocationTrace2021.rar
```

原有 Azure2019 可直接用以下命令。`--count-scale` 由 calibration 的实际容量选取并冻结；不得根据 test 结果改变：

```powershell
& $py -m systemlab azure2019-schedule --csv '<官方 invocations dXX CSV>' --columns 1 2 3 4 5 --manifest data/calibration_images.json --role calibration --users 16 --count-scale '<已登记比例>' --seed 1 --output data/azure2019_calibration_s1.json
```

## 图像扩展：按需要获取

第一阶段不需要完整 ImageNet。现有原尺寸 Imagenette 是 ImageNet 十类子集，官方另提供尺寸缩小版；继续使用当前原尺寸字节，避免同时改变上传载荷。[Imagenette 官方页](https://github.com/fastai/imagenette)

若需要更丰富图像：有合法 ImageNet 访问权限时，下载官方 ILSVRC2012 validation，预先按图像 SHA 选500–2000张并建立新角色清单；访问条款和许可由数据获取 agent 保存。[ImageNet 官方入口](https://www.image-net.org/download.php)

若需要第二个输入域，可从 [COCO 官方下载与条款页](https://cocodataset.org/#download) 获取 val2017。使用 ResNet-18 处理 COCO 只能作为分类执行工作负载/输入域变化；需要目标检测主张时另登记检测模型、权重、预处理、输出和真实服务校准。目前本原型仅实现 ResNet-18，不将分类结果伪装为检测结果。

新图片获取必须产出 `ACQUISITION.json`、归档 SHA、逐图 SHA/bytes/宽高、坏图清单、许可证/来源、角色清单、重复/近重复审查。用新图片池时需新增独立 manifest 导入器并审查，不覆盖原 `IMAGE_SPLITS.json`。

## 本次未做的获取

本次只核对官方来源并复用本地图片/权重，没有下载 Azure2021、ImageNet 或 COCO，没有重新下载 Alibaba。数据获取方案可以由其他 agent 按以上步骤执行。
