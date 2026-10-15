# 配对只读工程 pilot 的一次资源门槛与自身 MODEL 加载入口（SOURCE ONLY）

本入口目前未构造、未运行。实际资源／kernel／模型／权重加载／队列／工程收集／输出序列化均为零；人工检查不提供启动许可、用户授权、独立 PASS、完整 pilot 或正式 learned 结果。

## 前置证据与范围

`scripts/visual_pilot_driver.py` 的 `RealPairedReadonlyPilotDriver` 消费真实 `PilotPreparationArtifactBindings`，由现有 `RealCalValidationPilotPreparation` 核对当前完整源／保护输入、真实两臂 functional smoke、原 FAIL 连续性与 fresh 独立修复复审、无 optimizer 工程入口独立 Type-A，以及真实用户网络方向。这里不要求完成 pilot／正式 protocol／未来 test 日程，避免循环依赖；也不代答原 2MHz 条件或 Alibaba 五分钟协议。

所有新源都要先纳入最终新源锁及 fresh 审查，现有旧 smoke 准备已过期。本轮不重封旧计划、不运行 `run`／`_worker`，不重试额度不可用的审查者。资源保留 `run-experiment` 原 memory.used <500MiB 门槛及隔离环境规格；没有 CPU 神经回退、全局修改、终止未知进程或门槛放宽。

## 未来实际调用的源码顺序

1. 真实准备／两臂原自身 checkpoint 与 roundtrip/RNG 文件／完整 cal-validation 输入核对；建立与路径、标签、JSON 编码无关的永久科学身份锁。全部共同条件按 role／语义身份排列，每项 QLA 后 PPO，不按结果挑选，重命名重复科学条件拒绝。
2. 首次 CUDA 分配前，只调用一次原 `build_cuda_model`：原资源门槛、独立 seeded CUDA witness 后建立 unused QLA reference。它不执行 actor 或 optimizer，参数、配置、空 optimizer 状态在后续保持不变；不对自身已有显存重复 free-memory 门槛。
3. 每个 descriptor 从同进程原历史类建立 fresh controller，专用 pre-pilot 加载函数只读取 externally hash-bound 的原自身 smoke `MODEL_STATE.pt`。QLA 原 Full 结构与 PPO 原19/4/clamp保持，加载后真实参数、结构、空 optimizer、原 bytes 再核对。不读取 optimizer／H／MV／训练 RNG 状态，不使用初始化权重代替。该加载器不是原 validation-only loader 的权限复用，也不是 test、训练或 resume API。
4. 完整原 role measured pairs、真实10s KAIST位置、实际共同完整 schedule、固定 upload/秒/角色窗/drain 创建 fresh 环境，交给 `RealReadonlyPilotEpisodeCollector`；全部 N 步真实 deterministic act/value→执行→过去收据→固定 drain→GAE／mass／uncapped/drop/unfinished／参数和 RNG 不变。实际服务真值只进执行和审计，不进 actor；零完成／负面结果保留，不延长预算或 drain。
5. 保留每次实际 MODEL-load witness、全部单 episode 原输出及 marker，逐项 rehash；所有配对 episode 结束只能输出 `ACTUAL_PAIRED_EPISODES_PENDING_REVIEW.json`。它是完整候选首阶段无 optimizer 工程记录待核对，不是完整 pilot、工程参数冻结、metadata generation 许可、ten trainings 或120cells完成。独立全配对证据登记／pilot充分性、H/MV trainingcarry与预算规则确认、最终冻结仍是后续步骤。

失败永久保留整 driver 科学锁、当前阶段与已闭合单项证据；当前 collector 自己保存实际局部记录。无自动 append/resume/retry/合并局部臂；后续软件修复需要保存源与错误、独立 additive reviewed attempt。只释放自身不用的缓存，不做系统／驱动重置。

## 本轮可执行的人工检查

仅使用隔离 `environment/qla_np226/Scripts/python.exe -B -X utf8 scripts/check_visual_pilot_driver.py` 检查人工描述覆盖／固定语义顺序／科学身份／静态接线和缺实际 bindings 的早期类型拒绝。人工路径、SHA、尺度与参数身份字符串显式标为 ARTIFICIAL，不创建实际 MODEL、smoke、checkpoint、用户回复或 PASS。原保护字节另核对；不导入 Torch、不解析 test timing CSV、不调用 driver 或 loader 的真实路径。真实资源、权限、模型和输入保护路径仍未验收，需 fresh 独立完整入口审查。

首次人工 identity/AST 成功后，直接读 producer 源发现 driver 错读 `output_files_sha256`，而原 collector 实际输出键为 `files_sha256`；尚未运行真实 driver，未产生任何实际局部臂。先保存首次报告和三份当时源码，再修复 consumer，并增加基于两份 AST 的字段接口核对：保存的错字段与人工缺字段均被静态拒绝。错误记录 `20261011_194845_989974_pilot_driver/STATIC_INTERFACE_DEFECT_中文.md` 保留；这项作者静态补查不是实际加载／序列化通过或独立认证。
