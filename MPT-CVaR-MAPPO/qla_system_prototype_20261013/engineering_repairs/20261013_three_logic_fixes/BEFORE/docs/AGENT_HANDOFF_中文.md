# 可交给其他 agent 的实施交接

## 任务边界

实施用户两份新方案的“单机真实执行系统”，不是继续旧20MHz队列仿真。代码目录：

`X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/`

先读本文件、`EXPERIMENT_PLAN_中文.md`、`DATA_PLAN_中文.md`、`VALIDATION_REPORT_中文.md`。已停止的定时监控不重开。作者功能验证不等于独立完整入口复审。旧论文、RL源码、冻结数据/结果、旧失败记录都不覆盖或删除。

本次交付没有正式训练入口；使用历史 actor 的冻结部署。CPU是新原型明确的部署设置，不绕过原 CUDA 训练准入。不能因真实系统结果负面而修改科学设置让 QLA 获胜。

## 角色与交付

| 工作包 | 输入 | 需要产出 |
|---|---|---|
| A 软件/资源复核 | 源码、作者检查、4节点配置 | 独立源审查、真实4节点功能记录、资源与进程收据 |
| B 数据与校准 | 原 calibration JPEG、原模型 | 输入清单、并发低负载/容量 pilot、实际尺度及容量 |
| C 验证与入口复审 | B 尺度、冻结 actor、validation JPEG | 三方法完整验证、冻结协议、完整独立入口报告MD/JSON |
| D E1/E2 执行 | C 完整批准、固定矩阵和共同日程 | 每格原始记录/分析、配对重复结果、完整性审计 |
| E 新反馈贡献 | E1/E2可信基础、同预算二维矩阵 | E4、gamma=0需求项消融、交互差分、负结果 |
| F 新数据 | 官方 Azure2021 acquisition计划 | 原始归档/schema/单位审查/成员SHA/分窗与日程 |

可交给不同 agent，但 B→C→D 是依赖链；共享端口和物理CPU的运行不能并行抢占。一个 agent 对同一运行目录负责，不追加/合并失败臂。

## 0. 固定工作位置

以下 PowerShell 命令均在新代码目录执行。所有输出指定新的文件/目录；若已有相同输出，先阅读，不覆盖也不自动重跑。

```powershell
$repo = 'X:/Yi-one/workspace/MPT-CVaR-MAPPO'
$lab = "$repo/qla_system_prototype_20261013"
$py = "$repo/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe"
Set-Location $lab
& $py -m systemlab --help
& $py tools/verify_unit.py --output runs/M0_attempt_01/unit.json
& $py -m systemlab preflight --output runs/M0_attempt_01/environment.json
```

不安装或升级原环境。容器版另见第8节。保存完整命令、时间、PID、实际版本与源码SHA；不要把当前机器配置写成多物理边缘节点。

**源码收据范围（2026-10-13 修订）**：`source_receipt()` 只登记 `systemlab/` 与 `tools/` 下的 Python 文件，用于使冻结协议和入口复审绑定"交付源码"。临时编排脚本一律放在 `drive_scripts/`（不计入收据），否则一次性的驱动脚本会污染收据、导致已有 validation 证据和冻结协议失效。若确实需要修改 `systemlab/` 或 `tools/`，按第5节规则重跑受影响的 validation 并新增有限复审，不重写历史通过记录。

## 1. 先复核现有作者功能证据

`verification/real_http_attempt_4/`、`real_http_analysis_v2.json` 和 `actor_transfer_check_v2.json` 只是1节点、2用户、4请求、synthetic actor输入。先验证其输入/输出/source SHA。前两个失败尝试保留；不是当前背景任务。4节点功能证据以 `verification/real_qla_4edge_attempt_2/` 为准（attempt_1 是旧 launcher 的 PID 误判失败，仅作保留）。launcher 已改为按 RUNTIME.json 的 runtime PID 判定存活并逐步留痕 shutdown。

需要独立复现时，以新 attempt 运行，不覆盖已有结果：

```powershell
& $py -m systemlab run-local --config configs/functional_1edge.json --manifest verification/images.json --schedule verification/schedule.json --method score_softmax --output runs/M0_attempt_01/http
& $py -m systemlab analyze --client-dir runs/M0_attempt_01/http/client --controller-dir runs/M0_attempt_01/http/controller --output runs/M0_attempt_01/analysis.json
```

核对4任务各自唯一cutoff状态、真正的JPEG SHA、ResNet权重、execution_result、实际UDP接收、正常退出码、无遗留自有进程。辅助stub单元测试不能替代真实推理。

## 2. 导入原角色图片；登记共同请求

图片根目录是 `.../data/extracted`，因为成员路径已包含 `imagenette2/`。

```powershell
$splits = "$repo/dac_visual_calibration_20261009/protocol/IMAGE_SPLITS.json"
$imageRoot = "$repo/dac_visual_calibration_20261009/data/extracted"
& $py -m systemlab images --split-file $splits --image-root $imageRoot --role calibration --limit 2000 --output data/calibration_images.json
& $py -m systemlab images --split-file $splits --image-root $imageRoot --role validation --limit 500 --output data/validation_images.json
# test角色在协议冻结并批准后导入，不用于任何调参。
& $py -m systemlab schedule --manifest data/calibration_images.json --role calibration --users 16 --duration 30 --rate 20 --seed 1 --output data/calibration_rr_s1.json
```

`rate` 是全部16用户的聚合 requests/s，不是每用户。calibration/validation/test身份严格一致；不交换图像角色。相同实验配对共享同一个schedule文件，包括任务/user/JPEG/assignment_uniform。分钟Azure重构与Azure2021单位见数据文档；第一轮不用下载新数据。

## 3. 四节点共驻留实际校准

先审 `configs/local_4edge.json`：4进程、threads1/1/2/2、每节点单worker、32 waiting、16用户、50ms decision、200ms periodic、固定drain30s。它没有硬CPU隔离；如需更强隔离见Docker。不能把threads数写成GHz或者无线频宽。

```powershell
& $py -m systemlab run-local --config configs/local_4edge.json --manifest data/calibration_images.json --schedule data/calibration_rr_s1.json --method round_robin --output runs/M1_rr_s1_attempt_01
& $py -m systemlab analyze --client-dir runs/M1_rr_s1_attempt_01/client --controller-dir runs/M1_rr_s1_attempt_01/controller --output runs/M1_rr_s1_attempt_01/analysis.json
& $py -m systemlab calibrate --client-dir runs/M1_rr_s1_attempt_01/client --edges-dir runs/M1_rr_s1_attempt_01/edges --config configs/local_4edge.json --manifest data/calibration_images.json --output protocol/calibration_scales_v1.json
& $py tools/configure_pilot.py --base-config configs/local_4edge.json --feature-scales protocol/calibration_scales_v1.json --output protocol/validation_config_v1.json
```

每节点必须有实际calibration任务；全部runtime/退出/质量记录完整。实际容量参考为 `C_ref=sum_j 1/mean_service_j`；服务受并发调度影响，它不是稳定性定理。低负载后登记接近容量的calibration压力 pilot，检查队列、拒绝与客户端瓶颈；选定最终资源后从新attempt校准。同一资源版本下不要根据哪个设置有利QLA挑尺度。

候选正规窗按计划规则 `ceil_to_minute(max(120s,10000/(0.6*C_ref)))` 选定；validation也使用这个窗和固定drain。记录选择理由、CPU/RAM/进程观测、服务样本数与不足，不把4请求作者检查当M1。

## 4. 登记历史冻结 actor 与实际 validation

```powershell
$ckpt = "$repo/results_qla_formal_bothfix_20260930/validation/Full/seed_1/delay_4/KAIST/30u/poisson_bursty/checkpoint_LCC-MAPPO.pt"
$source = "$repo/results_qla_training_repair_pilot_20260930/source_snapshot"
& $py -m systemlab register-actor --checkpoint $ckpt --source-root $source --feature-scales protocol/calibration_scales_v1.json --edges 4 --allow-edge-count-transfer --output protocol/validation_actor_v1.json
& $py tools/check_actor_transfer.py --config protocol/validation_config_v1.json --bundle protocol/validation_actor_v1.json --output protocol/validation_actor_forward_v1.json
```

这是共享edgewise结构35→4的零样本迁移，且真实系统特征域改变，必须披露；不是重新训练4节点。旧 flat NoQL 不能裁剪输出冒充兼容baseline。`qla_no_bias_probe`仅是同一actor的推理bias探针，非原训练消融。

以calibration容量预登记validation的rate和窗，不读test。生成同一validation schedule，三种方法各新启动本次服务；顺序事先随机化。

```powershell
# 将 <H_SECONDS>/<VALIDATION_RATE> 换成已登记数值，不能按性能挑选。
& $py -m systemlab schedule --manifest data/validation_images.json --role validation --users 16 --duration <H_SECONDS> --rate <VALIDATION_RATE> --seed 101 --output data/validation_s101.json
& $py -m systemlab run-local --config protocol/validation_config_v1.json --manifest data/validation_images.json --schedule data/validation_s101.json --method qla --actor-bundle protocol/validation_actor_v1.json --output runs/M2_qla_s101_attempt_01
& $py -m systemlab run-local --config protocol/validation_config_v1.json --manifest data/validation_images.json --schedule data/validation_s101.json --method score_softmax --output runs/M2_score_s101_attempt_01
& $py -m systemlab run-local --config protocol/validation_config_v1.json --manifest data/validation_images.json --schedule data/validation_s101.json --method stale_shortest --output runs/M2_shortest_s101_attempt_01
```

三个方法各分析。检查完整任务质量守恒、正常关闭、每节点真实执行、过期cache、同输入公平性、预算/丢包/drain、客户端背压、actor前后参数hash、真实迁移行为。客户端背压是基础设施错误；零完成/负收益/集中卸载不是软件bug。若正式设置修订，保留本attempt，再新增cal/validation版本。

## 5. 冻结与独立完整入口复审

```powershell
& $py -m systemlab freeze --config protocol/validation_config_v1.json --validation-client-dirs runs/M2_qla_s101_attempt_01/client runs/M2_score_s101_attempt_01/client runs/M2_shortest_s101_attempt_01/client --output-dir protocol/frozen_v1
& $py -m systemlab register-actor --checkpoint $ckpt --source-root $source --feature-scales protocol/frozen_v1/FEATURE_SCALES.json --edges 4 --allow-edge-count-transfer --output protocol/formal_actor_v1.json
```

冻结后feature状态改变，因此必须重新登记formal bundle；旧fixture/validation bundle不能直接用于正式test。完整独立复审者需通读源码、计划、真实M1/M2记录并自己按文档做有限功能检查，输出完整MD及JSON；不能只做AST检查。检查清单：

- 输入角色/成员/JPEG字节/模型/checkpoint/完整可执行源绑定，启动前与结束连续性；
- 真正HTTP→FIFO→CPU forward→response；单worker、waiting32与in_service区别、重复/拒绝/超时后完成；
- cache只接纳已到达摘要、每节点source时间、乱序更新、陈旧特征，无实时私有queue进入actor；
- 压力项只用已发生提交，不读future schedule；同预算按实际receiver数计费，监测开销另披露；
- 真实时钟和planned-release E2E、固定cutoff、不截断/不补drop时延、所有状态唯一；
- actor结构迁移与域迁移、fairbaseline、参数不变、common schedule/uniform、资源一致；
- cal-only尺度、validation冻结、test未用于选择；CPU是显式新部署，非原训练fallback；
- 源码/依赖/实际资源/端口/PID/退出；只正常退出本次已核PID，不杀未知进程；
- 统计单位为配对run，CVaR样本不足披露；结果不可由fixture代替。

正式CLI必须传 `--entry-review`，JSON要求如下。复审者生成真实完整MD、实际SHA和当前source字典，不手工填假值。此机器可核对报告存在/哈希，但不能技术上认证填写者身份，操作者仍须确认为独立复审。

```json
{
  "schema": "systemlab-independent-entry-v1",
  "scope": "complete_system_entry",
  "status": "PASS",
  "blockers": [],
  "complete_report": true,
  "reviewer_identity": "实际独立复审agent或人员",
  "protocol_sha256": "FROZEN_SYSTEM_PROTOCOL.json实际SHA256",
  "source_sha256": {"每个当前Python文件的相对路径": "实际SHA256"},
  "report_path": "完整独立报告MD绝对路径",
  "report_sha256": "完整报告实际SHA256"
}
```

WARN只有blockers为空才可进入，保留全部warnings。`freeze`只是登记，绝不等于独立批准。源码变化导致原复审/冻结变旧，修复后新增有限复审，不重写历史通过记录。

## 6. E1→E2→E4，逐格启动与分析

```powershell
& $py tools/make_grid.py --feature-scales protocol/frozen_v1/FEATURE_SCALES.json --config protocol/frozen_v1/FORMAL_CONFIG.json --block E1 --output protocol/E1_grid.json
& $py -m systemlab images --split-file $splits --image-root $imageRoot --role test --limit 1000 --output data/test_images.json
```

先完成E1的18格再推进E2的48格；E4的48格在主结果可信后运行。每个rho/seed生成一次共同test schedule，聚合rate由grid读取、duration按冻结选择记录，不每方法重新随机生成。H和整个到达日程SHA另登记在矩阵执行清单；工具不会自动选H或启动矩阵。

```powershell
# 例：已登记的某个QLA格子；H/rate/seed/output必须对应grid实际值。
& $py -m systemlab run-local --config protocol/frozen_v1/FORMAL_CONFIG.json --manifest data/test_images.json --schedule data/E1_rho06_s1.json --method qla --actor-bundle protocol/formal_actor_v1.json --entry-review reviews/entry_v1/REVIEW_RESULT.json --output runs/E1_qla_rho06_s1_attempt_01
& $py -m systemlab analyze --client-dir runs/E1_qla_rho06_s1_attempt_01/client --controller-dir runs/E1_qla_rho06_s1_attempt_01/controller --output runs/E1_qla_rho06_s1_attempt_01/analysis.json
```

E2/E4变换不直接编辑冻结base配置。将登记的变换写成新JSON，例如 `{"injected_delay_s":0.1,"injected_loss":0.1}` 或 `{"mode":"coordinated"}`，再运行：

```powershell
& $py -m systemlab register-condition --base-config protocol/frozen_v1/FORMAL_CONFIG.json --changes-json protocol/condition_delay01_loss01.json --output-dir protocol/E2_delay01_loss01
```

每个派生条件先做非test角色功能验证并取得新scope完整入口记录，绑定派生协议SHA；同源码/actor/scales不变，共享基准pipeline验证，但不能伪称新延迟条件已跑过。正式命令使用派生config及该条件的review。条件工具仅允许登记feedback模式/周期/延迟/丢包/机制系数；不会改模型、角色、硬资源、预算。

配对分析按同rho和同delay/loss block组织至少3个真实重复，不能混入不同条件导致同system_id重复：

```powershell
& $py -m systemlab paired-compare --analyses <该block六份实际analysis.json> --treatment 'qla|periodic' --baseline 'score_softmax|periodic' --metric cvar99_s --output analysis/E1_rho06_paired_cvar99.json
& $py tools/interaction.py --analyses <同rho同传播条件下2x2共十二份analysis.json> --metric cvar99_s --output analysis/E4_rho06_interaction.json
```

同时汇总drop/unfinished/完成率/实际字节预算；不要只报conditional completed latency。gamma=0扩展要单独条件ID与矩阵，不与coordinated原臂合并。

## 7. 失败如何处理

完整保留FAILURE/LAUNCH_FAILURE/stdout/stderr/部分任务/旧配置/旧日程。只修复可定位软件错误；没有自动重试。新尝试新目录、source和planSHA，必要时重新校准/复审；两个未完整臂不能拼成一次成功比较。未退出自有PID先检查记录，正常shutdown；未知应用/系统/远程连接进程不终止。操作系统安装、重启、未知进程退出、资源或科研设置扩大需用户授权。

全部数据输出加上输入/程序版本、时间、hostname/PID、硬件资源和来源审计。正式结果独立审计后再向论文填入系统表/图/性能结论；论文旧主张不自动继承新结果。

## 8. Docker第二阶段（本机未验证）

本机作者只完成Windows多进程检查，Docker命令不可用、WSL未提供可用Linux发行版。需要用户完成安装/重启等系统动作后再继续，不擅自操作。

先生成新container配置：`bind_host=0.0.0.0`、edgeURL为`http://edge0:8101`等、controller为`http://controller:8100`、`feedback.udp_host=client`、clientbind`0.0.0.0`、权重`/inputs/resnet18-f37072fd.pth`、图片manifest根`/images`。这些地址/路径改变要新的容器cal/validation冻结，不能复用native正式收据。

`compose.yaml` quota1/1/2/2 CPU与1GiB memory为候选；实际cgroup收据和容器image digest必查。不够内存时保留OOM evidence，新增统一资源版本重校准，不能假称1GiB通过。volume环境变量包括SYSTEM_CONFIG/RESNET18_WEIGHTS/RUN_OUTPUT/SCHEDULE_FILE/IMAGE_MANIFEST/IMAGE_ROOT。

先 `docker compose up -d edge0 edge1 edge2 edge3`，核4个/health及模型runtime；再启动controller；后 `docker compose --profile replay run --rm client`。客户端command默认为score-softmax；QLA需要覆盖command并只读挂载实际checkpoint、历史source、registered bundle及同路径cal/validation/review证据。不要只挂bundle不挂依赖。每个格子独立project/run目录，正常down本project。容器模式必须与native一样保留完整输出；Dockerfile/compose是未验证模板，独立复核后才称可执行正式容器方案。

## 可复制的agent任务提示词

> 请从 X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/docs/AGENT_HANDOFF_中文.md 开始，按依赖顺序实施用户新授权的单机真实执行系统实验。先复核作者功能证据与源代码，然后做4节点calibration、三方法validation、冻结和独立完整入口复审，再按批准矩阵执行E1。复用原JPEG/权重/角色，保护旧论文/RL/冻结结果，CPU为新部署协议。所有尝试新目录，保存失败，不自动追加重试、不伪造复审、不按test/QLA胜负调参；正常关闭本次已核PID，不杀未知进程。Docker/系统安装重启须用户确认。记录实际阶段、证据路径与剩余事项，独立审计正式结果后再写论文结论。不要恢复已停止的定时监控。
