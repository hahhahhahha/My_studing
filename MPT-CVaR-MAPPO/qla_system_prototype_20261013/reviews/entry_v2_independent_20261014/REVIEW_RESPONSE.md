# E1 前置完整入口独立复审：FAIL

日期：2026-10-14。审查身份：gpt-6-astra xhigh / 独立上下文 Type-A / `/root/system_entry_v2_independent_review_20261014`。未参与本系统实现；依据文件、逐行源码、原始日志重算及一次新功能运行作判断。GPT 同家族审查不是 Type-B，不构成跨家族认证。身份标签不由文件哈希技术认证。

审查根目录 LAB：`X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013`。本报告中 `目录/文件:行` 均相对于该根目录；JSON 单行文件以第 1 行及字段定位。审查输出 OUT：`LAB/reviews/entry_v2_independent_20261014`。

## 1. 结论、阻断与边界

**当前正式 E1 入口不批准。** 不是因为“单机共驻一律不能研究”，也不是因为某个方法产生负结果。两个具体阻断如下：

| ID | 类型 | 发现 | 解除目标 |
|---|---|---|---|
| B1 | 软件正确性，P1 | 截止线程和响应线程可以给同一任务写入两条正式结局；`COMPLETE.protocol_complete` 仍为 true | 固定截止语义下原子地决定、只写一次正式结局；补受控线程交错回归，并按新源码重新 validation/freeze/registry/review |
| B2 | 历史来源完整性，P2 | 当前正式尺度引用的 M1 rr/stress 收据绑定旧 `systemlab/common.py`，但未找到匹配其 SHA 的实际文件；不能完成要求的历史 helper 字节闭包核验 | 提供该 SHA 的原文件并独立复核；若确实不可恢复，需另获授权登记替代来源/新校准及依赖验证，不能改旧收据冒充恢复 |

两项均为真实未解决事项，machine JSON 保留非空 `blockers`。B2 不表示 M1 数值伪造，也不推翻已能从原始日志重算的统计。干扰标定中的因果、完全隔离、保守估计等主张单列为 scope warnings，不借此捏造全面禁止共驻研究。

本次没有改待审源码、配置、协议、旧运行、旧计划或旧审查，没有训练、下载、E1 正式矩阵、强杀进程或新增自动化。所有新产物仅在 OUT。已结束测试不因后续上下文恢复而重跑。

## 2. 审查覆盖与读一致性

完整阅读了任务列明的 README、四份主要文档，11 个 `systemlab/*.py`、8 个 `tools/*.py`、2 个 `tests/*.py`、5 个指定驱动、全部 3 个 configs JSON，以及 frozen_v2、formal actor、grid、registry、validation registration、calibration scales。另读取了原 M1/M2 准备驱动、旧 edge/common 实现及实际 actor import 闭包。

`READ_INVENTORY.json` 对 333 个现有文件保存了实际字节 SHA、大小、JSON/JSONL 解析统计；包含所有指定 M1/M2c/旧功能运行实际输出、两套干扰尝试、protocol/data 和源码文档。原始 JSONL 全行读取并检查，不是只读取 analysis 摘要。`RAW_RUN_RECOMPUTATION.json` 是独立公式重算，`BINDINGS.json` 保存 39 个显式绑定检查和当前 19 文件 source 字典。修复报告、RESULT、UNIT_RESULT_attempt_02 也已完整阅读，其文件在最后封存清单补列。

3500 张 calibration/validation/test JPEG 逐字节读取、重算 SHA 和长度，并与原 `IMAGE_SPLITS.json` 成员比对：2000/500/1000 张；角色间 image ID 与内容 SHA 交集均为 0。原 split 实际 SHA：`8d6a3701fc3e046f8a2c898aaaf243ceb40449d2eb4d0fd7dfc2e0d8dcfb3b48`。源图片当前与 manifest 全部匹配。未把这些曾用于原视觉研究的角色包装成从未见过的新视觉泛化集。

全部对应 Poisson 日程的到达时间已按 seed/rate 重建；v2 六份日程还重新生成 user、image、assignment_uniform，逐项相同。没有生成或重采样新的正式日程。旧 v1 日程的 rate 小数截短与 v2 区别保留，没有混用。

当前实际 vision 权重 SHA 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`；actor checkpoint 为 `3444f039eb7359d37000ef0045ccd3efb859bb51ff1564b0c814e40d50213044`。117 个历史 Python 文件全部重哈希与 bundle 字典匹配。实际加载的 helper 闭包为历史 `mappo_gp_sf_dsr/{__init__,config,buffer,models}.py` 及 `repro_risk_averse_edge/{__init__,config}.py`；已读其内容，import 路径落在注册的 source_snapshot。

审查前后 333 文件、3500 图像、所有已核绑定和历史 actor 源均未变；`POST_EXECUTION_RECEIPT.json` 的变化列表为空。这是两个时点的哈希与运行自报连续性核对，**不是**持续文件监控、不可变镜像、数字签名或对“中途修改后又恢复”的排除证明。历史日志不能证明未记录的主机后台负载、频率、缓存状态或二进制依赖始终不变。

## 3. B1：截止窗双结局，确定性复现

定位：[systemlab/client.py:113](X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/systemlab/client.py:113)、[client.py:120](X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/systemlab/client.py:120)、[client.py:168](X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/systemlab/client.py:168)、[client.py:183](X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/systemlab/client.py:183)。

`send_task` 在响应后先记录 `end`，随后才取得 `outcome_lock`；只判断 `end <= cutoff`，不检查该任务是否已被最终登记。主线程在 cutoff 到达后取得相同锁，对尚不在 outcomes 中的任务写 `unfinished_at_cutoff`，但之后仍会等待发送线程。若发送线程在记录 end 后、获取锁前短暂被系统调度挂起，主线程可先结束登记；发送线程恢复后会覆盖内存 outcomes，并向同一 tasks.jsonl 再追加 `completed`。

本次没有改 client 源码。review 内的 `cutoff_race_fixture.py` 用合成 I/O、一个模拟任务和 trace hook，仅模拟上述可发生的线程暂停，直接调用原 `replay()`。其成功诊断目录为 `synthetic_cutoff_race_attempt_02/`：

- `RESULT.json`：`task_log_rows=2`，`unique_task_ids=1`，statuses 为 `unfinished_at_cutoff, completed`。
- `client/COMPLETE.json`：`protocol_complete=true`、`actual_outcomes=1`、`source_continuity_pass=true`。
- 记录的响应 end 为 `1079199.7676298`，cutoff 为 `1079199.8538304002`；响应计时实际在截止前约 86.2 ms，恢复发生在 CUTOFF 文件写完之后。
- 原 `analyze()` 正确拒绝：`ValueError('All offered tasks must have exactly one cutoff outcome')`（`analysis.py:22`）。因此后续分析有一道保护，但不能抵消运行完成标记错误或保证正式格子有效。

这是**合成并发正确性证据，不是额外真实系统运行或性能结果**。诊断辅助首次导入缺 Windows 进程变量，以及首次 FixtureSocket 构造器错误均完整保留：`.aris/FIXTURE_PRE_EXECUTION_ERROR.md`、`synthetic_cutoff_race_attempt_01/FIXTURE_FAILURE.md`。attempt_01 未执行任务；未与 attempt_02 拼接。唯一真实 HTTP/ResNet 功能尝试是第 10 节所列 attempt_01，未重试。

最小修复目标不是延长 drain，也不是把晚完成算入主指标，而是使“收到响应的真实 end 时间、正式 cutoff 分类、单次落盘”一致且每任务原子唯一次。可将原始响应收集与最终 cutoff 结局分开，等待有界 HTTP 调用收尾后仍按原固定 cutoff 判定，再对每任务只写一次正式结局。若采用 sealing 状态，必须明确处理 end 已在 cutoff 前但等待锁的响应，不能悄悄把时延定义改成拿锁时间。

必须回归的条件：截止前响应但锁前暂停、真正截止后响应、server timeout 后继续完成、client transport timeout、主线程早期异常且已发任务仍完成、队列拒绝及正常成功；每任务恰一条正式 outcome、迟到证据另记、全部状态守恒、COMPLETE 与日志一致、分析可解析，固定 horizon/drain 不变。

已核 M1/M2c 和本次真实功能日志未出现重复 outcome，late_response 均为 0；这些历史原字节可以保留为其源版本的有效观测，不需要删除，也不能仅因发现竞态就重写它们为失败。**它们不能被重新盖章成修复后源码的验证。** 修复后需要新 attempt 三方法 validation、随后新 frozen protocol、formal bundle/registry（重新绑定可复用的精确日程字节）及新独立复审。除非校准测量链受改动影响，B1 本身不自动要求重跑整套 M1；B2 的来源问题须另行解决。

## 4. B2：M1 原 helper 字节版本缺失

定位：`runs/M1_rr_s1_attempt_01/client/INPUTS.json:1` 与 `runs/M1_stress_s1_attempt_01/client/INPUTS.json:1` 的 `source_sha256["systemlab\\common.py"]`，以及 `protocol/calibration_scales_v1.json:1` 对 M1 rr INPUTS/COMPLETE/edge logs 的绑定。

两次 M1 登记的 common.py SHA 是 `6cedd7b3590da61d1a3e3ee12f20f6b5b0154954d5a0e8a5b54cc6418183a0d6`。当前文件 SHA 为 `2c1fdafd23653afd8938166c3ec9e03c094b789212645758b49af5622f09483c`；三修复 BEFORE 文件 SHA 为 `d72aca9edf50fe8806298a25870db58b2e497ea0ee7b22534c5636bf4fb76ec0`。二者均不匹配。

`LEGACY_SOURCE_RESOLUTION.json` 逐一核对 M1 收据中的源：其余文件可在 BEFORE 或原样移动到 drive_scripts 的文件中按 SHA 定位，唯独该 common.py 未解决。仓库内按 common.py 文件名搜索只定位到当前、BEFORE 和另一视觉项目的 common.py；该路径 Git 历史查询未返回可恢复版本。搜索有明确范围，不声称已穷尽所有外部备份或用户私有历史。

源码说明声称主要是 source_receipt 范围变更，但没有匹配 SHA 的原字节，不能用这一作者叙述替代独立核对。common 含请求、日志、哈希和运行环境 helper，完整入口要求其历史闭包可查。M1 逐任务日志、VisionEngine/edge 计时逻辑、实际统计均有大量可核证据；本项仅阻断“历史来源闭包已完整核验”的审批，不指称造假、不自动认定所有校准测量错误。

解除方式优先是提供上述 SHA 的原始 common.py，保留为独立历史副本，再核对与当前差异及对校准的影响。不得修改 M1 收据 SHA，不能把当前 common.py 复制后标作旧版本。若来源确实不可恢复，是否接受限定来源证据或登记新校准是需用户/协议所有者明确决定的额外授权；本审查没有擅自执行。

## 5. A/B：真实执行、指标与重算

**A PASS（限定执行工作负载，不是分类准确率研究）。** `edge.py:16–55` 真正按 SHA 加载 ResNet18，FP32 CPU、eval/inference_mode，解码原 JPEG、transform、forward、finite 检查及 argmax；没有抽取旧 service 分布代替本轮真实执行。HTTP body 在 `edge.py:209–211` 校验长度及 SHA；FIFO 等待与 inference 的单 worker 日志交叉一致。top1 是执行输出，不是 latency reference，也没有被拿来生成预测真值或分类标签。没有 ground-truth accuracy 主张，若以后要声称分类准确率必须另引入标签及相应指标。

**B WARN（公式正确，条件统计和小重复限制必须披露）。** `client.py:90–126` 的主 E2E 从计划 release 到客户端收完响应，包含分批等待和负载发生器迟滞；另有 send→response。`analysis.py:24–65` 只在 completed 集合计算未截断 mean/P95/P99/CVaR99；drop、timeout、unfinished 不填伪延迟、不用零替代。`common.py:81–91` 按上侧 1% 经验概率质量作分数边界加权，不是简单取超过样本分位点的可变数量均值。

从全部任务原始行独立计算如下；与已有 analysis 最大差异为浮点舍入约 1.11e-16。

| 运行 | 完成/提供 | 拒绝 | mean s | P95 s | P99 s | CVaR99 s | 尾部有效任务数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 rr | 599/599 | 0 | 0.065335 | 0.097981 | 0.114507 | 0.117515 | 5.99 |
| M1 stress | 15752/18916 | 3164 | 0.672809 | 1.346532 | 1.387114 | 1.407643 | 157.52 |
| M2c qla | 12565/12565 | 0 | 0.145194 | 0.337016 | 0.436857 | 0.486466 | 125.65 |
| M2c score_softmax | 12565/12565 | 0 | 0.182329 | 0.449863 | 0.682515 | 0.840482 | 125.65 |
| M2c stale_shortest | 12374/12565 | 191 | 0.416457 | 0.957830 | 1.316153 | 1.496935 | 123.74 |

M2c 是同一 seed101 的一次三臂 validation，不是三次独立重复；没有据此给出正式显著性、普适收益或方法排名结论。M1 rr 尾部质量仅 5.99 个任务，不能作为稳定 CVaR99 性能估计。本次 4 请求功能运行的尾部质量仅 0.04，更不能用于比较。

`paired_compare`（analysis.py:84–114）以同 schedule SHA/seed 的完整 run 配对，要求至少三对，不以任务当 IID；bootstrap 是描述性的。它只核对部分 comparison/budget 字段，未单独在 analysis 契约中纳入 source SHA、client_timeout/max_inflight、传播条件全体；因此正式汇总仍必须同时审原 registry/cell/INPUTS，不能仅凭散落的 analysis.json 跨条件拼表。无重试选择或删掉差结果的授权。

## 6. C/D/E：连续性、FIFO、迁移与公平性

**C FAIL，因 B2；当前 v2 身份和 M2c 原始数值检查通过。** 所有当前显式绑定 SHA 一致。M2c 三臂 INPUTS 的 source 字典与当前 19 文件相同，COMPLETE 的连续性检查为 true，全部 service RUNTIME 也匹配；旧 M1 的不同 source 身份如实保留。`freeze()` 确实要求新源码 validation，不是重贴旧验证标签，但 frozen protocol 仅绑定部分 client 文件，未把所有 edge/controller/process 收据或依赖二进制纳入门禁；本审查已独立检查这些额外文件。历史结束时 JPEG 文件及视觉权重没有全量再次哈希，服务内模型参数也无 vision 参数终态哈希，不能把 actor 的前后参数验证泛称成整个环境全闭包连续证明。

**D FAIL，因 B1；三项既有修复自身回归 PASS。** edge.py:107–156 将 dequeue→active 和 completion counters→active removal 放在一致锁下，队列守恒式可用。对全部已审 M1/M2c 的 admitted/execution_result 顺序、每节点 compute 时间不重叠、快照 `admitted=completed+failed+waiting+active` 与 `received=admitted+rejected` 均检查通过。waiting 容量 32 不含当前 active；M1 stress 和 stale-shortest 实际达到 32。每个服务一次仅一个 vision worker，不是 torch_threads 个并发任务。

18 项回归测试独立执行全通过：有限 FIFO/重复/守恒，cache 乱序与注入 loss，预算，fractional tail，角色与共同日程，Windows runtime PID、错误码/高位句柄/CloseHandle fail-closed，修复后成功/失败 completion 原子性，以及 v2 错 cell/actor/checkpoint/helper/generator/日程/报告等拒绝路径。测试并未覆盖 B1 的 cutoff 交错，所以“18/18”不能推出没有其他并发缺陷。

未发现 E1 实际路径中以 dead branch 冒充模型执行；`send_task` 的 weights/decision_start 参数未使用、analysis 的 csv import 等不影响该路径。`reference_service_s` 在 policy.py:64–65 的 load 比值代数抵消，load 实际等于任务数/request_scale；不能声称该 reference 参数本身调节在线 workload load。NoQL、NoBias、Azure、Docker 等代码存在不代表其已实际验证或是本次 E1 批准范围。

**E WARN，部署迁移和基线公平输入检查通过但不是原训练域等价证明。** policy.py:48–77 将位置/距离置 0，任务数及 payload 归一化，使用已送达摘要的 predicted_wait、service EMA、固定校准 transfer residual，age 按 source 时间算；queue pressure 截至 1，price 跨节点 min-max 正规化，price_scale log1p 后截至 10。这些是 actor 特征裁剪，不是最终 latency 截断。4 节点的 min/max 及 softmax 分母与 35 节点不同，属于显式零样本结构迁移加真实系统特征域迁移。

历史 models.py:182–265 的共享 edge/user 编码器支持 112维/35 输出改成19维/4输出的 actor-only 严格 state_dict 加载；当前仅取 Gaussian mean 再 softmax、减 0.5 price bias，log_std clamp 不参与随机采样。没有 optimizer、critic 更新或新的 CVaR 优化。实际重放过的 10759 条 M2c QLA 决策，用注册模型批量重新前向，最大 portfolio 误差 5.96e-8（FP32 批量算术差异）；actor 参数前后 SHA 均为 `13daaa5f2d076c188ce2810723a6e653bbe9bc6e8618a5c55b5e6e31630ba491`。

全部历史决策的 source timestamp/version 能在该用户实际接收、未注入 drop 且不晚于决策日志时间的反馈中找到；成本与 actor 特征重算最大差异 0，score_softmax 与 stale_shortest 的动作规则检查通过，共同 assignment_uniform 对应的任务目的节点全部相符。策略模块仅收到 policy 配置，不持有 monitor token；没有 fresh 私有 queue 读取。stale_shortest 实际最小化的是 queue+service+transfer 预测成本，而非仅 queue 长度，应准确命名/描述。

三臂共享 validation schedule SHA `8cf9e65ec45f375175a4b08dd8fb1427b5f6d3dd70e5741f696f21ed9f8273d5`、同 resource/初始 service EMA/warmup、16用户、50 ms decision、120 s horizon、30 s drain。每格重启服务并等初始完整反馈。策略不同导致后续 queue/EMA/cache 值不同是被测系统动力学，不应要求它们事后强制相同。

未发现用 test 性能选参数的证据；本次只读 test 输入，不执行 test。当前 validation registration 文字 rate_rule 缺少 round，而实际值/日程是 105/s；现有 run_m2_prep.py:14 明确 round(0.6*C_ref)。这属于历史文字不精确，不把 105 当成精确105.23721539481573。正式 v2 已用全精度 rates 并绑定，未偷偷放宽旧日程。

## 7. F/G：网络、预算、资源与进程

**F WARN。** 系统是真实 localhost HTTP 数据请求、真实 UDP datagram，软件在发送器处排队注入 delay，在收到 UDP 后按 seed/tick/user 决定注入 loss；没有真实无线链路、移动拓扑或跨机时钟同步验证。0 距离不是射频测量；不得恢复旧20 MHz仿真物理含义。

feedback.py:131–155 按发给所有实际逻辑 receiver 的序列化消息总字节扣 token，接收后注入丢包仍计发送成本。burst=65536、rate=65536 byte/s；不是每个 receiver 各自获得一份免计预算。监测 HTTP 不在此预算内，代码记录 response JSON bytes，不包含全部 HTTP 请求/头/内核开销。analysis 的 UDP/IPv4 28-byte 头是估计，不是抓包实测。M2c 120秒窗口应用反馈实际速率约 qla 65672.73、score 65787.82、stale 65742.27 byte/s；短窗均值略高于 rate 与 token bucket 的 burst 兼容，不能直接据此判“超预算”。periodic 200 ms 是候选发送周期，预算 deferral 会降低实际频率，不等于每用户精确 5Hz。

M2c 主窗监测 response JSON 总字节约 2.06/2.05/2.01 MB，控制器轮询、日志、Python/HTTP 线程、客户端和 actor 都共享本机资源，属于系统开销，不能把它们从真实时延解释中抹去。QLA P99 release→send 为53.42 ms，score/stale约50.90/50.97 ms；全日志无 client_backpressure。最大分别60.44/54.35/54.22 ms，说明客户端迟滞实际被包含。max_inflight 间接限制发生器能力，若未来出现 backpressure 必须保留该格且不能用于正常比较。

stale-shortest 有 6 次测量期间 monitor 错误，发生在相对开始8.81、32.46、54.86、57.70、59.81、114.01秒；另三臂各8次关闭后 monitor 错误。详见 FEEDBACK_DIAGNOSTICS.json。这并未制造 fresh 状态或破坏已核质量守恒，但须披露实际反馈延迟/缺失与监测线程共享资源，不能称监测链全程无异常。

**G PASS（Windows 本次进程管理），资源隔离 WARN。** launcher.py:15–68 使用 RUNTIME/health 的真实 runtime PID，Win32 API 指针宽度明确，Access denied、未知错误、WAIT_FAILED、CloseHandle失败都不当作已退出；未知PID不强杀。启动前环回端口 bind 检查，ready 后 health PID 与本次 runtime 对照，shutdown 也要求匹配。存在端口检查与真正绑定间的通常 TOCTOU 窗口，但后续身份核验限制了误接服务；不是敌对多用户环境的安全认证系统。

本次新功能运行的5个 runtime PID：controller58892、edge0=81388、edge1=50816、edge2=55712、edge3=58680，全部正常 `/shutdown`，runtime_exited=true，全部 shim exit0，无 lingering/unknown，forced_termination=false；独立结束后再查均已退出。没有关闭无关服务。

环境并非完全隔离：qla_np226 的 pyvenv.cfg 为 `include-system-site-packages=true`。NumPy2.2.6 在 venv，而 Torch2.11.0+cu128、torchvision0.26.0+cu128、Pillow11.1.0 实际来自 `X:/anaconda/ANACONDA/Lib/site-packages`。当前版本、module/metadata/RECORD和部分关键 native DLL/PYD SHA 已保存；它不是完整 hermetic 依赖锁，也未被现行 entry gate 全量验证。CPU 运行是显式部署配置，不是训练中的隐式 GPU fallback；安装 CUDA build 的 Torch 不等于本次用了 GPU。

4进程共用8物理核/16逻辑核及缓存、内存、功耗/频率域。torch_threads=1/1/2/2 不是 CPU quota；affinity mask 也不等于带宽、缓存、内存或功耗隔离。Docker cpuset 同样不能自动称完全独立物理资源。当前 frozen_v2 没有 affinity 配置；不得把诊断性 pinned 运行的隔离身份嫁接给正式 E1。

## 8. H：干扰标定重算与因果范围

**H WARN：数据描述部分可重算，报告强因果部分 FAIL，但不是共驻研究的全面禁令。** 两个目录完全分开审计；成功目录36份节点结果、每份200张，原失败目录保留12份 affinity error 和24份实际部分结果；没有拼接两套数据形成成功臂。各成功节点记录的 image_id 顺序、warmup20、线程数与 PLAN 一致，三条件每节点各三次。

独立按每run的200个 compute_s 求普通中位数，再对三run取均值：

| 节点 | solo ms | concurrent ms | concurrent_affinity ms |
|---|---:|---:|---:|
| edge0 | 28.05573 | 35.53010 | 29.22995 |
| edge1 | 27.70413 | 35.11492 | 31.80922 |
| edge2 | 19.19307 | 21.89648 | 18.46527 |
| edge3 | 19.42135 | 21.75248 | 23.14632 |

原 REPORT 表内数值有约0.01–0.02 ms的小差异，未完全精确复现其显示值；不影响“这些运行条件有观测差异”这一描述。不能据三次重复和这个表自动推论显著性或稳定因果。

关键限制，均直接来自 `drive_scripts/run_interference_calibration.py`：

1. **无 ready barrier**：worker在模型加载和warmup之后才 `start_event.wait()`（46–48行），主进程在 `p.start()` 后立即 `start.set()`（91、103行）。事件很可能早已set，不能保证各worker预热完后统一开始测量。
2. **无可核测量窗重叠**：结果只有持续 wall_s/cpu_s，无各进程绝对开始/结束时间和PID；且2线程run墙钟约5.3秒、1线程约8.1秒，即使同起点也不是全程同样并发程度。不能证明所有200图均在其他三节点持续测量下执行。
3. **PLAN 是后写**：110–114行在全部unit结束后才写 PLAN。seeded shuffle 可重建既定顺序，但现有 PLAN 本身不证明运行前已不可变预登记；同一顺序也在失败尝试重跑，不排除时间漂移。
4. **CPU量不是逐图CPU time**：49–56行用 process_time 包围整个200图循环，包括文件读取、预处理、forward及循环；rows只有 index/image_id/top1/preprocess_s/compute_s。REPORT:43 的“逐张CPU时间”不成立。
5. **CPU/墙钟比不变不代表CPU时间不变**：例如edge0 solo三次均值 cpu=6.255208 s、wall=6.421791 s；concurrent cpu=7.942708 s、wall=8.161044 s。两者都增大约27%，比值仍约0.97。REPORT:28 对此解释为“CPU工作量没变，只是墙钟拉长”不受数据支持。CPU时间也不等同于指令数；没有性能计数器或频率观测可归因缓存/抢占/睿频。
6. **拓扑证据有限**：原PLAN写8核和连续SMT配对，但未保存原查询输出、实际affinity读取回执及历史主机runtime。review已用只读 GetLogicalProcessorInformation 实测当前 core masks 为3,12,48,192,768,3072,12288,49152，支持当前8核和连续兄弟配对；它不是历史执行时的证明，也不测缓存域。
7. **失败不 fail-closed**：worker捕获异常写error但不重抛（61–63行），父进程仅查exitcode（92、106行）；故失败也可能exit0并进入最终PLAN。目录 exist_ok、write_text 允许同路径覆盖。此驱动不应直接作为后续可信采集入口，若再使用须新attempt、提前计划、ready barrier、异常聚合拒绝与source/runtime收据。它不在当前E1运行路径，不应把此风险假装成已运行E1的失败。
8. **原字节与来源**：worker只按image_id读取，不逐图记录原JPEG SHA、模型/依赖/runtime完整收据；当前图像/权重已核对，不能倒推标定历史中没有未记录变化。

允许表述：“在该主机、这些记录的条件下，同图序列的 compute 墙钟分布随共驻/亲和设置发生观测变化；残余干扰未被完全分离。”不允许“唯一原因确认是共享CPU竞争”“已排除时段/缓存/频率因素”“亲和完全隔离”“缓存/睿频机制已测得”“此结果必然是对QLA相对收益的保守估计”。最后一项尤其不能由服务变慢推出：共驻可能对不同策略产生不同方向的偏差。

有限共驻E1可在修复/来源阻断清除后研究**这一完整主机部署**的端到端表现；若要推广到独立边缘节点、或把路由收益从共享资源干扰中分离，需额外独立资源/受控干扰对照与更完整时间/硬件证据。这是加强论文主张的证据要求，不授权本review替换科学设置。

## 9. I：frozen_v2、具体cell准入与剩余门禁范围

当前实际SHA：

- FROZEN_SYSTEM_PROTOCOL：`68e0caa92cfcd30e2a70beb21a1b0955061a5cfbc42d902d7bd0981838b970da`
- FORMAL_CONFIG：`817b45713a1894890c7d2f8b2fdbe0899d8645661eaaa48e8e1c31a0e1e852c5`
- FEATURE_SCALES：`d68046e2f7cd208e7c66e2b698caeeb76b77bba834bbd812ba92b2e7e3971572`
- E1 grid v2：`a2a29f47d802bf6887fdaac10eb98a76363ac4f5b91bd956d935adce286b74d9`
- E1 test registry v2：`4e43cfe4cf86eb31d6cc9a0a0329be85033fc60e0625c9640a38a334c546752f`
- formal_actor_v2：`913dc6c6f9248b975c8c34324b45db3017552df6284d92cc2c62f75bb5997521`

registry/grid 当前18格、6份共同日程，seed1/2/3、rho .6/.9、三方法；rate分别105.23721539481573、157.8558230922236/s。日程任务数依次12629/12749/12567/18896/18940/18898。名义 C_ref=175.39535899135956/s 来自低负载 M1 逐节点 service 平均的倒数之和，不是容量/稳定性定理。压力pilot已出现3164拒绝且service均值由约29.58/29.82/18.56/18.46 ms增至38.25/38.26/24.24/24.23 ms，故不能把rho=.9解释为真实独立服务器负载率90%。

admission.py:22–32 核协议/当前源/validation文件；115–178 对test要求 registry、cell、review，核config/grid/scales/manifest/全部已登记schedule/generator，以及该方法actor bundle/checkpoint/完整历史源码和feature contract。正式actor与实际validation checkpoint匹配（95–103行）。actor helper全树启动前及结束重哈希；`run_e1_prep.py` 单独在registry固定，make_grid已在19文件source字典中。`run_e1_grid_v2.py` 只是调用这两者的准备wrapper，本身未入registry，但其输出grid/config/schedule均已绑定；没有据此声称所有临时驱动都被全闭包锁定。

本地launcher在建服务/输出目录前调用门禁（launcher.py:82），client在模型前调用（client.py:19），结束再调用并比较返回收据（187行）。18个实际cell在缺review情况下全部只读调用门禁并被拒绝；没有启动任何test服务。正向、错误格子、错误方法、SHA变化等路径由现有合成单元测试验证；没有创建假PASS审查来试开真实E1。

review要求完整报告存在及SHA、scope complete_system_entry、PASS/WARN且blockers为空、非self_audit、protocol/source/registry匹配。它不能认证作者身份、不能验证文本报告质量；因此独立上下文依然必要。当前本报告为FAIL，I的**最终准入结论FAIL**，即使现有绑定检查本身通过也不能越过B1/B2。

## 10. 已执行的有限复现与停止收据

以下是已执行命令的记录，不是重新执行授权。工作目录均为 LAB，解释器 PY 为 `X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe`；设置 `PYTHONDONTWRITEBYTECODE=1` 并使用 `-B`，不写历史source pycache。

```powershell
Set-Location 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013'
$env:PYTHONDONTWRITEBYTECODE='1'
$PY='X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe'
& $PY -B reviews/entry_v2_independent_20261014/audit_evidence.py
& $PY -B tools/verify_unit.py --output reviews/entry_v2_independent_20261014/UNIT_RESULT.json
& $PY -B -m systemlab run-local --config configs/functional_4edge.json --manifest verification/images.json --schedule verification/schedule_4edge.json --method qla --actor-bundle verification/actor_fixture_bundle.json --output reviews/entry_v2_independent_20261014/functional_4edge_attempt_01
& $PY -B -m systemlab analyze --client-dir reviews/entry_v2_independent_20261014/functional_4edge_attempt_01/client --controller-dir reviews/entry_v2_independent_20261014/functional_4edge_attempt_01/controller --output reviews/entry_v2_independent_20261014/FUNCTIONAL_ANALYSIS.json
& $PY -B reviews/entry_v2_independent_20261014/cutoff_race_fixture.py
& $PY -B reviews/entry_v2_independent_20261014/audit_completion.py
```

audit_evidence / audit_completion 仅审计读/计算与写新review证据；它们不是systemlab交付源码。启动前 netstat 和TCP/UDP bind检查记录于 PRE_EXECUTION_RECEIPT，8100–8104/8400–8404及9100/9400当时均可用；PowerShell网络查询cmdlet不可用，改用已有Windows netstat与socket检查，未安装任何组件。

唯一真实功能运行：4节点、16逻辑用户、4个pilot任务、0.4秒请求窗、1秒drain、warmup2。4/4完成，每节点恰1个 execution_result，实际UDP接收，ResNet权重/actor source真实加载，actor参数不变，optimizer0。该功能配置不是正式120+30秒资源测量，不能替代M2c。

停止依据：`functional_4edge_attempt_01/PROCESS_END.json` 与 `FUNCTIONAL_VERIFICATION.json`；5个自有runtime和shim均退出，shutdown authenticated，无强杀，stderr空。POST_EXECUTION_RECEIPT 重新检查这些PID均已退出。**本review没有仍在运行的实验或服务。**该判断仅指本review拥有的服务，不声称整台电脑没有其他用户进程。

## 11. A–I 汇总与允许/不允许主张

| 检查 | 结论 | 主要证据/限制 |
|---|---|---|
| A 真值/服务 | PASS | 真实JPEG/ResNet/时钟；不声称分类准确率 |
| B 统计定义 | WARN | 原始数值重算一致；conditional completed latency、单次validation、低样本tail |
| C 来源连续性 | FAIL | 当前v2及actor闭包匹配；M1 common.py原字节缺口B2 |
| D 并发与修复 | FAIL | 原三修复及18测试通过；新增cutoff双结局B1 |
| E 迁移/公平性 | WARN | 同缓存可见信息和共同输入通过；35→4及系统域迁移不能泛化 |
| F 网络/预算 | WARN | 真实localhost HTTP/UDP及per-recipient计费；无无线/全网络成本/独立资源证明 |
| G 进程/资源 | PASS/WARN | 正常关闭与Win32 fail-closed通过；线程、affinity不是全面隔离 |
| H 干扰标定 | WARN | 原始描述重算；屏障/重叠/CPU解释/预登记/因果主张不足 |
| I 当前正式准入 | FAIL | 实际绑定与缺项拒绝通过；B1/B2未清除，不批准E1 |

允许保留的表述：这是单机多进程真实执行原型；已用真实JPEG/HTTP/FIFO/ResNet/UDP跑通；当前M2c原始数值可重算；新独立功能检查完成；已发现并保留软件缺陷和来源限制。

当前不允许：E1已批准/已完成、正式性能胜负、跨家族认证、独立物理节点或真实无线验证、完全资源隔离、CVaR直接优化、零样本迁移保证、显著/普适收益、共驻偏差必然保守、已经证明缓存/睿频因果。修复B1也不会自动清除这些范围限制。

## 12. 交付与下一步

完整任务逐字保存于 REVIEW_REQUEST.md；本报告及 machine JSON 将在 `.aris/` 保存同字节副本与哈希索引。EXPERIMENT_AUDIT.md/JSON 以实验完整性scope汇总同一证据，不冒充另一次独立审批。

下一步仅建议，不在本次实施：先处理B1并新增精确回归；定位B2所需旧common.py；如需新校准/资源对照应明确取得授权。按源码改变重新产生新validation/freeze/registry/独立复审，保留现有全部记录。不得清空本报告blockers、直接改状态或用旧review SHA强行进入E1。
