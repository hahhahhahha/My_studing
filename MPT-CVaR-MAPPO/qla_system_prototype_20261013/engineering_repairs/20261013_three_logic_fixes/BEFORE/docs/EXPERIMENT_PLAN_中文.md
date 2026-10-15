# QLA-MAPPO 真实执行与反馈协同实验规划

日期：2026-10-13。适用代码：`qla_system_prototype_20261013/`。输入为用户提供的两份系统方案及现有仓库资产。目标交付是可复现实验和有边界的系统证据。

## 研究问题与主张

| 主张 | 最小证据 | 对应实验 |
|---|---|---|
| C1：冻结 QLA 决策器能在共享陈旧反馈下驱动真实 JPEG 推理和有限队列，并在相同请求下表现出可测的卸载价值 | 真实 HTTP 请求/推理记录、FIFO/drop/unfinished完整计数、相同图像与请求日程、与强部署基线比较、多次独立重复 | E1、E2 |
| C2（第二阶段候选）：基于已发生的队列变化、年龄与提交压力的反馈机制，在相同字节预算下具有价值，并与 QLA 存在互补 | 反馈×路由二维对照、实际发送字节/消息和年龄、同预算、压力项消融、交互差分及负结果 | E4 |

部署跑通只证明 C1 的执行可行性；C1 的性能比较以及 C2 的贡献都要等实际结果。CVaR99 是评估指标；这套部署程序没有直接优化 CVaR。不追加另一个 MARL 控制器。

## 与旧研究的关系

旧原型是实测 CPU service 分布校准的仿真；本次是实际请求在真实软件 FIFO 中等待，然后实际 ResNet forward。保留已有 service、reward 和 OOD 证据链。旧20MHz视觉入口 revision5 的 SHA blocker 属于旧仿真入口；本新目录在自己的输入、代码、运行和复审记录下推进。

单机默认4个进程、16个逻辑用户、ResNet18 FP32 CPU。部署型 actor 也明确运行 CPU1T，用于测其在线开销。旧 CUDA RL 训练的 `<1800MiB` 规则依旧适用于原训练协议；本原型没有隐式 GPU→CPU 回退路径。4节点共用物理资源，其结果称为“单机多进程/容器真实执行原型”。Docker quota 才能定义容器CPU配额；Windows初期 threads 差异不能等同于硬CPU隔离。

## 控制与数据平面

`Controller --私有HTTP监测--> Edge FIFO + Worker`；`Controller --实际UDP/可注入延迟--> 每用户版本缓存`；`User --缓存特征/Actor/共同uniform--> HTTP JPEG --> Edge --> HTTP结果`。

每节点一次只能执行一个任务，等待容量32，FIFO、整任务拒绝。客户端缓存按服务器 source timestamp 接纳更新；summary中每台服务器保留自己的源时间戳。source/send/receive三个时间分别记录。用户决策仅调用缓存，策略模块不持有私有监测 token。控制器压力项只用已经收到的任务累计计数变化，不知道未提交/未来请求。

反馈模式：periodic完整摘要、random随机单节点、queue_trigger队列变化、age最老状态优先、coordinated变化+年龄+已发生提交率×过去service EMA。它们共用 token bucket。预算是**应用层序列化字节×实际接收者数**，包含发出后在用户侧注入丢包的 datagram；UDP/IPv4头估计另报，控制器监测HTTP另报。发送延迟是明确的软件注入；任务数据HTTP传输本身是本机真实传输。

## 预实验与冻结设置

1. 核对新源码并执行6项核心测试、真实小请求功能测试、冻结 actor SHA/前向不变检查；审查这些作者测试的范围。
2. 用原 calibration JPEG，在4节点同时驻留、相同资源配置下进行低负载和接近容量的 round_robin pilot。默认先20 requests/s×30s，再按calibration容量登记负载扫描。每节点获取足够实际执行记录，不单独测一台后把总容量乘4。
3. `systemlab calibrate` 仅接受完整 calibration-role运行，拟合每节点服务均值、平均JPEG大小、传输/HTTP响应残差。它不拟合测试结果。
4. 在validation角色冻结候选设置做一次完整重放，核对异常/拒绝/客户端背压、FIFO、尾部定义、实际年龄、正常关闭以及actor权重不变。失败按错误类别修软件或如实记录容量/迁移失败；不调参直到 QLA 获胜。
5. 写 `FROZEN_SYSTEM_PROTOCOL.json`：server IDs、CPU quota/threads/memory、1worker、32waiting、用户数、模型/权重、全部特征映射/尺度、50ms decision、广播/预算/burst、到达生成与trace变换、窗口/角色、请求字节、测时起点、timeout、固定drain、种子、源码/依赖/checkpoint/image/schedule SHA、随机化执行顺序。
6. 按真实 pilot 确定主窗：建议采用规则 `max(120s, 10000/(0.6*C_ref))` 向上整到完整分钟，确保低负载格子有合理尾部样本；`C_ref=sum_j 1/mean_service_j` 从并发calibration得到并披露其近似性。正式时同时报告实际完成量，CVaR尾部有效任务数 `0.01*n_completed`。
7. 初始候选 drain30s，validation检查结束计数后冻结。正式运行即使留有unfinished也不临时延长drain改善数值。轮数/seeds固定后才能读test。
8. 此部署阶段没有optimizer、训练episode、GAE、critic或H训练carry；不要从原视觉训练的episode/H冻结字段制造部署训练完成记录。以后训练4-server系统策略时另登记训练协议并保持current-action-free critic。
9. 独立 agent 按入口文档执行一次小功能验证并审完整数据流/计时/预算/actor迁移。写完整MD/JSON与SHA，清除实际软件阻断后开始E1。

## 必做实验

### E1：真实执行主结果（18格）

路由：QLA冻结actor、score-softmax、stale-shortest。初始4节点/16用户；名义rho .6/.9；重复seed1/2/3；周期反馈，统一预算、burst与注入条件。共3×2×3=18格。先synthetic Poisson任务，之后用Azure到达形状补一个预登记外部工作负载，避免一次增加过多变量。

主表：未截断 completed E2E mean/P95/P99/CVaR99、完成/拒绝/传输失败/timeout/unfinished各项、吞吐、反馈成本、决策开销、实际年龄。不能仅比较完成时延而忽略drop。参考rho不是稳定性保证。

### E2：反馈传播稳健性（48格）

QLA/score-softmax × rho .6/.9 × 条件 `(delay,loss)=(0,0),(.1,0),(.3,0),(.1,.1)` × 3重复。延迟为真实控制发送器的软件注入，loss发生在真实收到UDP后。记录实际used age，并保持HTTP数据平面条件一致。出现负结果直接报告。

### E4：同预算反馈与路由二维对照（48格，第二阶段）

QLA/score-softmax × periodic/queue_trigger/age/coordinated × rho .6/.9 × 3重复。random是可选额外基线。相同B与burst；同时报告实际B使用率，不把“同上限”误写成“逐次字节数完全相同”。如果periodic因完整帧超过burst而不可能发出，增加所有机制共同的burst或使用登记分片方案，先在validation调整并重新冻结。

coordinated相对periodic的效果，在QLA和score中分别取差；交互差分 `I=(Coordinated_QLA-Periodic_QLA)-(Coordinated_Score-Periodic_Score)` 按同schedule/repetition配对计算。仅这个证据支持互补性；与原matched-age sharedness干预分别报告。增加 gamma=0 需求项移除对照，判断改进来自压力项还是单纯年龄/变化触发。

## 后续可选

E3 scalability先16/32/64逻辑用户，4/8服务；实际CPU/RAM/Actor时间/吞吐记录。8服务须新calibration与资源登记；不能据单机容器宣称多物理节点扩展。后续第二台机器增加时钟同步误差、独立网络与分布式trace验证。

NoBias当前代码提供同一actor推理bias置0的 `qla_no_bias_probe`，属于推理机制探针。它不等同于原重新训练的NoBias消融。`frozen_noql`仅接纳实际训练且结构/输出匹配的checkpoint；旧35-server flat NoQL不能改输出为4。需要该对照时，用现有训练框架独立训练兼容4-server策略后再登记；首个系统主结果不用等待此扩展。

联合训练反馈机制与actor属于之后的方法研究。本次提供固定策略的控制平面候选和对照代码，未实现/宣称联合训练优化已完成。

## 结果与统计规则

共同控制：同一请求release时间、JPEG、user和assignment_uniform、服务配置、模型、固定horizon/drain；真实service受调度扰动，不强行替换为共同随机service抽样。每格新启动已登记服务；服务EMA、缓存、队列重新初始化。

随机化完整格子顺序，记录日期/OS负载/其他应用，必要时按时间block重复。最少3个实际run重复；统计单位为配对run/repetition，不把任务或50ms槽当独立样本。报告每run分布、配对差、描述性bootstrap区间；3重复CI不能包装为高精度普适保证。更多重复按预登记预算增加。

主时延起点为schedule planned release，包含decision窗口等待和实际客户端拖延；另报send→response。服务器独立记录body读取、queue、preprocess、compute、service。HTTP端无法精确分离单向upload/download时，残差记为transport+protocol+return合项；跨机未经时钟校准不能直接减epoch时间戳求单向时延。

超时后服务可能继续完成：客户端cutoff结果保留，迟到response单列，服务最终事件保留；迟到结果不改正式截止窗指标。零完成tail metric为null。客户端backpressure意味着负载发生器成瓶颈：格子不能进正式比较，保留记录后修基础设施并登记新尝试。

## 实施节奏与停止条件

| 阶段 | 工作与交付 | 预估 |
|---|---|---|
| M0 | 文档/源码检查、数据hash、Windows功能链路 | 0.5–1天；已有小功能记录可复核 |
| M1 | 4节点共驻留pilot，Docker资源验证或native限制披露 | 1–3天；Docker环境目前未验证 |
| M2 | calibration/validation尺度、actor域迁移检查、协议冻结、独立入口审查 | 2–4天 |
| M3 | E1完整配对结果+原始记录审计 | 18格；纯窗口时间约18×(冻结H+drain)，另算加载/重复 |
| M4 | E2传播压力，失败分析和第一版系统章节 | 48格，按实际吞吐估算 |
| M5 | E4同预算新机制、需求项消融、交互分析 | 在E1/E2可信后推进，预计1–3周含分析 |

不把附件中的会议截止日期当已核实日程；以各会议官方CFP另核对。三周内若真实队列/反馈/测量仍未稳定，保留原仿真论文主线，单独继续系统工作。

Docker Desktop/WSL当前不可用时先native多进程验证。安装Docker/WSL可能涉及管理员操作和重启；交接agent应把具体安装步骤/版本准备好，再由用户执行必要系统操作，不擅自重启本机。

## 交付要求

每格：INPUTS/RUNTIME/PROCESSES、edge.jsonl、feedback_send/receive.jsonl、decisions.jsonl、tasks.jsonl、late_responses.jsonl、CUTOFF/COMPLETE/PROCESS_END、stdout/stderr和分析JSON。失败完整保留。只有实际结果经审计后再写论文性能结论；系统正常运行、负结果和软件异常分别解释。
