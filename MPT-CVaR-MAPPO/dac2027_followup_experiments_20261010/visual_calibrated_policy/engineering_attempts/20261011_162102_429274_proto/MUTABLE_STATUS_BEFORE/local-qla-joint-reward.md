# 本地 QLA 联合奖励消融环境账本

来源：正式 R1 全部25个Full training manifest与runtime记录；2026-10-09再次核对。

声明规格：Python3.13.5、NumPy2.2.6、PyTorch2.11.0+cu128、RTX4060Ti、CUDA；单工作进程、CPU算子线程1，不自动回退CPU。

规格文件：`results_qla_joint_reward_ablation_20261009/ENVIRONMENT_SPEC.json`。每次运行通过完整规格与源锁校验；不能以import或包元数据单独宣布ready。必须先通过种子123的CUDA矩阵乘法witness、GPU空闲检查与两臂冒烟测试。

2026-10-09检查：当前可读Anaconda包元数据为NumPy2.1.3，与正式结果不同；PyTorch导入Windows `_overlapped`报WinError10106，工具中NVML检查失败。默认与提权诊断都未证实可用。本记录不将此归因于用户主机驱动损坏。

状态：**NOT READY / BLOCKED PREFLIGHT**。没有训练进程启动，没有新结果。未安装、升级、重建环境，也未改变全局Python、系统网络或驱动。恢复隔离匹配环境/指定可用现有环境后，需重新做完整witness及文档调用检查。

完整失败输出：`results_qla_joint_reward_ablation_20261009/preflight.log`；状态：同目录`STATUS.json`。不能忽略版本差异继续复用旧Full作为严格配对训练对照。

## 2026-10-10 后续授权与隔离修复安排

用户授权遇到程序问题修复，并推进视觉校准策略实验、奖励消融和未见外生窗口测试。新的声明环境规格：`dac2027_followup_experiments_20261010/protocol/QLA_ENVIRONMENT_SPEC.json`，canonical SHA256 `75aa3aced99d49fa90a7318cd2b5dc8b6d8d320cd89c7ae740aff5f50cacbc88`。目标仅在新的venv中提供NumPy2.2.6，复用现有只读Python3.13.5/Torch2.11+cu128；不改全局Anaconda、驱动或网络设置。

当前仍NOT READY：尚未创建该venv/安装包/运行新CUDA witness或两臂smoke。现有CPU服务正式计时运行期间禁止这些重操作，先保护测量。初始化路径仅准备了process-local Windows变量恢复，不等于全局环境修复。服务结束后需实际版本、kernel/resource、模型smoke及fresh agent-follows-doc认证。原失败目录不覆盖，新输出使用20261010_repaired。

先重放Full验证历史指标与共同日程，再进入昂贵的25个NoJoint训练，避免预先浪费整个网格。若历史Full不能在相同规格重现，保存问题并建立明确的新公平配对训练，而不是放宽数值一致性门槛。未知GPU占用不终止他人任务，不自动CPU回退。

当前线程heartbeat `automation` 已扩展为三阶段推进：服务测量时15分钟，训练/评估时30分钟，长下载/处理按吞吐30–60分钟；正常等待安静，数值失败/负结果保留。服务校准成功后继续后续任务，不仅停在状态报告。执行交接见新目录MONITOR_HANDOFF.md。

## 2026-10-10 23:30–23:50：隔离构建完成，资源门槛尚未通过

venv `dac2027_followup_experiments_20261010/environment/qla_np226`实际构建成功：Python3.13.5、独立NumPy2.2.6、只读系统Torch2.11.0+cu128，base Anaconda没有修改。默认工具网络安装失败记录保留；获准公开网络访问后的新build attempt成功，路径在environment/attempts。规格SHA75aa3aced99d49fa90a7318cd2b5dc8b6d8d320cd89c7ae740aff5f50cacbc88未变。

NOT READY：23:27资源witness看到GPU6158/8188MiB已占用，超过原<500MiB门槛，在CUDA分配前停止。未做seeded CUDA kernel、两臂smoke、正式Full重放或训练；未知进程未终止，也没有CPU回退。已向用户询问该占用用途/释放时机。

新的奖励设计已prepare/check，25个旧Full checkpoint指纹匹配。`reward_followup_v2.py`是零执行状态下的新增一次性控制包装层，旧prepare绑定与设计不改；V2单独绑定和只读check通过。它防止smoke部分目录或正式run重复启动，不是算法/实验设置修订。GPU空闲后仍需实际两臂smoke和fresh代理按V2文档原样entry-check，之后才可25个Full重放→25训练→25测试。不能仅凭构建或只读check标ready。

## 2026-10-11 04:37–04:45：资源复核及保护测试

04:37:38完整进程内路径恢复后的witness看到GPU6168/8188MiB占用，仍未满足原<500MiB门槛，仍在CUDA分配前停止。没有seeded kernel、模型smoke或策略工作；未终止未知任务。新增合成入口保护测试首次在mock导入时遇到WinError10106，移到导入前调用现有process-local初始化后，实际四项测试通过、退出0。详见 `dac2027_followup_experiments_20261010/protocol/REWARD_V2_GUARD_FIXTURES_20261011_中文.md`。未改变封存的V2入口/设计，合成测试不是实际模型smoke或独立入口认证。30分钟heartbeat已实际更新成功。

2026-10-11 05:08:41 本轮单次witness检查：GPU6167/8188MiB占用，仍在原资源门槛处停止、没有CUDA分配/实际模型smoke/训练。失败attempt及资源记录保留。未重复轮询、终止未知任务或回退CPU；不涉及模型训练的视觉物理时间适配与基础测试可继续。

2026-10-11 05:38:24 本轮单次witness检查：GPU6168/8188MiB占用，仍在原资源门槛处停止，未作CUDA分配/实际模型smoke/训练。继续完成非策略的连续FIFO/预测压力广播工程边界测试；没有将工程重放混作RL结果，也未修改原环境规格/门槛或旧结果。

2026-10-11 06:08:25 本轮单次witness检查：GPU6167/8188MiB占用，仍在CUDA分配前停止。没有模型smoke/训练/重放旧checkpoint。继续完成只用NumPy/标准库、真实验证计时的共同日程/上传/动作前观测工程接入；明确标注路由夹具，不称策略测试或GPU环境就绪。

2026-10-11 06:38:34 单次witness：GPU6166/8188MiB占用，仍在CUDA分配前停止。方法特征/共享奖励合成记账检查用声明Python3.13.5/NumPy2.2.6，未导入Torch或调用模型工厂；历史模型类仅静态SHA/AST绑定。没有真实模型smoke/训练/重放或ready认证，未越过资源门槛。

2026-10-11 07:08 单次witness：GPU6167/8188MiB占用，仍未过原<500MiB门槛，未做CUDA分配或任何模型工作。07:14真实验证来源回执／共享奖励／两规则适配工程检查通过，包括drop和unfinished夹具；Python3.13.5/NumPy2.2.6，Torch未导入。没有QLA/PPO模型smoke/训练/正式测试或环境ready认证。终末异步学习回报尚未冻结，不能以规则工程成功越过模型入口门槛。

2026-10-11 07:38:20 单次witness：GPU6167/8188MiB占用，仍在CUDA分配前停止，无实际kernel/model/训练。07:43闭合回报候选与每用户时间GAE工程检查通过，使用只读旧验证工程轨迹和明确全零合成value；没有神经critic或优化器。旧Windows轨迹文本SHA与字节SHA标签错误通过严格CRLF→LF绑定并记录实际字节SHA修复新增检查，失败现场/旧文件不变。正式入口/协议仍未冻结，未越过资源门槛。

2026-10-11 08:08:24 单次witness：GPU6160/8188MiB占用，仍在CUDA分配前停止。08:14学习缓冲／真实类静态绑定和明确人工action/logp/value夹具检查通过，Torch未导入，真实控制器／模型／优化器从未调用。PPO-MV-A仍待原MV奖励的真实回执适配，已加语义拒绝检查防止把QLA奖励PPO冒充旧方法；未宣布环境或正式入口ready，原规格／门槛／结果不改。

2026-10-11 08:38:24 单次witness：GPU6159/8188MiB占用，仍在CUDA分配前停止，未做kernel/model/优化器/训练或CPU回退。08:45均值—方差奖励及方法语义缓冲工程检查通过：声明Python3.13.5/NumPy2.2.6、Torch未导入；隔离历史MVReward单类算术与5份既有只读验证来源回执逐值核对，人工buffer动作/logp/value另行标记，不称模型执行或完整独立审计。首次人工log零编码失败保存后仅修该样例；源路由/原始数据/资源门槛/环境规格不改。实际模型smoke及完整学习采集器/正式入口仍待完成。

2026-10-11 09:09:01 单次witness：GPU6159/8188MiB占用，仍在CUDA分配前停止，没有实际模型/优化器/训练或CPU回退。09:15学习采集器源码及4份明确人工行为样例的低层FIFO／回执／奖励闭合／GAE装配检查通过，声明Python3.13.5/NumPy2.2.6，Torch未导入。真实采集器构造和collect从未执行；不能称真实模型smoke/learned rollout/完整独立入口审计。源规则/原结果/测量记录保持只读，前次成功源快照保留。真实训练仍需cal-validation-only pilot冻结、原GPU资源/kernel门槛、模型smoke与fresh完整入口审查；环境不标READY。

2026-10-11 09:38:52 单次witness：GPU6159/8188MiB占用，仍在CUDA分配前停止，无实际kernel/model/训练。09:42完成NumPy-only校准／验证JPEG及物理位置的解析上传工作负荷诊断，未导入Torch、未改网络／rho或原测量；结果显示2MHz候选即使最快链路亦使大多数用户期望上传工作量超过该窗可用时间。数值不利结果保留，不作为软件bug放宽条件。已询问独立高带宽仿真条件或保留网络受限场景方向；不称环境READY／完整策略预实验通过，继续独立训练源码准备。

2026-10-11 10:08:29 单次witness（PID70432，environment/attempts/20261011_100829_189979_witness）：GPU6159/8188MiB占用，仍在原<500MiB门槛处停止，未作CUDA分配／模型smoke／训练／CPU回退。10:15新增MV跨episode连续统计器及人工低层装配检查通过：Python3.13.5／独立NumPy2.2.6、Torch未导入；既有验证回执0→2400→4800→7200→9600与旧单类公式逐值一致，校准人工装配0→480→960，均不是learner或优化器执行。219份保护输入／源和4份新增源指纹无变化。真实流式采集器仅源码接入，完整训练器／checkpoint／正式入口尚未完成，网络方向／原资源门槛未变，不标READY。

2026-10-11 10:39:02 单次witness（PID55444，environment/attempts/20261011_103902_790526_witness）：GPU6159/8188MiB占用，在CUDA分配前停止，没有模型smoke／训练／CPU回退。10:43共同预算／checkpoint事务保护工程通过：明确合成计数／统计数值及非模型纯文本存档字节，Torch未导入。真实模型导出接口只准备源码，未调用；224份受保护文件及3份新增文件SHA核对无变。完整训练器／虚拟H及随机状态、只读策略验证和checkpoint选择尚未完成，不能将文件保护测试或占位文件称作真实权重／模型就绪。原环境规格／门槛／网络设置未变，继续30分钟监视及独立源码准备。

2026-10-11 11:08:41 单次witness（PID67356，environment/attempts/20261011_110841_768962_witness）：GPU6159/8188MiB占用，仍在原门槛处停止，无CUDA分配／模型／优化器／CPU回退。11:17新增完整训练调度和只读验证源码及四份明确人工校准行为／H边界／精确回执前缀重建自检通过；Python3.13.5／NumPy2.2.6、Torch未导入，实际真实入口构造均被缺失冻结pilot／实际smoke／fresh完整审查保护拒绝。228份保护文件和3份新文件SHA不变。没有真实模型／权重导出／策略验证／训练／测试，不标环境或正式入口READY；原资源规格、网络和测量不改。

2026-10-11 11:39:00单次witness PID68816，及下一wake12:11:02单次witness PID32928，均6159/8188MiB且在原<500MiB门槛处停止，无CUDA分配／模型／优化器／CPU回退。视觉两臂smoke／本次权重加载入口新增源码；独立gpt-6-astra xhigh源审查完整FAIL报告保存后工具最终反馈额度中断，发现启动原数据绑定缺失和JSON字节SHA一次性锁缺口。保留原源／自检／FAIL后加严格原始输入／旧证据多点复核、规范化完整语义永久锁、完整本地历史Python源树绑定和回环数组证据导出源。12:11只读修复检查通过，232保护+170源共402条SHA零不符，明确mock哈希变化保护不改真实原始文件；Torch未导入、实际调用0。修复独立复审待完成，绝不将作者自检当PASS或READY，正式模型／重放／训练／测试仍0。源审查及额度错误轨迹已归档，环境规格、资源门槛、网络和测量不变，30分钟监视继续独立源码准备。

2026-10-11 12:38:50单次witness PID56096，environment/attempts/20261011_123850_284441_witness，GPU6159/8188MiB且仍在原门槛处停止，无CUDA分配／模型／优化器／CPU回退。12:45共同最终checkpoint规则／登记源及人工描述保护检查通过：382保护+3新文件SHA无变化，Torch未导入；人工十行描述不是实际权重或训练，0完成／drop／unfinished／长时延记录保留。真实登记类只以缺失授权保护样例拒绝，未register／seal／真实验证／选checkpoint；规则尚未pilot冻结。新增源使旧冒烟准备计划过期并实际只读拒绝，未重封／重跑；修复独立复审仍待完成，环境不标READY，原环境规格／网络／原测量不变。监视30分钟ACTIVE，继续独立固定checkpoint测试合同源码准备，不以测试结果选参。

2026-10-11 13:08:29单次witness PID53736，environment/attempts/20261011_130829_897043_witness，GPU6159/8188MiB仍超过原<500MiB门槛，在CUDA分配前停止；无模型／更新／smoke／重放／CPU回退。13:16只用NumPy／元数据的共同固定checkpoint测试合同检查通过，Torch未导入：原1000保留图片身份未变，人工120描述行共用10外生绑定、固定horizon／drain，拒绝早停／缺行／重复／字段变化；389保护+3新文件指纹无变，另做PowerShell字节复核。未读测试计时CSV、未生成实际测试日程、成功runtime构造或策略执行，正式120结果仍0；规则／网络／pilot未冻结，合同不准实际测试启动。新源让旧smoke计划过期，不重封／重跑，独立修复复审PENDING、旧FAIL保留，额度不可用审查者不反复调用。环境不标READY，继续30分钟ACTIVE及独立执行层／测试循环源码准备。

2026-10-11 13:39:02单次witness PID21144，environment/attempts/20261011_133902_630213_witness，GPU6166/8188MiB，仍在原<500MiB门槛处CUDA分配前停止。13:44保留执行配对表源和人工解析检查通过：原100项路径／SHA清单、8行虚构时长配对（包含1,000,000秒慢值）、缺失／重复／错身份／阶段／非有限／永久失败／副本边界；395保护+3新文件指纹无变。Path.open保护禁止实际test计时CSV，自检真实计时解析／catalog／模型／策略／训练测试调用0，无Torch／CPU神经回退。真实入口及cell边界重哈希仍须完整测试循环、pilot／实际smoke／fresh审查／最终权重，环境不标READY。进程CIM读取不可用后原生只读回退未见python/pythonw，不修改系统。新增源使旧smoke计划过期、不重封重跑，独立修复复审仍PENDING，旧FAIL及网络／数据权限方向不变，30分钟ACTIVE继续独立日程／评估循环源码准备。

2026-10-11 14:08:38单次witness PID53608，environment/attempts/20261011_140838_699167_witness，GPU6167/8188MiB，未过原<500MiB门槛，在CUDA分配前停止；原生进程快照无python/pythonw。14:13共同保留日程登记／读取保护源通过人工检查：4条手写任务／虚构图片pool，20种字段变更拒绝，时间边界／不可变索引／空到达／JSON重复键与双SHA保护，399保护+3新文件指纹无变；没有实际保留日程生成或加载／计时CSV／Torch／模型／策略／训练测试。真实完整入口／网络与pilot方向及独立修复复审仍未满足，环境不标READY，旧FAIL不变、旧smoke计划过期不重封重跑，继续30分钟ACTIVE及固定权重只读评估循环／证据导出源码准备。

2026-10-11 14:38:32单次witness PID72088，environment/attempts/20261011_143832_216702_witness，GPU6167/8188MiB，CUDA分配前门槛停止；原生快照无python/pythonw。14:50单cell只读四方法评估源码／人工报告保护检查通过：3组人工数据、12错报拒绝和人工语义独占锁；408保护+3新文件指纹无变，计时CSV／Torch／模型／队列／策略运行0。首次人工夹具长路径故障保存源／失败证据，仅改人工目录并另起attempt，不动正式锁或系统设置。当前无请求特征用默认网络，评估源明确拒绝非默认链路，另列高带宽须新适配／pilot／审查。原源FAIL／修复复审PENDING／资源与用户方向不变，旧smoke计划过期不重封重跑，不标READY；30分钟ACTIVE继续完整网格／正确模型加载／结果登记源准备，正式训练测试仍0。

2026-10-11 15:08:33单次witness PID59776，environment/attempts/20261011_150833_461357_witness，GPU6167/8188MiB，CUDA分配前原<500MiB门槛停止；原生只读进程快照无python/pythonw。15:18新增完整120cell登记源与纯人工覆盖保护检查：120描述共10绑定／每组12项，15种缺项／重复／配对／早停字段变更拒绝；416保护+3新文件SHA另做字节复核零不符，无Torch／test计时CSV／真实模型／evaluator／registrar构造或register／seal／队列策略运行。首次151614成功与源快照保留后加严生成数须从原共同请求文件重建，但真实登记解析／完成标志保护未实际测试，未伪造actual-complete／权重／PASS样例。仅源码准备，不是实际120结果或独立审计；最终源码只能登记完整真实cells后pending独立最终审计，不能自动统计／标complete_audited。新源让旧smoke计划过期，不重封运行，原源FAIL／修复复审PENDING／GPU与网络／数据用户方向不改，环境不标READY。奖励实际smoke／Full重放／训练和正式OOD／视觉cells仍0，30分钟ACTIVE，下轮继续独立共同请求日程生成／登记源码再接正确fresh模型网格启动。

2026-10-11 15:38:45单次witness PID65248，environment/attempts/20261011_153845_443323_witness，GPU6167/8188MiB，CUDA分配前原门槛停止；原生只读快照无python/pythonw，无CUDA／模型／CPU神经回退。15:45新增共同请求生成／登记源的纯人工RNG检查：虚构ID与两人工负载×五种子共332任务，与旧校准五流配方逐任务一致，重复／图片流隔离／全局NumPy状态／空请求／单用户周期26条无7cap／11设计拒绝通过；420保护+3新文件字节复核零不符，Torch未导入、原1000池采样／真实日程生成／真实准备权限保护／模型／队列／策略均0。准备权限与正式模型启动分离，尚无真实smoke／pilot／审查，不伪造这些证据。源码安排先实际smoke／cal-validation设计和元数据-only独立准备权限，后十份共同请求持久化，再装配不变的正式训练／测试协议，避免先需求未知日程SHA或在选完权重后重封其绑定；仅源阶段，不称独立PASS或冻结完成。旧源FAIL／修复复审PENDING／GPU、网络和Alibaba用户方向未变，旧smoke计划过期不重封／运行，不标环境READY。原数据／测量／论文不改，正式smoke／重放／训练／测试仍0；30分钟ACTIVE，下轮推进最终共同协议／环境指纹装配源再接fresh模型网格启动。
