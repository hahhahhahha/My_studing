# 小规模功能冒烟 V4：短目录规避实际 Windows 路径长度失败

V3完成B4源WARN（MD e915d3d3，JSON7736d6d0）后，一次prepare保存了train NPZ/JSON，但validation路径264字符被Windows拒绝，计划/registry未创建，CUDA/模型0。兄弟20261012_user_small_smoke_v3_windows_parent下的两份train原件与FAILED_PREPARATION.json保持，不自动补齐/重试。本V4目录仅改为短名20261012_s4；三个运行Python源与FROZEN_BASELINE逐字节等于已审查V3，科学配方/数据/RNG/门槛/规模不变，不修改系统长路径或驱动设置。新位置、文档和九项源绑定仍由独立复核后才准备新计划，不复用旧报告。

V2奖励的一次实际启动在任何GPU/kernel/模型前失败于严格getppid校验，Full未进入训练，第二臂未启动。原reward_smoke_attempts/20261012_2041_user_small_v2全部输出与永久锁、V2计划/两对日程、完整FAIL/WARN源报告及尚未运行的visual204021计划保持原样，不重试/追加。标准库+psutil的独立无CUDA诊断观察到逻辑parent69280→venv launcher51424→actual child44016，launcher正是声明的隔离python.exe且命令完整相同；这不是异进程篡改或硬件失败。

V3仅在另存源码中支持该精确至多一层Windows启动器，仍要求声明解释器和完整worker命令、活的记录逻辑父PID及创建时间、同入口run命令；拒绝任意祖先/额外中间层/未知exe/错命令/PID复用。父进程对完成的实际worker证据与Popen启动器PID再次核对。实际workerPID与launcherPID分别保存，不混称。所有资源/包/算法/数值/小规模及正式预算不变；新源完整复核后才一次登记新计划并执行，不能沿用V2旧计划/报告当V3许可。

用户明确要求“请修复并开始实验”，此前已授权缩小冒烟和严格显存已占用`<1800MiB`。原小v1完整FAIL与额度错误、旧043949/170计划、原奖励设计/V1/V2绑定均保留，不重封或运行。这里只启动功能测试，不修改正式预算、奖励/模型、数值门槛、KAIST旧奖励时间语义、2MHz网络或OOD数据协议。

V3保留V2规模：32训练/8验证槽，两臂Full/NoJointDemandReward，seed1/delay8，30用户/35服务器，原公式/参数/critic/bootstrap/drain。视觉：两方法每臂8决策，30用户/4服务器/50ms及固定30秒drain，与小v1算法路径相同。不是正式性能结果，也不能直接满足旧72/32读者或完整正式入口。

reward_input_integrity.py不导入模型。它在所有冻结代码动态导入/命令构造/worker启动之前和完成后检查原独立审查登记的115项完整执行源码成员与字节、两个包摘要、所有修复/干预helper、原设计/绑定/25个checkpoint/config。仅两项公共资源文件接纳人类明确的新版本，并核对旧BEFORE字节。实际sorted-first30 KAIST成员/字节来自原审查地图，不从今日发现的新文件重新选数据。并发外部写入的点时检查限制仍保留，不声称原子文件锁/永久GPU独占。

prepare一次性创建本attempt/schedules，使用原冻结canonical生成器原配方。train(seed1,32)/validation(seed10001,8)，相同原始轨迹最短长度、guard10及60/20/20划分、原分布/RNG/7请求上限。唯一存储变化是新独立目录，绝不写旧共享目录。完整NPZ/JSON两对的字节、内容和集合固定进新plan；运行不创建/修复/替换日程。缺文件、孤立半对、添加成员、改字节/registry均拒绝；原失败/孤立物不碰。两臂真实输出的日程content SHA还必须等于登记值。

第一轮V2完整FAIL原件、审查前六源和随后B3源WARN保留在兄弟目录20261012_user_small_smoke_v2_repair下的independent_repair_review、SOURCE_BEFORE_FIRST_V2_REVIEW、independent_repair_rereview。B1/B2/B3在V2后续复核源层面解决。本V3保留所有source_guard路径（包括prepare、父进程、私有worker、动态导入前）的解释器/Python/NumPy/Torch发行版本和NumPy归属校验，不导入模型；worker在资源/kernel通过后再检查自身实际导入的模块、GPU及intraop/interop均为1，独立保存ACTUAL_RUNTIME证据。冻结旧源、算法和数值标准不改。

V3第一轮完整FAIL/B4报告、修改前六源与完整命令修订WARN保留在兄弟20261012_user_small_smoke_v3_windows_parent下的independent_repair_rereview、SOURCE_BEFORE_COMMAND_REPAIR、independent_command_rereview。命令修订从已解析的明确参数重建唯一规范worker命令，同时核对真实进程current.cmdline()、程序参数和启动器命令；丢失-B、重复选项、变化次序/额外参数均拒绝，不用sys.argv自行证明其规范性。额外无模型诊断PID25456确认系统实际命令与隔离解释器/真实base native进程的关系。

必须先有本目录independent_command_rereview完整Type-A新源复核，status为PASS/WARN且blockers为空、reviewed_sources_sha256精确等于六个本地源/文档/登记文件加公共qla_environment.py/windows_runtime.py/环境spec九项当前SHA。作者不填写通过。报告MD实际存在且其SHA进入新计划。该复核只支持一次小功能执行，不是完整正式入口或Type-B。

源复核后，准备者执行一次：

```powershell
$a='X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_s4'
$py='X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe'
& $py -B -X utf8 "$a/reward_smoke_small.py" prepare
& $py -B -X utf8 "$a/visual_model_smoke_small.py" check
```

prepare只登记真实原配方小日程和新计划，check只非神经轻量自检；两者不是实际smoke。两份新PLAN各用其实际文件SHA，指定从未存在的独立输出后只运行一次：

```powershell
& $py -B -X utf8 "$a/reward_smoke_small.py" run --plan "$a/SMALL_REWARD_PLAN.json" --plan-sha <实际SHA> --output <新reward_smoke_attempts子目录>
& $py -B -X utf8 "$a/visual_model_smoke_small.py" run --plan <check生成的新PLAN路径> --plan-sha <实际SHA> --output <新visual_calibrated_policy子目录>
```

每臂独立进程；worker必须有活父进程记录，严格新1800和种子kernel在模型前执行，CUDA版本匹配，无CPU回退。锁在实际执行前独占保存，失败锁/partial/log保持，不自动重试、追加或双臂合并。此脚本没有formal/resume API。实际smoke全部通过后仍须真实证据独立审查/缩小证据采用及完整入口，先25冻结Full严格1e-8重放再正式变体训练；视觉先cal/validation设计冻结和人类网络方向，OOD等人类五分钟协议。负结果或数值失败不当软件bug“修”指标。
