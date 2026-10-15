# 小规模功能冒烟 V2：仅修复已审查的来源连续性缺陷

用户明确要求“请修复并开始实验”，此前已授权缩小冒烟和严格显存已占用`<1800MiB`。原小v1完整FAIL与额度错误、旧043949/170计划、原奖励设计/V1/V2绑定均保留，不重封或运行。这里只启动功能测试，不修改正式预算、奖励/模型、数值门槛、KAIST旧奖励时间语义、2MHz网络或OOD数据协议。

V2奖励：32训练/8验证槽，两臂Full/NoJointDemandReward，seed1/delay8，30用户/35服务器，原公式/参数/critic/bootstrap/drain。视觉：两方法每臂8决策，30用户/4服务器/50ms及固定30秒drain，与小v1算法路径相同。不是正式性能结果，也不能直接满足旧72/32读者或完整正式入口。

reward_input_integrity.py不导入模型。它在所有冻结代码动态导入/命令构造/worker启动之前和完成后检查原独立审查登记的115项完整执行源码成员与字节、两个包摘要、所有修复/干预helper、原设计/绑定/25个checkpoint/config。仅两项公共资源文件接纳人类明确的新版本，并核对旧BEFORE字节。实际sorted-first30 KAIST成员/字节来自原审查地图，不从今日发现的新文件重新选数据。并发外部写入的点时检查限制仍保留，不声称原子文件锁/永久GPU独占。

prepare一次性创建本attempt/schedules，使用原冻结canonical生成器原配方。train(seed1,32)/validation(seed10001,8)，相同原始轨迹最短长度、guard10及60/20/20划分、原分布/RNG/7请求上限。唯一存储变化是新独立目录，绝不写旧共享目录。完整NPZ/JSON两对的字节、内容和集合固定进新plan；运行不创建/修复/替换日程。缺文件、孤立半对、添加成员、改字节/registry均拒绝；原失败/孤立物不碰。两臂真实输出的日程content SHA还必须等于登记值。

必须先有本目录independent_repair_review完整Type-A源复核，status为PASS/WARN且blockers为空、reviewed_sources_sha256精确等于六个本地源/文档/登记文件加公共qla_environment.py/windows_runtime.py/环境spec九项当前SHA。作者不填写通过。报告MD实际存在且其SHA进入新计划。该复核只支持一次小功能执行，不是完整正式入口或Type-B。

源复核后，准备者执行一次：

```powershell
$a='X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/visual_calibrated_policy/engineering_attempts/20261012_user_small_smoke_v2_repair'
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
