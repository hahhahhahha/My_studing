# 共同保留请求日程登记／读取保护：源码与人工描述自检

新增`retained_test_schedule.py`为未来完整固定checkpoint测试入口准备不可变日程读取接口，不生成日程、不读服务时长、不运行策略。原`VisualRequestSchedule`仍拒绝test，未改旧保护。真实`RegisteredRetainedRequestSchedule`首先要求完成的固定权重合同，当前pilot／实际smoke／完整Type-A源绑定，再读取测试前登记的文件路径、外部字节SHA及完整请求语义SHA。当前真实构造没有成功或执行，不能用纯描述对象替代授权。

两负载×五种子各一份外生日程，四方法与三延迟共享同一图片、真实到达、原配对测量会话和任务分配uniform。日程没有方法／延迟字段；每条任务只含task_id／user／arrival_seconds／image_id／measured_session／assignment_uniform。未来执行层才查配对真值；actor／QL／critic／归一化／checkpoint选择不能获得日程、会话、uniform或实际未来时长。原1000图元数据身份及1–5原会话必须匹配；请求角色、共同网格／负载／burst规则须与pilot冻结合同逐项一致，不静默重抽样、过滤请求或按完成数量早停。

完整文档字段集、JSON重复键、非有限常数、布尔冒充整数、任务缺失／重复／重排、未知图片／会话、到达越界、uniform=1或条件变化全部拒绝。到达使用真实物理秒，按已有决策规则下一网格观察，包含时刻0任务和精确边界；没有外部轨迹加速／循环或终止条件修改。登记日程使用不可变任务tuple、不可变任务索引及副本序列化；接口查询不推进随机流、不重新生成任务。空到达人工案例保留，不因为任务少就删cell；真实低尾部样本数须如实报告。

语义SHA独立于JSON空白／键顺序；外部文件字节SHA也必须匹配，等价重编码仍不能替代登记文件。未来cell前后必须实际调用`verify_boundary()`复核日程／图片／源／权重登记，完整测试执行循环与一次性cell输出／失败保护尚待接入及独立检查。源码不是端到端无泄漏／完整120cell认证。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_retained_test_schedule.py'
```

命令仅用4条手写人工任务、2个虚构图片身份／虚构pool SHA和空到达案例检查描述保护，不使用原1000图生成实际测试日程，不打开原test计时CSV、不调用队列／Torch／模型／策略。真实构造只检查缺失授权拒绝，不伪造PASS／checkpoint／冻结协议。原独立源FAIL及修复复审PENDING、GPU原门槛、网络方向／pilot H／episode／terminal／load／时长尚未满足；新增源使旧smoke计划继续过期，不重封／重跑，正式训练测试仍0。还需在cal/validation-only pilot冻结设计后，以隔离随机流生成并登记10份真正共同日程，再完整评估循环／导出／独立审计。原服务数据／论文／旧结果不改。
