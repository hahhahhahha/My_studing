# 单条真实闭合工程轨迹收集源码（未执行）

`scripts/visual_pilot_episode.py` 只准备实际校准／验证 **无优化器** 工程收集，不是 launcher／训练／test／完整pilot或冻结器。当前实际构造、收集、模型动作、序列化均0；原资源／独立修复复审／工程许可及用户网络方向仍待满足。新增源使旧smoke准备过期，不重封运行。

入口必须消费同一真实 `RealCalValidationPilotPreparation`、它登记的共同方法描述，以及真实历史CUDA类、fresh MODEL-only加载的**自身实际functional-smoke checkpoint参数**。原parent／child冒烟报告、完整checkpoint字节和参数SHA均核对；随机初始化权重、手填action／value、CPU／stub、其它方法或恢复optimizer状态不能替代。模型factory、原空闲资源／kernel门槛和实际加载driver仍须单独完整准备／审查，本文件不构造／加载模型或执行资源探测，更不因为已有CUDA对象就宣布driver通过。

参数外的结构／行为配置也须等于原烟测：QLA原Full默认配置及30／4／19／705、默认feedback-age norm0.5，PPO原19／4及log-std限制-1.5／1；未用optimizer元数据与状态前后核对，不能只凭参数SHA忽略bias／clamp／配置变化。失败保存实际队列逐请求records及已闭合结果（若已有），仅作为执行诊断，不回流当前actor，也不造替代任务。

实际环境须是新原calibration或validation catalog／真30用户10秒移动／完整注册请求文件，role／seed／每条请求／全部N步／环境条件／校准来源共同尺度一致，覆盖generation+固定drain和原支持，不剪请求、不延长drain、不按完成数早停。旧FIFO、预测陈旧反馈、19维共同actor／705维动作前critic、真实JPEG／秒和过去收据链路不改；实测未来服务只在执行层。

每步用真实model的deterministic动作／value，保留完整行为、普通收据、append-only ACK、终末ledger、每用户物理GAE及实际RNG见证；关闭之后才检查固定结束的generated／accepted／complete／drop／unfinished、uncapped latencies与service秒守恒。参数、empty unused optimizer状态、Torch RNG、全部输入前后不变，绝不调用update／optimizer step。PPO MV为每条工程episode新diagnostic statistic（cal用原training-purpose算术接口但**无训练**，val用engineering-purpose）；H同样从0，训练carry/reset规则未被这个首阶段测试，不恢复烟测H／MV／优化器／RNG。

输出新短独立目录和永久科学身份锁。输出名／条件标签／日程文件位置／JSON编码／审查表述不解锁同一物理条件／参数；任意部分失败保留输入、观察、回执、行为／MV及失败锁，不自动追加／合并／重跑。完整marker仅 `actual_closed_no_optimizer_pilot_episode_pending_independent_check`，不是完整候选覆盖、pilot完成／最终冻结、独立审计或正式训练／120 cells。参数只在functional-smoke做过少量更新，本收集的指标不能当策略学习性能或论文结果。

下步仍需同预算工程driver／实际正确加载、所有共同条件与两方法live收集器的证据登记、独立held-out FIFO／mass／无泄漏审查及完整pilot确认，再到冻结／metadata许可／日程生成／正式完整入口、十项学习训练和120项固定测试。负面、零完成、drop与unfinished全保留，不以test结果或QLA优势选条件。

`check_visual_pilot_episode.py` 仅检查明确人工描述的科学身份、坏字段、静态真实接线与缺准备对象早拒绝，不伪造真实smoke／用户回复／PASS／完成权重／模型；不运行constructor成功路径／collect／catalog／queue／Torch。实际运行守卫和序列化未测试，作者检查不代替独立Type-A。
