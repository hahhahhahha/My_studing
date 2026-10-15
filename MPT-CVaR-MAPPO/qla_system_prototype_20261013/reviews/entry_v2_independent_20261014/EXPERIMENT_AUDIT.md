# 实验完整性审计：FAIL（不构成入口批准）

日期2026-10-14。独立上下文 Type-A reviewer：`/root/system_entry_v2_independent_review_20261014`，gpt-6-astra xhigh。不是Type-B或跨家族认证。

审查范围是已存在的M1校准/压力pilot、M2c三方法validation、干扰标定与失败尝试、三修复功能证据以及正式E1前置完整性。没有运行E1或训练，没有更改历史记录。完整细节、文件行号、限定条件与复现命令见同目录[REVIEW_RESPONSE.md](X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/reviews/entry_v2_independent_20261014/REVIEW_RESPONSE.md)。本文件不代表另一次独立审查，也不将“原始统计可重算”替换成“正式入口已批准”。

## 已核实

- 333个现有文件完整读取/解析并形成SHA清单；3500张图片按原角色、原始JPEG字节及长度核验，三角色无内容SHA重叠。
- 实际vision权重和actor checkpoint哈希匹配；117个历史Python文件全树匹配，实际helper import闭包位于注册目录。
- M1 rr 599/599完成；M1 stress 15752完成、3164拒绝；M2c qla与score各12565完成，stale12374完成、191拒绝。全部原始任务数、FIFO顺序、单worker不重叠、快照守恒、响应对应执行日志检查通过。
- M2c CVaR99独立重算为qla0.4864660398621918、score0.8404820931517881、stale1.4969345565801409秒，原analysis吻合。全部历史反馈来源、基线特征和共同uniform分配一致；10759次QLA决策批量重算最大误差5.96e-8。
- 当前18项回归测试独立全通过；唯一新真实4节点/QLA功能运行4/4完成，每节点1次实际推理，5个自有服务正常退出、无强杀、无遗留。详细停止收据为FUNCTIONAL_VERIFICATION.json和functional_4edge_attempt_01/PROCESS_END.json。
- 当前v2 protocol/source/grid/registry/actor/schedules字节绑定匹配；18个真实cell缺review均在只读门禁调用中被拒绝，没有启动test。

## 未解决阻断

1. **B1 截止窗双结局**：原client.replay在截止前响应计时、但outcome_lock之前被暂停时，可先写unfinished后写completed；合成受控交错复现为一任务两条日志，COMPLETE仍true，analyze拒绝。证据synthetic_cutoff_race_attempt_02/RESULT.json。需要保持固定cutoff语义的原子单次结局、受控交错/晚到/早期异常回归；修复源码后重新validation/freeze/registry/独立复审。
2. **B2 历史helper来源缺口**：M1收据绑定common.py SHA `6cedd7b3590da61d1a3e3ee12f20f6b5b0154954d5a0e8a5b54cc6418183a0d6`，当前与BEFORE均不匹配，未在已搜索范围定位原文件。其他M1源码与移动驱动已按SHA定位；不是把全部旧源笼统认定缺失。需要提供匹配SHA的原字节，或另获明确授权登记替代来源路径，不能改旧收据。

这些阻断不证明历史数据造假。已核任务没有重复结局，可原样保留为对应旧源码观测，不能重新标成修复后验证。

## 不应越界的主张

latency是completed条件分布，必须连同拒绝/超时/unfinished报告；M2c只有一个配对seed，不是正式多重复优势证据。低负载C_ref是名义归一化而非硬容量。35→4结构加载不是零样本性能保证，当前没有新CVaR训练优化。

共驻和亲和的原始compute分布可以描述；无ready barrier、缺历史绝对测量时间与实际affinity回读、PLAN后写，不能保证全程严格配对并发或事前不可变登记。CPU时间是整个200图循环的进程总量，不是逐图CPU时间；CPU/墙钟比相近不推出CPU总时间不变。没有证据把残差归因缓存/睿频，或称相对方法收益为保守估计。

单机真实HTTP/UDP不是无线/多物理节点验证；torch线程数与cpuset/affinity都不提供全面资源隔离。环境使用system-site-packages，并非完整封闭依赖镜像。原始字节前后相同也不是持续不可变性证明。

## 审计状态

整体FAIL；不批准E1。scope warnings不等于全面禁止共驻系统研究，但B1/B2未解决前不得以本审计启动正式矩阵。所有已有失败与诊断失败保留，不拼接、不清空blockers。无活动的本审查实验/服务。
