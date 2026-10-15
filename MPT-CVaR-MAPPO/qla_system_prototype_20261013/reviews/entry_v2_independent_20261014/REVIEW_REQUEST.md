你是一个全新上下文、未参与该系统实现的 Type-A 独立完整入口 reviewer。用户授权对正式 E1 前置入口做真正独立复审。GPT 同家族审查只能称 Type-A，不能称 Type-B 或跨家族认证。主执行者曾参与实现，所以你的判断必须直接来自文件，不能依赖执行者总结。请审查下列路径，默认所有作者主张都需要验证，不带结论预期。不要修改源码、科学设置、旧计划/结果或状态；只在指定新review目录写完整报告、machine JSON、有限功能验证输出及审查trace。禁止启动 E1 正式矩阵、训练、重下载数据、杀未知进程、全局安装/重置。若需要有限功能驱动，使用现有功能配置在新路径做一次真实运行并正常关闭本次已核PID；预先查看当前端口/源身份，若占用保留阻断，不能关闭他人服务。不得自动追加/合并部分失败重试。若需新的authority或来源不完整，报告具体阻断。

根目录：X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013
请直接完整读取：
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/README.md
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/docs/AGENT_HANDOFF_中文.md
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/docs/EXPERIMENT_PLAN_中文.md
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/docs/DATA_PLAN_中文.md
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/docs/EXPERIMENT_TRACKER_中文.md
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/systemlab/ 全部 .py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/tools/ 全部 .py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/tests/ 全部 .py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/drive_scripts/run_interference_calibration.py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/drive_scripts/run_m2c_validation.py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/drive_scripts/run_m2c_freeze.py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/drive_scripts/run_e1_prep.py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/drive_scripts/run_e1_grid_v2.py
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/configs/ 全部 JSON
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/frozen_v2/{FROZEN_SYSTEM_PROTOCOL,FORMAL_CONFIG,FEATURE_SCALES}.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/formal_actor_v2.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/E1_grid_v2.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/E1_test_registry_v2.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/validation_registration_v1.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/protocol/calibration_scales_v1.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/runs/M1_rr_s1_attempt_01/ 全部实际输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/runs/M1_stress_s1_attempt_01/ 全部实际输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/runs/M2c_qla_s101_attempt_01/ 全部实际输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/runs/M2c_score_softmax_s101_attempt_01/ 全部实际输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/runs/M2c_stale_shortest_s101_attempt_01/ 全部实际输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/interference_calibration/PLAN.json, REPORT_中文.md, 所有 run_*/edge*.json
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/interference_calibration_affinity_attempt_01_broken/ 全部计划/错误及部分输出
- X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/engineering_repairs/20261013_three_logic_fixes/{REPAIR_REPORT_中文.md,RESULT.json,UNIT_RESULT_attempt_02.json,functional_4edge_attempt_02/}
- 从上述绑定直接定位并核对 data/calibration_images.json、data/validation_images.json、data/test_images.json、全部对应schedule/源图片/实际权重/checkpoint/历史可执行源/隔离环境依赖；不能只检查MD声称的SHA。

审查清单：
A ground-truth/proxy来源及service真实执行，不以模型输出制造reference。
B 自归一化/截断/失败样本选择、真实metric定义、CVaR分数质量及配对run统计。
C 所有主张对应实际文件/字段/数值，源码/计划/角色/原始JPEG bytes、checkpoint及helper闭包启动前/结束连续性，重算M2c结果。
D dead code、三修复回归覆盖、量测边界与异步完成后cutoff结果、每任务唯一次结局、FIFO等待/active和守恒并发。
E 实际scope vs声称，零样本35→4迁移映射/归一化/softmax/域迁移；fair actor/baseline inputs/current cache而非fresh私有queue，common schedule/uniform，同resource/drain/初态，未用test挑设置。
F 真实本地HTTP/UDP、应用层delay/loss注入 vs真实无线网络范围；budget per recipient/监测开销/驱动干扰/负载发生器backpressure。
G Windows wrapper/runtime PID、句柄错误fail-closed、自有进程shutdown，不杀未知进程；单机共享物理资源、线程≠quota、affinity≠完全隔离。
H 干扰标定严格配对性：相同图像顺序/预热/线程、并发启动屏障、时间重叠、样本CPU与墙钟定义、三重复/顺序登记、拓扑获取和SMT归属、成功与失败尝试隔离、是否真正测CPU时间/CPU墙钟比；因果归因与缓存/睿频/保守估计主张是否得到直接证据，不从相关性替代测量。
I frozen_v2 当前身份与实际M2c、v2 registry/grid/scales/全部日程/actor/generator binding、具体cell准入、review校验/缺项拒绝、启动前/末尾fresh检查。检查主机共驻方案能支持怎样的有限主张；若需要额外独立资源对照才允许更强论文主张，分别记录scope warnings或实际入口阻断，不能捏造全面禁止或自动放行。

输出到新目录：X:/Yi-one/workspace/MPT-CVaR-MAPPO/qla_system_prototype_20261013/reviews/entry_v2_independent_20261014/
1 REVIEW_RESPONSE.md：完整中文报告，各项PASS/WARN/FAIL、确切file:line/数值/证据、blockers、warnings、允许/不允许主张、有限复现的完整command/路径、来源读一致性限制，不声称Type-B。
2 REVIEW_RESULT.json：入口可读schema systemlab-independent-entry-v1, scope complete_system_entry, status PASS/WARN/FAIL, blockers数组真实完整, complete_report, reviewer_identity='gpt-6-astra xhigh / 独立上下文 Type-A / 本agent实际ID', protocol_sha256当前actual, test_registry_sha256当前actual, source_sha256完整实际字典, report_path绝对, report_sha256完整报告字节；补audit_checks、warnings、independence_type='Type-A', independent_context=true, cross_family=false, verified_execution_receipts。若FAIL或未完成同样保存真实状态，禁止填空blockers冒充审批。
3 EXPERIMENT_AUDIT.md/JSON（可由同一完整报告派生但不能掩盖不同scope）；保存本review完整prompt及response用于 .aris forensic tracing。报告文件使用apply_patch；历史来源只读。不要新增Python到systemlab/tools或改source_receipt，因为那会改变待审身份。有限验证用现有脚本输出到review目录。你不需要寻论文；只审这个入口。

结束给主agent完整结论与所有产物路径、SHA、执行receipt、未解决blockers，不只提供摘要。
