# 系统实验执行清单

更新日期：2026-10-14。

最新修复状态：独立入口复审（reviews/entry_v2_independent_20261014）结论 FAIL，两项阻断已解除——(B1) 截止双结局竞态已在 systemlab/client.py 修复（unfinished 写到 pool.shutdown 之后），新增 tests/test_cutoff_race.py 三回归；(B2) M1 收据绑定的旧 common.py（SHA 6cedd7b3…）已精确恢复到 legacy_source/systemlab/common.py（重建 SHA 精确匹配，恢复报告 legacy_source/RECOVERY_REPORT.json）。当前 E1 仍未启动；须由真实独立 reviewer 按新源码（M2d/frozen_v3/E1 v3）重新出完整入口复审（非 self_audit）。

| ID | 工作 | 状态 | 证据/下一动作 |
|---|---|---|---|
| S001 | 新原型代码与6项核心测试 | 作者测试通过 | tests/test_core.py（现全套 21 测试通过） |
| S002 | 真实ResNet/HTTP/FIFO/UDP小请求 | 作者功能验证通过 | verification/real_http_attempt_3：4完成；attempt1/2环境错误保留 |
| S003 | 35→4冻结edge-wise actor加载/前向 | 作者功能验证通过 | verification/actor_transfer_check.json；synthetic观测，权重不变 |
| S004 | 4节点共驻留、16用户功能验证 | 作者功能验证通过 | verification/real_qla_4edge_attempt_2：4任务QLA真实推理、5服务正常shutdown、runtime PID判定修复后复测通过；attempt_1为旧launcher误判保留 |
| S005 | Docker/cgroups资源验证 | TODO | Docker命令未发现；WSL不可用，先native或准备管理员安装 |
| S006 | 并发calibration与负载量级 | 完成（作者校准） | protocol/calibration_scales_v1.json；runs/M1_rr_s1_attempt_01 + runs/M1_stress_s1_attempt_01（来源闭包已由 B2 恢复补齐） |
| S007 | validation完整重放/域迁移边界 | 完成（B1修复后 M2d） | runs/M2d_{qla,score_softmax,stale_shortest}_s101_attempt_01：qla cvar99=0.556、score 0.794、stale 1.319（12317+248拒绝）；全部 infra_valid、source_continuity_pass=true。M2/M2b/M2c 为旧源码保留 |
| S008 | 新系统冻结协议与fresh入口独立复审 | 冻结完成（frozen_v3），独立复审待做 | protocol/frozen_v3/{FROZEN_SYSTEM_PROTOCOL,FORMAL_CONFIG,FEATURE_SCALES}.json + protocol/formal_actor_v3.json；冻结只是登记，独立完整入口复审仍未进行 |
| S009 | E1：18格实际执行 | v3 矩阵/日程已登记，未启动 | protocol/E1_grid_v3.json + protocol/E1_test_registry_v3.json（18格、6共同日程、started=false）；须 --entry-review（含 test_registry_sha256、非 self_audit）才能运行 |
| S016 | 共驻干扰标定（solo vs 4节点并发） | 完成（结论已降级） | interference_calibration/REPORT_中文.md：观测到条件间 compute 墙钟变化（1线程 +27%、2线程 +12~14%），但仅陈述"观测变化"，不作唯一因果/缓存睿频/保守估计归因 |
| S010 | Azure2019到达形状复用 | TODO | 显式分钟内合成与count-scale |
| S011 | Azure2021官方获取+schema/单位核对 | OPTIONAL_NEXT | docs/DATA_PLAN_中文.md，当前未下载 |
| S012 | E2：48格 | TODO_AFTER_E1 | UDP注入传播条件，实际age/drop完整 |
| S013 | E4：48格与gamma=0对照 | TODO_SECOND_STAGE | 同预算、交互差分、负结果 |
| S014 | 多物理节点/E3/联合训练 | DEFERRED | 先完成单机可复现主结果 |
| S015 | 结果独立审计和论文系统章节 | TODO | 所有预登记格子实际完成后 |

当前没有长期服务运行，没有新监控自动化。这里的作者功能验证不是独立认证，也没有完成正式性能实验。

2026-10-13 launcher 修复记录：Windows venv 壳进程 PID 与真实服务 PID 不同导致正常关闭误判（attempt_1 实际已全部退出）。修复后以 RUNTIME.json 的 runtime PID 为存活判定基准，每步 shutdown 留痕（systemlab/launcher.py；tests/test_core.py 新增第 9 项测试）。attempt_1 连同失败记录保留不覆盖。
