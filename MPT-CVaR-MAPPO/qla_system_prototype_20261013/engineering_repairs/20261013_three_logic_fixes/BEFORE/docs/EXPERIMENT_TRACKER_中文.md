# 系统实验执行清单

更新日期：2026-10-13。

| ID | 工作 | 状态 | 证据/下一动作 |
|---|---|---|---|
| S001 | 新原型代码与6项核心测试 | 作者测试通过 | tests/test_core.py |
| S002 | 真实ResNet/HTTP/FIFO/UDP小请求 | 作者功能验证通过 | verification/real_http_attempt_3：4完成；attempt1/2环境错误保留 |
| S003 | 35→4冻结edge-wise actor加载/前向 | 作者功能验证通过 | verification/actor_transfer_check.json；synthetic观测，权重不变 |
| S004 | 4节点共驻留、16用户功能验证 | 作者功能验证通过 | verification/real_qla_4edge_attempt_2：4任务QLA真实推理、5服务正常shutdown、runtime PID判定修复后复测通过；attempt_1为旧launcher误判保留 |
| S005 | Docker/cgroups资源验证 | TODO | Docker命令未发现；WSL不可用，先native或准备管理员安装 |
| S006 | 并发calibration与负载量级 | 完成（作者校准） | protocol/calibration_scales_v1.json；runs/M1_rr_s1_attempt_01（599/599 低负载）+ runs/M1_stress_s1_attempt_01（90%C_ref=158/s 饱和证据：3164拒绝、max_waiting=32、无客户端背压） |
| S007 | validation完整重放/域迁移边界 | 完成（绑定当前源码） | runs/M2b_{qla,score_softmax,stale_shortest}_s101_attempt_01：三方法全部完成、source_continuity_pass=true、无背压、正常退出。M2（旧源码收据）作为历史尝试保留 |
| S008 | 新系统冻结协议与fresh入口独立复审 | 冻结完成，独立复审待做 | protocol/frozen_v1/{FROZEN_SYSTEM_PROTOCOL,FORMAL_CONFIG,FEATURE_SCALES}.json + protocol/formal_actor_v1.json；冻结只是登记，独立完整入口复审仍未进行 |
| S009 | E1：18格实际执行 | 网格已登记，未启动 | protocol/E1_grid.json（18格、预登记随机序、started=false）；须先取得 --entry-review 才能运行 |
| S010 | Azure2019到达形状复用 | TODO | 显式分钟内合成与count-scale |
| S011 | Azure2021官方获取+schema/单位核对 | OPTIONAL_NEXT | docs/DATA_PLAN_中文.md，当前未下载 |
| S012 | E2：48格 | TODO_AFTER_E1 | UDP注入传播条件，实际age/drop完整 |
| S013 | E4：48格与gamma=0对照 | TODO_SECOND_STAGE | 同预算、交互差分、负结果 |
| S014 | 多物理节点/E3/联合训练 | DEFERRED | 先完成单机可复现主结果 |
| S015 | 结果独立审计和论文系统章节 | TODO | 所有预登记格子实际完成后 |

当前没有长期服务运行，没有新监控自动化。这里的作者功能验证不是独立认证，也没有完成正式性能实验。

2026-10-13 launcher 修复记录：Windows venv 壳进程 PID 与真实服务 PID 不同导致正常关闭误判（attempt_1 实际已全部退出）。修复后以 RUNTIME.json 的 runtime PID 为存活判定基准，每步 shutdown 留痕（systemlab/launcher.py；tests/test_core.py 新增第 9 项测试）。attempt_1 连同失败记录保留不覆盖。
