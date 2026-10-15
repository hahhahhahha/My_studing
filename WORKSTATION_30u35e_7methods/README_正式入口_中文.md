# WORKSTATION_30u35e_7methods additive formal entry

此目录是从 `qla_30u35e_v3` 派生的 additive 版本。原 v3 源码、pilot、标定和
baseline bundle 不在本目录覆盖。当前交付只完成入口、协议、registry、grid 和
fixture；`formal_entry_approved` 仍为 `false`，没有启动任何正式 test cell。

## 已绑定设计

- `qla`、`ppo_mv`、`ppo_dsr`、`mfc_di`、`qeco`、`jseq`、`lyfoto_lc`
- 共同工作负载 seed：1--5，共 35 cells
- 30 用户、35 逻辑节点、62 请求/秒、120 秒到达、30 秒 drain
- 反馈预算 2 MiB/s、burst 1 MiB、名义周期 0.2 秒
- test 图像来自原 split 的 test 成员，manifest 绑定 JPEG 字节 SHA
- 453.44/s 不用于生成正式负载，也不作为饱和容量声明

## 生成顺序（不启动服务）

```powershell
Set-Location 'C:\Users\Administrator\WORKSTATION_30u35e_7methods'
$py = 'C:\Users\Administrator\qla_env_3135\Scripts\python.exe'
& $py prepare_formal_entry.py `
  --pilot-root 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_live_validation_pilot_62rps_20261015_01' `
  --calibrated-config 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_actual_scale_binding_20261015_01\CONFIG_CALIBRATED.json' `
  --pilot-summary 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_live_validation_pilot_62rps_20261015_01\VALIDATION_PILOT_SUMMARY.json' `
  --pilot-plan 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_live_validation_pilot_62rps_20261015_01\PLAN.json' `
  --qla-bundle 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_actual_scale_binding_20261015_01\qla_actor_bundle.json' `
  --baseline-root 'C:\Users\Administrator\qla_30u35e_v3\remote_runs\M2_actual_scale_binding_20261015_01\baseline_bundles' `
  --image-split 'C:\Users\Administrator\qla_experiments_20261015\MPT-CVaR-MAPPO\dac_visual_calibration_20261009\protocol\IMAGE_SPLITS.json' `
  --image-root 'C:\Users\Administrator\qla_assets_20261015\images' `
  --output 'C:\Users\Administrator\WORKSTATION_30u35e_7methods'
```

实际已生成的 `FORMAL_CONFIG.json`、`FROZEN_SYSTEM_PROTOCOL.json`、`E1_GRID.json`、
五份 schedule 和 `E1_TEST_REGISTRY.json` 应在独立复审前保持不变。若重新生成，
必须使用全新的 additive 目录。

## 独立复审后才能运行

复审者须将完整收据写入 `protocol\INDEPENDENT_ENTRY_REVIEW.json`，其中至少包括：
`status` 为 `PASS` 或无阻断 `WARN`、`blockers: []`、`protocol_sha256`、
`test_registry_sha256`、`source_sha256`、报告路径及报告 SHA。然后把 registry 的
`formal_entry_approved` 更新为 `true` 并重新计算 registry SHA；否则
[run_formal_30u35e.ps1](./run_formal_30u35e.ps1) 会在启动前拒绝。

运行器按 registry 顺序逐 cell 新建目录；任何准入、协议、任务结局、进程退出或
分析错误都会停止并保留失败目录，不覆盖、不追加、不自动重试。
