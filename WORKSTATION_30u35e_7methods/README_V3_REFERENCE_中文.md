# WORKSTATION1：30用户 / 35逻辑服务器，六 baseline 推理接入 v3

> 下方历史 v1 说明保留供溯源；v2 操作以本节和 ADAPTATION_PROTOCOL_中文.md 为准。

v3 仅修复独立 v2 报告 B1：服务器与 JSEQ 共用 `SERVER_RESPONSE_TIMEOUT` 状态常量，并新增跨层契约回归。v2 FAIL、旧源码与失败尝试均保留。其他已披露域适配/警告没有静默更改。

## v3 操作：仅准备与冻结模型功能检查

六方法 CLI：ppo_mv、ppo_dsr、mfc_di、qeco、jseq、lyfoto_lc；加 qla 共七方法。
前四种严格加载原 checkpoint；后两种为原规则方法，不造假 checkpoint。
新代码必须放在 qla_30u35e_v3；不要覆盖 v1/v2 或旧结果。
每项输出必须为新路径。报错后停止，保留原尝试，不自动重试。

```powershell
Set-Location "$HOME\qla_30u35e_v3"
$py = "$HOME\qla_env_3135\Scripts\python.exe"
& $py tools/verify_unit.py --output UNIT_RECEIPT.json
if ($LASTEXITCODE -ne 0) { throw '单元检查失败，停止' }
& $py prepare_protocol.py --assets "$HOME\qla_assets_20261015" --output prepared_v3 --check-ports
if ($LASTEXITCODE -ne 0) { throw '准备失败，停止' }
& $py tools/register_baselines.py --config prepared_v3/CANDIDATE_CONFIG.json --assets "$HOME\qla_baseline_assets_30u35e_v2" --output baseline_bundles_v3
if ($LASTEXITCODE -ne 0) { throw '资产/源绑定失败，停止' }
& $py tools/check_baseline_inference.py --config prepared_v3/CANDIDATE_CONFIG.json --bundles baseline_bundles_v3 --output baseline_inference_v3
if ($LASTEXITCODE -ne 0) { throw '推理失败，保留目录并停止' }
```

此处实际模型推理使用人工缓存，只验证加载/维度/冻结，不启动35节点、图片forward、HTTP或正式实验。
本地既有25项回归测试加7项适配回归检查。候选 service=.031秒、transfer=.001秒仍是占位。
源绑定包含 systemlab 与 tools 的全部 .py，资产绑定完整122文件成员/字节；模型结束时再次校验。

## v2 后续真实执行顺序

1. 完成新版本独立源审查；资产通过与模型功能通过不能替代。
2. 30u35e目标资源 calibration：预先登记真实JPEG、到达率、时域、完整共同日程及30秒固定drain；不再回到16用户/4节点测试。测节点服务能力与真实反馈年龄/字节率，不用旧4节点容量外推。
3. 根据真实 calibration 生成新 config/scales，重新绑定七方法；占位 bundle 不作为正式特征冻结。
4. 同一完整 validation 日程运行七方法，保留拒绝、超时、unfinished与反压，不仅统计完成任务。
5. 冻结设置，登记新的七方法test网格，取得新鲜完整独立入口复审后才允许正式实验。

新六方法 test 角色明确拒绝；旧三方法网格/review 不自动适用。不开监控、不重训、不杀其他进程、不改环境全局设置。

---

# v1 历史说明（不作为 v2 启动指令）

用户已选择复用原30用户/35服务器 checkpoint，并同意总反馈预算2MiB/s、突发额度1MiB、200ms名义周期。这里的服务器是同一物理机的逻辑进程，不是35台独立物理节点。

## 本版本实际修改

- 从已同步源码逐字节派生新目录；旧仓库、四节点记录及原 checkpoint 不改动。
- 仅修复校准 `round_robin`：按全局决策批次轮转，而非 `user_id % edge_count`。否则30用户最多覆盖30台服务器，35节点校准必然遗漏5台。每批内请求仍共享目的节点，未改六个比较基线算法。
- 新配置生成器登记30用户、35节点、统一CPU1线程/节点作为同构初始候选，不做亲和性或硬配额。0.031秒初始service和0.001秒传输值仅为待标定占位，不能当作测量值。
- 反馈总预算2097152 bytes/s、burst1048576 bytes，名义周期0.2秒。完整轮次仍按实际30名接收者计费；不得把名义周期写成实测周期。

## 运行准备（WORKSTATION1）

先把整个目录复制到 `C:\Users\Administrator\qla_30u35e_v1`。然后使用已有独立环境：

```powershell
Set-Location "$HOME\qla_30u35e_v1"
& "$HOME\qla_env_3135\Scripts\python.exe" -m unittest discover -s tests -v
& "$HOME\qla_env_3135\Scripts\python.exe" prepare_protocol.py --assets "$HOME\qla_assets_20261015" --output "$HOME\qla_30u35e_v1\prepared_v1" --check-ports
```

生成器核验已有ResNet权重、广播序列化夹具和可选空闲端口；只生成新的候选配置和带哈希登记，不运行模型、服务、训练、validation或正式E1。输出必须为新目录，失败完整保留，不自动重试。

## 仍未完成，不能冒充部署成功

原运行入口目前只接入QLA与原规则路由；PPO-MV、PPO-DSR、MFC、QECO、JSEQ和LyFOTO的真实系统适配尚未全部接入。30/35的形状匹配只消除了已发现的维度冲突，不验证真实特征域迁移。不能在CLI中改方法名字冒充基线。

此目录故意不复制旧冻结协议或旧test网格。补齐适配、源/权重绑定、目标资源校准、完整validation和独立入口复审之前，不运行正式矩阵。旧35用户/4节点容量不能乘比例当成35节点容量。需要新模型权重时按原冻结SHA单独传输，禁止随机初始化、裁剪输出或复制QECO用户模型替代。
