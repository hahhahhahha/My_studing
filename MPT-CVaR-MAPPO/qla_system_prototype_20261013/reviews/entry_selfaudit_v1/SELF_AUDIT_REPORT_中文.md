# 系统入口复审 —— 执行者自查报告（非独立认证）

- **范围**：complete_system_entry（完整系统入口）
- **复审者身份**：执行者自查（**非独立认证**）。本报告由实施者本人编写，不构成协议要求的"独立复审者"身份；不可直接作为 `--entry-review` 的独立批准依据。
- **日期**：2026-10-13
- **冻结协议**：`protocol/frozen_v1/FROZEN_SYSTEM_PROTOCOL.json`
  - `protocol_sha256 = ef5b869e0b43a5a3f1f0766529d2d560fbd8fa12652b69c0d54d61f8141f2f69`
- **源码收据范围**：`systemlab/`（11 文件）+ `tools/`（6 文件）= 17 文件；临时驱动脚本在 `drive_scripts/`，不计入收据。

## 结论

自查通过 9 项检查清单的**功能性核对**，未发现代码层面的伪造/篡改路径；但存在一个**阻断项**：本报告缺少独立复审者身份，因此不能进入正式 E1 执行门禁。

---

## 逐项核对

### 1. 输入角色/成员/JPEG 字节/模型/checkpoint/完整可执行源绑定，启动前后连续性
- **结论：通过（自查）**
- 图片清单 `import_images` 按 `image_id` + 内容哈希排序，逐张校验 `image_sha256`、`payload_bytes`、宽高，且路径必须落在 `image_root` 内（`data.py:13-33`）。
- 角色绑定：`schedule`/`import_images` 强制 `role` 与 `manifest.role` 一致（`data.py:47-48`）；`client.replay` 校验 `schedule.role == manifest.role == role` 且 `users` 一致（`client.py:21-23`）。
- 权重 `weight_sha256` 在 `VisionEngine` 加载前强校验（`edge.py:22-23`）；实际权重文件 SHA 与配置一致：`f37072fd...07ec`。
- checkpoint 与源码绑定在 `register_actor`/`Router` 中双重校验（`policy.py:89-93, register_actor:35-43`）。
- 运行前后源码连续性：`client.replay` 在结尾比对 `initial_sources == source_receipt()` 与 schedule/manifest SHA（`client.py:204-211`），并写入 `source_continuity_pass`。三份 M2b 均为 `True`。

### 2. 真实 HTTP→FIFO→CPU forward→response；单 worker、waiting32 与 in_service 区别、重复/拒绝/超时后完成
- **结论：通过（自查）**
- 真实链路：`edge.py` 的 `ThreadingHTTPServer` 接收 `/infer` → `EdgeQueue.submit`（FIFO `queue.Queue(capacity)`，capacity 仅计等待）→ 单 worker 线程 `_work` → `VisionEngine`（真实 ResNet-18 CPU forward）→ 回写 `task.result` → `/infer` 阻塞等待 `task.done` 后返回。
- `waiting_capacity=32` 为等待队列上限；`in_service` 由 `active` 字段单独追踪（`edge.py:144-154`），质量守恒断言 `admitted == completed + failed + waiting + active`（`edge.py:147-148`）。
- 重复 → `duplicate`(409)；满 → `queue_rejected`(429)；超时 → `server_response_timeout`(504) 但后台仍完成、以服务器事件为准（`edge.py:214-216`）。

### 3. cache 只接纳已到达摘要、每节点 source 时间、乱序更新、陈旧特征；无实时私有 queue 进入 actor
- **结论：通过（自查，问题设定边界已复核）**
- `FeedbackCache.accept` 仅接受 UDP 实际送达的包，按 `source_epoch_ns` 单调更新（乱序丢弃旧的，`feedback.py:84-88`），含真实丢包注入。
- 在线 Actor `Router.portfolio` 只读 `cache.observation(user)`（`client.py:161-164` → `policy.py:148-169`）。**真实服务器队列快照不进入 actor**；控制器读 `/snapshot` 仅用于调度/广播/实验记录。满足用户要求"真实队列不能直接传给在线 Actor"。

### 4. 压力项只用已发生提交、不读 future schedule；同预算按实际 receiver 数计费；监测开销另披露
- **结论：通过（自查）**
- 客户端按 `decision_period_s` 分桶，仅在到达时处理该 tick 内已到期任务；决策仅基于当前 cache 状态（`client.py:148-184`）。
- 反馈预算 `ByteBudget` 按实际序列化字节计费，且按 `users` 每位接收者分别生成 packet 计费（`feedback.py:131-137`）；监测开销计入 `polling_response_bytes` 单独披露（`feedback.py:122`）。
- `analyze` 另计 `udp_ipv4_header_bytes_estimated` 与 `feedback_application_bytes_per_s`，应用字节与头部开销分列。

### 5. 真实时钟、planned-release E2E、固定 cutoff、不截断/不补 drop 时延、状态唯一
- **结论：通过（自查）**
- 任务按 `planned_release_s` 释放，E2E 定义为 `planned_release → client_response`（`analysis.py:53`）。
- `cutoff = begin + duration_s + fixed_drain_s` 固定登记（`client.py:111-112`）；迟到的响应进 `late_responses.jsonl`、绝不回填 cutoff 指标（`client.py:141-145`，`COMPLETE.json` 说明 "late data never improves the cutoff metrics"）。
- `analyze` 强制每 task 恰好一个 cutoff 结局（`analysis.py:22-23`）；`caps_or_drop_latency_imputation = false`，不截断也不补齐丢弃时延。

### 6. actor 结构迁移与域迁移、fair baseline、参数不变、common schedule/uniform、资源一致
- **结论：通过（自查）**
- 迁移为 `registered_zero_shot_edge_count_and_system_feature_transfer`（35→4），`register_actor` 显式登记并要求 edgewise 结构、`actor_obs_dim = 3*source_edges+7`（`policy.py:26-32`）；域迁移（真实系统特征）在 `transfer` 字段披露。
- 参数不变：`verify_frozen()` 前后 `parameter_sha256` 比对 + checkpoint/source SHA 比对（`policy.py:137-146`）；M2b QLA 运行 `parameters_unchanged=True`。
- 公平 baseline：`score_softmax`/`stale_shortest`/`round_robin` 共享同一 `features()` 与 `comparison_contract`；`frozen_noql` 强制真正匹配训练的 NoQL actor（`policy.py:114-117`），不允许裁剪旧 flat actor 冒充。
- 共同日程/资源：三方法用同一 `schedule_sha256`、同一 `assignment_uniform` 抽样（`sampled_index`），`analyze`/`paired_compare` 强制 `comparison_contract` 与 `feedback_budget_contract` 一致。

### 7. cal-only 尺度、validation 冻结、test 未用于选择；CPU 是显式新部署，非原训练 fallback
- **结论：通过（自查）**
- 尺度仅由 calibration 拟合：`calibrate` 强制 `role=='calibration'`；`features()` 强制 `fitted_role=='calibration'`（`policy.py:51-52`）。
- test 角色在 `replay` 中强制 `feature_scales.status == 'frozen_after_calibration_validation'`（`client.py:24-25`）。
- `freeze` 由 cal+validation 证据生成冻结协议，未读 test（`analysis.py:157-200`）。test 图片清单与日程在冻结之后、本轮才生成（`data/test_images.json` + 6 个共同日程），未参与任何尺度选择。
- 部署为显式 CPU（`VisionEngine` `.cpu()`、`torch.set_num_threads`），原 CUDA 训练准入不被绕过。

### 8. 源码/依赖/实际资源/端口/PID/退出；只正常退出本次已核 PID，不杀未知进程
- **结论：通过（自查）**
- `preflight` 记录 python/torch/torchvision/Pillow 版本与 CPU 数；`RUNTIME.json` 记录 `executable`、`logical_cpus`、`pid`、模型 receipt。
- 端口在 launcher 启动前 `sock.bind` 探测占用（`launcher.py:26-32`）。
- **本轮修复**：关闭/存活判定一律以 RUNTIME.json 的 runtime PID 为准（Windows 用 `OpenProcess+WaitForSingleObject`），每步 `/health`→`/shutdown` 写入 `PROCESS_END.json`；从不 `kill` 未知进程，`forced_termination=false`。M1/M2b/E1-prep 各次运行 `known_children_still_alive=[]`。

### 9. 统计单位为配对 run、CVaR 样本不足披露、结果不可由 fixture 代替
- **结论：通过（自查）**
- `paired_compare`/`interaction` 以完整 run 对为统计单位，做 paired repetition bootstrap，并明确 `no_task_level_iid_CI=True`（`analysis.py:108-113`、`interaction.py:39-45`）。
- `tail_mean` 尾部样本数记录为 `cvar_tail_effective_tasks`（`analysis.py:56`）；`quantile` 空样本返回 `None` 而非 0。
- fixture 输入明确标注 `input_is_synthetic_fixture=true`、`performance_evidence=false`（`check_actor_transfer.py:35-38`、`verify_unit.py`）。

---

## 阻断项

| 项 | 说明 |
|---|---|
| 独立复审者身份缺失 | 本报告为执行者自查，`reviewer_identity` 为"执行者自查（非独立认证）"。协议要求独立复审者另读源码/计划/真实 M1/M2 记录并自行做有限功能检查。**本报告不得作为 `--entry-review` 直接放行 E1。** |

## 附：关键 SHA

- 冻结协议 `FROZEN_SYSTEM_PROTOCOL.json`：`ef5b869e...f2f69`
- `FORMAL_CONFIG.json`：`9e77871b...59886f`
- `E1_grid.json`：`efd0ff9e...7833696`
- 权重文件：`f37072fd...607ec`
- checkpoint：`3444f039...13044`（见 `protocol/formal_actor_v1.json`）
