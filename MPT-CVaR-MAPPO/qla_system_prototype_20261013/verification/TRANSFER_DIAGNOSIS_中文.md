# 35→4 零样本策略迁移诊断报告

日期：2026-10-14。目的：检查当前 4 节点系统所冻结的 QLA actor（历史 35 节点 checkpoint 零样本迁移）是否退化为无意义策略，从而**避免把"迁移失效"误判为"算法在真实任务下失效"**。这是行为诊断，不是性能结论。

## 1. 迁移机制审查

`EdgeWisePriceGaussianActor`（`source_snapshot/mappo_gp_sf_dsr/models.py:182-265`）采用**置换等变**结构：`user_net`（Linear 7→h）、`edge_net`（Linear 5→h）、`score_net`（Linear 2h→2）的**可学习参数都不依赖 edge 数 n_edges**；`n_edges` 只影响 forward 时 obs 的切分与 softmax 分母。

因此 35→4 迁移是合法的**结构迁移**：`register_actor` 用 `EdgeWisePriceGaussianActor(3*4+7=19, 4, hidden, …)` 构造，`load_state_dict(strict=True)` 成功（参数形状完全一致）。复审已独立复核：重放 10759 条决策的批量前向最大 portfolio 误差 5.96e-8，参数前后 SHA 均为 `13daaa5f…`。

但**特征域发生了漂移**：`policy.py features()` 中的 `price` 用跨边 min-max 归一化、`price_scale = log1p(sum/price)/m`、`age` 归一化分母含 m——这些在 35 节点训练域与 4 节点真实系统域之间取值分布不同。这是显式披露的零样本结构+系统特征域迁移，不是重新训练。

## 2. 行为诊断（迁移是否退化）

### 2.1 合成探针（`verification/transfer_behavior_probe_v3.json`）

构造 4 个场景，观测迁移后 actor 的 portfolio：

| 场景 | 权重 | 结论 |
|---|---|---|
| 四边相同（uniform_idle） | [0.272, 0.217, 0.193, 0.318] | 大致均匀（服务/传输微差导致） |
| edge0 高负载（wait 2.0s，余 0.1s） | [**0.176**, 0.275, 0.275, 0.275] | edge0 权重**最低** ✓ |
| edge0 快服务 | [**0.353**, 0.215, 0.214, 0.219] | edge0 权重**最高** ✓ |
| edge0 空闲、余高负载 | [**0.34**, 0.22, 0.22, 0.22] | edge0 权重**最高** ✓ |

检查项：`price_aware_loaded_drops=true`、`idle_dominates=true`、`responsive_to_input=true`、`not_degenerate_uniform=true`。→ **迁移后 actor 保留了 price/queue 感知，未退化为均匀/随机。**

### 2.2 真实 M2d 决策（10759 条）

- portfolio spread（max−min）分布：min 0.1105 / 中位 0.1197 / p90 0.1262 / max 0.1338——**非均匀**（均匀为 0）。
- argmax 分布（前 2000 条）：edge3 1150（57.5%）、edge2 611（30.6%）、edge0 113、edge1 126——**稳定偏好 2 线程的 edge2/edge3**（service_estimate 更低），符合 cost/price 感知。

### 2.3 诊断中的一次假警报（已修正并保留）

首版探针把 `age` 的 `source_epoch_ns` 设成了 `1e10` 尺度，而 `features()` 用真实 `time.time_ns()`（≈1.79e18）作 `now`，导致 age 特征爆炸到 3.58e9，把 latent mean 压到饱和值 −1.94e7，softmax 后呈假均匀。已修正时钟尺度；错误的 `transfer_behavior_probe_v3_buggy_scale.json` 原样保留。**这提醒：迁移诊断必须以与系统一致的时钟尺度构造特征，否则会制造"迁移失效"的假象。**

## 3. 结论与边界

1. **迁移未失效**：35→4 零样本迁移后的 actor 行为语义保留（price 感知、queue 感知、非退化、稳定偏好低成本节点），数值接入严格正确。
2. **但"未退化" ≠ "有效"**：迁移后的策略是"35 节点训练权重 + 系统特征域漂移"的产物，与"若在 4 节点重新训练的 QLA"可能不同。它是否真的在真实系统上优于 baseline，是 E1 实验要回答的，本诊断不替代。
3. **表述边界**：若 E1 中 QLA 表现不佳，正确表述应为"这个 35→4 零样本迁移版本的 QLA 在真实系统下表现为 X"，**不能**据此笼统宣称"QLA 算法失效"。
4. **严格区分迁移失败 vs 算法失效的方法**：需要"在 4 节点重新训练的 QLA"作为对照臂。当前系统没有训练入口（`optimizer_calls=0`，S014 联合训练 DEFERRED），因此**无法在当前范围内严格分离这两者**——这是必须保留的 scope 限制，不应在论文中暗示已分离。

## 数据

- `verification/transfer_behavior_probe_v3.json`（合成探针，含 checks）
- `verification/transfer_behavior_probe_v3_buggy_scale.json`（尺度 bug 版本，保留）
- `drive_scripts/run_transfer_probe.py`（探针脚本）
- `runs/M2d_qla_s101_attempt_01/client/decisions.jsonl`（真实决策 portfolio）
