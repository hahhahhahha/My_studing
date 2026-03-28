# My_studing

这个仓库用于复现 P2TS 任务调度核心逻辑，并补齐 RL 训练链路（Environment + Reward + P2TD3）。

## 文件用途

- `p2ts.py`  
  论文中 P2M/P2TS 的优先级与抢占判定核心逻辑（如 `execution_priority`、`preemptive_evaluation`）。

- `rl_env.py`  
  Gym 风格 `P2MEnvironment`：  
  - 状态包含每个节点 AQ/CQ/PZ/SZ  
  - 状态包含带宽矩阵和链路传输队列（links）  
  - 动作可按任务批量映射到目标节点  
  - Step 内包含跨节点传输时延建模（链路队列 + 带宽消耗）与节点抢占执行
  - Reward 返回 `R(t)=mu1*E - mu2*D - mu3*W`
  - 负载方差 `W` 按“剩余计算量工作负载”计算：`CQ` 计算量总和 + `PZ` 剩余计算量 + `SZ` 剩余计算量（而非任务数量）

- `p2td3.py`  
  PyTorch 版本 P2TD3：Actor/Target、双 Critic/Target、ReplayBuffer、目标平滑噪声与延迟策略更新。

- `testP2TS.py`  
  P2M 基础逻辑单元测试。

- `test_rl_env.py`  
  Environment 测试：状态、奖励、跨节点传输时延、动作批量映射策略。

- `test_p2td3.py`  
  P2TD3 基础测试：采样与训练步运行。

## 运行顺序（建议）

### 1) 安装依赖

在仓库根目录执行：

```bash
cd <repo_root>
pip install numpy torch
```

### 2) 先验证 P2M 核心逻辑

```bash
cd <repo_root>
python -m unittest testP2TS -v
```

你会看到每个测试用例的 `... ok`，表示基础优先级与抢占逻辑正确。

### 3) 再验证 RL Environment

```bash
cd <repo_root>
python -m unittest test_rl_env -v
```

重点可看到以下能力被测试通过：
- reward 三要素 `E/D/W` 计算
- 跨节点传输任务会进入链路队列并受带宽影响分时传输
- 动作条目少于任务数时，批次映射会自动复用动作（broadcast/repeat）

### 4) 验证 P2TD3 Agent

```bash
cd <repo_root>
python -m unittest test_p2td3 -v
```

你会看到：
- ReplayBuffer 能正确 add/sample（shape 正确）
- `train_step` 能产出 `critic1_loss/critic2_loss/actor_loss`

### 5) 最后跑全量测试

```bash
cd <repo_root>
python -m unittest -v
```

## 运行后输出及含义

### 单元测试输出

典型输出形式：

```text
test_xxx (...) ... ok
...
Ran N tests in X.XXXs
OK
```

含义：
- `ok`：该测试通过
- `Ran N tests`：总测试数量
- `OK`：全部通过（无失败/错误）

### Environment 的 `step(...)` 返回解释

`next_state, reward, done, info = env.step(action)` 中：

- `next_state`  
  - `nodes[i]["AQ/CQ/PZ/SZ"]`：节点队列与执行区状态  
  - `links`：链路传输队列状态（源节点、目标节点、带宽、在途任务ID、剩余传输量）
- `reward`  
  - 由 `mu1*E - mu2*D - mu3*W` 计算
- `done`  
  - 是否到达 episode 结束（无新任务且系统/链路均清空）
- `info`  
  - `E`: 成功率（完成率）  
  - `D`: 完成任务平均时延  
  - `W`: 节点负载方差（基于剩余计算工作负载，而不是任务个数）  
  - `completed_task_ids`: 本步完成任务  
  - `dropped_task_ids`: 本步丢弃任务  
  - `transferred_task_ids`: 本步完成链路传输并进入目标节点的任务

## 论文对齐说明（重要假设）

### 1) 关于网络拓扑与路由

当前环境为了仿真与训练效率，使用的是**端到端可用带宽矩阵**抽象（`bandwidth_matrix[src][dst]` + per-link queue），而不是论文中的“路由器+多跳+Floyd 动态最短路”严格复现。  
这意味着：本实现是高效近似模型，适用于 RL 训练迭代；若需要严格网络层复现，需要引入图拓扑与路径算法（多跳链路带宽联合约束）。

### 2) 关于时间槽（slot）与物理时间

环境内部每步按 `1.0 slot` 扣减（例如 `remaining_compute_delay -= 1.0`、`left_delay -= 1.0`）。  
因此，输入任务参数 `compute_delay` 与 `tolerable_delay` 需要按 **slot 单位**给出（例如先把秒数按 `ΔT` 换算为 slot 数）。  
如果直接传入秒值而不换算，仿真时间尺度会与论文物理时间不一致。
