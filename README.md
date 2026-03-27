# My_studing
task_offloading

## P2TS 代码复现(最小可运行版)

根据仓库中的论文 `P2TS_A_Preemptive_Approach_for_Priority-Aware_Task_Scheduling_in_Computing_Power_Networks.pdf`，
已在 `p2ts.py` 中复现执行层核心机制（P2M）的关键逻辑：

- 多因素动态优先级计算（对应论文公式(7)）
- CQ 按优先级重排
- 抢占评估两种场景（Situation A / B，对应论文算法与公式(9)(10)）
- 评估后任务执行顺序输出

### 运行测试

```bash
# 在仓库根目录执行
python -m unittest -v
```
