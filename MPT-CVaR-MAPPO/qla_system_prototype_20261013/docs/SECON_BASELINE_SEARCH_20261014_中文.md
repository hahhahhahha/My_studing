# SECON 相关基线：检索与实施建议

检索日期：2026-10-14。用户所称 SECOM 按上下文理解为 IEEE SECON，不与 ACM/IEEE SEC 混用。本文件是文献筛选建议，不是实验入口批准，也不改变 frozen_v2、E1 18 格或任何测试注册。

## 结论

存在与边缘卸载方向相关的 SECON 工作，但“同会议发表”“较新”不等于在当前共享陈旧反馈、有限 FIFO、单机 CPU 共驻及任务级 CVaR99 设置下已证明 SOTA。本次未确认一个无需适配即可直接比较的完整实现。

优先研究 LeDRL（2026）的作者实现；若目标是节省时间，先完成当前冻结 E1 的入口修复与有限系统证据，再单独登记外部算法对照。不要为追求 SOTA 标签把 LLM 服务加入当前测量链路，也不要把论文表中数字拿来直接比较。

## 已核实的候选

### 1. LeDRL，SECON 2026：优先候选，但不是即插即用

- 论文：Hao Guo, Kaixiang Xv, Ziwu Ge, Lei Yang，*LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing*。
- [作者全文](https://arxiv.org/html/2605.05727v1)、[作者代码](https://github.com/GalleyG5/LeDRL)。作者仓库明确标注 accepted by SECON 2026；本次 IEEE DOI 页面未成功读取。
- 方法结合 LLM 指导与 DRL；作者原型采用 Jetson、YOLOv8、有线 LAN，LLM 服务使用 Qwen3-4B/RTX 4090。原文主要评价成功率，不能据此推定 CVaR99 或陈旧广播性能。
- 仓库已公开，不等于本机已安装、可完整复现或已通过审计。复用前需检查许可证、固定 commit、依赖与模型获取方式、完整训练/执行路径。
- 公平适配必须约束为相同可见反馈、相同四服务器动作和共同负载，并计入 LLM/路由推理开销。若去掉 LLM 或改变核心结构，应明确称 adapted / ablation，不称完整 LeDRL。
- 公开入口包含 MAPPO、MASAC、VDN、RATC 等注册项（[注册源码](https://github.com/GalleyG5/LeDRL/blob/main/agent_register.py)），可作为低成本代码检查起点，但不是本项目已验证的实现。
- 特别注意：[LLM 配置](https://github.com/GalleyG5/LeDRL/blob/main/ledrl/ledrl_config.py)的 endpoint/model 默认为空；[teacher 源码](https://github.com/GalleyG5/LeDRL/blob/main/ledrl/llm_teacher.py)在调用失败或响应无效时可退回本地执行。必须保存实际请求/成功调用/回退计数，不能把未配置 LLM 的回退运行宣称为完整 LeDRL。兼容 OpenAI 的客户端接口并不意味着必须使用 OpenAI 模型或付费服务。

### 2. SE-MASAC，SECON 2024：多智能体方法候选

- Yueqian Song, Guiyuan Yuan, Qingtian Zeng, Geng Chen，*Resource Allocation and Task Offloading for Slicing-Based Communication and Computing in Space-Air-Ground Integrated Networks*。
- DOI：[10.1109/SECON64284.2024.10934865](https://doi.org/10.1109/SECON64284.2024.10934865)；可读取的[会议论文集摘要](https://www.proceedings.com/content/079/079494webtoc.pdf)。
- 摘要确认 state-encoding multi-agent SAC / CTDE，但联合网络切片与资源分配，问题不完全一致。本次未核实官方公开代码或完整算法全文；不是已完成复现的基线。
- 若固定额外动作并替换观测，须单独披露适配，不能将通用 MASAC 代码冒充此论文实现。

### 3. 较早的相关工作：备用阅读，不作为已验证 SOTA

| 年份 | 论文 | DOI | 当前证据状态 |
|---|---|---|---|
| 2022 | Ce Shang, Yan Sun, Hong Luo：*A Hybrid Deep Reinforcement Learning Approach for Dynamic Task Offloading in NOMA-MEC System* | [10.1109/SECON55815.2022.9918560](https://doi.org/10.1109/SECON55815.2022.9918560) | [会议目录](https://www.proceedings.com/content/065/065987webtoc.pdf)确认条目；本次未读到全文、未核实作者代码。NOMA 设定不能直接搬入回环网络实验。 |
| 2021 | Xin Li, Xinglin Zhang, Tiansheng Huang：*Asynchronous Online Service Placement and Task Offloading for Mobile Edge Computing* | [10.1109/SECON52354.2021.9491595](https://doi.org/10.1109/SECON52354.2021.9491595) | 文献元数据确认；本次未读到完整方法、未核实作者代码。“异步”不应自动理解为本项目的陈旧共享反馈。 |

## 最省时间的比较路线（建议，尚未注册/执行）

1. 保留当前 score-softmax、stale-shortest、冻结 35→4 QLA 的有限 E1。这个阶段不宣称击败外部 SOTA，也不将 validation 的排序作为正式性能结果。
2. 增加 NoQL 作为机制消融、4-edge-trained QLA 作为迁移对照时，使用新的训练/验证/测试版本；两者都不是外部 SOTA。不得追加进当前冻结注册或用测试选模型。
3. 外部学习对照只选一个开始：优先检查 LeDRL 作者实现的可复现性与适配成本。若完整 LLM 路线成本超预算，可研究其公开非 LLM 对照，但准确标为该论文中的对照/适配实现，不能称 LeDRL 本体。
4. 先独立烟测和校准/验证，再冻结新比较协议；只有此前条件满足，才能运行新的配对正式网格。

## 外部对照接入检查表

- 固定源码 commit、依赖、模型 SHA、训练预算、种子、checkpoint 选择规则；所有调参与选择只用训练/校准/验证。
- 相同图像 bytes、到达时间、共同日程、CPU 线程/亲和配置、FIFO 容量、固定截止与 drain；不按方法删除慢任务。
- 显式列出 actor 的每个输入及时间戳。所有方法使用同一个陈旧反馈缓存；额外新鲜队列信息只能作为单独标注的 oracle 对照。
- 相同动作含义；不得给某算法独有迁移、拒绝、资源分配或多跳能力却混作同一问题。
- 报告全部任务分类、拒绝/超时/未完成、完成任务时延及其条件性；CPU 共驻干扰、路由/LLM 开销、反馈开销均披露。
- 在同一评价流水线真实重跑，多次配对重复。论文原有数字只能用于介绍原设定，不能替代本机结果。
- 如果发生部分失败，完整保留该尝试；不得追加、合并或只选有利重复。

本文件不授权下载大模型、调用收费 API、训练新策略或运行新正式网格；这些需要明确选择和单独注册。
