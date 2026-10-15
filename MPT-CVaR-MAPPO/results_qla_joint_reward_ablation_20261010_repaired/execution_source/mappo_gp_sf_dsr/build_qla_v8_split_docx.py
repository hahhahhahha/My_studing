"""Build a split v8 QLA paper (main manuscript + supplement).

The latest audited evidence draft remains the numerical source of truth.  This
builder performs three editorial operations only:

1. convert legacy ASCII/text equations into Pandoc math so Word receives OMML;
2. separate the compact manuscript from detailed audits and supplementary data;
3. add protocol specifications that are explicitly labelled as pending
   implementation, without presenting them as completed experiments.

The source v8 DOCX is never overwritten.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_draft_priceedge_qtbias_cn_v6_literature_audited.md"
REFERENCE = ROOT / "paper_draft_qla_cn_v7_formula_fixed.docx"

MAIN_MD = ROOT / "paper_draft_qla_cn_v8_main_editable.md"
SUPP_MD = ROOT / "paper_draft_qla_cn_v8_supplement_editable.md"
MAIN_DOCX = ROOT / "paper_draft_qla_cn_v8_main_provisional.docx"
SUPP_DOCX = ROOT / "paper_draft_qla_cn_v8_supplement_provisional.docx"


def between(text: str, start: str, end: str | None = None) -> str:
    left = text.index(start)
    if end is None:
        return text[left:]
    right = text.index(end, left)
    return text[left:right]


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


MATH_BLOCKS = {
    "A_j(t) = sum_{i=1}^N a_i(t) w_ij(t).": r"A_j(t)=\sum_{i=1}^{N}a_i(t)w_{ij}(t).",
    "P_i(s_{t+1} | s_t, w_i) != P_i(s_{t+1} | s_t, w_i, W_{-i}).": (
        r"P_i\!\left(s_{t+1}\mid s_t,\mathbf w_i(t)\right)"
        r"\ne P_i\!\left(s_{t+1}\mid s_t,\mathbf w_i(t),\mathbf W_{-i}(t)\right)."
    ),
    "o_i(t) = [x_i(t), r_i(t), d_i(t), B(t-δ), δ].": (
        r"\mathbf o_i(t)=\left[\mathbf x_i(t),\mathbf r_i(t),\mathbf d_i(t),"
        r"\mathbf B(t-\delta),\delta\right]."
    ),
    "G = <N, S, {O_i}, {A_i}, P, R, γ>.": (
        r"\mathcal G=\left\langle\mathcal N,\mathcal S,"
        r"\{\mathcal O_i\},\{\mathcal A_i\},P,R,\gamma\right\rangle."
    ),
    "x(t+1) = x(t) - G x(t-δ),   G = η N.": (
        r"x(t+1)=x(t)-Gx(t-\delta),\qquad G=\eta N."
    ),
    "G_crit(δ) = 2 sin( π / (4δ + 2) ).": (
        r"G_{\mathrm{crit}}(\delta)=2\sin\!\left(\frac{\pi}{4\delta+2}\right)."
    ),
    "B(t-δ) → π_ij^QL → edge-wise shared actor → w_i(t) → A_j(t) → Q_j(t+1).": (
        r"\mathbf B(t-\delta)\longrightarrow\pi_{ij}^{\mathrm{QL}}(t)"
        r"\longrightarrow\text{edge-wise shared Actor}\longrightarrow\mathbf w_i(t)"
        r"\longrightarrow A_j(t)\longrightarrow Q_j(t+1)."
    ),
    "π_ij^Q(t) = ā_i(t) · W_j^{δ,norm}(t),": (
        r"\pi_{ij}^{Q}(t)=\bar a_i(t)W_j^{\delta,\mathrm{norm}}(t)."
    ),
    "π_ij^L(t) = V · τ_ij(t),": r"\pi_{ij}^{L}(t)=V\tau_{ij}(t).",
    "π_ij^QL(t) = π_ij^Q(t) + π_ij^L(t).": (
        r"\pi_{ij}^{\mathrm{QL}}(t)=\pi_{ij}^{Q}(t)+\pi_{ij}^{L}(t)."
    ),
    "stale_margin_ij(t) = κ_δ · (δ / δ_max) · |W_j^{δ,norm}(t) - W_j^{δ,norm}(t-1)|.": (
        r"U_{ij}^{\delta}(t)=\kappa_{\delta}\frac{\delta}{\delta_{\max}}"
        r"\left|W_j^{\delta,\mathrm{norm}}(t)-W_j^{\delta,\mathrm{norm}}(t-1)\right|."
    ),
    "π_ij ← (π_ij - min_k π_ik) / (max_k π_ik - min_k π_ik + ε).": (
        r"\pi_{ij}\leftarrow\frac{\pi_{ij}-\min_k\pi_{ik}}"
        r"{\max_k\pi_{ik}-\min_k\pi_{ik}+\varepsilon}."
    ),
    "s_i^π = log(1 + mean_j [π_ij^raw]^+).": (
        r"s_i^{\pi}=\log\!\left(1+\frac{1}{M}\sum_{j=1}^{M}"
        r"[\pi_{ij}^{\mathrm{raw}}]^+\right)."
    ),
    "h_i^u = f_u(user_i, feedback_age, s_i^π).": (
        r"\mathbf h_i^u=f_u(\mathbf u_i,\delta,s_i^{\pi})."
    ),
    "e_ij(t) = [d_ij(t), B_j(t-δ), π_ij^QL(t), δ, s_i^π(t)].": (
        r"\mathbf e_{ij}(t)=\left[d_{ij}(t),\mathbf B_j(t-\delta),"
        r"\pi_{ij}^{\mathrm{QL}}(t),\delta,s_i^{\pi}(t)\right]."
    ),
    "h_ij^e = f_e(e_ij),   m_ij^θ = f_s(h_i^u, h_ij^e).": (
        r"\mathbf h_{ij}^e=f_e(\mathbf e_{ij}),\qquad "
        r"m_{ij}^{\theta}=f_s(\mathbf h_i^u,\mathbf h_{ij}^e)."
    ),
    "μ_ij(t) = m_ij^θ(t) - λ_π · π_ij^QL(t).\nz_i(t) ~ N(μ_i(t), σ_i^2(t)),   w_i(t) = softmax(z_i(t)).": (
        r"\mu_{ij}(t)=m_{ij}^{\theta}(t)-\lambda_{\pi}\pi_{ij}^{\mathrm{QL}}(t),"
        "\n"
        r"\mathbf z_i(t)\sim\mathcal N(\boldsymbol\mu_i(t),"
        r"\operatorname{diag}(\boldsymbol\sigma_i^2(t))),\qquad"
        r"\mathbf w_i(t)=\operatorname{softmax}(\mathbf z_i(t))."
    ),
    "r_t = -tanh( max{0, Φ_t} ).": (
        r"r_t=-\tanh\!\left(\max\{0,\Phi_t\}\right)."
    ),
    "O(N · M · C_actor + N · M).": r"O(NMC_{\mathrm{actor}}+NM).",
    "D_stale(M) = (M_{δ=8} - M_{δ=0}) / M_{δ=0}.": (
        r"D_{\mathrm{stale}}(Y)=\frac{Y_{\delta=8}-Y_{\delta=0}}{Y_{\delta=0}}."
    ),
}


def convert_text_math(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        formula = MATH_BLOCKS.get(raw)
        if formula is None:
            # Keep genuine non-mathematical text blocks unchanged.
            return match.group(0)
        return "$$\n" + formula + "\n$$"

    text = re.sub(r"```text\n(.*?)\n```", repl, text, flags=re.DOTALL)

    # High-impact inline ASCII formulae in the narrative.
    replacements = {
        "w_i(t) ∈ Δ^(M-1)": r"$\mathbf w_i(t)\in\Delta^{M-1}$",
        "π_ij^QL = π_ij^Q + π_ij^L": (
            r"$\pi_{ij}^{\mathrm{QL}}=\pi_{ij}^{Q}+\pi_{ij}^{L}$"
        ),
        "π_ij → μ_ij → w_ij": (
            r"$\pi_{ij}^{\mathrm{QL}}\rightarrow\mu_{ij}\rightarrow w_{ij}$"
        ),
    }
    for old, new in replacements.items():
        # Do not replace inside existing inline/display math.
        text = re.sub(rf"(?<![$\\]){re.escape(old)}(?![$])", lambda _m, v=new: v, text)

    # Convert common ASCII mathematics embedded in pseudocode table cells.
    inline_patterns = {
        "o_i(t) = {x_i(t), r_i(t), d_i(t), B(t-δ), δ}": (
            r"$\mathbf o_i(t)=\{\mathbf x_i(t),\mathbf r_i(t),\mathbf d_i(t),"
            r"\mathbf B(t-\delta),\delta\}$"
        ),
        "π_ij^QL(t) = π_ij^Q(t) + π_ij^L(t)": (
            r"$\pi_{ij}^{\mathrm{QL}}(t)=\pi_{ij}^{Q}(t)+\pi_{ij}^{L}(t)$"
        ),
        "μ_ij = m_ij^θ - λ_π · π_ij^QL": (
            r"$\mu_{ij}=m_{ij}^{\theta}-\lambda_{\pi}\pi_{ij}^{\mathrm{QL}}$"
        ),
        "z_i(t) ~ N(μ_i, σ_i^2)": (
            r"$\mathbf z_i(t)\sim\mathcal N(\boldsymbol\mu_i,"
            r"\operatorname{diag}(\boldsymbol\sigma_i^2))$"
        ),
        "w_i(t) = softmax(z_i(t))": (
            r"$\mathbf w_i(t)=\operatorname{softmax}(\mathbf z_i(t))$"
        ),
        "r_t = -tanh(max{0, Φ_t})": (
            r"$r_t=-\tanh(\max\{0,\Phi_t\})$"
        ),
    }
    for old, new in inline_patterns.items():
        text = text.replace(old, new)
    text = text.replace("动作空间为概率单纯形 Δ^(M-1)", r"动作空间为概率单纯形 $\Delta^{M-1}$")
    text = text.replace(
        "总导数 d μ_ij / d π_ij 恒为负",
        r"总导数 $\mathrm d\mu_{ij}/\mathrm d\pi_{ij}$ 恒为负",
    )
    text = text.replace(
        "Compute QL risk signal π_ij^QL(t) = π_ij^Q(t) + π_ij^L(t) for all edge servers j.",
        r"Compute QL risk signal $\pi_{ij}^{\mathrm{QL}}(t)=\pi_{ij}^{Q}(t)+\pi_{ij}^{L}(t)$ for all edge servers $j$.",
    )
    text = text.replace(
        "Compute QL risk signal π_ij^QL(t) = ā_i W_j^{δ,norm} + V τ_ij.",
        r"Compute $\pi_{ij}^{\mathrm{QL}}(t)=\bar a_iW_j^{\delta,\mathrm{norm}}+V\tau_{ij}$.",
    )
    text = text.replace(
        "因此 0 < G < G_crit(δ) 是该标量局部模型的稳定区间，且 G_crit(δ) 随 δ 单调减小。",
        r"因此 $0<G<G_{\mathrm{crit}}(\delta)$ 是该标量局部模型的稳定区间，且 $G_{\mathrm{crit}}(\delta)$ 随 $\delta$ 单调减小。",
    )
    text = text.replace(
        "| π_ij^Q, π_ij^L, π_ij^QL |",
        r"| $\pi_{ij}^{Q},\pi_{ij}^{L},\pi_{ij}^{\mathrm{QL}}$ |",
    )
    text = text.replace(
        "| μ_ij(t), σ_i(t) |",
        r"| $\mu_{ij}(t),\sigma_i(t)$ |",
    )
    return text


NO_Z_REWARD = r"""### 4.4 最终 Full-noZ 团队目标

所有用户共享 Actor 参数，并采用集中训练、分散执行（CTDE）。Actor 在线阶段只读取局部状态和滞后摘要；centralized critic 的全局信息仅用于训练。最终主方法关闭 tail-state $Z$，代码中的有界团队目标由队列压力、集中度辅助状态、平均时延和 portfolio migration 组成。

令基于执行 portfolio 的期望到达为

$$
\widehat A_j(t)=\sum_{i=1}^{N}K_i(t)c\,w_{ij}(t),\qquad
s_j^{A}(t)=\frac{\widehat A_j(t)}{\sum_{k=1}^{M}\widehat A_k(t)+\varepsilon}.
$$

代码将队列压力裁剪并归一化为

$$
\bar q_j(t)=\frac{\operatorname{clip}\!\left(
Q_j(t)/(\mu_j\Delta t),0,q_{\max}\right)}{q_{\max}},
\qquad
q_t=\sum_{j=1}^{M}\bar q_j(t)s_j^{A}(t).
$$

归一化需求集中度为 $C_g(t)$，其预算为

$$
\bar C=\max\!\left\{0.02,\frac{\chi-1}{M-1}\right\}.
$$

集中度辅助状态及其有界压力为

$$
H(t+1)=\left[H(t)+0.05\bigl(C_g(t)-\bar C\bigr)\right]^+,
\qquad
\bar H(t)=\tanh\!\left(\frac{H(t)}{5}\right).
$$

令 $\widetilde T_t$ 为系统平均时延除以 `latency_norm` 后的归一化值，$M_t$ 为平均 portfolio migration。最终代价与团队奖励严格按代码写为

$$
\Phi_t=\max\!\left\{0,
q_t+\bar H(t)\bigl(C_g(t)-\bar C\bigr)
+V\bigl(\widetilde T_t+\lambda_m M_t\bigr)
\right\},
\qquad
r_t=-\tanh(\Phi_t).
$$

因此，CVaR99 是本文的任务级评价指标，而不是 Full-noZ 代码直接优化的闭式奖励项；本方法也不声称具有标准 drift-plus-penalty 的 $O(1/V)$ 性能界或全局队列稳定保证。tail-state $Z$ 与 stale margin 的删除依据作为负结果审计放入补充材料 Appendix F。
"""


PROTOCOL_SUPPLEMENT = r"""# QLA-MAPPO 补充材料（v8，可编辑中文版）

> **协议状态。** 本补充材料把“当前兼容性实现事实”和“投稿级冻结协议”分开书写。标为“待实现并重跑”的条目不是已完成实验；其目的在于先锁定语义，避免后续根据结果改变规则。

## Appendix A 队列守恒、删失与评估协议

### A.1 当前兼容性实现及其局限

当前兼容性环境首先为所有生成任务记录通信、排队和服务时延，然后按

$$
Q_j(t+1)=\min\!\left\{Q_j^{\max},
\left[Q_j(t)+A_j(t)+A_j^{\mathrm{bg}}(t)-\mu_j(t)\Delta t\right]^+
\right\}
$$

更新工作量队列。若全部服务器均不满足 admission 检查，代码记录一次 `overflow_count`，但仍抽样服务器；超过 $Q_j^{\max}$ 的后续工作量被静默截去，没有显式形成 dropped workload、timeout 或未完成任务记录。因此旧结果中的 `overflow rate` 不是 drop probability，事后 `capped@10s` 也不是模拟器 timeout。该实现不能支持投稿级可靠性结论。

### A.2 投稿级有限缓冲守恒协议（待实现并重跑）

正式协议采用“先服务旧 backlog，再接纳新工作量”的离散时隙语义。定义

$$
S_j(t)=\min\{Q_j(t),\mu_j(t)\Delta t\},\qquad
Q_j^{-}(t)=Q_j(t)-S_j(t),
$$

$$
F_j(t)=\left[Q_j^{\max}-Q_j^{-}(t)\right]^+,
$$

$$
A_j^{\mathrm{acc}}(t)=\min\{A_j(t)+A_j^{\mathrm{bg}}(t),F_j(t)\},
\qquad
A_j^{\mathrm{drop}}(t)=A_j(t)+A_j^{\mathrm{bg}}(t)-A_j^{\mathrm{acc}}(t),
$$

$$
Q_j(t+1)=Q_j^{-}(t)+A_j^{\mathrm{acc}}(t).
$$

每个时隙必须满足质量守恒恒等式

$$
Q_j(t)+A_j(t)+A_j^{\mathrm{bg}}(t)
=S_j(t)+A_j^{\mathrm{drop}}(t)+Q_j(t+1).
$$

实现应逐服务器记录 generated、accepted、served、dropped 和 remaining workload，并用自动化测试验证误差小于数值容差。所有方法共享同一 admission、drop 和服务规则。

### A.3 任务时延、drop、deadline 与评估终点（待实现并重跑）

被接纳任务按 FCFS 进入服务器队列，任务时延不截断。测试窗口结束后停止生成新任务并执行 drain-only 阶段，直至全部已接纳任务完成或达到预注册的最大 drain horizon。正式报告同时给出：

$$
p_{\mathrm{drop}}=
\frac{\sum_{t,j}A_j^{\mathrm{drop}}(t)}
{\sum_{t,j}\left(A_j(t)+A_j^{\mathrm{bg}}(t)\right)},
$$

$$
p_{\mathrm{ddl}}=
\frac{\sum_r\mathbb 1\{T_r>T_{\mathrm{ddl}}\}}
{N_{\mathrm{accepted}}},
$$

以及测试终点仍未完成的任务数 $N_{\mathrm{unfinished}}$。平均时延、CVaR95 和 CVaR99 只在具有完整 uncapped completion latency 的已接纳任务上计算，并始终与 $p_{\mathrm{drop}}$、$p_{\mathrm{ddl}}$ 和 $N_{\mathrm{unfinished}}$ 联合报告，防止通过拒绝任务获得虚假的低时延。不得为 dropped task 任意赋予一个巨大伪时延，也不得把事后 10 s cap 称为 timeout。

### A.4 独立 train/validation/test 划分（无需旧结果即可预注册）

每个轨迹数据集按时间顺序划分为互不重叠、禁止回绕的 60% train、20% validation 和 20% test 片段。相邻片段之间保留至少

$$
g\ge \delta_{\max}+b_B
$$

个时隙的 guard interval，以清空反馈缓冲和广播相位。训练 seed、参数选择 seed 和最终测试 seed 使用不相交的命名空间；所有超参数与 checkpoint rule 只在 validation 上锁定，test 只执行一次。若个别轨迹长度不足，必须缩短 episode，而不能循环回到片段起点。

### A.5 Shared evaluation schedule（待实现并重跑）

对每个 $(\text{dataset},\text{split},\text{seed},N)$ 单元，预先持久化逐时隙外生变量：用户位置索引、请求数、任务属性、用于 categorical action 的公共均匀随机数、容量事件、隐藏背景负载、广播生成时间戳、summary loss 和 feedback age。不同方法读取相同 schedule；其动作仍可不同，但任务到服务器的随机抽样使用相同公共随机数，从而形成可解释的 paired comparison。

### A.6 Checkpoint 规则与 provenance（无需性能结果即可冻结）

所有学习型方法使用相同的 10,000 training slots，并采用最终 10k checkpoint；不使用 test early stopping。各方法可保留原论文定义的更新频率、网络结构和 replay/rollout 机制，但不得拥有额外环境交互预算。每个正式结果单元必须保存：

- 代码版本或 source-tree hash；
- 完整配置及 SHA-256；
- 数据集、split ID 与 schedule SHA-256；
- checkpoint SHA-256；
- training/evaluation seed；
- Python、PyTorch、CUDA、设备与线程信息；
- 结果文件和 latency-sample 文件路径。

任何缺少 schedule hash 或 checkpoint hash 的历史 cell 只能标为 legacy/provisional。
"""


def build_sources() -> tuple[str, str]:
    source = SOURCE.read_text(encoding="utf-8")
    source = convert_text_math(source)
    source = replace_section(source, "### 4.4 Bounded team-risk objective", "### 4.5 训练与在线执行算法", NO_Z_REWARD)

    front = source[: source.index("## 3 系统模型与问题定义")]
    system = (
        "## 3 系统模型与问题定义\n\n"
        + between(source, "### 3.1 协作式部分可观测随机博弈", "### 3.5 局部 delayed-response 稳定性分析")
        + between(source, "### 3.6 假设、动作映射与技术挑战", "### 3.7 符号表").replace("### 3.6", "### 3.5")
    )
    system = system.replace(
        "投稿级版本必须改为不截断真实 backlog，或显式记录 drop/timeout 及其可靠性代价。",
        r"""投稿级版本必须改为不截断真实 backlog，或显式记录 drop/timeout 及其可靠性代价。

**投稿级冻结语义（待实现并重跑）。** 本文选择有限缓冲且显式记录 drop。先服务旧 backlog，随后接纳新到达：

$$
Q_j^{-}(t)=\left[Q_j(t)-\mu_j(t)\Delta t\right]^+,qquad
A_j^{\mathrm{acc}}(t)=\min\!\left\{A_j(t),Q_j^{\max}-Q_j^{-}(t)\right\},
$$

$$
A_j^{\mathrm{drop}}(t)=A_j(t)-A_j^{\mathrm{acc}}(t),qquad
Q_j(t+1)=Q_j^{-}(t)+A_j^{\mathrm{acc}}(t).
$$

正式实现必须逐槽验证 $Q_j(t)+A_j(t)=S_j(t)+A_j^{\mathrm{drop}}(t)+Q_j(t+1)$。被接纳任务采用 uncapped completion latency；dropped task 不赋予伪时延，而与 deadline violation、unfinished task 联合报告。完整语义见补充材料 Appendix A。""",
    )
    method = between(source, "## 4 QLA-MAPPO 方法", "## 5 实验设置")
    setup = between(source, "## 5 实验设置", "## 6 实验结果")
    setup = setup.replace(
        "本文实验围绕四个假设组织：",
        r"""**投稿级数据与评估协议（已预注册，待实现并重跑）。** 每个轨迹按时间顺序划分为互不重叠且禁止回绕的 60% train、20% validation 和 20% test 片段，片段间设置不小于 $\delta_{\max}+b_B$ 的 guard interval。所有方法读取同一持久化 evaluation schedule；该文件固定位置索引、请求、任务属性、categorical 公共随机数、容量/背景负载事件、广播时间戳及 source age。超参数只在 validation 上锁定，所有学习方法采用最终 10k checkpoint，不使用 test early stopping。每个结果 cell 保存配置、schedule 与 checkpoint 的 SHA-256。当前表格仍来自旧兼容性协议，故保持 provisional；完整规范见补充材料 Appendix A。

本文实验围绕四个假设组织：""",
    )

    sec60 = between(source, "### 6.0 参数析因审计", "### 6.1 主 baseline 对比")
    sec61 = between(source, "### 6.1 主 baseline 对比", "### 6.2 问题存在性与负载区间扫描")
    baseline_core = sec61.split("**直接 delayed-information 机制基线扩展。**", 1)[0]
    baseline_extra = "**直接 delayed-information 机制基线扩展。**" + sec61.split(
        "**直接 delayed-information 机制基线扩展。**", 1
    )[1]
    sec62 = between(source, "### 6.2 问题存在性与负载区间扫描", "### 6.3 Feedback age 分解")
    sec63 = between(source, "### 6.3 Feedback age 分解", "### 6.4 核心结构消融")
    sec64 = between(source, "### 6.4 核心结构消融", "### 6.5 同 checkpoint 风险信号干预")
    flat_marker = "**Flat-Bias-QL 反馈年龄机制（不重训）。**"
    flat_extra = flat_marker + sec64.split(flat_marker, 1)[1] if flat_marker in sec64 else ""
    ablation_core = sec64.split(flat_marker, 1)[0]
    if flat_extra:
        ablation_core += (
            r"\n**反馈年龄交互摘要。** 现有 cell 的 paired DiD 显示，Flat 在 $\delta=0$ 可略优，"
            "但其 average latency 与 CVaR99 随反馈年龄的退化分别约为 Main 的 6 倍和 5 倍；"
            r"详细逐 $\delta$ CI、斜率和 DiD 见补充材料 Appendix F。\n\n"
        )
    sec65 = between(source, "### 6.5 同 checkpoint 风险信号干预", "### 6.6 辅助状态与奖励诊断")
    sec66 = between(source, "### 6.6 辅助状态与奖励诊断", "### 6.7 固定 checkpoint 跨环境测试")
    sec67 = between(source, "### 6.7 固定 checkpoint 跨环境测试", "### 6.8 KAIST N=40 高负载压力测试")
    sec68 = between(source, "### 6.8 KAIST N=40 高负载压力测试", "### 6.9 尾部分布")
    sec69 = between(source, "### 6.9 尾部分布", "### 6.10 推理效率")
    sec610 = between(source, "### 6.10 推理效率", "### 6.11 Stale-feedback 机理诊断")
    sec611 = between(source, "### 6.11 Stale-feedback 机理诊断", "## 7 讨论与局限")
    ending = between(source, "## 7 讨论与局限")

    param_summary = r"""### 6.1 参数筛选与配置锁定

在删除 stale margin 与 tail-state $Z$ 后，本文对 $\lambda_\pi\times H_{\mathrm{target}}\times V$ 完成 $2^3+$中心点审计，并额外完成 $V\in\{0.25,0.5,1.0,2.0\}$ 的 36-cell seed-block sensitivity。局部析因中，增大 $\lambda_\pi$ 会提高 CVaR99 和 migration，较高熵目标降低 migration 但略损害时延，$V$ 的平均主效应接近零；独立 $V$ sweep 的各项 paired CI 均跨 0，因此保留预定义中心值 $V=0.5$。这些结果用于参数敏感性与冻结合理性说明，不用于依据 test 结果重新选取主配置。完整设计、效应区间和 $V$ sweep 放在补充材料 Appendix D。
"""

    robustness_summary = r"""### 6.6 现实反馈与跨环境诊断

现有固定-checkpoint 48-cell 跨环境结果和四条件信息结构实验只作为 provisional 诊断：前者来自 legacy checkpoint provenance，后者尚未覆盖随机时变 delay、per-server age、summary loss、真正异步更新、容量突变和隐藏背景负载。完整图表与限制见补充材料 Appendix G。正式稿将在队列守恒与 shared evaluation schedule 冻结后，用 Full-noZ checkpoint 和 5 个共同 evaluation seeds 重评；在此之前，正文不声称跨环境相对优势或广泛 stale-feedback 鲁棒性。
"""

    main_results = (
        "## 6 实验结果\n\n"
        + param_summary
        + sec62.replace("### 6.2", "### 6.2", 1)
        + baseline_core.replace("### 6.1", "### 6.3", 1)
        + sec63.replace("### 6.3", "### 6.4", 1)
        + sec69.replace("### 6.9", "#### 6.4.1", 1)
        + ablation_core.replace("### 6.4", "### 6.5", 1)
        + sec65.replace("### 6.5", "#### 6.5.1", 1)
        + robustness_summary
        + sec68.replace("### 6.8", "### 6.7", 1)
        + sec610.replace("### 6.10", "### 6.8", 1)
    )

    main = front + system + method + setup + main_results + ending
    main = main.replace(
        "> **Provisional draft (v8, 2026-07-27).**",
        "> **Provisional main manuscript (v8-split).**",
        1,
    )

    appendix = (
        PROTOCOL_SUPPLEMENT
        + "\n\n## Appendix B 局部机制分析与完整符号\n\n"
        + between(source, "### 3.5 局部 delayed-response 稳定性分析", "### 3.6 假设、动作映射与技术挑战").replace("### 3.5", "### B.1")
        + between(source, "### 3.7 符号表", "## 4 QLA-MAPPO 方法").replace("### 3.7", "### B.2")
        + "\n## Appendix C Delayed-information baselines 与适配审计\n\n"
        + baseline_extra
        + "\n## Appendix D 参数析因与 $V$ sensitivity\n\n"
        + sec60.replace("### 6.0", "### D.1", 1)
        + r"\n独立 $V$ sweep 使用 $V\in\{0.25,0.5,1.0,2.0\}$、3 seeds 与 $\delta\in\{0,4,8\}$。"
          "现有 36 cells 无需重训；按 seed 汇总三个 delay 后计算的 paired CI 均跨 0，支持保留预定义 $V=0.5$，"
          "但不证明 $V=0.5$ 为全局最优。原始与 seed-block 汇总位于 `results_qla_simplified_rerun/lcc_v_sweep/`。\n"
        + "\n## Appendix E 完整统计来源\n\n"
        + "主表与消融的唯一权威 paired CI 来源为 `results_qla_simplified_rerun/final_analysis/`。"
          "重采样单位为 training seed，delay 作为 seed block 内的实验条件；不得将 25 个 seed-delay cells 当作 25 个独立重复。\n"
        + "\n## Appendix F 补充消融与负结果\n\n"
        + flat_extra
        + sec66.replace("### 6.6", "### F.2", 1)
        + "\n## Appendix G 跨环境与现实 stale-feedback 诊断\n\n"
        + sec67.replace("### 6.7", "### G.1", 1)
        + sec611.replace("### 6.11", "### G.2", 1)
        + "\n## Appendix H 尾部分布与可靠性审计说明\n\n"
        + r"正文保留 $\delta=8$ CCDF 与关键 tail audit；在守恒协议重跑后，本附录将补充每个 seed-delay 的任务数、"
          "tail count、uncapped CVaR95/CVaR99、drop、deadline violation、unfinished 和 drain horizon。\n"
        + "\n## Appendix I 可复现性与结果替换清单\n\n"
        + "正式 artifact 不在 PDF 中粘贴大段源代码，而提供匿名仓库、环境锁文件、配置、schedule manifest、"
          "checkpoint manifest、原始 CSV 和绘图脚本。下列结果必须在守恒协议和 shared schedule 完成后替换："
          "(i) 主 5×5 网格；(ii) JSEQ-A 与正式 baselines；(iii) N=40；(iv) cross-env；"
          "(v) tail audit/CCDF；(vi) paired bootstrap 与摘要效应量。\n"
    )
    return convert_text_math(main), convert_text_math(appendix)


def polish_docx(path: Path) -> None:
    doc = Document(path)
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.72)
        section.left_margin = Inches(0.72)
        section.right_margin = Inches(0.72)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(3)

    for style_name, size in (("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11.5)):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    for paragraph in doc.paragraphs:
        xml = paragraph._p.xml
        text = paragraph.text.strip()
        if "<w:drawing" in xml or "<m:oMath" in xml or "<m:oMathPara" in xml:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if text.startswith(("Fig.", "图 ", "表 ", "Table ")):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(9)

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        for row in table.rows:
            for cell in row.cells:
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.size = Pt(8.5)
    doc.save(path)


def convert(md: Path, docx: Path) -> None:
    cmd = [
        "pandoc",
        str(md),
        "--from=markdown+tex_math_dollars+raw_tex",
        "--to=docx",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        f"--resource-path={ROOT}",
        f"--reference-doc={REFERENCE}",
        "--metadata=lang:zh-CN",
        "--output",
        str(docx),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    polish_docx(docx)


def main() -> None:
    main_md, supp_md = build_sources()
    MAIN_MD.write_text(main_md, encoding="utf-8")
    SUPP_MD.write_text(supp_md, encoding="utf-8")
    convert(MAIN_MD, MAIN_DOCX)
    convert(SUPP_MD, SUPP_DOCX)
    print(MAIN_DOCX)
    print(SUPP_DOCX)


if __name__ == "__main__":
    main()
