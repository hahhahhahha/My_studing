from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper_draft_facr_cn_v8_compact_formula_checked.docx"
DST = ROOT / "paper_draft_facr_cn_v9_evidence_checked.docx"
EVIDENCE = ROOT / "results_missing_evidence"
FIGURES = ROOT / "figures"


def read_rows(name: str) -> list[dict[str, str]]:
    with (EVIDENCE / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def add_table_before(doc: Document, anchor, rows: list[list[str]], widths: list[float] | None = None) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = text
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if r == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    run.font.size = Pt(8 if len(rows[0]) > 4 else 9)
                    if r == 0:
                        run.bold = True
            if widths and c < len(widths):
                cell.width = Inches(widths[c])
    anchor._p.addprevious(table._tbl)


def add_paragraph_before(anchor, text: str, style: str | None = None, align: int | None = None):
    paragraph = anchor.insert_paragraph_before(text)
    if style:
        paragraph.style = style
    if align is not None:
        paragraph.alignment = align
    return paragraph


def add_image_before(anchor, image: Path, caption: str, width: float = 5.8) -> None:
    paragraph = anchor.insert_paragraph_before("")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(image), width=Inches(width))
    cap = anchor.insert_paragraph_before(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(9)


def replace_inline_numbers(doc: Document) -> None:
    replacements = {
        "4.454": "4.263",
        "3.546": "3.538",
    }
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            text = run.text
            for old, new in replacements.items():
                text = text.replace(old, new)
            run.text = text


def build_main_ci_table() -> list[list[str]]:
    rows = read_rows("main_baseline_bootstrap_ci.csv")
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["metric"]] = row
    order = ["FACR-MAPPO", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    table = [["方法", "Avg latency 95% CI", "CVaR99 95% CI", "Queue osc. 95% CI", "Migration 95% CI"]]
    for method in order:
        m = by_method[method]
        table.append(
            [
                method,
                f"{fmt(m['average_latency']['mean'])} [{fmt(m['average_latency']['bootstrap95_low'])}, {fmt(m['average_latency']['bootstrap95_high'])}]",
                f"{fmt(m['cvar99']['mean'])} [{fmt(m['cvar99']['bootstrap95_low'])}, {fmt(m['cvar99']['bootstrap95_high'])}]",
                f"{fmt(m['queue_oscillation']['mean'])} [{fmt(m['queue_oscillation']['bootstrap95_low'])}, {fmt(m['queue_oscillation']['bootstrap95_high'])}]",
                f"{fmt(m['portfolio_migration_rate']['mean'], 4)} [{fmt(m['portfolio_migration_rate']['bootstrap95_low'], 4)}, {fmt(m['portfolio_migration_rate']['bootstrap95_high'], 4)}]",
            ]
        )
    return table


def build_paired_test_text() -> str:
    rows = read_rows("main_baseline_paired_tests.csv")
    lookup = {(r["baseline"], r["metric"]): r for r in rows}
    lat = lookup[("PPO-MV-A", "average_latency")]
    tail = lookup[("QECO-A", "cvar99")]
    qosc = lookup[("PPO-MV-A", "queue_oscillation")]
    mig = lookup[("PPO-MV-A", "portfolio_migration_rate")]
    return (
        "Paired seed-delay tests further confirm the main comparison. Against the strongest non-FACR average-latency "
        f"baseline PPO-MV-A, FACR reduces latency by {fmt(lat['mean_improvement_baseline_minus_facr'])} s "
        f"({float(lat['relative_improvement_vs_baseline']) * 100:.1f}%, 95% CI "
        f"[{fmt(lat['bootstrap95_low'])}, {fmt(lat['bootstrap95_high'])}], one-sided paired p<0.001). "
        f"Against the strongest non-FACR CVaR99 baseline QECO-A, FACR reduces CVaR99 by {fmt(tail['mean_improvement_baseline_minus_facr'])} s "
        f"({float(tail['relative_improvement_vs_baseline']) * 100:.1f}%, 95% CI "
        f"[{fmt(tail['bootstrap95_low'])}, {fmt(tail['bootstrap95_high'])}], p<0.001). "
        f"For queue oscillation and migration, the improvements over PPO-MV-A are {float(qosc['relative_improvement_vs_baseline']) * 100:.1f}% "
        f"and {float(mig['relative_improvement_vs_baseline']) * 100:.1f}%, respectively."
    )


def build_tail_table() -> list[list[str]]:
    rows = read_rows("latency_tail_quantiles_delta8.csv")
    order = ["FACR-MAPPO", "PPO-MV-A", "PPO-DSR-A", "QECO-A", "TODA-MAB-A", "LyFOTO-LC"]
    by_method = {r["method"]: r for r in rows}
    table = [["方法", "samples", "mean", "P95", "P99", "max"]]
    for method in order:
        row = by_method[method]
        table.append([method, row["samples"], fmt(row["mean"]), fmt(row["p95"]), fmt(row["p99"]), fmt(row["max"])])
    return table


def build_queue_corr_table() -> list[list[str]]:
    rows = read_rows("queue_state_correlation_summary.csv")
    table = [["δ", "ρ_Q mean", "std", "解释"]]
    for row in rows:
        delta = int(row["delta"])
        rho = float(row["rho_Q_mean"])
        if delta <= 2:
            note = "强时间相关，可作为低置信度参考"
        elif delta == 4:
            note = "相关性明显下降"
        else:
            note = "严重滞后时可能误导，不能当作 fresh queue"
        table.append([str(delta), fmt(rho), fmt(row["rho_Q_std"]), note])
    return table


def build_async_table() -> list[list[str]]:
    rows = read_rows("asynchrony_score_softmax_summary.csv")
    keep = [
        ("simultaneous", "0", "同步、fresh summary"),
        ("simultaneous", "8", "同步、同一 stale summary"),
        ("mixed_age", "mixed_0_1_2_4_8", "不同用户反馈年龄异质"),
        ("sequential", "0", "顺序执行且可见 fresh 更新"),
    ]
    lookup = {(r["execution_mode"], r["feedback_delay"]): r for r in rows}
    table = [["设置", "Avg latency", "CVaR99", "Queue osc.", "Migration"]]
    for mode, delay, label in keep:
        row = lookup[(mode, delay)]
        table.append(
            [
                label,
                fmt(row["average_latency_mean"]),
                fmt(row["cvar99_mean"]),
                fmt(row["queue_oscillation_mean"]),
                fmt(row["portfolio_migration_rate_mean"], 3),
            ]
        )
    return table


def build_adaptation_table() -> list[list[str]]:
    rows = read_rows("baseline_adaptation_transparency.csv")
    table = [["Baseline", "保留机制", "本文适配", "主要限制/透明说明"]]
    for row in rows:
        table.append(
            [
                row["baseline_name"],
                row["kept_from_original"],
                row["adapted_for_this_problem"],
                row["not_kept_or_limitation"],
            ]
        )
    return table


def main() -> None:
    doc = Document(str(SRC))
    replace_inline_numbers(doc)

    anchor = None
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith("Heading") and paragraph.text.strip().startswith("6 "):
            anchor = paragraph
            break
    if anchor is None:
        raise RuntimeError("Could not find Discussion heading anchor.")

    add_paragraph_before(anchor, "5.6 统计显著性、尾部分布与假设敏感性补充", "Heading 3")
    add_paragraph_before(
        anchor,
        "为补强 mean±std 之外的统计证据，本文基于主 baseline 对比的 25 个 paired seed-delay 单元（5 个训练种子 × δ={0,1,2,4,8}）补充 bootstrap 95% 置信区间和配对符号翻转检验。表中所有指标均以较低为优。可以看到，FACR-MAPPO 的置信区间与主要 baseline 基本不重叠；配对检验进一步显示，FACR 在平均时延、CVaR99 与队列振荡上的优势均具有稳定统计支撑。",
    )
    add_table_before(doc, anchor, build_main_ci_table(), widths=[1.25, 1.55, 1.55, 1.55, 1.45])
    add_paragraph_before(anchor, "Table 2. 主 baseline 对比的 bootstrap 95% 置信区间。单位：时延为秒，其余为无量纲比例或归一化指标。", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph_before(anchor, build_paired_test_text())

    add_paragraph_before(
        anchor,
        "仅报告 CVaR99 仍不足以展示尾部形态，因此本文在代表性严重滞后场景 KAIST、N=30、poisson-bursty、δ=8 下绘制 latency CCDF。FACR 的曲线在右尾区域整体更靠左，且 P99 未触及系统上限；多项 adapted baseline 的 P99 或最大值达到 10 s 上限，说明其平均时延改善不足以避免少量极端排队事件。",
    )
    add_image_before(anchor, FIGURES / "fig_latency_ccdf_delta8.png", "Fig. 9. Representative latency CCDF under KAIST, N=30, poisson-bursty workload and δ=8.", width=5.8)
    add_table_before(doc, anchor, build_tail_table(), widths=[1.35, 1.0, 0.85, 0.85, 0.85, 0.85])
    add_paragraph_before(anchor, "Table 3. δ=8 代表性场景下的 latency tail quantiles。", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_paragraph_before(
        anchor,
        "为解释 stale prior 的信息边界，本文计算真实队列时间序列的状态相关性 ρ_Q(δ)=corr(Q(t−δ),Q(t))。结果显示，δ=1 和 δ=2 下队列仍具有较强时间相关性，δ=4 时相关性明显下降，而 δ=8 时在 naive score-softmax 高并发响应下甚至可能转为负相关。这一证据支持本文对 stale summary 的保守解释：它不是当前真实队列的估计器，而是低置信度结构参考；严重滞后时必须通过 AgeGate、prior 权重和平滑融合限制其影响。",
    )
    add_image_before(anchor, FIGURES / "fig_queue_state_correlation.png", "Fig. 10. Queue-state temporal correlation ρ_Q(δ) under score-softmax stress dynamics.", width=4.9)
    add_table_before(doc, anchor, build_queue_corr_table(), widths=[0.65, 1.05, 0.85, 3.7])
    add_paragraph_before(anchor, "Table 4. 真实队列状态相关性随 feedback age 的变化。", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_paragraph_before(
        anchor,
        "同步决策假设也需要单独检查。本文补充 score-softmax stress test：当 δ=0 且顺序执行可观察到 slot 内更新时，CVaR99 从同步 fresh 的 5.848 s 降至 1.636 s；但在 δ=4 或 δ=8 时，sequential 与 simultaneous 几乎一致，因为用户仍依据同一份滞后摘要决策。进一步地，当不同用户的 feedback age 在 {0,1,2,4,8} 中异质采样时，CVaR99 为 3.661 s，显著低于所有用户共享 δ=8 stale summary 的 10.037 s。这说明本文的核心压力并非单纯“同时计算动作”，而是多用户对同一 stale load indication 的同步追逐。",
    )
    add_image_before(anchor, FIGURES / "fig_mixed_age_sensitivity.png", "Fig. 11. Heterogeneous feedback ages reduce score-softmax tail-risk amplification.", width=4.9)
    add_table_before(doc, anchor, build_async_table(), widths=[2.2, 1.0, 1.0, 1.0, 1.0])
    add_paragraph_before(anchor, "Table 5. 同步、顺序与 heterogeneous-age score-softmax stress test。", align=WD_ALIGN_PARAGRAPH.CENTER)

    add_paragraph_before(anchor, "5.7 Baseline 适配透明度与数值一致性审计", "Heading 3")
    add_paragraph_before(
        anchor,
        "由于本文统一采用 stale-summary portfolio offloading 环境，近期 MEC 方法不能逐项照搬原系统模型。表 X 明确列出每个 adapted baseline 保留的原始思想、为本文场景做出的接口适配以及未完整复现的部分。这样的写法比简单宣称“复现原论文”更稳健，也能避免审稿人误解 baseline 结果来自不透明削弱。",
    )
    add_table_before(doc, anchor, build_adaptation_table(), widths=[1.05, 1.65, 1.9, 1.9])
    add_paragraph_before(anchor, "Table 6. Adapted baselines 的机制保留、场景适配与限制说明。", align=WD_ALIGN_PARAGRAPH.CENTER)
    add_paragraph_before(
        anchor,
        "数值一致性方面，本文统一以 results_facr_ablation_analysis/facr_ablation_summary_mean_std.csv 作为 FACR 消融的权威来源。Actor-only MAPPO 的 CVaR99 统一为 4.263 s，Prior-only 的 CVaR99 统一为 3.538 s；稿件早期旧版或部分聚合口径中的数值已不再作为正文引用。完整 FACR 的平均 D_KL(w_i||p_i^L)=0.0234，而 Prior-only 为 0.0104，说明完整方法不是简单复制 prior，而是在 prior 附近学习小幅 residual correction；同时 Prior-only 的 migration 约为完整 FACR 的 3.45 倍，表明学习型 actor 与保守融合对动作稳定性仍有实际价值。",
    )

    doc.save(str(DST))
    print(DST)


if __name__ == "__main__":
    main()
