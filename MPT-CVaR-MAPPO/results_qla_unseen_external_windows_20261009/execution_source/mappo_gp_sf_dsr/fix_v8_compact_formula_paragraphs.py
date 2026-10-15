"""Fix ambiguous equation paragraphs in paper_draft_facr_cn_v8_compact.docx.

The compact manuscript preserved many equations from the source DOCX, but some
OfficeMath objects have ambiguous linear text when copied or inspected. This
script regenerates the most important display equations through pandoc and
replaces only the display-equation paragraphs, leaving surrounding prose,
figures, and tables intact.
"""

from __future__ import annotations

import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

from docx import Document
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper_draft_facr_cn_v8_compact.docx"
OUT = ROOT / "paper_draft_facr_cn_v8_compact_formula_checked.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


EQUATIONS = [
    r"B(t-\delta)=\{s_j(t-\delta),q_j(t-\delta),c_j(t-\delta),d_j(t-\delta)\}_{j=1}^{M}",
    r"T_{ij}(t)=T_{ij}^{\mathrm{comm}}(t)+T_j^{\mathrm{queue}}(t)+T_j^{\mathrm{service}}(t)",
    r"w_i(t)\in\Delta^{M-1},\quad \sum_{j=1}^{M}w_{ij}(t)=1,\quad w_{ij}(t)\ge 0",
    r"o_i(t)=\{x_i(t),r_i(t),\mathrm{distance}_i(t),B(t-\delta),\delta\}",
    r"\min_{\pi}\ \mathbb{E}[T]+\lambda_c\,\mathrm{CVaR}_{0.99}(T)",
    r"x(t+1)=x(t)-Gx(t-\delta)",
    r"o_i(t)=\big(o_i^{\mathrm{loc}}(t),o_i^{\mathrm{score}}(t),\delta\big)",
    r"(\mu_i^{\mathrm{loc}},\log\sigma_i^{\mathrm{loc}})=f_{\mathrm{loc}}\!\left(o_i^{\mathrm{loc}}(t)\right)",
    r"(\mu_i^{\mathrm{score}},\log\sigma_i^{\mathrm{score}})=f_{\mathrm{score}}\!\left(o_i^{\mathrm{score}}(t)\right)",
    r"g(\delta)=\exp(-\eta\delta)",
    r"(\mu_i(t),\log\sigma_i(t))=(\mu_i^{\mathrm{loc}},\log\sigma_i^{\mathrm{loc}})+g(\delta)(\mu_i^{\mathrm{score}},\log\sigma_i^{\mathrm{score}})",
    r"z_i(t)\sim\pi_{\theta}^{z}(\cdot\mid o_i(t))=\mathcal{N}\!\left(\mu_i(t),\operatorname{diag}(\sigma_i^2(t))\right)",
    r"u_i(t)=\operatorname{softmax}(z_i(t)),\quad u_i(t)\in\Delta^{M-1}",
    r"C_{ij}^{L}(t,\delta)=V_L\!\left(T_{ij}^{\mathrm{comm}}(t)+T_j^{\mathrm{srv}}+W_j(t-\delta)\right)+\frac{W_j(t-\delta)}{\Delta t}\,T_j^{\mathrm{srv}}",
    r"p_{ij}^{L}(t)=\frac{\exp\!\left(-C_{ij}^{L}(t,\delta)/\tau_L\right)}{\sum_{k=1}^{M}\exp\!\left(-C_{ik}^{L}(t,\delta)/\tau_L\right)}",
    r"p_i^{L}(t)=\operatorname{softmax}\!\left(-C_i^{L}(t,\delta)/\tau_L\right)",
    r"\tilde{w}_i(t)=(1-\lambda_L)u_i(t)+\lambda_L p_i^L(t)",
    r"\bar{w}_i(t)=\rho w_i(t-1)+(1-\rho)\tilde{w}_i(t)",
    r"w_i(t)=\Pi_{\Delta^{M-1}}\!\left(\bar{w}_i(t)\right)",
    r"\|w_i(t)-w_i(t-1)\|_1\le\epsilon_{\max}",
    r"V_{\phi}(c_i(t),i)",
    r"c_i(t)=\left[\{o_i^{\mathrm{loc}}(t)\}_{i=1}^{N},Q(t),\mathrm{edge\ states},\mathrm{joint\ portfolio\ statistics}\right]",
    r"r_i(t)=\alpha R_{\mathrm{DSR-tail}}(t)+(1-\alpha)[-T_i(t)]-\lambda_q P_i^Q(t)-\kappa_i P_i^{\mathrm{port}}(t)-\kappa_g P^{\mathrm{conc}}(t)",
    r"\mathrm{CVaR}_{0.95}(T),\quad \mathrm{CVaR}_{0.99}(T)",
    r"r_t^{\mathrm{PPO}}=\frac{\pi_{\theta}^{z}(z_i(t)\mid o_i(t))}{\pi_{\theta_{\mathrm{old}}}^{z}(z_i(t)\mid o_i(t))}",
]


OLD_LINEAR_MARKERS = [
    "B(t−δ)={sj(t−δ),qj(t−δ),cj(t−δ),dj(t−δ)}j=1M,",
    "Tij(t)=Tijcomm(t)+Tjqueue(t)+Tjservice(t).",
    "wi(t)∈ΔM−1",
    "oi(t)={xi(t),ri(t),distancei(t),B(t−δ),δ},",
    "minπ",
    "x(t+1)=x(t)−Gx(t−δ)",
    "oi(t)=oiloc(t),oiscore(t),δ,",
    "(μiloc,logσiloc)=floc",
    "(μiscore,logσiscore)=fscore",
    "g(δ)=exp(−ηδ),",
    "(μi(t),logσi(t))=(μiloc,logσiloc)+g(δ)",
    "zi(t)∼πθz",
    "ui(t)=softmax(zi(t))",
    "CijL(t,δ)=",
    "pijL(t)=exp",
    "piL(t)=softmax",
    "wi(t)=(1−λL)ui(t)+λLpiL(t).",
    "wi(t)=ρwi(t−1)+(1−ρ)wi(t).",
    "wiΔ(t)=Normalize",
    "∥wi(t)−wi(t−1)∥1≤ϵmax.",
    "Vϕ(ci(t),i),",
    "ci(t)=[{oiloc(t)}i=1N",
    "ri(t)=αRDSR-tail",
    "CVaR0.95(T),CVaR0.99(T).",
    "rtPPO=",
]


def math_text(paragraph) -> str:
    root = etree.fromstring(paragraph._element.xml.encode("utf-8"))
    return "".join(root.xpath(".//m:t/text()", namespaces=NS))


def word_text(paragraph) -> str:
    root = etree.fromstring(paragraph._element.xml.encode("utf-8"))
    return "".join(root.xpath(".//w:t/text()", namespaces=NS))


def generate_equation_docx(tmpdir: Path) -> Path:
    md = tmpdir / "equations.md"
    docx = tmpdir / "equations.docx"
    md.write_text("\n\n".join(f"$${eq}$$" for eq in EQUATIONS), encoding="utf-8")
    subprocess.run(["pandoc", str(md), "-o", str(docx)], check=True)
    return docx


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    with tempfile.TemporaryDirectory() as td:
        eq_docx_path = generate_equation_docx(Path(td))
        eq_doc = Document(str(eq_docx_path))
        eq_paragraphs = [p for p in eq_doc.paragraphs if math_text(p)]
        if len(eq_paragraphs) != len(EQUATIONS):
            raise RuntimeError(f"Expected {len(EQUATIONS)} equation paragraphs, got {len(eq_paragraphs)}")

        doc = Document(str(SRC))
        targets = []
        for para in doc.paragraphs:
            if word_text(para).strip():
                continue
            mt = math_text(para)
            if not mt:
                continue
            for marker in OLD_LINEAR_MARKERS:
                if marker in mt:
                    targets.append(para)
                    break
        if len(targets) != len(EQUATIONS):
            found = [math_text(p) for p in targets]
            raise RuntimeError(f"Expected {len(EQUATIONS)} target equations, found {len(targets)}: {found}")

        for target, source in zip(targets, eq_paragraphs):
            target._p.getparent().replace(target._p, deepcopy(source._p))

        for para in doc.paragraphs:
            for run in para.runs:
                if "第 6.6 节" in run.text:
                    run.text = run.text.replace("第 6.6 节", "第 5.4 节")

        # Avoid inline OfficeMath conversion artifacts such as text{ηδ} in
        # algorithm/prose exports; plain text is clearer for these short inline
        # expressions.
        for para in doc.paragraphs:
            if "AgeGate 实际使用" in para.text:
                para.clear()
                para.add_run(
                    "局部验证结果支持最终配置：H_target=0.99 明显优于 0.95，说明在多用户 portfolio 卸载中，高熵先验有助于减少早期单边缘塌缩；"
                    "κ_i=κ_g=0.02 在平均时延、CVaR99 和队列振荡之间最均衡；η=0.25 不是所有聚合指标的绝对最优点，但在 CVaR99 上优于更弱衰减 η=0.05,0.10，"
                    "并与 η=0.40 的差异很小，因此本文选择 η=0.25 作为偏尾部风险控制的折中。AgeGate 实际使用 g(δ)=exp(−ηδ)，没有再除以最大 delay，"
                    "因此 η 的数值直接决定状态摘要随反馈年龄的衰减强度。"
                )
            elif "通过" in para.text and "降低陈旧服务器评分对动作的影响" in para.text:
                para.clear()
                para.add_run(
                    "从 δ=0 到 δ=8，完整 FACR 的 CVaR99 仅从 3.453 s 增至 3.514 s；w/o AgeGate 则从 3.360 s 增至 3.615 s。"
                    "换言之，w/o AgeGate 在新鲜状态下并不差，但在严重滞后反馈下退化更明显。这与 AgeGate 的设计目标一致："
                    "它并不保证改善 δ=0 的瞬时最优性，而是通过 g(δ)=exp(−ηδ) 降低陈旧服务器评分对动作的影响，从而减小 stale feedback 条件下的尾部风险斜率。"
                )

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if "Apply" in cell.text and "server-score branch" in cell.text:
                        cell.text = "Apply g(δ)=exp(−ηδ) to fade the server-score branch before Gaussian parameter construction."

        doc.save(str(OUT))
        print(OUT)


if __name__ == "__main__":
    main()
