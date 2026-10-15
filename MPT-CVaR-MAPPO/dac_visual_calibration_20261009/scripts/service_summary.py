"""Describe measured service distributions without substituting simulated data."""
from __future__ import annotations

import argparse
from datetime import datetime
import math

from common import BASE, read, save, sha, csv_write
from measurement_audit import audit_role, audit_all, STAGES


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered or not 0 < probability <= 1:
        raise ValueError("Nonempty data and 0<p<=1 required")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def statistics(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"n": n, "mean": mean, "population_std": std, "CV": std / mean,
            "median": quantile(values, .5), "P90": quantile(values, .9),
            "P95": quantile(values, .95), "P99": quantile(values, .99),
            "minimum": min(values), "maximum": max(values)}


def validate_distribution():
    rule_path = BASE / "protocol/SERVICE_DISTRIBUTION_VALIDATION_RULE.json"
    rule = read(rule_path)
    if rule["schema"] != "empirical_service_distribution_rule_v1" or rule["test_used_for_fit_or_acceptance"]:
        raise RuntimeError("Invalid validation rule")
    cal_audit, cal_rows = audit_role("calibration")
    val_audit, val_rows = audit_role("validation")
    errors = []
    table = []
    for profile in ("CPU-1T", "CPU-4T"):
        cal = statistics([float(r["service_ms"]) for r in cal_rows if r["profile_id"] == profile])
        val = statistics([float(r["service_ms"]) for r in val_rows if r["profile_id"] == profile])
        for metric, criterion in rule["metrics"].items():
            error = abs(cal[metric] - val[metric]) / val[metric]
            limit = criterion["maximum_absolute_relative_error"]
            row = {"profile": profile, "metric": metric, "calibration_ms": cal[metric],
                   "validation_ms": val[metric], "absolute_relative_error": error,
                   "tolerance": limit, "passed": error <= limit}
            table.append(row)
            if not row["passed"]:
                errors.append(row)
    fingerprints = {**cal_audit["measurement_sha256"], **val_audit["measurement_sha256"]}
    report = {"schema": "empirical_service_distribution_validation_v1",
              "accepted": not errors, "datetime": datetime.now().isoformat(),
              "method": rule["model"], "criteria_sha256": sha(rule_path),
              "test_accessed": False, "results": table, "failed_criteria": errors,
              "measurement_sha256": fingerprints, "scope": rule["scope"],
              "simulator_queue_execution_verified": False}
    save(BASE / "calibration/VALIDATION_REPORT.json", report)
    csv_write(BASE / "calibration/validation_errors.csv", table)
    if errors:
        raise RuntimeError("Held-out validation failed frozen service-distribution tolerances; report retained")
    # This freezes the calibration request table, not a scalar average service rate.
    csv_write(BASE / "calibration/paired_empirical_service_table.csv", cal_rows)
    report_path = BASE / "calibration/VALIDATION_REPORT.json"
    review_path = BASE / "protocol/FORMAL_RUNNER_REVIEW.json"
    review = read(review_path)
    if review.get("passed") is not True:
        raise RuntimeError("Independent protocol/source review is required")
    for relative, digest in review["source_sha256"].items():
        if sha(BASE / relative) != digest:
            raise RuntimeError(f"Reviewed source changed: {relative}")
    save(BASE / "calibration/CALIBRATION_FREEZE.json", {
        "schema": "visual_calibration_acceptance_v1", "accepted": True,
        "datetime": datetime.now().isoformat(),
        "environment_spec_sha256": sha(BASE / "protocol/ENVIRONMENT_SPEC.json"),
        "image_split_sha256": sha(BASE / "protocol/IMAGE_SPLITS.json"),
        "measurement_sha256": fingerprints,
        "validation_report_sha256": sha(report_path),
        "empirical_table_sha256": sha(BASE / "calibration/paired_empirical_service_table.csv"),
        "independent_protocol_source_review_sha256": sha(review_path),
        "acceptance_scope": rule["scope"],
        "simulator_queue_execution_verified": False})
    return report


def summarize():
    audits, role_rows = audit_all()
    save(BASE / "analysis/MEASUREMENT_AUDIT.json", {
        "passed": True, "datetime": datetime.now().isoformat(),
        "formal_rows": sum(audits[role]["rows"] for role in ("calibration", "validation", "test")),
        "pilot_rows_excluded": audits["pilot"]["rows"], "roles": audits})
    summaries, session_rows, stage_rows, payload_rows = [], [], [], []
    for role in ("calibration", "validation", "test"):
        rows = role_rows[role]
        for profile in ("CPU-1T", "CPU-4T"):
            selected = [r for r in rows if r["profile_id"] == profile]
            summaries.append({"role": role, "profile": profile, "unit": "ms",
                              "sessions": len({r["session_id"] for r in selected}),
                              "unique_images": len({r["image_id"] for r in selected}),
                              **statistics([float(r["service_ms"]) for r in selected])})
            for session in sorted({int(r["session_id"]) for r in selected}):
                group = [r for r in selected if int(r["session_id"]) == session]
                session_rows.append({"role": role, "profile": profile, "session": session,
                                     "first_request": min(r["measurement_timestamp"] for r in group),
                                     "last_request": max(r["measurement_timestamp"] for r in group),
                                     **statistics([float(r["service_ms"]) for r in group])})
            total = sum(float(r["service_ms"]) for r in selected)
            for stage in STAGES:
                stage_total = sum(float(r[stage]) for r in selected)
                stage_rows.append({"role": role, "profile": profile, "stage": stage,
                                   "mean_ms": stage_total / len(selected),
                                   "fraction_of_total_service": stage_total / total})
        image_rows = read(BASE / "protocol/IMAGE_SPLITS.json")[role]
        payload_rows.append({"role": role, "unit": "bytes", **statistics([r["payload_bytes"] for r in image_rows])})
    csv_write(BASE / "analysis/service_time_summary.csv", summaries)
    csv_write(BASE / "analysis/service_time_by_session.csv", session_rows)
    csv_write(BASE / "analysis/service_stage_summary.csv", stage_rows)
    csv_write(BASE / "analysis/jpeg_payload_summary.csv", payload_rows)
    csv_write(BASE / "analysis/formal_service_requests.csv", [r for role in ("calibration", "validation", "test") for r in role_rows[role]])
    cis = bootstrap_test(role_rows["test"])
    save(BASE / "analysis/test_session_bootstrap.json", cis)
    make_figure(role_rows, summaries, stage_rows, session_rows)
    write_report(summaries, payload_rows, cis)
    return summaries


def bootstrap_test(rows):
    import numpy as np
    profiles = ("CPU-1T", "CPU-4T")
    sessions = sorted({int(r["session_id"]) for r in rows})
    images = sorted({r["image_id"] for r in rows})
    lookup = {(r["profile_id"], int(r["session_id"]), r["image_id"]): float(r["service_ms"]) for r in rows}
    arrays = {p: np.array([[lookup[p, s, i] for i in images] for s in sessions]) for p in profiles}
    rng = np.random.default_rng(20261009)
    repetitions = 2000
    boot = {p: {m: [] for m in ("mean", "P95", "P99")} for p in profiles}
    ratios = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(sessions), size=len(sessions))
        means = {}
        for p in profiles:
            values = arrays[p][chosen].reshape(-1)
            ordered = np.sort(values)
            means[p] = float(values.mean())
            boot[p]["mean"].append(means[p])
            boot[p]["P95"].append(float(ordered[math.ceil(.95 * len(values))-1]))
            boot[p]["P99"].append(float(ordered[math.ceil(.99 * len(values))-1]))
        ratios.append(means["CPU-1T"] / means["CPU-4T"])
    intervals = {p: {m: [quantile(v, .025), quantile(v, .975)] for m, v in by_metric.items()} for p, by_metric in boot.items()}
    return {"method": "paired session-block bootstrap; fixed held-out image pool retained whole",
            "resamples": repetitions, "seed": 20261009, "sessions": len(sessions),
            "unique_images": len(images), "unit": "ms", "95pct_intervals": intervals,
            "mean_service_ratio_1T_over_4T": float(arrays["CPU-1T"].mean() / arrays["CPU-4T"].mean()),
            "mean_service_ratio_95pct_interval": [quantile(ratios, .025), quantile(ratios, .975)],
            "interpretation": "same-machine repeatability conditional on fixed images; not multi-device or cross-day generalization"}


def make_figure(role_rows, summaries, stage_rows, session_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"CPU-1T": "#0072B2", "CPU-4T": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), layout="constrained")
    images = read(BASE / "protocol/IMAGE_SPLITS.json")["test"]
    x = np.sort([r["payload_bytes"] / 1024 for r in images])
    axes[0, 0].step(x, np.arange(1, len(x)+1)/len(x), where="post", color="#009E73")
    axes[0, 0].set(xlabel="Original JPEG payload (KiB)", ylabel="Empirical CDF", title="(a) Held-out request payloads")
    for p in colors:
        values = np.sort([float(r["service_ms"]) for r in role_rows["test"] if r["profile_id"] == p])
        # P(S>x); zero at the largest sample is omitted only on the logarithmic display.
        survival = (len(values) - np.arange(1, len(values)+1)) / len(values)
        axes[0, 1].step(values[:-1], survival[:-1], where="post", label=p, color=colors[p])
    axes[0, 1].set(xlabel="Measured service time (ms)", ylabel="P(service time > x)",
                   yscale="log", title="(b) Held-out service tails")
    axes[0, 1].legend(frameon=False)
    bottom = np.zeros(2)
    stage_colors = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]
    for stage, color in zip(STAGES, stage_colors):
        means = [next(r["mean_ms"] for r in stage_rows if r["role"] == "test" and r["profile"] == p and r["stage"] == stage) for p in colors]
        axes[1, 0].bar(list(colors), means, bottom=bottom, color=color, label=stage.replace("_ms", ""))
        bottom += means
    axes[1, 0].set(ylabel="Mean service time (ms)", title="(c) Timed service stages")
    axes[1, 0].legend(frameon=False, fontsize=8)
    for p in colors:
        rows = [r for r in session_rows if r["profile"] == p]
        axes[1, 1].plot(range(1, len(rows)+1), [r["mean"] for r in rows], "o-", color=colors[p], label=p)
    axes[1, 1].axvline(5.5, ls=":", color="0.6")
    axes[1, 1].axvline(8.5, ls=":", color="0.6")
    axes[1, 1].set(xlabel="Measurement session (5 calibration / 3 validation / 5 test)",
                   ylabel="Mean service time (ms)", title="(d) Same-day session variation")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(BASE / "figures/fig_visual_service_measurement.pdf", bbox_inches="tight")
    fig.savefig(BASE / "figures/fig_visual_service_measurement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summaries, payload_rows, cis):
    test = [r for r in summaries if r["role"] == "test"]
    payload = next(r for r in payload_rows if r["role"] == "test")
    lines = ["# 原始视觉负载与正式服务时间测量结果", "",
             f"生成时间：{datetime.now().isoformat()}（本地北京时间）。", "",
             "## 已完成的范围", "",
             "官方原尺寸 Imagenette 下载、13,394 张 JPEG 解码检查、官方 ResNet-18 IMAGENET1K_V1 权重校验完成。图片角色在计时前冻结，跨角色精确哈希和 dHash≤2 筛选未发现重复。",
             "",
             "正式记录共33,000条：校准20,000、验证3,000、保留测试10,000；另有200条pilot不计入。两配置为同一CPU的1线程与4线程，均为FP32、batch=1、interop=1，每个块独立进程并预热200次。没有训练视觉模型。",
             "",
             "计时范围：原JPEG已经在内存中，随后解码/RGB转换、官方预处理、推理、argmax与字典生成。读盘、网络、排队、控制器、模型加载和预热不计入。未裁剪慢请求。",
             "", "## 保留测试池服务分布", "",
             "| 配置 | 请求数 | 图片数 | 会话数 | 均值(ms) | P95(ms) | P99(ms) | 最大值(ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in test:
        lines.append(f'| {r["profile"]} | {r["n"]} | {r["unique_images"]} | {r["sessions"]} | {r["mean"]:.3f} | {r["P95"]:.3f} | {r["P99"]:.3f} | {r["maximum"]:.3f} |')
    lines += ["", f'保留图片原JPEG均值为{payload["mean"]/1024:.2f} KiB，P99为{payload["P99"]/1024:.2f} KiB。它是原图请求体字节量，不是224×224模型张量大小，也不含HTTP/TLS协议开销。',
              "", f'测试池均值的1T/4T比值为{cis["mean_service_ratio_1T_over_4T"]:.3f}；配对会话块bootstrap的95%区间为[{cis["mean_service_ratio_95pct_interval"][0]:.3f}, {cis["mean_service_ratio_95pct_interval"][1]:.3f}]。这个比值不是预设四倍加速。',
              "", "## 数据与图表", "",
              "- `analysis/formal_service_requests.csv`：全部正式逐请求记录（含来源角色、会话、图片、载荷和阶段时长）。",
              "- `measurements/`：每个独立测量块的原始CSV与元数据。",
              "- `analysis/service_time_summary.csv`：按角色和资源配置的均值、标准差、CV、中位数、P90/P95/P99。",
              "- `analysis/service_time_by_session.csv`、`service_stage_summary.csv`、`jpeg_payload_summary.csv`：会话、阶段与上传字节摘要。",
              "- `figures/fig_visual_service_measurement.pdf`：载荷、服务尾部、阶段分解和会话变化。",
              "- `analysis/MEASUREMENT_AUDIT.json`：逐块、逐行校验与数据指纹。",
              "- `calibration/VALIDATION_REPORT.json`：预先固定容差下的保留验证误差；不使用测试记录拟合。",
              "", "## 解释边界与下一步", "",
              "这是同日单机重复测量，不是多设备、跨日或校园实体部署。会话块区间仅描述固定图片池下的同机重复性；温度不可读、频率没有连续采样，不推断热因果。",
              "",
              "Imagenette/ResNet-18提供真实视觉分类服务负载，不代表校园建筑识别准确率。服务分布验收不等于模拟器FIFO/队列执行验收，也不能据此声称QLA平均/尾部时延收益已经验证。校准模拟与策略比较需另行执行；正文、附录和旧实验未改动。",
              ""]
    (BASE / "analysis/服务时间测量结果_中文.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "summarize"))
    args = parser.parse_args()
    if args.command == "validate":
        validate_distribution()
    else:
        summarize()
