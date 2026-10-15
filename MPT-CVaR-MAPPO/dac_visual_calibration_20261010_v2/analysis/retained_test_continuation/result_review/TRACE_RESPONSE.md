# Reviewer final response — trace-only archive

The following is the final audit response delivered after the resumed review, copied verbatim from the available conversation. This addendum does not modify the audit reports, their hashes, or any original experiment evidence/state.

---

独立审计完成：总体 **WARN**；记录完整性 **PASS**；测试六项冻结门槛 **PASS**。无阻断性发现。仅新增审计文件，原证据与最终状态均未修改。

| 配置 | 指标 | 原始实测 ms | 独立重算 95% CI |
|---|---|---:|---|
| CPU-1T | Mean | 30.928228 | [29.995592404, 31.796249718] |
| CPU-1T | P95 | 35.8829 | [35.2080025, 36.46892] |
| CPU-1T | P99 | 45.1566 | [43.3202, 50.7459075] |
| CPU-4T | Mean | 13.05232274 | [12.7430802645, 13.3656134045] |
| CPU-4T | P95 | 16.4708 | [15.57734, 17.6302075] |
| CPU-4T | P99 | 28.8099 | [27.09397, 34.356155] |

六项误差依次为 8.6329%、5.2769%、9.8794%、6.4338%、4.0296%、0.8934%，均满足原 10%/15%/20% 门槛。

已独立验证：

- 全部 23,200 条新记录、170 块、完整 CSV 逐字段逐顺序一致。
- 从 20,000 条 V1 校准记录重建全部配对模型；重算全部 1,000 个测试邻居集合，测试耗时未参与拟合。
- 262 个 V1、558 个确认保护指纹及 5,000 个 JPEG 指纹匹配。
- 五会话间隔均约 905.36 秒；日期、空闲门槛、元数据、任务日志一致。
- 测试和确认各 500 次 bootstrap 端点均独立重算一致；PDF 24 个柱形和 12 个标注、PNG 图像核对通过。

主要 WARN：

1. 冻结浮点 CDF 取第 4751/4951 位，精确经验秩为 4750/4950；最大差 0.0906 ms，精确秩下仍全部通过。未替换原估计器。
2. RUN_STATUS 文档过时；测试结果第 15 行复用了确认加权描述。
3. CI 仅描述实测分布，不是模型误差 CI；单机同日、少会话及稀疏格不支持视觉质量、队列、部署或 QLA 收益结论。
4. 本次是 Type-A，不是跨家族认证。

报告：[EXPERIMENT_AUDIT.md](X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac_visual_calibration_20261010_v2/analysis/retained_test_continuation/result_review/EXPERIMENT_AUDIT.md)

SHA256：

- MD：`795ef05a8a189ba81548a175d5c37858582a8c3e6c3c92bd43cb6c79f9d54418`
- JSON：`e689c1fe565ed0692d6118b59815be92721bcf7bb88dac6a9b961890827826c8`
- 独立重算 JSON：`b8ce31fbd254240e08b48938be2a560d88caca595fd2eea4628b0aa520e1b23b`

完整文件指纹见同目录 `REPORT_HASHES.json`。
