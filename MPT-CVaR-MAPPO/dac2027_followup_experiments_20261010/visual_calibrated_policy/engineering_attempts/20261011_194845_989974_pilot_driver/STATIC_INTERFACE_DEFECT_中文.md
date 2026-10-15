# 真实路径尚未执行时发现的静态 producer/consumer 接口错误

2026-10-11 本 wake，首次人工身份／AST 接线检查 `RESULT.json` SHA `9c0657d6763ba33dca3987e03bf220b6783c3787af8817a952f70fb0f56608a0` 成功，但它未核对 collector 的实际输出键。随后直接读取原 `visual_pilot_episode.py` 的完整 marker 构造发现：producer 注册键为 `files_sha256`，新 driver consumer 却读取 `output_files_sha256`，未来实际成功 episode 后会抛 KeyError。没有运行 driver、加载模型或收集工程 episode；没有真实失败臂或数据被改写。

首次成功报告不改，三份当时源码按报告指纹保存在 `SOURCE_BEFORE_PRODUCER_CONSUMER_FIELD_FIX`。修复仅新增 driver consumer 键，并补 boundary 中实际 method report 的参数指纹对照；原 producer 与所有历史源／数据／论文／模型／分割／环境／门槛不改。

加严人工检查解析原 producer `complete=dict(...)` 与 driver `complete[...]`，核对全部字段集合。保留错源的 `output_files_sha256` 和内存 AST 人工缺键均被拒绝；不构造真实 MODEL/checkpoint/smoke/PASS/权限或 actual collector stub，真实路径仍零执行。后续独立完整 review 仍必须进行，此静态错误修复及作者自检不得充当实际模型／运行保护验收。
