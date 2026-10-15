你是一个全新上下文、未参与该系统实现的 Type-A 独立完整入口 reviewer。用户授权对正式 E1 前置入口做真正独立复审。GPT 同家族审查只能称 Type-A，不能称 Type-B 或跨家族认证。主执行者曾参与实现，所以你的判断必须直接来自文件，不能依赖执行者总结。请审查下列路径，默认所有作者主张都需要验证，不带结论预期。不要修改源码、科学设置、旧计划/结果或状态；只在指定新 review 目录写完整报告、machine JSON、有限功能验证输出及审查 trace。禁止启动 E1 正式矩阵、训练、重下载数据、杀未知进程、全局安装/重置。若需要有限功能驱动，使用现有功能配置在新路径做一次真实运行并正常关闭本次已核 PID；预先查看当前端口/源身份，若占用保留阻断，不能关闭他人服务。不得自动追加/合并部分失败重试。若需新的 authority 或来源不完整，报告具体阻断。

根目录：X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013

## 本次复审的背景（上一轮 FAIL 的两个阻断已由执行者处理，需你独立验证）

上一轮独立复审 `reviews/entry_v2_independent_20261014/REVIEW_RESPONSE.md` 结论 FAIL，两项阻断：

- **B1 截止双结局竞态**：执行者已修改 `systemlab/client.py`（把"写 unfinished_at_cutoff"的循环移到 `pool.shutdown(wait=True)` 之后，使每任务恰好一条正式 outcome），并新增 `tests/test_cutoff_race.py`（3 个回归：锁前暂停/正常完成/截止后响应）。你需要独立验证该修复正确、回归真实覆盖竞态窗口、且未引入新的结局语义变化。
- **B2 历史 common.py 缺口**：执行者已从 `engineering_repairs/20261013_three_logic_fixes/BEFORE/systemlab/common.py`（SHA `d72aca9e…`）把 `source_receipt` 恢复为原始 `rglob('*.py')` 版本，重建后 SHA 精确等于 M1 收据绑定的 `6cedd7b3…`，保存为 `legacy_source/systemlab/common.py`（恢复报告 `legacy_source/RECOVERY_REPORT.json`）。你需要独立验证该恢复的 SHA 匹配、与当前 `systemlab/common.py` 的逐行 diff、以及"恢复仅影响 source_receipt 范围、不影响 M1 校准数值链"这一结论。

由于 B1 改变了 `systemlab/client.py`，当前待审身份是 **M2d / frozen_v3 / E1 v3**（不是上一轮的 M2c / frozen_v2 / E1 v2）。旧的 M1/M2/M2b/M2c、frozen_v1/v2、E1_grid/registry v1/v2、entry_v2 复审及其失败记录均原样保留，不作为新源码的批准依据。

**本轮复审期间执行者的两项补充修正（也需你独立核验，不是替代你的判断）**：
- **测试覆盖差异已修复**：`tests/test_cutoff_race.py` 原先用 pytest 函数风格，而 `tools/verify_unit.py` 用 `unittest discover` 只收集 `unittest.TestCase` 子类，导致 3 项 cutoff 回归会被 unittest 跳过。已改写为 `unittest.TestCase` 风格（`CutoffRaceTests`），现在两个 runner 都能收集运行：`verify_unit.py` 报 `Ran 21 tests ... OK`（含 3 项 cutoff）、pytest 报 `3 passed`。`tests/` 不在 `source_receipt` 范围，故不改变 frozen_v3 身份。
- **校准链影响已验证**：`legacy_source/CALIBRATION_CHAIN_IMPACT.json` 记录 AST 级逐函数/类比对 `legacy_source/systemlab/common.py` vs `systemlab/common.py`——校准链 10 个单元（Journal/request_json/sha/runtime/quantile/tail_mean/local_windows_runtime/read/write/encoded）全部 byte-identical，仅 `source_receipt` 不同，因此恢复内容不影响 M1 校准数值链。

## 请直接完整读取

- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/README.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/docs/AGENT_HANDOFF_中文.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/docs/EXPERIMENT_PLAN_中文.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/docs/DATA_PLAN_中文.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/docs/EXPERIMENT_TRACKER_中文.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/docs/SECON_BASELINE_SEARCH_20261014_中文.md
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/systemlab/ 全部 .py（含 client.py、admission.py、common.py 当前版本）
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/tools/ 全部 .py
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/tests/ 全部 .py（含 test_cutoff_race.py）
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/drive_scripts/run_m2d_validation.py、run_freeze_v3.py、run_e1_prep.py、run_interference_calibration.py、run_b2_recover.py
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/configs/ 全部 JSON
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/frozen_v3/{FROZEN_SYSTEM_PROTOCOL,FORMAL_CONFIG,FEATURE_SCALES}.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/formal_actor_v3.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/E1_grid_v3.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/E1_test_registry_v3.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/validation_registration_v1.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/protocol/calibration_scales_v1.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/legacy_source/systemlab/common.py 与 X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/legacy_source/RECOVERY_REPORT.json（B2 恢复）
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/runs/M1_rr_s1_attempt_01/ 全部实际输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/runs/M1_stress_s1_attempt_01/ 全部实际输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/runs/M2d_qla_s101_attempt_01/ 全部实际输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/runs/M2d_score_softmax_s101_attempt_01/ 全部实际输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/runs/M2d_stale_shortest_s101_attempt_01/ 全部实际输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/data/calibration_images.json、validation_images.json、test_images.json、E1_common_v3/ 全部日程
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/interference_calibration/PLAN.json、REPORT_中文.md、全部 run_*/edge*.json
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/interference_calibration_affinity_attempt_01_broken/ 全部计划/错误及部分输出
- X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013/engineering_repairs/20261013_three_logic_fixes/{REPAIR_REPORT_中文.md,RESULT.json,UNIT_RESULT_attempt_02.json,functional_4edge_attempt_02/,BEFORE/}
- 从上述绑定直接定位并核对源图片、实际权重、checkpoint、历史可执行源、隔离环境依赖；不能只检查 MD 声称的 SHA。

## 审查清单（A–I）

A ground-truth/proxy 来源及 service 真实执行，不以模型输出制造 reference。
B 自归一化/截断/失败样本选择、真实 metric 定义、CVaR 分数质量及配对 run 统计。
C 所有主张对应实际文件/字段/数值，源码/计划/角色/原始 JPEG bytes、checkpoint 及 helper 闭包启动前/结束连续性，重算 M2d 结果。
D 三修复回归覆盖、B1 截止竞态修复是否真正消除双结局、每任务唯一次结局、FIFO 等待/active 与守恒并发、dead code。
E 实际 scope vs 声称，零样本 35→4 迁移映射/归一化/softmax/域迁移；fair actor/baseline inputs、current cache 而非 fresh 私有 queue、common schedule/uniform、同 resource/drain/初态、未用 test 挑设置。
F 真实本地 HTTP/UDP、应用层 delay/loss 注入 vs 真实无线网络范围；budget per recipient、监测开销、驱动干扰、负载发生器 backpressure。
G Windows wrapper/runtime PID、句柄错误 fail-closed、自有进程 shutdown、不杀未知进程；单机共享物理资源、线程≠quota、affinity≠完全隔离。
H 干扰标定严格配对性与因果边界：相同图像顺序/预热/线程、并发启动屏障、时间重叠、样本 CPU 与墙钟定义、三重复/顺序登记、拓扑获取与 SMT 归属、成功与失败尝试隔离、是否真正测 CPU 时间/CPU 墙钟比；"唯一因果/缓存睿频/保守估计"主张是否得到直接证据，不能从相关性替代测量。
I frozen_v3 当前身份与实际 M2d、v3 registry/grid/scales/全部日程/actor/generator binding、具体 cell 准入、review 校验/缺项拒绝、启动前/末尾 fresh 检查；B2 恢复的 common.py 与 M1 收据 SHA 匹配及影响分析。

## 关键 artifact 实际 SHA（供核对，勿改）

| 文件 | SHA-256 |
|---|---|
| E1_grid_v3.json | `6bf149fe674e4f996e39e0bc7d05607d100a3870ab6d47f01250ca55ba702ab0` |
| E1_test_registry_v3.json | `51df80fba22c1f75ef30e064eff152a157988cc42a3e949790bca20bcceaa2fd` |
| actor checkpoint | `3444f039eb7359d37000ef0045ccd3efb859bb51ff1564b0c814e40d50213044` |
| calibration_scales_v1.json | `2f9e813a18797732ebf85565c650107db3d14b567d47ada1910a0470c6727f5e` |
| data/E1_common_v3/common_01_s1.json | `8f09c0c2fa21c8b86718a402fd5e4c7de2efd9e7ab12d2541cd16187e4fae0c9` |
| data/E1_common_v3/common_02_s2.json | `56ddc44bf8197ca31563db66cd556bc77b16e5f91dca02d072bcedd4a95b5186` |
| data/E1_common_v3/common_03_s3.json | `b66e9996d4b78d4a8432a4645c1d0193176989eda79fe427dbef47af3a575528` |
| data/E1_common_v3/common_04_s1.json | `65bf45be310add8449ba0cbd4b9c07b63f0d9854fcc3c08e4dd3ba126c57af7c` |
| data/E1_common_v3/common_05_s2.json | `449d0e6082a47a73ff50de6e46b942496decbea0fda31f9ffb271b9717058fa8` |
| data/E1_common_v3/common_06_s3.json | `65de243d09c033ea7ca48f527f65ae1fd14accb42f30d4f311f60b2bb217c566` |
| data/calibration_images.json | `b082bc5708db00c6dd6fb1f0fd0897f79d15ae84b2849bbaf15ac85ad08782ed` |
| data/test_images.json | `d8355dbaf2aec4a82698da4642ba6b2b466ec993269ec895b7cb99bd5e5b948a` |
| data/validation_images.json | `69c01f12b194e8ef67e7880d76f6f318c929b96de5d49195757a7742c7af223c` |
| formal_actor_v3.json | `913dc6c6f9248b975c8c34324b45db3017552df6284d92cc2c62f75bb5997521` |
| frozen_v3/FEATURE_SCALES.json | `d68046e2f7cd208e7c66e2b698caeeb76b77bba834bbd812ba92b2e7e3971572` |
| frozen_v3/FORMAL_CONFIG.json | `6026d8d8c616294fe79ea47fe7ca62670a584b44898ab4027aa42ce05baee64b` |
| frozen_v3/FROZEN_SYSTEM_PROTOCOL.json | `cf7acc96346a2aeb400ebf3638abb6c64a5c185d6b2137ce1ba648c6f2cde46d` |
| legacy_source/RECOVERY_REPORT.json | `23ca98f6f9b0e9cc1506db818ce54f392d2d1fb72a6c690dbfcf1b6b3bcbcbf8` |
| legacy_source/systemlab/common.py | `6cedd7b3590da61d1a3e3ee12f20f6b5b0154954d5a0e8a5b54cc6418183a0d6` |
| runs/M1_rr_s1_attempt_01/client/COMPLETE.json | `7b0b1ad75d0a1acce36df0407568f10d22ee0952115b76f27385c6c39c7b4f8d` |
| runs/M1_rr_s1_attempt_01/client/INPUTS.json | `d2dafaeaa8cd9a72acb746e7cd1de7a689d11e0317938d5e8537752355150e71` |
| runs/M1_rr_s1_attempt_01/client/tasks.jsonl | `5f2f3c90db9067f1b638a4bb42e19ef1d2c7ee15ccada292a6124c1ff35d31e6` |
| runs/M1_stress_s1_attempt_01/client/COMPLETE.json | `f198bb2acc1d5725183140e5a0d4a8d24b46dff4446fb6a8d333bc07d5611375` |
| runs/M1_stress_s1_attempt_01/client/INPUTS.json | `98e81e7a17ee1fe0a0aaff29ff907f1660254cabd5750c7db65deaa9be31b894` |
| runs/M1_stress_s1_attempt_01/client/tasks.jsonl | `2055742a4b5c464af6fbd50fc943a071c31b35f8d0e1f60cd6a2a5ebd92d63a4` |
| runs/M2d_qla_s101_attempt_01/client/COMPLETE.json | `7b54bb199a5c532bf50a853ebdc09c682bfa3b4b843f18083ccd22d8937b35ff` |
| runs/M2d_qla_s101_attempt_01/client/CUTOFF.json | `f98262d3724a6d19eb15e9bc76cb1e1de801aa112f41c06b41d8b2ac44032e54` |
| runs/M2d_qla_s101_attempt_01/client/INPUTS.json | `c6f679290f73c7a90da53049d788dcb3ac7b3ef93ce94aea2bf46af8e27f1924` |
| runs/M2d_qla_s101_attempt_01/client/decisions.jsonl | `af3ecd8568e1167fcdc7e3d19ef92eeb52331c82d16a11c51ee89292da9cbed6` |
| runs/M2d_qla_s101_attempt_01/client/tasks.jsonl | `71d78259795378799ce982aa6001184f6928b00b15229b5b4626721fa665ee8d` |
| runs/M2d_score_softmax_s101_attempt_01/client/COMPLETE.json | `8507f0b5088ec3855231cd0d2da960be2855263d2fc1fe1b8a63dc22be91741f` |
| runs/M2d_score_softmax_s101_attempt_01/client/CUTOFF.json | `1bc7a73c0a3ac6fe1b13030c767fc5e57874962c11d954f0b7a87bb6f8f8b5b7` |
| runs/M2d_score_softmax_s101_attempt_01/client/INPUTS.json | `45acce13f1f645e1406256e411337ce57ae09be97204ca91d22fcbaca939009e` |
| runs/M2d_score_softmax_s101_attempt_01/client/decisions.jsonl | `1d106e0aadb2d1a064392e613b0aac9605b454de79efe0af8450cb5effd3f1b7` |
| runs/M2d_score_softmax_s101_attempt_01/client/tasks.jsonl | `dd108733f6bb06af066fed0336b2ab07afb0a73849868993cf0e13e94a7c96a2` |
| runs/M2d_stale_shortest_s101_attempt_01/client/COMPLETE.json | `5a074b3044ca9aba3de0dafa3ca81749489804a63613a9855373d66565d638d3` |
| runs/M2d_stale_shortest_s101_attempt_01/client/CUTOFF.json | `661903cdc112e1865c2a0f3552e2a6e7c815037fbc3af95a9bdd656f62904997` |
| runs/M2d_stale_shortest_s101_attempt_01/client/INPUTS.json | `19b2109953226bb16695004e57a1349402f9d4c92c0623ae541958678d7bd9ba` |
| runs/M2d_stale_shortest_s101_attempt_01/client/decisions.jsonl | `d3e76b69f1dcdb603f080e6ab75cd79b6bb23d9ecf29a82ca07e418c2007cbb2` |
| runs/M2d_stale_shortest_s101_attempt_01/client/tasks.jsonl | `8ed897e64aa41c2713152ec5fec5a3d0757af4736b3d6d00e12484b81bba53f9` |
| tests/test_cutoff_race.py | `826d42a12cc4e4c91df22c496979ebd85f4f78d725aeb3cc0dcecc002a9b3b11` |
| validation_registration_v1.json | `1e2311696e86eec07beb9be7f2b893b0eb8ebff36b2722955ced6cd417f7fdb9` |
| vision weight | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

当前 source_receipt 共 19 个交付源码文件，其字典如下（machine JSON 须原样引用 `source_sha256`）：

```json
{
  "systemlab\\__init__.py": "a590a53454b85f9630437329607c03ea3ee3a125aae2868ac7ff7ffe420e7f44",
  "systemlab\\__main__.py": "10ab8eb03edf6e6164f7568099be3ca5c020f991767b80cb47c1d29ddb87a042",
  "systemlab\\admission.py": "5c6bdd3b8086adb464a25e10c61ce4337fc9ff13a3ade64c975b7b0aca4ff96c",
  "systemlab\\analysis.py": "19402d7f475fa09a86cce0d0f785138a78db79bd0a082238ca7ff2cd4b468bc5",
  "systemlab\\client.py": "09e9cba325d4657ad04d58f7b75cee2e82dd7aed706576fe1ccfc9b89fbef869",
  "systemlab\\common.py": "2c1fdafd23653afd8938166c3ec9e03c094b789212645758b49af5622f09483c",
  "systemlab\\data.py": "939f2a67577b44f6a54c5bba63f4007ac036a1e472874647a8580636c7265e73",
  "systemlab\\edge.py": "4475962034043cbfc6e8811c1d8a04bbb0891a3b4aabdd1744d5ed6e85681171",
  "systemlab\\feedback.py": "97ac868dcbae31d2efd660458cfd316aaae2024213bd677bda8cb50508f9c4f0",
  "systemlab\\launcher.py": "9bd05356f9d44939a95631475b7dadc8c1a8ea3997d8f659974490a142113485",
  "systemlab\\policy.py": "6db8dbeb68ce7d3cedf0a0b36f80bdbc2a6ffe066b8f27f95f790d6706edae78",
  "tools\\build_manifest.py": "399fb4de86f9db7940ec0be4da83a2b5737c94d31af85eb6e8232dd5694af0bc",
  "tools\\check_actor_transfer.py": "7197f27272d0571cec9a84f9de27736ea29f1ee48b95addbf7f63df2b9917c18",
  "tools\\configure_pilot.py": "0a484202c34b6d6fb0adec21ba215905602f07b685aaa99d3e34880211c6ae42",
  "tools\\interaction.py": "c5f7f57c1b4e75fac6dc3ec7092c8b87d6ca409049d54e12ed20073e715b18da",
  "tools\\make_grid.py": "4ab6ca42b436e3664a4f3c45851c44db308db133c4a72d718b23636abd8162aa",
  "tools\\register_test.py": "5365246fa130c70e16f930d96faf6d50ab1aa2470a755f9c4cdfa36aa04e3f74",
  "tools\\stop_owned_attempt.py": "2285c3535c0c1e763b8de9067b9ecf4a21d4d0e93d62b3e53ef4b704e0098c95",
  "tools\\verify_unit.py": "cbb76d608941e32fa9ed1e0367dbf970d08f530a1294b79285e3d8f1ef695cdf"
}
```

## 输出到新目录：X:\Yi-one\workspace\MPT-CVaR-MAPPO\qla_system_prototype_20261013\reviews\entry_v3_independent_20261014

1. `REVIEW_RESPONSE.md`：完整中文报告，各项 PASS/WARN/FAIL、确切 file:line/数值/证据、blockers、warnings、允许/不允许主张、有限复现的完整 command/路径、来源读一致性限制，不声称 Type-B。
2. `REVIEW_RESULT.json`：入口可读 schema `systemlab-independent-entry-v1`、scope `complete_system_entry`、status PASS/WARN/FAIL、blockers 数组真实完整、complete_report、reviewer_identity、`protocol_sha256`（当前 frozen_v3 actual）、`test_registry_sha256`（当前 E1_test_registry_v3 actual）、`source_sha256` 完整实际字典、report_path 绝对、report_sha256 完整报告字节；补 audit_checks、warnings、independence_type='Type-A'、independent_context=true、cross_family=false、verified_execution_receipts。若 FAIL 或未完成同样保存真实状态，禁止填空 blockers 冒充审批。
3. `EXPERIMENT_AUDIT.md`/`.json`（可由同一完整报告派生但不能掩盖不同 scope）；保存本 review 完整 prompt 及 response 用于 .aris forensic tracing。报告文件使用 apply_patch；历史来源只读。不要新增 Python 到 systemlab/tools 或改 source_receipt，因为那会改变待审身份。有限验证用现有脚本输出到 review 目录。你不需要寻论文；只审这个入口。

结束给主 agent 完整结论与所有产物路径、SHA、执行 receipt、未解决 blockers，不只提供摘要。
