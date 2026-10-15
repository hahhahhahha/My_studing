# 只读人工边界诊断失败（非实际数据／pilot）

184601首次几何自检成功后，实际执行了独立只读诊断，退出码1。没有构造catalog、真实轨迹、模型或权限记录；原源保存在 SOURCE_BEFORE_DECIMAL_TIME_FIX。

明确人工输入：拟合窗[0,0.1]，placement seed1，隔离gap0.1，决策0.05，固定drain0.2；校准从0.1秒开始、16步，验证从1.2秒开始、1步；人工支持[0,10]。校准完整结束应为1.1，加已声明gap应为1.2，数学上精确符合条件，但原浮点加法将其误拒绝。

实际工具输出末段：

```text
  File "X:\Yi-one\workspace\MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts\visual_pilot_preparation.py", line 77, in pilot_role_physical_windows
    raise ValueError('Validation overlaps calibration execution/drain/geometry fitting or explicit role gap')
ValueError: Validation overlaps calibration execution/drain/geometry fitting or explicit role gap
EXPLICIT_ARTIFICIAL_EXACT_DECIMAL_GAP_ONLY
```

修复使用已登记十进制秒的精确有理数，不用容差放宽科学条件、不移动窗口或改变角色。后续检查另测精确边界和真正低于边界，原报告／错误不覆盖。这不是实际数据接受失败，也不是更改负面QLA结果的依据。
