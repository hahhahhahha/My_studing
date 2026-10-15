# 人工解析参考的 dtype 错误：不放宽数值标准

首次人工检查在手算两步GAE与纯数组GAE比较处AssertionError。FAILURE.json SHA `c2b578663774f82c46d93e0c4fd1795f228a10de9b67763030756881f4869141` 与当时三源保留；实际模型／registrar／NPZ／catalog／FIFO调用0，未打开test timing CSV。

只读独立Python进程对原显式人工QLA数组诊断：primitive value dtype float32，gamma×value仍float32；作者解析参考先乘后加，误差 `2.2120060982899759e-07`。原per_user_temporal_gae在计算前把rewards/value/nextvalue全部转float64。把人工解析参考value显式转float64后，最大差异为0，**原atol1e-15/rtol0仍通过**。

仅修改作者check中的解析参考转换与失败证据保护；登记source、原GAE、原模型、服务统计／原数据／任何科学门槛未改，没有削峰／裁剪／不利结果删除／数值标准放宽。不是实际策略接受失败，更不是QLA收益。修复后另新检查目录，不覆盖初失败或其源快照。
