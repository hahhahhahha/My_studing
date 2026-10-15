# Windows人工夹具路径故障保留与有界修复

首次自检14:49:04在`_claim_once → _write_once → Path.open('x')`报FileNotFoundError，目标为本attempt下`EXPLICIT_ARTIFICIAL_FILE_LOCK_FIXTURE_NOT_FORMAL_LOCKS/<64位身份>.json`。父目录已创建，路径过长；发生在人工锁文件创建时，未运行任何实际模型／队列／策略，也未打开原test计时CSV。FAILURE.json及已创建空夹具目录保留，不删除／续写。

当前三份源／说明已Copy-Item保存到SOURCE_BEFORE_PATH_FIX，SHA在新检查入口固定绑定；正式evaluator源及锁逻辑保持不变。只把人工I/O夹具的长标签目录改为ART_LOCK，并另起attempt。短人工目标完整路径239字符，正式锁候选183字符；不改全局Windows长路径设置、网络／驱动／Anaconda。额外实验统计或科研条件均未改变。

该修复是文件路径软件故障，不是数值失败；人工质量守恒／时延夹具不冒充实际结果。正式入口仍未满足GPU／pilot／实际smoke／独立复审／最终权重与日程门槛，独立源FAIL原文保留。后续真正端到端入口仍须fresh独立源审查，作者夹具通过不标READY或120结果完成。
