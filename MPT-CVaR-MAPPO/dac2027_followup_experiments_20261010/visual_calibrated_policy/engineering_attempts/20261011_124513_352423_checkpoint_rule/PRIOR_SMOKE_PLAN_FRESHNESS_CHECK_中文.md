# 旧冒烟计划鲜度保护核对

本轮新增两个脚本，使当前完整源树不同于12:11准备计划；没有修改／重封该旧计划，也没有启动run或_worker。

隔离Python仅导入进程内Windows初始化和只读计划检查，读取旧计划原始SHA `094caefc9bc78a689769ff56317492364011e35d441f3d2ea6bb1b06e2cf521c`。当前`read_plan`实际拒绝，输出：

```text
OLD_PREPARED_PLAN_REJECTED_AFTER_ADDED_SOURCE_DEPENDENCIES Exact complete prepared source/input/settings schema required
```

退出0表示预期拒绝断言通过；同时断言Torch未导入。不是实际smoke成功，没有模型／队列／权重工作，也未创建smoke锁。原独立FAIL与修复复审待完成状态不变。日后执行前必须重新准备当前完整源身份并完成独立复审，不复用旧计划、同锁重试或拼部分结果。
