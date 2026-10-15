# 保留测试入口独立审查：FAIL（未启动）

审查时间：2026-10-10T19:32:14.7824248+08:00。文档命令原样执行，退出 0；原 CPU kernel witness 与确认 13,200 行/120 块检查通过。独立哈希遍历核对 554 个保护文件，全部一致。原 V2 冻结五会话、50 个 profile-specific 测量块/10,000 行计划及 V1 原 1,000 图测试池保持不变，未测测试服务时间。

启动前阻断项：

1. 同日守卫位于 prerequisites/frozen_guard 公用路径，故 statistics-worker、export-worker 及完成后的只读 audit-all 也被限制为原测量日期。后续日期核查有效的同日测量记录会被拒绝。应仅约束测量启动/执行，并在审计中核对原记录日期。
2. 可直接调用的 export-worker 只验证完整性与计划绑定，不验证 TEST_REPORT.accepted 或六项通过。因此有效原始行但数值失败的报告仍可导出带 complete 含义的文件。正常 run 的失败停止路径正确，但独立导出入口缺门禁。

本次只写入口审查档案；没有修源码、重封存、训练、模拟或测试计时。未来需修订新增入口并重新冻结/独立审查，原实验协议与失败/确认数据保持不变。

当前绑定：

- continuation SHA256：`06babb5dd485eda38ad41aabff65ce4c4d428d68c0bb76b76beecb46ce056a7a`
- 入口源码 SHA256：`1e60e73a23d228efa81a5729aedf345dd69f3d03f41bc7c068762bbb9e83ccab`
- 原 PLAN SHA256：`bbddc7b6e15e68625deb41992707d64bf32faa184910de8d8c01fbab1a64f282`

此结论仅针对入口；不代表未来测试结果失败或通过。完整命令输出、文件核对与限制见本目录 JSON 档案。
