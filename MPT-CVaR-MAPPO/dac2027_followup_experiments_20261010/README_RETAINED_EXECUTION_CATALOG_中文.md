# 保留测试执行专用配对服务表：仅源码准备

新增`retained_test_service_catalog.py`只在真实构造后用于执行层。没有模块级计时读取；构造首先要求已经完成的`FrozenTestExecutionContract`及外部绑定的pilot、实际两臂CUDA冒烟、完整fresh Type-A源审查和全部10个最终权重／训练验证证据。当前这些条件不存在，**真实构造／查表／策略执行均未发生**。不能用人工描述行或伪造PASS启用此入口。

原`ExecutionServiceCatalog`和`VisualRequestSchedule`仍拒绝test，未改这两份保护源码。新表为未来单独测试日程／执行循环提供同样的执行私有接口；旧策略接收不可变动作前观测，不接收catalog、日程、测量会话或未来服务真值。预测器仍是原calibration-only模型；归一化只准校准池。新接口只导出宽高／JPEG字节的元数据，明确拒绝拟合入口。完成后的过去回执可按既定算法使用，实际未来时长不能进入actor／QL／critic或可行性筛选。

启动、载入结束以及每个cell前后应核对原配对服务／最终审查文件SHA、当前原图片身份、当前完整源／权重证据。原`STRICT_TEST_AUDIT.json` SHA `8c256aad126f3631986bcb99b68bdbb19e494836f990ad9155f20fe6ef9a9d4f`绑定50份CSV及50份会话元数据，最终独立审查SHA保持`e689c1fe565ed0692d6118b59815be92721bcf7bb88dac6a9b961890827826c8`。未来真实载入只取这些注册路径，不扫描其他数据替换。1000图×5原会话×2CPU配置完整配对，一次共同日程指定每请求的同一图片／会话，全方法共享该对；四虚拟服务器中重复资源配置使用同一测量值，不捏造四设备并发实测。

解析器核对角色／身份／SHA／实际JPEG、文件会话／资源配置、有限正服务秒及四阶段之和；缺行、重复、错误状态、非有限值或路径替换立即失败，不平均重复、不补缺、不删慢记录。返回配对数值副本，不能通过调用方修改内部真值。哈希核对在显式cell边界进行，不为每个请求重复重算所有文件；未来完整测试入口必须实际连接`verify_boundary()`，不能把此源码表称端到端权限隔离认证。

本轮自检只读取既有审计的路径／哈希清单和保护源文件，禁止打开真实test计时CSV。配对解析样例是2个虚构图片×2个虚构会话×2资源的8行**明确人工**时长；包含1,000,000秒慢值并保留。未计算原测试分布、生成实际日程、创建真实catalog、运行FIFO／模型／策略或拟合／选择任何参数。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_retained_test_service_catalog.py'
```

后续仍需实际冻结并登记共同测试日程、完整固定checkpoint测试循环、执行边界复核与全部结果导出及独立入口／结果审计；作者解析检查不是独立PASS或120cells。网络方向、pilot H／episode／terminal／load／时长、GPU原门槛和独立修复复审仍待完成；新增源继续使旧smoke计划过期，不重封／重跑。原服务计时／模型／划分／门槛／失败证据／论文均不改。
