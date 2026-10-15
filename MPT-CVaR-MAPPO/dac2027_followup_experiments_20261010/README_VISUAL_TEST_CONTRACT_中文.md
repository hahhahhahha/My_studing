# 固定checkpoint共同测试合同：元数据／源级准备，不是120个结果

合同准备原矩阵4方法×2负载×3名义延迟（0、P、4P）×5种子。每个负载／种子只绑定一份共同外生日程，跨四方法和三个延迟共用图片、实际到达、配对测量会话及分配uniform；这里的SHA绑定不是已经生成完整测试日程。所有方法使用共同固定动作窗和drain，不按完成10,000任务早停，不因drop／unfinished延长drain。原方案`max(120s,10000/lambda_low)`只实现为未冻结候选起点，以校准／独立pilot的lambda输入向50ms向上取整，未使用测试完成数选择长度。

新增原保留图片**元数据**池只核对原`IMAGE_SPLITS.json`、`IMAGE_MANIFEST.csv`及冻结服务PLAN的既有SHA，保留原官方val的1000图片和跨角色身份／哈希隔离；只返回宽高／实际JPEG字节。类别、标签、未来实际时长或抖动不进入可观察接口。该池明确拒绝服务时长查询和拟合／归一化API，**不读取任何测试计时CSV，不拟合、不统计测试大小来选场景**。图片身份检查不等于权重已经冻结或测试已经启动。

真实合同构造源码须先满足外部冻结pilot／实际smoke／fresh完整Type-A入口、在测试前登记当前矩阵／源、两方法×五种子全部最终权重的外部SHA登记及未变的训练／验证证据。当前这些前提未满足，真实构造不能成功。合同本身没有执行真值API、模型加载或run方法；真实配对时长执行适配器、完整策略测试循环和结果导出仍待接入／审查，不能把合同当端到端120cells完成。

网络、rho、动作／drain时长和物理移动窗仍未pilot冻结，本轮不选择新网络条件。检查用人工低／高请求率与人工SHA描述，原1000图片元数据仅做身份和访问边界核对。30用户／4虚拟服务器／50ms决策／100ms广播仍为工程候选，2MHz不改。任何实际测试只能在共同校准／验证规则及权重冻结后开始，测试结果不得反馈拟合或checkpoint选择。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_test_contract.py'
```

命令只核对元数据身份、明确人工120行合同及篡改／缺失授权拒绝，不导入Torch、不跑队列／模型／训练／策略测试，不读test服务真值。作者自检不替代独立复审；原独立源FAIL、GPU门槛和待用户的网络／Alibaba协议方向不改变。
