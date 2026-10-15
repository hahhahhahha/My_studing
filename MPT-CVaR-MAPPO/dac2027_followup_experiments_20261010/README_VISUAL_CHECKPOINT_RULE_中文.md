# 共同最终checkpoint登记候选：源码与纯描述检查，不是权重冻结

原视觉方案第10节要求两个学习方法在共同训练／验证规则、预算和条件下训练，再固定权重测试120个cells；它没有指定选最优或最终checkpoint。本入口准备候选“共同预算全部完成后的最终checkpoint”，**不按validation/test时延、奖励或QLA优势排名**，未替用户或pilot把规则冻结。原方案、旧结果、GPU门槛和网络不改。

候选两方法×五种子，10×1000仍是共同10,000环境决策的未冻结示例；选末段只能在全部段完整采集→更新→存档后，不从早停／部分预算／失败段挑最优。每个种子的两方法必须共享paired-design及有序验证日程。验证只核对执行／参数／记录完整，不以较差时延、drop、unfinished或0完成排除checkpoint；全部uncapped完成时延与可靠性记录保留。

`visual_checkpoint_registration.py`纯函数只检查**描述**，不读取模型、不证明训练发生。真实登记类源码要求外部冻结pilot、实际smoke、fresh完整Type-A入口及本规则在训练前登记／源锁；缺一即拒绝。它只消费实际完整预算文件、每段已存archive和正确最终权重的实际只读evaluator及验证文件，核对外部SHA／实际当前参数，全部10项齐全前不能seal。源码在seal前复核已登记输入；一次性锁使用规范化规则+冻结协议身份，不能靠输出名／JSON空白重开。登记输出只标“pending test entry”，不是测试启动授权或最终结果审计。

不加载优化器／H／MV状态，不支持续训／自动重试，不读取保留测试测量或测试结果，不改变神经结构。此类真实构造／register／seal当前从未成功或执行；没有真实checkpoint被选定／冻结、真实验证／训练／测试。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_checkpoint_rule.py'
```

该只读保护检查用明确人工ledger／SHA字符串／不利验证计数，所有“checkpoint描述”都不是权重文件。它不导入Torch、不运行模型／队列、没有真实validation/evaluator，也不创建登记锁／真实冻结文件；缺失实际授权的真实构造必须被拒绝。完整训练、模型加载与只读验证仍需原GPU资源门槛、修复冒烟入口独立复审及cal/validation-only pilot后fresh完整审查；当前作者自检不能替代任何独立审计。
