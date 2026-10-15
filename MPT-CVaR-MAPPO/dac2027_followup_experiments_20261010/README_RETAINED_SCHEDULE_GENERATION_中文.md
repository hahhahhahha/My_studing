# 共同请求日程生成／登记（仅源码与虚构元数据检查）

`scripts/retained_test_schedule_generation.py` 实际构造／生成／登记均0。没有采样原1000保留图片、打开test计时CSV、读取服务标签、运行模型／队列／策略或冻结正式协议。本轮只在**明确虚构图片ID／人工负载／虚构poolSHA**上检查随机数与描述边界，不是实际测试日程、模型行为或补造服务记录。

生成源保持现有校准生成器的NumPy2.2.6 PCG64／`SeedSequence(seed).spawn(5)`配方：计数和burst、到达offset、图片、实测会话、分类uniform分别用五个隔离流；固定step→user→task生成，按真实arrival／user／task存档。burst概率和倍率修正基础强度，使声明的是无条件总请求率；每用户没有7请求截断、按表现过滤、补齐或重采样。两负载共享种子配方，计数不同会消耗不同长度的独立请求流，**不称两负载独立随机重复**；每个负载／种子各有完整不可变日程供四方法×三延迟共同使用。原工程生成器的test角色拒绝未修改。

实际生成只准使用原完整1000图片的元数据身份与全部五个测量会话均匀抽样。每task存arrival、image ID、session、assignment uniform；原JPEG字节／宽高由原身份表确定，两个配置的实测服务配对由执行层按同一image/session查找，不在各策略运行时顺序抽样。移动位置／链路／资源标签／广播规则仍由最后公共环境指纹锁定；生成器不含策略／延迟字段，不知道路由、未来服务或结果。

## 非循环冻结顺序

不能要求生成器先消费“含这十份尚不存在日程SHA”的最终测试合同，也不能在十份权重选择完后改写它们所绑定的协议。准备采用：

1. 原资源／源码复审允许后完成真实两方法smoke；完成cal/validation-only工程pilot，按工程依据冻结日程负载、horizon／drain、burst和物理网格。
2. 独立Type-A只对“生成共同请求元数据”提供准备权限，绑定真实pilot、smoke、完整当前源与保护输入；不准模型训练／测试。该权限与正式完整入口明确分开，目前两者均不存在，不伪造PASS。
3. 永久科学设计锁下生成十份共同请求文件和字节／语义SHA登记。完整产出也只pending正式完整入口，不是120个结果、环境READY或complete_audited。
4. **再**将十份日程／矩阵与全部训练／验证设置、原输入纳入唯一完整正式协议，fresh完整Type-A入口通过后才进行十份共同预算训练／验证／最终权重登记；权重与最后120项测试始终使用这一不变协议绑定，不因生成日程而重封权重。

这只是待独立检查的源码阶段安排，不授予当前执行权限、不改原负载／网络／split／门槛。准备审查要求模型gpt-6-astra、xhigh、Type-A、无阻断，scope为`retained_metadata_schedule_generation_only_after_cal_validation_pilot_and_actual_smoke`；只准`generate_common_request_metadata_only`。旧基础审查、原FAIL或作者人工自检不能替代它。实际smoke报告／完整源树／原保护输入／真实cal-validation工程证据在生成前后重哈希；实际池身份不符立即停止。任何不利校准／验证诊断仍保留，不按测试标签或QLA收益选参。

永久一次性设计锁排除输出名称／审查装饰路径／JSON编码；失败保留已写日程、输入与RNG状态指纹，不自动追加／换目录重采样／删除锁。十份文件不是匿名数组：每份有完整schema、全部请求、语义SHA；登记表另记录文件字节SHA、五流前后状态SHA、实际生成数量、固定horizon／drain候选矩阵与输入文件绑定。实际请求数0亦保留；尾部样本不足以最后可靠性指标解释，不按完成数延长测试。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_retained_test_schedule_generation.py'
```

本命令只检查十份虚构元数据描述的重复确定性、与旧校准配方的逐任务一致、图片流变化不推进其余流、全局NumPy随机状态不变、空请求和人工高率无7请求cap、缺失准备权限拒绝。没有真实已完成smoke／pilot／审查证据，真实准备权限解析／输出登记保护未实际测试；不伪造这些证据来冒充测试通过。正确fresh模型加载与完整网格启动源仍待。GPU门槛、原源FAIL／独立修复复审PENDING以及网络／Alibaba协议的用户方向不变，旧smoke计划仍过期不重封／不运行。
