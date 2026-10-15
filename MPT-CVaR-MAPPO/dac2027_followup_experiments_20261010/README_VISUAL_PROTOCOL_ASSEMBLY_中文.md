# 最终公共协议装配（仅源码／人工环境身份检查）

2026-10-11 源级顺序修复：pilot 内的 `test_environment_templates_by_load_seed` 仅冻结完整环境条件，**必须缺少** `schedule_sha256`，不接受未来占位／提前生成或旧日程SHA。先真实 cal-validation-only pilot＋独立元数据生成权限，后实际共同请求生成；原同一成功 generator 及外部指纹核验后，装配器只把实际十份共同日程的语义SHA机械加入深副本，再派生30条件。其余字段原样保留，不重写pilot、改条件或读取测试标签／结果。避免要求pilot先知道尚未获准生成的日程指纹。完整不可变正式协议仍只在实际生成后、正式训练及权重选择前装配，仍须fresh完整Type-A，当前实际构造／生成／冻结／模型工作0。改前源码和旧成功报告已保留；这是源码修复，不代表真实pilot运行或独立通过。

`scripts/visual_protocol_assembly.py` 没有真实构造／装配／写出正式冻结协议，没有真实pilot／smoke／最终入口审查。没有拟合、服务计时行解析、模型／队列／策略运行。源码完整程度和作者人工检查不能称正式训练／测试入口ready或独立审计PASS。

低层函数只组合已给定的**描述**：十个负载／种子模板派生30个名义延迟条件供四方法120项共享，每项哈希对应现有实际环境`environment_binding`字段，不插入method-specific外生输入或未来服务字段。所有动作对齐的四资源标签、原30移动文件身份、物理秒、位置、32waiting、链路与原catalog身份必须完整；同负载／种子只改变名义delay，其他输入不变。返回实际环境证明数0。当前无请求特征仅支持默认链路，非默认带宽继续拒绝而非静默改变；这不替用户选择网络条件或宣称2MHz可行。

真实装配源只接受同一实际generator成功完成的十份共同请求文件及外部SHA，重核真实smoke／cal-validation-only pilot／元数据生成准备权限和当前源。pilot须提供完整显式字段，不用缺省值填补未冻结归一化、H、终末、预算或优化器。原测试catalog身份从既有原审计清单读取；只核对字节，不重跑服务统计或把测试服务值用于拟合。cal/validation catalog保持完整，原预测器SHA固定；服务器位置与位置归一化只按pilot训练轨迹窗和种子重建，真实10秒移动时间支持覆盖完整各角色窗及drain、明确角色隔离间隔，不wrap或重映射。

有序训练段／只读验证日程按角色、完整决策数、物理网格及文件字节／语义SHA核对。两个学习方法共享预算、尺度、终末、优化器及外生环境条件；配对设计哈希严格对应现有TrainingTransactionLedger的公式。不同方法自身结构／QL/MV奖励不混同。此原型训练及策略验证条件均固定名义P；若pilot选择其他验证设计或buffer／非默认链路，须新增有保存证据的适配及独立入口审查，不能直接换字段。这里不按验证性能排名checkpoint，最终共同预算规则仍以真实pilot确认。

十份实际日程先产生，随后完整训练／测试协议**一次性**写出，之后十份训练／验证／最终权重及120项测试沿用同一协议SHA，不在权重选完后改绑定。即使未来写出FROZEN_PROTOCOL_PENDING_COMPLETE_ENTRY.json，也只是已冻结设计文件，不是review/模型启动权限：fresh gpt-6-astra xhigh完整Type-A入口必须另行真实完成并绑定该SHA，原GPU/kernel／实际两方法smoke等门槛仍有效。当前这些真实输入缺失，写出数0；不制作假冻结／PASS样例。

科学条件独占锁不按输出名称／审查装饰路径解锁。新目录、输入前后字节复核和失败现场／部分输入保存，不append／重封／重试旧或已经权重绑定的协议，不修改原论文／测量／旧RL输出。原源FAIL和修复独立复审PENDING保留，新增源使旧smoke计划过期不重封运行。

首次16:18人工描述检查及三份源已保存，随后新增保护输入合并拒绝同路径换SHA、原实际generator矩阵／完整文件／输入身份核对，以及训练几何窗也按calibration角色参与 held-out窗隔离；装配后再次核对原generator／pilot保护。新的只读检查另起attempt；同路径换SHA／坏SHA仅在人工字典上实际测试，原generator／窗口／真实序列化保护仍未实际执行，不能称这些真实路径已通过。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_protocol_assembly.py'
```

仅明确人工SHA／不存在的文件名／手填位置和尺度的纯描述：10模板→30条件→120描述覆盖、13种错角色／错配对／未来字段／非法坐标／无效链路等拒绝、深副本／编码独立及未执行ledger的配对公式检查。不会读原test计时CSV、构造真实geometry／catalog／generator／assembler或写真实冻结／审查文件；真实pilot验证、目录／真实序列化／完整源检查路径仍未实际执行，不能从人工检查推断它们成功。下一步正确fresh模型加载／共同网格启动源及真实入口独立检查，实际模型／新权重／训练／测试仍0。
