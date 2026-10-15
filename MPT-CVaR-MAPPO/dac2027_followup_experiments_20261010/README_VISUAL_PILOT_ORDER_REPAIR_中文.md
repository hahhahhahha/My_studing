# Pilot 与共同请求的顺序修复（仅源码）

发现的源码问题：共同请求生成器要求先有真实完成的 cal-validation-only pilot 和独立元数据-only权限；原装配器又要求该pilot中的测试环境模板提前含有尚未生成请求的SHA。实际所有生成／pilot／冻结仍为0，未伪造这些文件；当前阶段修改未冻结的准备源码，不改服务PLAN、原测量、模型、分割或门槛。

修复后的顺序为：真实模型smoke和明确网络方向 → cal-validation-only pilot冻结环境条件（测试模板不含任何请求SHA）→ 独立metadata-only准备权限 → 同一实际生成器写出十份共同请求 → 装配器核对该真实完成对象／外部SHA／原矩阵与文件，机械注入十份实际日程SHA的深副本 → 完整不可变正式协议／fresh Type-A → 十份共同预算训练和验证／自身final权重 → 120固定checkpoint测试及独立最终审计。

`bind_generated_test_environment_schedules` 不生成或预计算请求、读取图片池、选择窗口／条件、拟合参数或使用测试时长／结果。pilot模板仅含原角色、四资源、位置、物理窗口、广播、零delay基模板、32等待、默认链路、原catalog与30移动文件指纹；接受的字段恰好是完整环境字段减去日程SHA。任何未来占位SHA、旧SHA、多余字段、方法专属条件、非法角色／延迟／配置拒绝。输出十份完整条件再由原函数派生30条件，共用四方法120项。原冻结pilot没有被重写，正式协议一次性语义锁及原边界核验保留。

原 `visual_protocol_assembly.py`、其README与未修改的生成器源码保存在 `visual_calibrated_policy/engineering_attempts/20261011_174154_pilot_order_repair/SOURCE_BEFORE_REPAIR`。新作者检查仅用明确人工SHA／不存在文件／手填位置，不构造实际generator／pilot／装配器，不读取test计时CSV、不导入Torch，不创建actual-complete／权重／PASS记录；真实准备权限／序列化／物理窗保护仍未执行，也不是独立Type-A审查。

GPU门槛、网络用户方向、独立修复复审仍未满足；不得借此启动run／worker或冻结pilot。下一步仍需准备实际pilot证据入口，不能把本修复当作pilot已经完成。所有原失败证据、服务与论文保持。
