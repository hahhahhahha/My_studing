# 真实视觉策略两方法冒烟／权重回环入口（准备中，尚未执行）

只读检查命令：

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/visual_model_smoke.py' check
```

该命令不导入Torch、不建模型／优化器、不跑GPU或队列，不生成实际冒烟PASS。它绑定原工程报告／旧输入指纹，解析新入口与加载源码，并生成带源锁的新`SMOKE_CANDIDATE_PLAN.json`及明确`actual_cuda_smoke_pass=false`的准备报告。不是正式训练条件冻结，也不是完整独立入口审查。

实际run仍须原GPU空闲门槛与seeded CUDA witness通过，仅用同一校准池／物理KAIST／原2MHz／相同16×30样本日程，30请求/秒为**低负载功能检查**，不把它当正式rho=.75或上传可行性验收。完整episode与30秒固定drain先闭合，再各自唯一优化器更新、真实存档、加载实际新权重，核对保存／加载参数SHA与确定性act输出逐值一致。没有性能改善门槛，负值或无收益不是bug；无参数更新／非有限／加载不一致等失败保留且不自动重试。

两种方法按QLA、PPO顺序各在独立新进程运行，子进程结束才启动下一种。每个子进程首次建模前核对原GPU门槛；自身模型的同进程回环克隆直接用已核验历史类，不再次把自己的CUDA分配当外部占用。实际父入口按**计划内容SHA**在统一smoke_locks目录写永久独占启动记录；重新生成相同计划或另取输出名字不能绕过锁，私有子入口须匹配正在运行的父PID／输出／计划。失败修复须归档、另增源／计划并审查，不能复用成功的部分结果拼假PASS。

加载只接受已核验外部完成标志SHA的本次实际checkpoint；QLA保持原历史loader，PPO仅加载本次自己保存的严格state_dict。先核对文件SHA再反序列化，不读外来未验证文件。只加载模型权重，不恢复优化器、H、MV或训练随机状态，不支持resume。历史QLA新构造可能重新设定全局Torch／NumPy随机种子，回环发生在smoke训练结束后；不从该克隆继续训练。确定性动作前后另核对Torch RNG不变。

当前缺GPU空闲，真实构造／act／优化器／权重保存或加载都没有执行。入口只准备到可做功能smoke的程度，不等于已通过模型就绪、独立完整入口、cal/validation-only科学pilot、最终checkpoint规则、10项训练或120测试cells。网络方向仍等用户，原测量／协议／旧结果／正文保持不变。后续需先审查准备入口，实际smoke后才准完成pilot与fresh完整Type-A入口冻结。
