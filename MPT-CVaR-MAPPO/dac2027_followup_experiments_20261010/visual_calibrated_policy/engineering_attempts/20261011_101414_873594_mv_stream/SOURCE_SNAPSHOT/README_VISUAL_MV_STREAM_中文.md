# MV统计器跨episode连续：新增接口与工程核对

旧历史`train_towards_ppo`在整个训练循环外创建一次MVReward；为了闭合FIFO回执而分段采集，不应无意把这个统计器每段清零。新增`PersistentMVStream`在固定用户顺序下延续count／mean／m2／previous ratio；新`StreamingEpisodeAssembly`在GAE之前使用连续MV奖励；`RealStreamingEpisodeCollector`源码接入该接口，但**真实构造／模型采集从未执行**。

原`visual_mean_variance_reward.py`、旧采集器和全部工程记录保留，不覆盖其每episode重置候选。后续拟用新增连续接口消除重置偏差，不悄悄重写旧结果。真实训练器与checkpoint仍待接通，正式协议仍未冻结。

## 顺序、角色与失败保护

一个训练种子只创建一个统计器，episode必须按1、2、3…且具有唯一key闭合；用户仍按0…N−1逐一调用同一统计器，无回执输入0约定、population variance和二值奖励公式不改。状态只在整段合法成本递推后提交；失败保留已提交和部分递推状态，实例永久拒绝追加／重试。下游buffer或模型失败也必须停止该stream，不能回滚后静默再试。

训练用途只接收校准角色；工程验证回执用途另建对象，不能安装到真实训练采集器，不能把验证数学检查的统计值拿去初始化训练。没有自动resume／外部状态restore，也没有公开reset按钮。真实采集器仍只接受真实历史CUDA控制器，不调用优化器。

## 此次检查边界

只读复用四份已保存的验证来源工程轨迹（每份80×30调用），逐段状态必须从0→2400→4800→7200→9600。将实际过去回执／终末代理的成本矩阵送入从冻结历史源码AST隔离出的MVReward，与连续接口每一项奖励及每一步均值／方差／计数／比值逐值核对。另检查乱序、重复闭合、错误角色、无任务零输入及失败后禁止继续。

低层流式装配与每用户GAE再用明确人工动作／logp／value的校准来源小型工程episode连接；它们不是模型输出或训练效果，不访问验证服务来训练，也不访问保留测试。真实采集器仅静态入口布局／错误对象拒绝检查，Torch和模型调用数为0。

```powershell
& 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/environment/qla_np226/Scripts/python.exe' -B -X utf8 'X:/Yi-one/workspace/MPT-CVaR-MAPPO/dac2027_followup_experiments_20261010/scripts/check_visual_mv_stream.py'
```

连续统计器修复不等于旧方法完整逐槽复现：输入仍为真实过去回执，固定drain后的成本合并与unfinished代理仍是已披露视觉适配；队列初态、episode划分、虚拟H状态／终末信用与物理折扣还需统一pilot／审查。网络选项、GPU资源和Alibaba五分钟协议等待用户方向；不因此变更科学条件或正式启动训练。
