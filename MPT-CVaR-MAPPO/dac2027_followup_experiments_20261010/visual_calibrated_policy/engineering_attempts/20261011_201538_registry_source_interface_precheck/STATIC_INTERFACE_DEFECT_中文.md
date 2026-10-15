# 尚未执行真实路径时的 counts 接口静态修复

20:15:38源码核对发现原 `ClosedEpisodeCounts` 字段为 `action_interval_seconds`，初写登记 source 的 consumer 使用 `physical_seconds`。尚未进行人工成功检查，也没有运行任何实际 registrar／model／driver／collector／NPZ或replay。实际初写源保存后计算的 SHA 为 `71b45557d6cdd38faa883d55bbcc94acfa7b290312db7357efee7a2174b4088d`，由后续报告重新核对。

初写源先保存于本目录 `visual_pilot_registration.py`，再仅修复新登记 consumer，原producer、既有代码、测量、模型、论文、分割与门槛均不改。新增人工 AST dataclass字段检查应识别保存的错键，且完整 parent/episode/load witness 字段要与实际 producer源码匹配；没有造假实际marker／smoke／checkpoint／用户／PASS来测试运行门槛。

加严 source 也保存失败前已核对summary／当前descriptor和stage、拒绝重复NPZ数组名、绑定正确自身MODEL payload路径／字节并核对原共同H+drain。这些都是源码保护，不是实际调用或独立完整入口认证。实际记录必须通过真实先前门槛后再产生，不使用本人工预检代替。
