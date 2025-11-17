# CogMind
A lightweight LLM training framework built from the ground up for educational purposes.

## 学习计划
```
📅 CogMind 项目学习计划回顾
阶段一：奠基 (1-2周) - 核心组件打造
目标： 实现 Transformer 的所有基本模块，并具备完整的单元测试。

Week 1: ✅ 注意力机制、✅ 位置编码、✅ 前馈网络

Week 2: ✅ LayerNorm、✅ 残差连接、⬜ Embedding 层

阶段二：集成 (1-2周) - 模型组装与数据流
目标： 将模块组装成完整的 Encoder、Decoder 和 Transformer 模型，并建立数据加载流程。

Week 3: ⬜ EncoderLayer/DecoderLayer, ⬜ 完整 Transformer

Week 4: ⬜ BPE 分词器、⬜ 数据加载器、⬜ 简单训练循环

阶段三：强化 (2-3周) - 训练效率与稳定性
目标： 引入高级训练技术，让框架变得高效、稳定、可用。

Week 5: ⬜ 优化器、⬜ 学习率调度、⬜ 模型初始化

Week 6: ⬜ 混合精度训练、⬜ 激活检查点

Week 7: ⬜ 模型保存/加载、⬜ 日志记录、⬜ 基础评测

阶段四：进阶 (2-3周) - 工业级特性
目标： 集成最前沿的优化技术，让你的框架具备"工业强度"。

Week 8: ⬜ FlashAttention 集成、⬜ 分布式训练基础

Week 9: ⬜ DeepSpeed ZeRO 集成、⬜ LoRA 微调

Week 10: ⬜ 最终测试、⬜ 性能基准评估、⬜ 撰写项目文档
```

### day01
- ✅ 实现了缩放点积注意力机制 (ScaledDotProductAttention)
- ✅ 支持因果掩码和填充掩码
- ✅ 编写了完整的测试用例
- ✅ 理解并实现多头注意力机制，掌握分头计算与合并的原理
- ✅ 前馈网络

### day02
- ✅ 层归一化
- ✅ 残差连接