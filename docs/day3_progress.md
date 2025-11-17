# day3 - 词嵌入实现

## 🎯 学习目标
实现词嵌入层，将离散文本转换为连续向量表示，完成文本处理管道

## 📚 核心组件

### TokenEmbedding
- 将词索引映射为连续向量
- 支持padding符号处理
- 使用Xavier初始化保证训练稳定

### Embeddings（完整嵌入层）
- 组合词嵌入和位置编码
- 提供端到端的文本到向量转换

## 💡 关键技术点

### 1. 嵌入矩阵
```
词汇表: [词1, 词2, ..., 词N]
嵌入矩阵: [N, d_model]
映射: 词索引i → 向量embedding_matrix[i]
```

### 2. 缩放因子
```python
embeddings = self.embedding(tokens) * math.sqrt(self.d_model)
平衡词嵌入和位置编码的数值范围

提高训练稳定性
```

### 3. Padding处理
Padding符号的嵌入向量初始化为0, 防止模型学习无意义的填充位置

### 4. 位置编码集成
词索引 → 词嵌入 → 位置编码 → 完整输入表示