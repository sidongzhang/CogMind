# Day 1 Progress - 注意力机制实现
## 实现的组件
### 1. 缩放点积注意力 (ScaledDotProductAttention)
- 支持因果掩码和填充掩码
- 核心公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

### 2. 多头注意力 (MultiHeadAttention)
- 将输入线性映射到多个头，每个头独立计算注意力
- 最后合并所有头的输出并通过线性变换

### 3. 位置编码 (PositionalEncoding)
- 使用正弦和余弦函数为输入添加位置信息
- 支持不同长度的序列（最大长度由max_len决定）

### 4. 前馈网络 (FeedForward)
- 两个线性变换中间加激活函数
- 通常中间维度比输入输出维度大

## 关键知识点

### 注意力机制中的掩码
- 因果掩码：防止解码器看到未来信息
- 填充掩码：忽略填充符号的影响

### 多头注意力的维度变换
- 分头：将d_model拆分为num_heads个d_k
- 合并：将多个头的输出拼接回d_model

### 位置编码的数学原理
- 使用不同频率的正弦和余弦函数
- 允许模型学习相对位置信息


# 今日疑问
## 1.d_k是什么
答：d_k 是键向量和查询向量的维度。
 - Q (Query)：查询向量，表示"我在找什么"
 - K (Key)：键向量，表示"我有什么信息"（可以提供的特征）
 - V (Value)：值向量，表示"我的实际内容是什么"（具有的信息）
### （额外问题）为什么需要 d_k？（简述）
    缩放因子 1/sqrt(d_k)
    为了防止点积结果过大，导致 softmax 梯度消失。

## 2.Q,K,V的维度不太理解
 [batch_size, seq_len, d_k] 的含义：
 - batch_size: 同时处理多少个样本（如2个句子）
 - seq_len: 序列长度（如每个句子4个词）
 - d_k: 每个词的向量维度（如8维）

## 3.为啥会有两次掩码
 - 第一次掩码 - 因果掩码 (Causal Mask)：
     - 用途：确保在生成文本时，每个词只能看到它之前的词，不能"偷看"未来的词
     - 场景：主要用于解码器，保证自回归生成
 - 第二次掩码 - 填充掩码 (Padding Mask)：
     - 用途：忽略填充符号（padding tokens），不让模型关注无意义的填充位置
     - 场景：处理变长序列时使用

## 4.掩码那里具体怎么实现的，为什么这样写可以实现
掩码的核心思想：在 softmax 之前，把要屏蔽的位置设为负无穷
// ...existing code...
```python
if self.causal:
    seq_len = scores.size(-1)
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=scores.device),
        diagonal=1
    ).bool()
    scores.masked_fill_(causal_mask, float('-inf'))
```
// ...existing code...
### eg 步骤分解：
#### 1. 创建掩码矩阵:
    要屏蔽的位置为1，其他为0
#### 2. 使用masked_fill:
    把掩码为1的位置替换为 -inf
#### 3. softmax 计算：
    e^(-inf) = 0，所以这些位置的权重为0
### eg:
    scores = [ [1.2, 0.8, 0.5],   # 原始得分
           [0.9, 1.1, 0.7],
           [0.6, 0.4, 1.0] ]

    mask = [ [0, 0, 1],           # 要屏蔽第三个位置
         [0, 0, 1], 
         [0, 0, 1] ]

#### 应用掩码后：
    scores = [ [1.2, 0.8, -inf],  # 第三个位置被屏蔽
           [0.9, 1.1, -inf],
           [0.6, 0.4, -inf] ]

#### softmax 后:
    -inf 的位置权重为0
## 5.为什么在view中使用 -1
```
K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
```
-1 是 PyTorch 中的一个特殊符号，表示 "**自动计算这个维度的大小**"

#### 为什么用-1？
    让代码更灵活，如果序列长度变化也能自动适应
    避免硬编码，减少出错可能

### 6.如何体现每个头独立计算？
```
# 变换前：所有头混在一起
Q = [batch_size, seq_len, d_model]  # 如 [2, 4, 8]

# 分头变换后：每个头独立
Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
# 形状变为：[batch_size, num_heads, seq_len, d_k] 如 [2, 2, 4, 4]

# 现在张量结构：
# 第0个头：Q[0, 0, :, :] -> [4, 4]  # 第一个batch，第一个头
# 第1个头：Q[0, 1, :, :] -> [4, 4]  # 第一个batch，第二个头
```
实际计算时：
```
# self.attention 内部处理的是4D张量
# 但它会在 batch_size * num_heads 这个维度上并行计算
# 相当于同时计算 2(batch) * 2(heads) = 4 个独立的注意力机制

```
### 7.如何合并多个头的输出？
这是分头操作的逆过程：
```
# 注意力计算后的输出形状：[batch_size, num_heads, seq_len, d_k]
output = [2, 2, 4, 4]  # 2个batch，2个头，序列长4，每个头维度4

# 步骤1：转置，把num_heads维度移到后面
output = output.transpose(1, 2)  # [2, 4, 2, 4]
# 现在形状：batch_size, seq_len, num_heads, d_k

# 步骤2：合并最后两个维度
output = output.contiguous().view(batch_size, seq_len, d_model)  # [2, 4, 8]
# 把 [2, 4] 合并成 [8]，因为 2 heads * 4 d_k = 8 d_model
```

### 8.为什么用正弦余弦？
    唯一性：每个位置有唯一的编码
    相对位置：PE(pos + k) 可以表示为 PE(pos) 的线性函数
    周期性：不同频率的正弦波可以捕捉不同距离的依赖关系

### 9.频率衰减的意义
    低频分量（接近1.0）：捕捉长距离依赖
    高频分量（接近0.001）：捕捉局部依赖
