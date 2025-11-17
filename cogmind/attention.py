import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    实现缩放点积注意力机制

    Args:
        d_k:键/查询向量的维度
        dropout:注意力权重的dropout率
        causal:是否为因果注意力（用于解码器）
    """

    def __init__(self, d_k, dropout=0.1, causal=False):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

        # 缩放因子：1 / sqrt(d_k)
        self.scale_factor = 1 / math.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q:[batch_size, seq_len_q, d_k]
            K:[batch_size, seq_len_k, d_k]
            V:[batch_size, seq_len_v, d_v](seq_len_v should be equal to seq_len_k)
            mask:[batch_size, seq_len, seq_len_k] or [seq_len, seq_len_k]

        Returns:
            output:[batch_size, seq_len_q, d_v]
            attention_weights:[batch_size, seq_len_q,seq_len_k]
        """
        # 1.计算点积并缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor  # [batch_size, seq_len_q, seq_len_k]

        # 2.如果启用因果注意力， 创建因果mask
        if self.causal:
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=scores.device),  diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

        # 3.应用用户提供的mask
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        # 4. 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. 应用注意力权重到V
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - 让模型同时关注不同方面的信息

    Args:
        d_model:模型的总维度
        num_heads:注意力头的数量
        dropout:注意力权重的dropout率
        causal:是否为因果注意力（用于解码器）
    
    """
    def __init__(self, d_model, num_heads, dropout=0.1, causal=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"\
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 创建Q、K、V的线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=False) # 查询变换
        self.W_k = nn.Linear(d_model, d_model, bias=False) # 键变换
        self.W_v = nn.Linear(d_model, d_model, bias=False) # 值变换
        self.W_o = nn.Linear(d_model, d_model) # 输出变换

        self.attention = ScaledDotProductAttention(d_k=self.d_k, dropout=dropout, causal=causal)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q:[batch_size, seq_len, d_model]
            K:[batch_size, seq_len, d_model]
            V:[batch_size, seq_len, d_model]
            mask:[batch_size, seq_len, seq_len]

        Returns:
            output:[batch_size, seq_len, d_model]
            attention_weights:[batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len = Q.size(0), Q.size(1)

        # 1.线性变换并分割为多头
        # Q: [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 为每个头扩展mask(如果存在)
        if mask is not None:
            # mask:[batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1) # [batch_size, 1, seq_len, seq_len]
        
        # 3.计算多头注意力(每个头独立计算)
        output, attention_weights = self.attention(Q, K, V, mask=mask)

        # 4.合并多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5.最终线性变换
        output = self.W_o(output)
        output = self.dropout(output)
        return output, attention_weights



    
# 测试代码

# 简单测试缩放点积注意力机制
def test_attention():
    """简单测试函数"""
    batch_size, seq_len, d_k, d_v = 2, 4, 8, 16
    attention = ScaledDotProductAttention(d_k=d_k)
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)

    output, weights = attention(Q, K, V)
    print("✅ 注意力测试通过!")
    print(f"输入 Q shape: {Q.shape}")
    print(f"输出 output shape: {output.shape}")
    print(f"注意力权重 weights shape: {weights.shape}")
    
    return output, weights


# 简单测试多头注意力机制
def test_multihead_attention():
    """测试多头注意力"""
    batch_size, seq_len, d_model, num_heads = 2, 4, 8, 2
    multihead_attn = MultiHeadAttention(d_model, num_heads)
    
    # 创建输入（所有输入相同，因为是自注意力）
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = multihead_attn(x, x, x)
    
    print("✅ 多头注意力测试通过!")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}") 
    print(f"注意力权重形状: {weights.shape}")
    print(f"注意力权重有 {num_heads} 个头，每个头学习不同的注意力模式")

if __name__ == "__main__":
    test_attention() # 测试缩放点积注意力
    test_multihead_attention() # 测试多头注意力