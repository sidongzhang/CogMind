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
    
# 测试代码
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

if __name__ == "__main__":
    test_attention()