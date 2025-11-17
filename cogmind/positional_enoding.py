import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码 - 为Transformer添加序列位置信息

    Transformer本身没有递归结构，所以需要显式地告诉模型每个词的位置
    使用正弦和余弦函数来编码位置信息

    Args:
        d_model: 词嵌入维度
        max_len: 序列的最大长度
        dropout: dropout概率
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 创建位置索引[0, 1, 2, ..., max_len-1]，形状：[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率项：1/(10000^(2i/d_model))
        # 这里使用对数空间的计算方式，更数值稳定
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 对偶数位置应用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)

        # 对奇数位置应用余弦函数 
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为buffer，不作为模型参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码加到输入词向量上

        Args:
            x: 输入词向量，形状为 [batch_size, seq_len, d_model]

        Returns:
            添加位置编码后的词向量，形状为 [batch_size, seq_len, d_model]
        """
        # 将位置编码加到输入上（只取前seq_len部分）
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
def test_positional_encoding():
    """测试位置编码"""
    d_model, seq_len, batch_size = 8, 6, 2
    pos_encoding = PositionalEncoding(d_model)
    
    # 创建输入（可以全是0，因为我们只关心位置编码）
    x = torch.zeros(batch_size, seq_len, d_model)
    
    output = pos_encoding(x)
    
    print("✅ 位置编码测试通过!")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码矩阵形状: {pos_encoding.pe.shape}")
    
    # 可视化第一个batch的第一个位置编码
    print(f"\n第一个位置的位置编码（前8个值）:")
    print(pos_encoding.pe[0, 0, :].detach().numpy())
    
    return output

if __name__ == "__main__":
    test_positional_encoding()