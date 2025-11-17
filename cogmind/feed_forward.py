import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    前馈网络 - Transformer中的位置级前馈层
    
    这是Transformer中每个位置独立计算的前馈神经网络
    包含两个线性变换和一个激活函数
    
    Args:
        d_model: 输入输出维度
        d_ff: 中间层维度（通常为d_model的4倍）
        dropout: dropout比率
        activation: 激活函数，默认为ReLU
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        
        # 如果没有指定d_ff，默认为d_model的4倍
        if d_ff is None:
            d_ff = d_model * 4
        
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层：扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层：恢复维度
        self.dropout = nn.Dropout(dropout)

         # 选择激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 第一个线性变换 + 激活函数
        x = self.linear1(x)        # [batch_size, seq_len, d_ff]
        x = self.activation(x)     # [batch_size, seq_len, d_ff]
        x = self.dropout(x)        # [batch_size, seq_len, d_ff]

        # 第二个线性变换
        x = self.linear2(x)        # [batch_size, seq_len, d_model]
        
        return x

class PositionWiseFeedForward(FeedForward):
    """
    位置级前馈网络的别名，与原始论文保持一致
    """
    pass

def test_feed_forward():
    """测试前馈网络"""
    batch_size, seq_len, d_model = 2, 4, 8
    d_ff = 32
    
    # 测试ReLU激活
    ff_relu = FeedForward(d_model, d_ff, activation="relu")
     
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ff_relu(x)
    
    print("✅ 前馈网络测试通过!")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"中间层维度 d_ff: {d_ff}")
    # 测试GELU激活
    ff_gelu = FeedForward(d_model, d_ff, activation="gelu")
    output_gelu = ff_gelu(x)
    print(f"GELU激活输出形状: {output_gelu.shape}")
    
    # 验证输入输出维度相同
    assert x.shape == output.shape, f"输入输出形状不匹配: {x.shape} vs {output.shape}"
    
    return output

if __name__ == "__main__":
    test_feed_forward()