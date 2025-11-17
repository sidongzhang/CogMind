import torch
import torch.nn as nn
from .layer_norm import LayerNorm

class ResidualConnection(nn.Module):
    """
    残差连接 - 将输入直接添加到子层输出中
    
    这是解决深度神经网络梯度消失问题的关键技术
    公式: output = LayerNorm(x + Sublayer(x))
    
    Args:
        d_model: 特征维度（用于LayerNorm）
        dropout: dropout比率
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        前向传播
        
        Args:
            x: 输入张量
            sublayer: 子层函数（如注意力或前馈网络）
            
        Returns:
            残差连接后的输出
        """
        # 残差连接: x + Dropout(Sublayer(LayerNorm(x)))
        # 注意：有些实现先做LayerNorm，有些后做。这里采用pre-norm结构
        return x + self.dropout(sublayer(self.layer_norm(x)))


class PreNormResidual(ResidualConnection):
    """
    Pre-Norm 残差连接的别名
    在子层之前进行LayerNorm，训练更稳定
    """
    pass


class PostNormResidual(nn.Module):
    """
    Post-Norm 残差连接
    原始Transformer论文中的实现：在子层之后进行LayerNorm
    
    Args:
        d_model: 特征维度
        dropout: dropout比率
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        前向传播 - Post-Norm版本
        
        Args:
            x: 输入张量
            sublayer: 子层函数
            
        Returns:
            残差连接后的输出
        """
        # Post-Norm: LayerNorm(x + Dropout(Sublayer(x)))
        return self.layer_norm(x + self.dropout(sublayer(x)))


def test_residual_connection():
    """测试残差连接"""
    print("=== 测试Pre-Norm残差连接 ===")
    batch_size, seq_len, d_model = 2, 3, 4
    
    # 创建Pre-Norm残差连接
    pre_norm_residual = PreNormResidual(d_model)
    
    # 创建一个简单的子层（恒等映射）
    def identity_sublayer(x):
        return x * 0.5  # 简单的缩放
    
    # 创建输入
    x = torch.ones(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    print(f"输入数据 (第一个样本):\n{x[0]}")
    
    # 应用残差连接
    output = pre_norm_residual(x, identity_sublayer)
    print(f"输出形状: {output.shape}")
    print(f"输出数据 (第一个样本):\n{output[0]}")
    
    # 验证残差连接的效果
    expected = x + 0.5 * x  # 因为子层是0.5倍缩放
    print(f"期望输出 (第一个样本):\n{expected[0]}")
    
    diff = torch.abs(output - expected).max()
    print(f"实际与期望的最大差异: {diff}")
    
    print("\n=== 测试Post-Norm残差连接 ===")
    post_norm_residual = PostNormResidual(d_model)
    output_post = post_norm_residual(x, identity_sublayer)
    print(f"Post-Norm输出形状: {output_post.shape}")
    print(f"Post-Norm输出数据 (第一个样本):\n{output_post[0]}")
    
    print("✅ 残差连接测试通过!")
    return output, output_post


if __name__ == "__main__":
    test_residual_connection()