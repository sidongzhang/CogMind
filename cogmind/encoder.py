import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .residual import PreNormResidual

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层 - 完整的神经网络层
    
    包含：
    1. 多头自注意力机制 + 残差连接 + 层归一化
    2. 前馈网络 + 残差连接 + 层归一化
    
    Args:
        d_model: 模型维度
        nhead: 注意力头的数量
        dim_feedforward: 前馈网络中间层维度
        dropout: dropout比率
        activation: 前馈网络激活函数
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1, activation="relu"):
        super().__init__()
        
        # 如果没有指定前馈网络维度，默认为d_model的4倍
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.d_model = d_model
        self.nhead = nhead
        
        # 第一个子层：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.residual1 = PreNormResidual(d_model, dropout)
        
        # 第二个子层：前馈网络
        self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.residual2 = PreNormResidual(d_model, dropout)
        
    def forward(self, src, src_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 [batch_size, src_len, src_len]
            
        Returns:
            编码后的序列 [batch_size, src_len, d_model]
        """
        # 第一个子层：多头自注意力 + 残差连接
        def self_attn_sublayer(x):
            # 自注意力：Q=K=V=src
            output, attn_weights = self.self_attn(x, x, x, src_mask)
            return output
            
        src = self.residual1(src, self_attn_sublayer)
        
        # 第二个子层：前馈网络 + 残差连接
        src = self.residual2(src, self.ffn)
        
        return src


class TransformerEncoder(nn.Module):
    """
    Transformer编码器 - 堆叠多个编码器层
    
    Args:
        encoder_layer: 编码器层实例
        num_layers: 编码器层数量
    """
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, src_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            
        Returns:
            编码后的序列 [batch_size, src_len, d_model]
        """
        output = src
        
        # 逐层处理
        for layer in self.layers:
            output = layer(output, src_mask)
            
        return output


def test_encoder_layer():
    """测试单个编码器层"""
    print("=== 测试Transformer编码器层 ===")
    
    batch_size, src_len, d_model, nhead = 2, 5, 12, 3
    dim_ff = 48
    
    # 创建编码器层
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=0.1,
        activation="relu"
    )
    
    # 创建输入序列
    src = torch.randn(batch_size, src_len, d_model)
    print(f"输入序列形状: {src.shape}")
    
    # 测试无mask情况
    output = encoder_layer(src)
    print(f"输出序列形状: {output.shape}")
    
    # 验证输入输出形状相同
    assert output.shape == src.shape, f"形状不匹配: {output.shape} vs {src.shape}"
    
    # 测试带mask情况
    print("\n=== 测试带掩码的编码器层 ===")
    src_mask = torch.ones(batch_size, src_len, src_len)
    src_mask[:, :, 3:] = 0  # 屏蔽后2个位置
    
    output_masked = encoder_layer(src, src_mask)
    print(f"带掩码输出形状: {output_masked.shape}")
    
    print("✅ 编码器层测试通过!")
    return output, output_masked


def test_transformer_encoder():
    """测试完整的Transformer编码器（多层堆叠）"""
    print("\n=== 测试完整Transformer编码器 ===")
    
    batch_size, src_len, d_model, nhead, num_layers = 2, 6, 16, 4, 3
    
    # 创建编码器层模板
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=0.1
    )
    
    # 创建多层编码器
    encoder = TransformerEncoder(encoder_layer, num_layers)
    
    # 创建输入
    src = torch.randn(batch_size, src_len, d_model)
    print(f"输入形状: {src.shape}")
    print(f"编码器层数: {num_layers}")
    
    # 前向传播
    output = encoder(src)
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == src.shape, f"形状不匹配: {output.shape} vs {src.shape}"
    
    print("✅ Transformer编码器测试通过!")
    return output


if __name__ == "__main__":
    test_encoder_layer()
    test_transformer_encoder()
    print("\n🎉 编码器实现完成！我们已经构建了完整的Transformer编码器！")