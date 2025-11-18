import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .residual import PreNormResidual

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层 - 生成目标序列的核心组件
    
    包含三个子层：
    1. 掩码多头自注意力 + 残差连接 + 层归一化
    2. 编码器-解码器注意力 + 残差连接 + 层归一化  
    3. 前馈网络 + 残差连接 + 层归一化
    
    Args:
        d_model: 模型维度
        nhead: 注意力头的数量
        dim_feedforward: 前馈网络中间层维度
        dropout: dropout比率
        activation: 前馈网络激活函数
    """
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1, activation="relu"):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.d_model = d_model
        self.nhead = nhead

        # 第一个子层：掩码多头自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, causal=True)
        self.residual1 = PreNormResidual(d_model, dropout)

        # 第二个子层：编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.residual2 = PreNormResidual(d_model, dropout)

        # 第三个子层：前馈网络
        self.ffn = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.residual3 = PreNormResidual(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列 [batch_size, tgt_len, d_model]
            memory: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码（用于掩码自注意力）
            memory_mask: 源序列掩码（用于编码器-解码器注意力）
            
        Returns:
            解码后的序列 [batch_size, tgt_len, d_model]
        """
        # 第一个子层：掩码多头自注意力 + 残差连接
        def self_attn_sublayer(x):
            output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
            return output
        
        tgt = self.residual1(tgt, self_attn_sublayer)

        # 第二个子层：编码器-解码器注意力 + 残差连接
        def cross_attn_sublayer(x):
            output, cross_attn_weights = self.cross_attn(x, memory, memory, memory_mask)
            return output
        
        tgt = self.residual2(tgt, cross_attn_sublayer)

        # 第三个子层：前馈网络 + 残差连接
        tgt = self.residual3(tgt, self.ffn)
        return tgt

class TransformerDecoder(nn.Module):
    """
    Transformer解码器 - 堆叠多个解码器层
    
    Args:
        decoder_layer: 解码器层实例
        num_layers: 解码器层数量
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            tgt: 目标序列 [batch_size, tgt_len, d_model]
            memory: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码
            memory_mask: 源序列掩码
            
        Returns:
            解码后的序列 [batch_size, tgt_len, d_model]
        """
        output = tgt

        # 逐层处理
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        
        return output

def test_decoder_layer():
    """测试单个解码器层"""
    print("=== 测试Transformer解码器层 ===")
    batch_size, src_len, tgt_len, d_model, nhead = 2, 6, 4 ,12, 3
    dim_ff = 48

    # 创建解码器层
    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_ff)

    # 创建模拟数据
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    print(f"目标序列形状: {tgt.shape}")
    print(f"编码器输出形状: {memory.shape}")

    # 测试无mask情况
    output = decoder_layer(tgt, memory)
    print(f"解码器输出形状: {output.shape}")
    # 验证输入输出形状相同
    assert output.shape == tgt.shape, f"形状不匹配: {output.shape} vs {tgt.shape}"
    
    # 测试带mask情况
    print("\n=== 测试带掩码的解码器层 ===")
    tgt_mask = torch.ones(batch_size, tgt_len, tgt_len)
    # 创建因果掩码：只能看到当前和之前的位置
    for i in range(tgt_len):
        tgt_mask[:, i, i+1:] = 0
        
    memory_mask = torch.ones(batch_size, tgt_len, src_len)
    memory_mask[:, :, 4:] = 0  # 屏蔽源序列后2个位置
    
    output_masked = decoder_layer(tgt, memory, tgt_mask, memory_mask)
    print(f"带掩码输出形状: {output_masked.shape}")
    
    print("✅ 解码器层测试通过!")
    return output, output_masked


def test_transformer_decoder():
    """测试完整的Transformer解码器（多层堆叠）"""
    print("\n=== 测试完整Transformer解码器 ===")
    
    batch_size, src_len, tgt_len, d_model, nhead, num_layers = 2, 8, 5, 16, 4, 2
    
    # 创建解码器层模板
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4
    )
    
    # 创建多层解码器
    decoder = TransformerDecoder(decoder_layer, num_layers)
    
    # 创建模拟数据
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    print(f"目标序列形状: {tgt.shape}")
    print(f"编码器输出形状: {memory.shape}")
    print(f"解码器层数: {num_layers}")
    
    # 前向传播
    output = decoder(tgt, memory)
    print(f"解码器输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == tgt.shape, f"形状不匹配: {output.shape} vs {tgt.shape}"
    
    print("✅ Transformer解码器测试通过!")
    return output


if __name__ == "__main__":
    test_decoder_layer()
    test_transformer_decoder()
    print("\n🎉 解码器实现完成！我们已经构建了完整的Transformer解码器！")