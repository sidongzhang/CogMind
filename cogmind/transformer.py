from sympy import O
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .embedding import Embeddings, SharedEmbedding

class Transformer(nn.Module):
    """
    完整的Transformer模型 - Encoder-Decoder架构
    
    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        nhead: 注意力头数量
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈网络维度
        dropout: dropout比率
        activation: 激活函数
        share_embedding: 是否共享编码器解码器嵌入权重
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", share_embedding=False):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # 创建嵌入层
        self.src_embedding = Embeddings(src_vocab_size, d_model, dropout=dropout)

        if share_embedding and src_vocab_size == tgt_vocab_size:
            # 共享权重（适用于类似语言的任务，如文本摘要）
            self.tgt_embedding = self.src_embedding
        else:
            # 独立的嵌入层（适用于机器翻译等任务）
            self.tgt_embedding = Embeddings(tgt_vocab_size, d_model, dropout=dropout)

        # 创建编码器
        from .encoder import TransformerEncoderLayer, TransformerEncoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # 创建解码器
        from .decoder import TransformerDecoderLayer, TransformerDecoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 如果共享嵌入，将输出投影层的权重与目标嵌入层绑定
        if share_embedding and src_vocab_size == tgt_vocab_size:
            self.output_projection.weight = self.tgt_embedding.token_embedding.embedding.weight
        
        self._rest_parameters()

    def _rest_parameters(self):
        """初始化参数"""
        # 使用Xavier初始化线性层
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列词索引 [batch_size, src_len]
            tgt: 目标序列词索引 [batch_size, tgt_len]  
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（因果掩码）
            memory_mask: 编码器-解码器注意力掩码
            
        Returns:
            输出logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 1. 源序列嵌入和编码
        src_embedded = self.src_embedding(src) # [batch_size, src_len, d_model]
        memory = self.encoder(src_embedded, src_mask)# [batch_size, src_len, d_model]

        # 2. 目标序列嵌入
        tgt_embedded = self.tgt_embedding(tgt) # [batch_size, tgt_len, d_model]

        # 3.解码
        decoder_output = self.decoder(tgt_embedded, memory, tgt_mask, memory_mask) # [batch_size, tgt_len, d_model]

        # 4.投影输出
        output = self.output_projection(decoder_output)# [batch_size, tgt_len, tgt_vocab_size]

        return output

def test_transformer():
    """测试完整Transformer模型"""
    print("=== 测试完整Transformer模型 ===")
    
    # 配置参数
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    d_model = 32
    nhead = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    batch_size, src_len, tgt_len = 2, 6, 4
    # 创建Transformer模型
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers
    )
    
    print(f"模型配置:")
    print(f"  源词汇表: {src_vocab_size}")
    print(f"  目标词汇表: {tgt_vocab_size}") 
    print(f"  模型维度: {d_model}")
    print(f"  注意力头: {nhead}")
    print(f"  编码器层数: {num_encoder_layers}")
    print(f"  解码器层数: {num_decoder_layers}")
      
    # 创建模拟输入
    src_tokens = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_tokens = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"\n输入数据:")
    print(f"  源序列形状: {src_tokens.shape}")
    print(f"  目标序列形状: {tgt_tokens.shape}")
    
    # 前向传播
    output = transformer(src_tokens, tgt_tokens)
    print(f"输出形状: {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, tgt_len, tgt_vocab_size)
    assert output.shape == expected_shape, f"期望{expected_shape}, 实际{output.shape}"
    
    print("✅ 完整Transformer模型测试通过!")
    return output

if __name__ == "__main__":
    test_transformer()
    print("\n🎉 完整Transformer架构实现完成！我们构建了一个真正的深度学习模型！")