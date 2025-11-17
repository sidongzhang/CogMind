import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cogmind.encoder import TransformerEncoderLayer, TransformerEncoder
from cogmind.positional_encoding import PositionalEncoding

def demo_complete_encoding_process():
    """演示完整的编码过程"""
    print("=== 完整编码过程演示 ===")
    
    # 配置参数
    batch_size = 2
    src_len = 8
    d_model = 16
    nhead = 4
    num_layers = 2
    vocab_size = 100
    
    print(f"配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  序列长度: {src_len}")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头: {nhead}")
    print(f"  编码器层数: {num_layers}")
    
    # 1. 创建词嵌入（模拟）
    embedding = torch.randn(vocab_size, d_model)
    
    # 2. 创建输入序列（词索引）
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_len))
    print(f"\n1. 输入词索引形状: {src_tokens.shape}")
    print(f"   示例: {src_tokens[0]}")
    
    # 3. 词嵌入
    src_embed = embedding[src_tokens]
    print(f"2. 词嵌入后形状: {src_embed.shape}")
    
    # 4. 位置编码
    pos_encoder = PositionalEncoding(d_model)
    src_encoded = pos_encoder(src_embed)
    print(f"3. 位置编码后形状: {src_encoded.shape}")
    
    # 5. 创建编码器
    encoder_layer = TransformerEncoderLayer(d_model, nhead)
    encoder = TransformerEncoder(encoder_layer, num_layers)
    
    # 6. 编码过程
    memory = encoder(src_encoded)
    print(f"4. 编码后形状: {memory.shape}")
    
    print(f"\n🎯 演示完成！")
    print(f"   从词索引 {src_tokens.shape} → 编码表示 {memory.shape}")
    
    return memory

def demo_attention_patterns():
    """演示注意力模式"""
    print("\n=== 注意力模式演示 ===")
    
    # 创建简单的编码器层
    encoder_layer = TransformerEncoderLayer(d_model=8, nhead=2)
    
    # 创建有意义的输入序列
    # 假设序列: "猫 喜欢 吃 鱼"
    src = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 猫
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 喜欢
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 吃
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 鱼
    ]])
    
    print(f"输入序列形状: {src.shape}")
    print("输入序列: 4个不同的one-hot向量")
    
    # 获取注意力权重（需要修改EncoderLayer来返回注意力权重）
    output = encoder_layer(src)
    print(f"输出形状: {output.shape}")
    print("输出: 每个词都包含了整个序列的上下文信息")
    
    return output

if __name__ == "__main__":
    demo_complete_encoding_process()
    demo_attention_patterns()