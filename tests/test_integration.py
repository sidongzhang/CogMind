import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cogmind.attention import MultiHeadAttention
from cogmind.feed_forward import FeedForward
from cogmind.residual import PreNormResidual

def test_attention_with_residual():
    """测试注意力机制与残差连接的集成"""
    print("=== 测试注意力 + 残差连接 ===")
    
    batch_size, seq_len, d_model, num_heads = 2, 4, 8, 2
    
    # 创建组件
    attention = MultiHeadAttention(d_model, num_heads)
    residual = PreNormResidual(d_model)
    
    # 创建输入（自注意力：Q=K=V）
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 定义注意力子层
    def attention_sublayer(x):
        output, weights = attention(x, x, x)
        return output
    
    # 应用残差连接 + 注意力
    output = residual(x, attention_sublayer)
    print(f"输出形状: {output.shape}")
    
    # 验证形状不变
    assert output.shape == x.shape, f"形状不匹配: {output.shape} vs {x.shape}"
    
    print("✅ 注意力 + 残差连接测试通过!")
    return output

def test_feedforward_with_residual():
    """测试前馈网络与残差连接的集成"""
    print("\n=== 测试前馈网络 + 残差连接 ===")
    
    batch_size, seq_len, d_model = 2, 4, 8
    d_ff = 32
    
    # 创建组件
    feed_forward = FeedForward(d_model, d_ff)
    residual = PreNormResidual(d_model)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 应用残差连接 + 前馈网络
    output = residual(x, feed_forward)
    print(f"输出形状: {output.shape}")
    
    # 验证形状不变
    assert output.shape == x.shape, f"形状不匹配: {output.shape} vs {x.shape}"
    
    print("✅ 前馈网络 + 残差连接测试通过!")
    return output

def test_complete_flow():
    """测试完整的数据流：输入 → 注意力+残差 → 前馈+残差"""
    print("\n=== 测试完整数据流 ===")
    
    batch_size, seq_len, d_model, num_heads = 2, 5, 12, 3
    d_ff = 48
    
    # 创建所有组件
    attention = MultiHeadAttention(d_model, num_heads)
    feed_forward = FeedForward(d_model, d_ff)
    residual1 = PreNormResidual(d_model)
    residual2 = PreNormResidual(d_model)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"初始输入形状: {x.shape}")
    
    # 第一层：注意力 + 残差
    def attention_sublayer(x):
        output, weights = attention(x, x, x)
        return output
    
    x = residual1(x, attention_sublayer)
    print(f"注意力+残差后形状: {x.shape}")
    
    # 第二层：前馈网络 + 残差
    x = residual2(x, feed_forward)
    print(f"前馈+残差后形状: {x.shape}")
    
    # 最终形状应与初始相同
    assert x.shape == (batch_size, seq_len, d_model)
    
    print("✅ 完整数据流测试通过!")
    return x

if __name__ == "__main__":
    test_attention_with_residual()
    test_feedforward_with_residual() 
    test_complete_flow()
    print("\n🎉 所有集成测试通过！组件可以协同工作！")