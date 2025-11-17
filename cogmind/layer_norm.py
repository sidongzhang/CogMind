import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    层归一化 - 对每个样本的最后一个维度进行归一化
    
    与BatchNorm不同，LayerNorm对每个样本独立归一化，不依赖batch中其他样本
    这使得LayerNorm更适合变长序列和小batch_size情况
    
    Args:
        d_model: 特征维度
        eps: 防止除零的小常数
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # 可学习参数：缩放(gamma)和偏移(beta)
        self.gamma = nn.Parameter(torch.ones(d_model))   # 初始化为1
        self.beta = nn.Parameter(torch.zeros(d_model))   # 初始化为0
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model] 或 [batch_size, d_model]
            
        Returns:
            归一化后的张量，与输入形状相同
        """
        # 在最后一个维度计算均值和方差
        mean = x.mean(dim=-1, keepdim=True) # 保持维度便于广播
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 无偏估计设为False

        # 归一化：(x - mean) / sqrt(var + eps)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移： gamma * x_normalized + self.beta
        output = self.gamma * x_normalized + self.beta

        return output

def test_layer_norm():
    """测试层归一化"""
    print("=== 测试1: 3D输入 (Transformer典型输入) ===")
    batch_size, seq_len, d_model = 2, 3, 4
    layer_norm = LayerNorm(d_model)
    
    # 创建输入数据
    x = torch.tensor([[
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0], 
        [3.0, 4.0, 5.0, 6.0]
    ], [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0]
    ]], dtype=torch.float32)
    
    print(f"输入形状: {x.shape}")
    print(f"输入数据:\n{x}")
    
    output = layer_norm(x)
    print(f"输出形状: {output.shape}")
    print(f"归一化后数据:\n{output}")
    
    # 验证每个位置的均值和方差
    print("\n验证归一化效果:")
    for i in range(batch_size):
        for j in range(seq_len):
            vec = output[i, j]
            print(f"样本{i}位置{j}: 均值={vec.mean():.6f}, 方差={vec.var(unbiased=False):.6f}")
    
    print("\n=== 测试2: 2D输入 ===")
    # 测试2D输入情况
    x_2d = torch.randn(3, 6)  # [batch_size, d_model]
    layer_norm_2d = LayerNorm(6)
    output_2d = layer_norm_2d(x_2d)
    print(f"2D输入形状: {x_2d.shape}")
    print(f"2D输出形状: {output_2d.shape}")
    
    print("✅ 层归一化测试通过!")
    return output


if __name__ == "__main__":
    test_layer_norm()