import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cogmind.layer_norm import LayerNorm

def test_layer_norm_manual():
    """手动验证层归一化计算"""
    print("=== 手动验证层归一化 ===")
    
    # 创建简单的输入
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])  # [2, 3]
    
    print(f"输入:\n{x}")
    
    # 手动计算
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_manual = (x - mean) / torch.sqrt(var + 1e-5)
    
    print(f"手动计算 - 均值:\n{mean}")
    print(f"手动计算 - 方差:\n{var}") 
    print(f"手动计算 - 归一化结果:\n{x_manual}")
    
    # 使用LayerNorm（gamma=1, beta=0）
    layer_norm = LayerNorm(3)
    with torch.no_grad():
        layer_norm.gamma.data.fill_(1.0)
        layer_norm.beta.data.fill_(0.0)
    
    x_layer_norm = layer_norm(x)
    print(f"LayerNorm结果:\n{x_layer_norm}")
    
    # 验证是否一致
    diff = torch.abs(x_manual - x_layer_norm).max()
    print(f"最大差异: {diff}")
    assert diff < 1e-6, "手动计算与LayerNorm结果不一致!"
    
    print("✅ 手动验证通过!")

if __name__ == "__main__":
    test_layer_norm_manual()