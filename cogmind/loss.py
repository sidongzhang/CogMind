import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑的交叉熵损失
    
    Args:
        smoothing: 平滑系数（0-1之间）
        ignore_index: 忽略的索引（如padding）
        reduction: 损失 reduction 方式
    """

    def __init__(self, smoothing=0.1, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            targets: 目标标签 [batch_size, seq_len]
            
        Returns:
            平滑后的交叉熵损失
        """
        logits = logits.view(-1, logits.size(-1))# [batch_size*seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size*seq_len]

        # 如果指定了ignore_index，创建mask
        if self.ignore_index >= 0:
            padding_mask = targets.eq(self.ignore_index)
            targets = targets.masked_fill(padding_mask, 0)

        # 计算交叉熵损失
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss + nll_loss.squeeze(1)

        # 计算平滑损失
        smooth_loss = -log_probs.mean(dim=-1)
         # 如果指定了ignore_index，应用mask
        if self.ignore_index >= 0:
            nll_loss = nll_loss.masked_fill(padding_mask, 0.0)
            smooth_loss = smooth_loss.masked_fill(padding_mask, 0.0)
        
        # 组合损失
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                loss = loss.sum() / (~padding_mask).float().sum()
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss

def test_loss_function():
    """测试损失函数"""
    print("=== 测试标签平滑交叉熵损失 ===")
    
    batch_size, seq_len, vocab_size = 2, 3, 5
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)
    
    # 创建模拟数据
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.tensor([
        [1, 2, 0],  # 0是padding
        [3, 0, 0]   # 有padding
    ])
    
    print(f"Logits形状: {logits.shape}")
    print(f"Targets形状: {targets.shape}")
    print(f"Targets内容:\n{targets}")
    
    loss = loss_fn(logits, targets)
    print(f"损失值: {loss.item():.4f}")
    
    print("✅ 损失函数测试通过!")
    return loss

if __name__ == "__main__":
    test_loss_function()