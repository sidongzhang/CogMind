import torch
import math
from torch.optim import Optimizer

class NoamOptimizer(Optimizer):
    """
    Noam优化器 - Transformer论文中使用的学习率调度策略
    
    学习率 = factor * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    Args:
        params: 模型参数
        d_model: 模型维度
        factor: 缩放因子
        warmup_steps: 预热步数
        betas: Adam的beta参数
        eps: 数值稳定性常数
        weight_decay: 权重衰减
    """
    def __init__(self, params, d_model, factor=1.0, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9, weight_decay=0):
        self.d_model = d_model
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.step_num = 0

        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def get_lr(self):
        """计算当前学习率"""
        step = self.step_num + 1  # 避免除零
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    # TODO
    def step(self, closure=None):
        """
        执行单步优化
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        self.step_num += 1
        lr = self.get_lr()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NoamOptimizer不支持稀疏梯度')
                
                state = self.state[p]
        # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # 一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 二阶动量
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 更新一阶和二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = lr / bias_correction1
        # 计算更新量
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group['eps']
                
                # 应用权重衰减
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-lr * group['weight_decay'])
                
                # 更新参数
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

class WarmupScheduler:
    """
    带预热的线性学习率调度器
    
    Args:
        optimizer: 优化器
        d_model: 模型维度
        warmup_steps: 预热步数
        factor: 缩放因子
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _get_lr(self):
        """计算学习率"""
        step = self.step_num
        warmup = self.warmup_steps
        
        return self.factor * (
            self.d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [self._get_lr()]

def test_optimizer():
    """测试优化器"""
    print("=== 测试Noam优化器 ===")
    
    # 创建简单模型
    model = torch.nn.Linear(10, 1)
    optimizer = NoamOptimizer(
        model.parameters(),
        d_model=512,
        warmup_steps=100,
        factor=1.0
    )
    # 模拟训练步骤
    for step in range(5):
        # 模拟损失计算和反向传播
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        lr = optimizer.get_lr()
        print(f"步骤 {step+1}: 学习率 = {lr:.6f}, 损失 = {loss.item():.4f}")
    
    print("✅ 优化器测试通过!")

def test_scheduler():
    """测试学习率调度器"""
    print("\n=== 测试学习率调度器 ===")
    
    model = torch.nn.Linear(5, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # 初始学习率为0
    scheduler = WarmupScheduler(optimizer, d_model=256, warmup_steps=10)
    
    learning_rates = []
    for step in range(15):
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        learning_rates.append(lr)
        print(f"步骤 {step+1}: 学习率 = {lr:.6f}")
    
    # 验证学习率变化
    assert learning_rates[0] < learning_rates[5] < learning_rates[9], "学习率应该在预热阶段上升"
    assert learning_rates[14] < learning_rates[10], "学习率应该在预热后下降"
    
    print("✅ 学习率调度器测试通过!")
    return learning_rates


if __name__ == "__main__":
    test_optimizer()
    test_scheduler()

                
