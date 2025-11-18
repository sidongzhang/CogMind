import torch
import torch.nn as nn
import time
import os
from tqdm import tqdm


class Trainer:
    """
    基础训练器 - 管理模型训练过程
    
    Args:
        model: 要训练的模型
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备 ('cuda' 或 'cpu')
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        
        # 将模型移到设备
        self.model.to(device)
        
    def train_epoch(self, dataloader, description="Training"):
        """
        训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            description: 进度条描述
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=description)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # 前向传播
            output = self.model(src, tgt[:, :-1])  # 输入到解码器的是前n-1个词
            
            # 计算损失 - 目标是后n-1个词
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化步骤
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.epoch += 1
        
        return avg_loss
    
    def validate(self, dataloader):
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for batch in progress_bar:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}'
                })
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, filepath):
        """
        保存训练检查点
        
        Args:
            filepath: 检查点文件路径
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # 确保目录存在（允许直接保存到当前目录）
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        加载训练检查点
        
        Args:
            filepath: 检查点文件路径
        """
        if not os.path.exists(filepath):
            print(f"检查点文件不存在: {filepath}")
            return
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"检查点已加载: {filepath}")
        print(f"从 epoch {self.epoch} 继续训练")


def test_trainer():
    """测试训练器"""
    print("=== 测试训练器 ===")
    
    # 创建简单的模型和数据加载器
    from cogmind.transformer import Transformer
    from cogmind.loss import LabelSmoothingCrossEntropy
    
    # 配置
    vocab_size = 100
    d_model = 32
    batch_size = 2
    seq_len = 5
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)
    
    # 创建训练器
    trainer = Trainer(model, optimizer, criterion, device='cpu')
    
    # 创建模拟数据加载器
    class MockDataLoader:
        def __init__(self, num_batches=3):
            self.num_batches = num_batches
            
        def __iter__(self):
            for i in range(self.num_batches):
                yield {
                    'src': torch.randint(1, vocab_size, (batch_size, seq_len)),
                    'tgt': torch.randint(1, vocab_size, (batch_size, seq_len + 1))
                }
                
        def __len__(self):
            return self.num_batches
    
    # 测试训练和验证
    train_loader = MockDataLoader(3)
    val_loader = MockDataLoader(2)
    
    print("开始训练...")
    train_loss = trainer.train_epoch(train_loader)
    print(f"训练损失: {train_loss:.4f}")
    
    print("开始验证...")
    val_loss = trainer.validate(val_loader)
    print(f"验证损失: {val_loss:.4f}")
    
    # 测试检查点保存和加载
    print("测试检查点...")
    trainer.save_checkpoint('test_checkpoint.pth')
    trainer.load_checkpoint('test_checkpoint.pth')
    
    # 清理
    if os.path.exists('test_checkpoint.pth'):
        os.remove('test_checkpoint.pth')
    
    print("✅ 训练器测试通过!")


if __name__ == "__main__":
    test_trainer()