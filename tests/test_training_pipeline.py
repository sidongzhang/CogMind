import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_complete_training_pipeline():
    """测试完整训练流程"""
    print("=== 测试完整训练流程 ===")
    
    from cogmind.transformer import Transformer
    from cogmind.loss import LabelSmoothingCrossEntropy
    from cogmind.optimizer import NoamOptimizer
    from cogmind.trainer import Trainer
    from cogmind.dataset import TranslationDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # 配置
    vocab_size = 100
    d_model = 32
    batch_size = 4
    num_samples = 20
    
    print("1. 创建模型...")
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    print("2. 创建损失函数和优化器...")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)
    optimizer = NoamOptimizer(
        model.parameters(),
        d_model=d_model,
        warmup_steps=10,
        factor=1.0
    )
    
    print("3. 创建训练器...")
    trainer = Trainer(model, optimizer, criterion, device='cpu')
    
    print("4. 创建数据集和数据加载器...")
    dataset = TranslationDataset(
        num_samples=num_samples,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    
    print("5. 运行一个训练周期...")
    train_loss = trainer.train_epoch(dataloader)
    print(f"训练损失: {train_loss:.4f}")
    
    print("6. 运行验证...")
    val_loss = trainer.validate(dataloader)
    print(f"验证损失: {val_loss:.4f}")
    
    print("7. 测试检查点...")
    trainer.save_checkpoint('test_pipeline_checkpoint.pth')
    
    # 创建新的训练器并加载检查点
    trainer2 = Trainer(model, optimizer, criterion, device='cpu')
    trainer2.load_checkpoint('test_pipeline_checkpoint.pth')
    
    # 清理
    if os.path.exists('test_pipeline_checkpoint.pth'):
        os.remove('test_pipeline_checkpoint.pth')
    
    print("🎉 完整训练流程测试通过!")
    return train_loss, val_loss


if __name__ == "__main__":
    test_complete_training_pipeline()