import torch
from torch.utils.data import Dataset, DataLoader
import random

class TranslationDataset(Dataset):
    """
    简单的翻译数据集（用于演示）
    
    在实际应用中，你会从文件加载真实的翻译数据
    这里我们生成随机数据来演示流程
    """
    def __init__(self, num_samples=1000, src_vocab_size=100, tgt_vocab_size=100, max_len=10, min_len=3):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.min_len = min_len

        # 生成随机数据
        self.data = []
        for _ in range(num_samples):
            src_len = random.randint(min_len, max_len)
            tgt_len = random.randint(min_len, max_len)

            src_seq = torch.randint(1, src_vocab_size, (src_len,))
            tgt_seq = torch.randint(1, tgt_vocab_size, (tgt_len,))
            
            self.data.append({
                'src': src_seq,
                'tgt': tgt_seq
            })
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    """
    批处理函数 - 将不同长度的序列填充到相同长度
    
    Args:
        batch: 批数据
        src_pad_idx: 源序列填充索引
        tgt_pad_idx: 目标序列填充索引
    """
    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]

    #  找到最大长度
    src_max_len = max(len(seq) for seq in src_seqs)
    tgt_max_len = max(len(seq) for seq in tgt_seqs)
    # 填充序列
    src_batch = torch.full((len(batch), src_max_len), src_pad_idx, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max_len), tgt_pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, :len(src)] = src
        tgt_batch[i, :len(tgt)] = tgt
    
    return {
        'src': src_batch,
        'tgt': tgt_batch
    }
