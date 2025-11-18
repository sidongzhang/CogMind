import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from .positional_encoding import PositionalEncoding

class TokenEmbedding(nn.Module):
    """
    词嵌入层 - 将离散的词索引转换为连续的向量表示
    
    Args:
        vocab_size: 词汇表大小
        d_model: 嵌入维度（与模型维度一致）
        padding_idx: 填充符号的索引（可选）
    """
    
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 创建嵌入矩阵 [vocab_size, d_model]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        
        # 初始化权重（使用Xavier均匀初始化）
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化嵌入权重"""
        # 使用Xavier均匀初始化，有助于训练稳定
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # 如果有padding_idx，将其对应的权重设为0
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
                
    def forward(self, tokens):
        """
        前向传播
        
        Args:
            tokens: 词索引张量 [batch_size, seq_len] 或 [seq_len]
            
        Returns:
            词嵌入向量 [batch_size, seq_len, d_model] 或 [seq_len, d_model]
        """
        # 应用嵌入矩阵并乘以sqrt(d_model)进行缩放
        embeddings = self.embedding(tokens) * math.sqrt(self.d_model)
        return embeddings


class SharedEmbedding(TokenEmbedding):
    """
    共享权重嵌入 - 编码器和解码器共享相同的嵌入矩阵
    
    在某些架构中，编码器的输入嵌入和解码器的输出投影层共享权重
    这可以减少参数数量，并可能提高性能
    """
    
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__(vocab_size, d_model, padding_idx)
        
    def get_embedding_weight(self):
        """获取嵌入权重，用于输出投影层"""
        return self.embedding.weight


class Embeddings(nn.Module):
    """
    完整的嵌入层 - 结合词嵌入和位置编码
    
    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        max_len: 最大序列长度
        padding_idx: 填充符号索引
        dropout: dropout比率
    """
    
    def __init__(self, vocab_size, d_model, max_len=5000, padding_idx=None, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
    def forward(self, tokens):
        """
        前向传播：词嵌入 + 位置编码
        
        Args:
            tokens: 词索引 [batch_size, seq_len]
            
        Returns:
            嵌入表示 [batch_size, seq_len, d_model]
        """
        # 词嵌入
        x = self.token_embedding(tokens)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.positional_encoding(x)   # [batch_size, seq_len, d_model]
        
        return x


def test_token_embedding():
    """测试词嵌入层"""
    print("=== 测试词嵌入层 ===")
    
    vocab_size, d_model, batch_size, seq_len = 1000, 16, 2, 5
    
    # 创建词嵌入层
    embedding = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    
    # 创建输入词索引
    tokens = torch.tensor([
        [1, 2, 3, 4, 0],  # 0是padding
        [5, 6, 0, 0, 0]   # 序列长度不同，用0填充
    ])
    
    print(f"输入词索引形状: {tokens.shape}")
    print(f"输入词索引:\n{tokens}")
    
    # 获取嵌入表示
    embeddings = embedding(tokens)
    print(f"嵌入表示形状: {embeddings.shape}")
    
    # 验证padding位置的嵌入是否为0
    padding_embed = embeddings[0, -1]  # 第一个序列的最后一个位置（padding）
    print(f"Padding位置嵌入范数: {padding_embed.norm()}")
    assert padding_embed.norm() < 1e-6, "Padding位置嵌入应该接近0"
    
    print("✅ 词嵌入层测试通过!")
    return embeddings


def test_complete_embeddings():
    """测试完整嵌入层（词嵌入 + 位置编码）"""
    print("\n=== 测试完整嵌入层 ===")
    
    vocab_size, d_model, batch_size, seq_len = 500, 8, 2, 4
    
    # 创建完整嵌入层
    embeddings = Embeddings(vocab_size, d_model, padding_idx=0)
    
    # 创建输入
    tokens = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 0, 0]   # 有padding
    ])
    
    print(f"输入词索引形状: {tokens.shape}")
    
    # 获取完整嵌入
    output = embeddings(tokens)
    print(f"完整嵌入形状: {output.shape}")
    
    # 验证形状
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✅ 完整嵌入层测试通过!")
    return output


if __name__ == "__main__":
    # 需要导入PositionalEncoding
    from .positional_encoding import PositionalEncoding
    
    test_token_embedding()
    test_complete_embeddings()