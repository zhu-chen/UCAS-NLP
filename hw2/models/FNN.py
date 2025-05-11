import torch 
from torch import nn as nn
from torch.nn import functional as F


class FNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size=2, hidden_dim=100):
        super(FNN, self).__init__()
        actual_context_size = 2 * context_size  # 左右各context_size个词
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(actual_context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        initrange = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        
        # 线性层使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        self.linear1.bias.data.zero_()
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        self.linear2.bias.data.zero_()

    def forward(self, input):
        """前向传播"""
        embeds = self.embeddings(input)  # [batch_size, context_size] -> [batch_size, context_size, embedding_dim]
        
        embeds = embeds.reshape(embeds.size(0), -1)  # [batch_size, context_size * embedding_dim]
        
        hidden = F.relu(self.linear1(embeds))
        out = self.linear2(hidden)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

