import torch 
from torch import nn as nn
from torch.nn import functional as F


class FNN(nn.Module):
    def __init__(self,vocab_size, embedding_dim,context_size=2,hidden_dim=100):
        super(FNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        initrange = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear1.weight.data.uniform_(-0, 0)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.uniform_(-0, 0)
        self.linear2.bias.data.zero_()

    def forward(self, input):
        """前向传播"""
        embeds = self.embeddings(input)
        hidden = F.relu(self.linear1(embeds))
        out = self.linear2(hidden)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

