import torch 
from torch import nn as nn
from torch.nn import functional as F

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=100):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        # 嵌入层使用均匀分布初始化
        initrange = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        
        # RNN 权重使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.rnn.weight_ih_l0, nonlinearity='tanh')
        nn.init.orthogonal_(self.rnn.weight_hh_l0)  # 对于循环连接权重，使用正交初始化
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()
        
        # 线性层使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        self.linear.bias.data.zero_()

    def forward(self, input):
        """前向传播"""
        embeds = self.embeddings(input)  
        rnn_out, _ = self.rnn(embeds)    
        out = self.linear(rnn_out[:, -1, :])  # 取最后一个时间步的输出
        log_probs = F.log_softmax(out, dim=1)
        return log_probs