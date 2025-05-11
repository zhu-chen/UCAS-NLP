import torch 
from torch import nn as nn
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=100):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        initrange = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        
        # LSTM 权重使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.lstm.weight_ih_l0, nonlinearity='sigmoid')
        nn.init.orthogonal_(self.lstm.weight_hh_l0)  # 对于循环连接权重，使用正交初始化
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        
        # 设置遗忘门的偏置为正值，以减少梯度消失问题
        n = self.lstm.bias_ih_l0.size(0)
        forget_gate_bias = 1.0
        self.lstm.bias_ih_l0.data[n//4:n//2].fill_(forget_gate_bias)
        
        # 线性层使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        self.linear.bias.data.zero_()

    def forward(self, input):
        """前向传播"""
        embeds = self.embeddings(input)  
        lstm_out, _ = self.lstm(embeds)  
        out = self.linear(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        log_probs = F.log_softmax(out, dim=1)
        return log_probs