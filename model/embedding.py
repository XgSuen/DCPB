import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class PositionalEncoding(nn.Module):
    """
    非参数更新的固定位置编码
    """
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        # 初始化pe, 并设置非梯度传播
        pe = torch.zeros(max_len, d_model, requires_grad=False)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)

        pe = pe.unsqueeze(0)
        # 不会更新，但会作为模型的一部分被保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEncoding(nn.Module):
    """
    对于输入的pop序列进行一维卷积，卷积核大小为3，将其映射到d_model维度
    """
    def __init__(self, c_in, d_model):
        super(TokenEncoding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x

class TemporalEncoding(nn.Module):
    """
    可学习的时间编码，包括day，hour，minute 和 second
    """
    def __init__(self, d_model):
        super(TemporalEncoding, self).__init__()
        day=2; hour = 24; minute=60; second=60

        Embed = nn.Embedding

        self.day_embed = Embed(day, d_model)
        self.hour_embed = Embed(hour, d_model)
        self.minute_embed = Embed(minute, d_model)
        self.second_embed = Embed(second, d_model)

    def forward(self, x, embed_type='pop'):
        x = x.long()
        if embed_type == 'pop':
            day_x = self.day_embed(x[:, :, 0])
            hour_x = self.hour_embed(x[:, :, 1])
            minute_x = self.minute_embed(x[:, :, 2])
            second_x = self.second_embed(x[:, :, 3])
        else:
            day_x = self.day_embed(x[:,0])
            hour_x = self.hour_embed(x[:,1])
            minute_x = self.minute_embed(x[:,2])
            second_x = self.second_embed(x[:,3])

        return day_x + hour_x + minute_x + second_x

class InfluenceEncoding(nn.Module):
    """
    度影响力编码，将节点度数映射为可学习向量
    """
    def __init__(self, d_model):
        super(InfluenceEncoding, self).__init__()
        Embed = nn.Embedding
        deg_count = 2048
        self.inf_embed = Embed(deg_count, d_model)

    def forward(self, x):
        x = x.long()
        inf_x = self.inf_embed(x)
        return inf_x

class InitalEmbedding(nn.Module):
    """
    节点初始化编码，所有节点向量均为可学习向量
    """
    def __init__(self, d_model, N):
        super(InitalEmbedding, self).__init__()
        self.embed = nn.Embedding(N, d_model)

    def forward(self, x):
        x = x.long()
        return self.embed(x)

class NodeEmbedding(nn.Module):
    """
    级联节点的输入合成
    """
    def __init__(self, d_model, N, dropout):
        super(NodeEmbedding, self).__init__()

        self.init_embedding = InitalEmbedding(d_model, N)
        self.inf_embedding = InfluenceEncoding(d_model)
        self.temp_embedding = TemporalEncoding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ids, deg, time):
        x = self.init_embedding(ids) + self.inf_embedding(deg) + self.temp_embedding(time, embed_type='node')
        return self.dropout(x)

class PopEmbedding(nn.Module):
    """
    流行度序列嵌入
    """
    def __init__(self, c_in, d_model, dropout):
        super(PopEmbedding, self).__init__()

        self.token_embedding = TokenEncoding(c_in, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.temp_embedding = TemporalEncoding(d_model)

        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t_pos):
        x = x.unsqueeze(1)
        x = self.token_embedding(x).permute(0, 2, 1)
        y = x + self.pos_embedding(x) + self.temp_embedding(t_pos, embed_type='pop')
        return y

