import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import DGNConv
from dgl.nn.pytorch import GlobalAttentionPooling
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

class PeerLevelLayer(nn.Module):
    """
    对等层，让时间序列的隐藏向量信息和参数与图的保持一致
    """
    def __init__(self, net_params):
        super(PeerLevelLayer, self).__init__()
        in_size = net_params['in_size']
        self.linear1 = nn.Linear(in_size, 512)
        self.linear2 = nn.Linear(512, in_size)
        self.layernorm = nn.LayerNorm(in_size)
        self.dropot = nn.Dropout(p=net_params['dropout_rate'])

    def forward(self, x):
        x = self.dropot(x)
        out = self.linear1(x)
        out = self.linear2(out)
        # post norm
        return self.layernorm(x + out)



class DGN(nn.Module):
    """
    有向GNN, 用于生成级联图表示
    """
    def __init__(self, net_params):
        super(DGN, self).__init__()
        # 有向GNN的参数
        in_size = net_params['in_size']
        out_size = net_params['out_size']
        # 选择聚合方法, mean, max ,,,
        aggregators = net_params['aggregators']
        scalers = net_params['scalers']
        num_layers = net_params['L']
        # 平均度数
        avg_d = net_params['avg_d']

        self.dgn = nn.ModuleList([DGNConv(in_size=in_size,
                                          out_size=out_size,
                                          aggregators=aggregators,
                                          scalers=scalers,
                                          delta=avg_d) for _ in range(num_layers)])
        # 全局attention池化
        self.pooling = GlobalAttentionPooling(gate_nn=nn.Linear(out_size, 1))

    def forward(self, graph, node_feat, edge_feat, eig_vec):
        for dgn in self.dgn:
            h_t = dgn(graph, node_feat, edge_feat, eig_vec)
            node_feat = h_t
        graph.ndata['h'] = node_feat

        hg = self.pooling(graph, graph.ndata['h'])
        return hg


class MircoTransformer(nn.Module):
    """
    微观Transformer模块, 用于嵌入流行度序列
    """
    def __init__(self, net_params):
        super(MircoTransformer, self).__init__()
        layers = net_params['L']
        d_model = net_params['d_model']
        nhead = net_params['nhead']
        ffn_dim = net_params['ffn_dim']
        self.mts = nn.ModuleList([TransformerEncoderLayer(d_model = d_model, nhead=nhead,
                                 dim_feedforward=ffn_dim, batch_first=True) for _ in range(layers)])

    def forward(self, x):
        # 4, 6, 32
        for mt in self.mts:
            x = mt(x)
        return x.sum(-2)

class TransVAE(nn.Module):
    # VAE模块, 用于捕获不确定性
    def __init__(self, net_params):
        super(TransVAE, self).__init__()
        # node_emb = tf.keras.layers.Dense(FLAGS.emb_dim)(bn_casflow_inputs)
        # node_mean = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
        # node_log_var = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
        # node_z = Sampling3D()((node_mean, node_log_var))
        input_dim = net_params['input_dim']
        hidden_dim = net_params['hidden_dim']
        self.layer1 = nn.Linear(input_dim, input_dim)
        self.layer2 = nn.Linear(input_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, hidden_dim)
        self.var_layer = nn.Linear(hidden_dim, hidden_dim)

        self.layer3 = nn.Linear(hidden_dim, input_dim)
        self.layer4 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # encoder
        h = self.layer1(x)
        h = self.layer2(h)
        mu = self.mean_layer(h)
        var = self.mean_layer(h)

        #sample
        std = torch.exp(0.5*var)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)

        # decoder
        h1 = self.layer3(z)
        h1 = self.layer4(h1)

        return z, h1, mu, var

class PopDecoder(nn.Module):
    def __init__(self, net_params):
        super(PopDecoder, self).__init__()
        layers = net_params['L']
        d_model = net_params['d_model']
        nhead = net_params['nhead']
        ffn_dim = net_params['ffn_dim']
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=ffn_dim)
                                      for _ in range(layers)])

    def forward(self, x):
        for de in self.decoder:
            x = de(x)
        return x.sum(-2)

class GTAELoss(nn.Module):
    def __init__(self):
        super(GTAELoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, x, h, mu, var):
        # node_ce_loss = tf.reduce_mean(tf.square(bn_casflow_inputs - node_rec))
        # node_kl_loss = -.5 * tf.reduce_mean(node_log_var - tf.square(node_mean) - tf.exp(node_log_var) + 1)
        ce_loss = torch.mean(torch.square(x - h))
        kl_loss = -.5 * torch.mean(var - torch.square(mu) - torch.exp(var) + 1)
