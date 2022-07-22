import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import DGNConv
from dgl.nn.pytorch import GlobalAttentionPooling
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from model.embedding import NodeEmbedding, PopEmbedding
from dgl import LaplacianPE
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score, mean_squared_error

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
        in_size, out_size, aggregators, scalers, dgn_layers, avg_d = net_params
        # 有向GNN
        self.dgn = nn.ModuleList([DGNConv(in_size=in_size,
                                          out_size=out_size,
                                          aggregators=aggregators,
                                          scalers=scalers,
                                          delta=avg_d) for _ in range(dgn_layers)])

        # 全局attention池化
        self.pooling = GlobalAttentionPooling(gate_nn=nn.Linear(out_size, 1))

    def forward(self, graph, node_feat, edge_feat, eig_vec):
        for dgn in self.dgn:
            h_t = dgn(graph, node_feat, edge_feat, eig_vec)
            node_feat = F.relu(h_t)
        graph.ndata['h'] = node_feat
        hg = self.pooling(graph, graph.ndata['h'])
        return hg

class GraphEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim, aggregators, scalers, dgn_layers,
                 avg_d, N, dropout_rate, k, seq_len, batch_size):
        super(GraphEmbedding, self).__init__()
        self.k = k
        self.dgn = DGN([d_model, hidden_dim, aggregators, scalers, dgn_layers,
                 avg_d])
        self.NE = NodeEmbedding(d_model, N, dropout_rate)

        self.outs = torch.zeros(batch_size, seq_len, hidden_dim)
        self.register_buffer("dgn_outs", self.outs)

    def forward(self, batch_graphs):
        # 要循环处理所有位置的级联图，保存结果
        for i, graph in enumerate(batch_graphs):
            # 节点输入特征
            node_input = self.NE(graph.ndata['nid'], graph.ndata['deg'], graph.ndata['time'])
            # 边特征
            edata = graph.edata
            # laplacian 编码
            transform = LaplacianPE(k=self.k, feat_name='eig')
            graph = transform(graph)
            eig = graph.ndata['eig']
            # dgn conv
            out = self.dgn(graph, node_input, edata, eig)
            self.outs[:, i, :] = out
        # batch_size, seq_len, hidden_dim
        return self.outs

class TransVAE(nn.Module):
    # VAE模块, 用于捕获不确定性
    def __init__(self, input_dim, hidden_dim):
        super(TransVAE, self).__init__()
        # node_emb = tf.keras.layers.Dense(FLAGS.emb_dim)(bn_casflow_inputs)
        # node_mean = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
        # node_log_var = tf.keras.layers.Dense(FLAGS.z_dim)(node_emb)
        # node_z = Sampling3D()((node_mean, node_log_var))
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

class PopEncoderDecoder(nn.Module):
    def __init__(self, d_model, hidden_dim, nhead, ffn_dim, dropout_rate, act, num_layers, is_vae):
        super(PopEncoderDecoder, self).__init__()
        # 判断使用哪个激活函数
        if act == 'relu':
            activation = F.relu
        else:
            activation = F.gelu
        # 建立Transformer encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
                                                     dropout=dropout_rate, activation=activation,
                                                     batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers)
        # 建立Transformer decoder
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
                                                     dropout=dropout_rate, activation=activation,
                                                     batch_first=True)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers)
        # 建立VAE
        self.is_vae = is_vae
        if self.is_vae:
            self.vae = TransVAE(d_model, hidden_dim)

    def forward(self, x, tgt, masks):
        en_h, mu, var = None, None, None
        # 先进行encoder
        en_x = self.encoder(x)
        # 将encoder的输出送到vae中捕获不确定性
        if self.is_vae:
            en_z, en_h, mu, var = self.vae(en_x)
        else: en_z = en_x
        # 将vae的输出送到decoder中充当memory
        de_out = self.decoder(tgt=tgt, memory=en_z, tgt_mask=masks)
        return de_out, en_z, en_h, en_x, mu, var

class GTAEModel(pl.LightningModule):
    def __init__(self,d_model, hidden_dim, aggregators, scalers, dgn_layers,
                 avg_d, N, dropout_rate, k, seq_len, batch_size, nhead, ffn_dim, act,
                 num_layers, is_ge, is_vae, is_reg, classes, lam):
        super(GTAEModel, self).__init__()
        self.seq_len = seq_len
        self.is_reg = is_reg
        self.is_vae = is_vae
        self.is_ge = is_ge
        self.classes = classes
        if self.is_ge:
            self.GE = GraphEmbedding(d_model, hidden_dim, aggregators, scalers, dgn_layers,
                    avg_d, N, dropout_rate, k, seq_len, batch_size)

        self.POPE = PopEmbedding(1, d_model, dropout_rate)
        self.PED = PopEncoderDecoder(d_model, hidden_dim, nhead, ffn_dim, dropout_rate, act,
                                     num_layers, is_vae)
        self.dropout = nn.Dropout(p=dropout_rate)
        # loss
        self.gate_loss = GTAELoss(self.is_reg, self.is_vae, lam)

        self.classifier = nn.Linear(hidden_dim, classes)
        self.regressier = nn.Linear(hidden_dim, 1)

        self.automatic_optimization = False

    def forward(self, graphs, pops, t_pos, tgt_pops, tgt_pos, masks):
        # 流行度数值嵌入+位置嵌入+时间位置嵌入
        pope = self.POPE(pops, t_pos)
        tgt_pope = self.POPE(tgt_pops, tgt_pos)
        # 级联图嵌入
        if self.is_ge:
            ge = self.GE(graphs)
            out = ge + pope
        else:
            out = pope
        out = self.dropout(out)
        # PED过程, 返回decoder的output, vae的隐藏状态, vae的output, encoder的output, 均值及方差
        de_out, en_z, en_h, en_x, mu, var = self.PED(out, tgt_pope, masks)
        # 分类预测
        if self.is_vae:
            class_out = self.classifier(en_z.mean(1))
        else:
            class_out = self.classifier(en_x.mean(1))
        # 回归预测
        regress_out = self.regressier(de_out)

        return class_out, regress_out, en_h, en_x, mu, var

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def training_step(self, batch_data, batch_idx):
    #     # batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch
    #     # decoder output, encoder vae output, encoder output, mu, var
    #     batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch_data
    #     # 切分ori和tgt的时间位置
    #     t_pos_ori, t_pos_tgt = t_pos_tr[:,:self.seq_len], t_pos_tr[:,self.seq_len:]
    #     class_out, regress_out, en_h, en_x, mu, var = self(batch_graphs, pop_tr, t_pos_ori, pop_labels_tr, t_pos_tgt, masks)
    #     class_loss, reg_loss, vae_loss = self.gate_loss(class_out, regress_out.squeeze(-1), en_h, en_x, mu, var, labels, pop_labels_tr)
    #     self.log_dict({'loss':class_loss, 'reg_loss':reg_loss, 'vae_loss':vae_loss})
    #     return class_loss

    def training_step(self, batch_data, batch_idx):
        # batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch
        # decoder output, encoder vae output, encoder output, mu, var
        opt = self.optimizers(use_pl_optimizer=True)
        batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch_data
        # 切分ori和tgt的时间位置
        t_pos_ori, t_pos_tgt = t_pos_tr[:, :self.seq_len], t_pos_tr[:, self.seq_len:]
        class_out, regress_out, en_h, en_x, mu, var = self(batch_graphs, pop_tr, t_pos_ori, pop_labels_tr, t_pos_tgt,
                                                           masks)
        class_loss, reg_loss, vae_loss = self.gate_loss(class_out, regress_out.squeeze(-1), en_h, en_x, mu, var, labels,
                                                        pop_labels_tr)
        opt.zero_grad()
        self.manual_backward(class_loss, retain_graph=True)
        opt.step()
        self.log_dict({'loss': class_loss, 'reg_loss': reg_loss, 'vae_loss': vae_loss})
        # return class_loss

    def validation_step(self, batch_data, batch_idx):
        batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch_data
        # 切分ori和tgt的时间位置
        t_pos_ori, t_pos_tgt = t_pos_tr[:, :self.seq_len], t_pos_tr[:, self.seq_len:]
        class_out, regress_out, en_h, en_x, mu, var = self(batch_graphs, pop_tr, t_pos_ori, pop_labels_tr, t_pos_tgt,
                                                           masks)
        # loss = self.gate_loss(class_out, regress_out, en_h, en_x, mu, var, labels, pop_labels_tr)
        return {'class_pre':class_out, 'regress_pre':regress_out.squeeze(-1), 'classes':labels, 'regress':pop_labels_tr}

    def validation_epoch_end(self, outputs):
        class_pre = torch.cat([o['class_pre'] for o in outputs])
        regress_pre = torch.cat([o['regress_pre'] for o in outputs])
        classes = torch.cat([o['classes'] for o in outputs])
        regress = torch.cat([o['regress'] for o in outputs])
        acc = accuracy(preds=class_pre.softmax(-1), target=classes)
        mse = mean_squared_error(preds=regress_pre, target=regress)
        weight_acc = accuracy(preds=class_pre.softmax(-1), target=classes, average='weighted', num_classes=self.classes)
        self.log_dict({'val_acc':acc, 'val_mse':mse, 'val_weight_acc':weight_acc})

    def test_step(self, batch_data, batch_idx):
        batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch_data
        # 切分ori和tgt的时间位置
        t_pos_ori, t_pos_tgt = t_pos_tr[:, :self.seq_len], t_pos_tr[:, self.seq_len:]
        class_out, regress_out, en_h, en_x, mu, var = self(batch_graphs, pop_tr, t_pos_ori, pop_labels_tr, t_pos_tgt,
                                                           masks)
        # loss = self.gate_loss(class_out, regress_out, en_h, en_x, mu, var, labels, pop_labels_tr)
        return {'class_pre':class_out, 'regress_pre':regress_out.squeeze(-1), 'classes':labels, 'regress':pop_labels_tr}

    def test_epoch_end(self, outputs):
        class_pre = torch.cat([o['class_pre'] for o in outputs])
        regress_pre = torch.cat([o['regress_pre'] for o in outputs])
        classes = torch.cat([o['classes'] for o in outputs])
        regress = torch.cat([o['regress'] for o in outputs])
        acc = accuracy(preds=class_pre.softmax(-1), target=classes)
        mse = mean_squared_error(preds=regress_pre, target=regress)
        weight_acc = accuracy(preds=class_pre.softmax(-1), target=classes, average='weighted')
        self.log_dict({'test_acc':acc, 'test_mse':mse, 'test_weight_acc':weight_acc})

    @staticmethod
    def setting_model_args(parent_parser):
        parser = parent_parser.add_argument_group("GTAE")
        parser.add_argument('--d_model', type=int, default=32, help="The input dimension of models.")
        parser.add_argument('--hidden_dim', type=int, default=32, help="The hidden dimension.")
        parser.add_argument('--classes', type=int, default=282, help="Num of classes.")
        parser.add_argument('--dgn_layers', type=int, default=2, help="Num of graph encoder layers.")
        parser.add_argument('--aggregators', type=str, default='dir1-av,dir1-dx,sum', help="aggregators.")
        parser.add_argument('--scalers', type=str, default='identity,amplification', help="scalers.")
        parser.add_argument('--act', type=str, default='gelu', help="Active function.")
        parser.add_argument('--avg_d', type=float, default=2.1, help="The average degree.")
        parser.add_argument('--lam', type=float, default=1.0, help="The weight of classification loss.")
        parser.add_argument('--N', type=int, default=365756, help="Num of Nodes.")
        parser.add_argument('--seq_len', type=int, default=6, help="input length.")
        parser.add_argument('--nhead', type=int, default=8, help='The num heads of attention.')
        parser.add_argument('--ffn_dim', type=int, default=128, help='FFN dimension.')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout prob.')
        parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='Attention dropout prob.')
        parser.add_argument('--num_layers', type=int, default=6, help="Num of Transformer encoder layers.")
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
        parser.add_argument('--k', type=int, default=3, help="The smallest k eigenvalues.")
        parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
        parser.add_argument('--clip_val', type=float, default=5.0, help="Gradient clipping values. ")
        parser.add_argument('--total_epochs', type=int, default=100, help="Max epochs of model training.")
        parser.add_argument('--lr_decay_step', type=int, default=25, help="Learning rate decay step size.")
        parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
        parser.add_argument('--gpu_lst', nargs='+', type=int, help="Which gpu to use.")
        # parser.add_argument('--accelerator', nargs='cpu', type=int, help="Which gpu to use.")
        # parser.add_argument('--device', nargs='+', type=int, help="Which gpu to use.")
        parser.add_argument('--observation', type=str, default="")
        parser.add_argument('--data_name', type=str, default="weibo")
        parser.add_argument('--is_ge', action='store_false', default="add cascade graph embedding.")
        parser.add_argument('--is_vae', action='store_false', default="add vae module.")
        parser.add_argument('--is_reg', action='store_false', default="add regression task loss.")
        return parent_parser

class GTAELoss(nn.Module):
    def __init__(self, is_reg, is_vae, lam):
        super(GTAELoss, self).__init__()
        self.is_reg = is_reg
        self.is_vae = is_vae
        self.lam = lam
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, class_out, regress_out, en_h, en_x, mu, var, labels, pop_labels_tr):
        # node_ce_loss = tf.reduce_mean(tf.square(bn_casflow_inputs - node_rec))
        # node_kl_loss = -.5 * tf.reduce_mean(node_log_var - tf.square(node_mean) - tf.exp(node_log_var) + 1)
        class_loss, regress_loss, ce_loss, kl_loss = 0, 0, 0, 0
        #分类loss
        class_loss = self.lam * self.cross_entropy(class_out, labels)
        # 回归loss
        if self.is_reg:
            regress_loss = self.mse(regress_out, pop_labels_tr)
            class_loss = class_loss + regress_loss
        # vae loss
        if self.is_vae:
            ce_loss = torch.mean(torch.square(en_h - en_x))
            kl_loss = -.5 * torch.mean(var - torch.square(mu) - torch.exp(var) + 1)
            class_loss = class_loss + ce_loss + kl_loss
        return class_loss, regress_loss, ce_loss+kl_loss
