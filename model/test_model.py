import torch
import dgl
import torch.nn as nn
from argparse import ArgumentParser
from MyDataset import get_batch_data
from embedding import NodeEmbedding, PopEmbedding
from GTAE import DGN, GTAEModel
from dgl import LaplacianPE
# from GTAE import MircoTransformer, TransVAE

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path = '../dataset/weibo/train.pkl'
loader = get_batch_data(path, 4)
parser = ArgumentParser()
parser = GTAEModel.setting_model_args(parser)
args = parser.parse_args()

gtae = GTAEModel(args.d_model, args.hidden_dim, args.aggregators.split(','), args.scalers.split(','), args.dgn_layers,
                 args.avg_d, args.N, args.dropout_rate, args.k, args.seq_len, args.batch_size,
                 args.nhead, args.ffn_dim, args.act, args.num_layers, args.is_ge, args.is_vae, args.is_reg, args.classes,
                 args.lam)
print(gtae)
for i, batch in enumerate(loader):
    batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids, masks = batch
    # print(len(batch_graphs))
    # print(pop_tr.size())
    # print(t_pos_tr.size())
    # print(labels.size())
    # print(pop_labels_tr.size())
    # print(masks.size())
    class_out, regress_out, en_h, en_x, mu, var = gtae(batch_graphs, pop_tr, t_pos_tr[:,:6], pop_labels_tr, t_pos_tr[:,6:], masks)
    print(class_out)
    print(class_out.size())
    print(regress_out.squeeze(-1))
    print(regress_out.squeeze(-1).size())
    break
    # print(t_pos_tr.size())
    # NE, PE = NodeEmbedding(32, 365756, 0.1), PopEmbedding(1, 32, 0.1)
    # batch_graph = batch_graphs[0]
    # node_input = NE(batch_graph.ndata['nid'], batch_graph.ndata['deg'], batch_graph.ndata['time'])
    # print(node_input.size())
    # print(node_input)
    # pop_input = PE(pop_tr.unsqueeze(1), t_pos_tr[:,:6])
    # print(pop_input.size())
    # print(pop_input)
    # print(batch_graph.adjacency_matrix(transpose=True))


    # net_params = {}
    # net_params['in_size'] = 32
    # net_params['out_size'] = 32
    # net_params['aggregators'] = ['dir1-av', 'dir1-dx', 'sum']
    # net_params['scalers'] = ['identity', 'amplification']
    # net_params['L'] = 2
    # net_params['avg_d'] = 2.1
    # net_params['d_model'] = 32
    # net_params['nhead'] = 8
    # net_params['ffn_dim'] = 64
    #
    # dgn = DGN(net_params)
    # transform = LaplacianPE(k=3, feat_name='eig')
    # batch_graph = transform(batch_graph)
    # eig = batch_graph.ndata['eig']
    # h = dgn(batch_graph, node_input, batch_graph.edata, eig)
    #
    # mt = MircoTransformer(net_params)
    # h1 = mt(pop_input)
    #
    # net_params['input_dim'] = 32
    # net_params['hidden_dim'] = 64
    # vae = TransVAE(net_params)
    # z2, h2, mu, var = vae(h1+h)
    #
    # print(h)
    # print(h1)
    #
    # print("z2:",z2)
    # print("h2:",h2)
    # print(mu)
    # print(var)
    # break