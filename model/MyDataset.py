import torch
import torch.nn as nn
import torch.utils.data as Data
import networkx as nx
import numpy as np
import pickle
from collections import Counter
import dgl
import pprint

class CasData(Data.Dataset):
    def __init__(self, path):
        super(CasData, self).__init__()
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        # cid, graph_lst, pop_lst, classes, pop_label
        self.cid = dataset['cid']
        self.graph_lst = dataset['graph_lst']
        self.pop_lst = dataset['pop_lst']
        self.t_pos = dataset['t_pos']
        minclass = min(dataset['classes'])
        self.classes = [c - minclass for c in dataset['classes']]
        self.pop_label = dataset['pop_label']

    def __getitem__(self, item):
        return self.cid[item], self.graph_lst[item], self.pop_lst[item], \
            self.t_pos[item], self.classes[item], self.pop_label[item]

    def __len__(self):
        return len(self.cid)

def collen_fn(batch):
    """
     1. 将networkx的graph转为dgl graph，并batch起来
     2. 将输入流行度序列转换为bx6x5的tensor
    :param batch:
    :return: batch_graphs, pop_tr, labels, pop_labels_tr, cids
    """
    cid, graph_lst, pop_lst, t_pos, classes, pop_label = zip(*batch)
    # 处理batch graph，将bath中对应每个位置的图batch起来，生成batch序列
    batch_graphs = []
    for i in range(len(graph_lst[0])):
        batch_g = [gs[i] for gs in graph_lst]
        dgl_graph = [dgl.from_networkx(g, node_attrs=['time', 'deg', 'nid'], edge_attrs=['weight']) for g in batch_g]
        batch_graphs.append(dgl.batch(dgl_graph))
    pop_tr = torch.FloatTensor(pop_lst)
    t_pos_tr = torch.LongTensor(t_pos)
    labels = torch.LongTensor(classes)
    pop_labels_tr = torch.FloatTensor(pop_label)
    cids = torch.LongTensor([int(c) for c in cid])
    b, l = pop_labels_tr.size(0), pop_labels_tr.size(1)
    masks = torch.FloatTensor([torch.tril(torch.ones(l, l)) for _ in range(b)])

    return batch_graphs, pop_tr, t_pos_tr, labels, pop_labels_tr, cids

def get_batch_data(path, batch_size, type='train'):
    dataset = CasData(path)
    if type == 'train':
        # Class-balanced sampling
        with open(path, 'rb') as f:
            train_data = pickle.load(f)
        classes = train_data['classes']
        # 计算各个类别的样本数量
        counts = Counter(classes)
        # 将每个类别的采样权重设置为1/C, 加一个小值协调采样概率
        weights = {k:1/(v+1/len(classes)) for k, v in counts.items()}
        sample_weights = torch.tensor([weights[t] for t in classes])
        # 生成采样策略
        sampler = Data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=False)  # replacement控制是否重复采样
        train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collen_fn, sampler=sampler)
        return train_loader
    else:
        return Data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collen_fn)

# if __name__ == '__main__':
#     path = '../dataset/weibo/train.pkl'
#     loader = get_batch_data(path, 64)
#     for i, batch in enumerate(loader):
#         print(i)
        # pop = batch[1]
        # print("pop", pop)
        # pos = batch[2]
        # print("pos", pos)
        # bg = batch[0][2]
        # print("batch", bg.batch_size)
        # print("time:", bg.ndata['time'])
        # print("deg:", bg.ndata['deg'])
        # print("nid:", bg.ndata['nid'])
        # print("weight:", bg.edata['weight'])
        # break