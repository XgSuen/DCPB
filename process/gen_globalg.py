import networkx as nx
import numpy as np
from global_params import get_params
import pickle

def get_graph(dataname):
    observation, _, _, _, _, _, rppath, wpath = get_params(dataname)
    # 生成全局图
    node2time = dict()
    G = nx.DiGraph()
    with open(rppath, 'r') as f:
        for line in f:
            cascade = line.split('\t')[4].split(' ')
            n2t = dict()
            cid = line.split('\t')[0]
            for i, c in enumerate(cascade):
                edge = c.split(':')[0].split('/')
                t = c.split(':')[-1]
                n2t[edge[-1]] = t
                if i == 0:
                    G.add_node(edge[0])
                else:
                    G.add_edge(edge[-2], edge[-1])
            node2time[cid] = n2t
    with open(wpath + 'global_g.pkl', 'wb') as f:
        pickle.dump(G, f)
    degrees = {dg[0]:dg[1] for dg in G.out_degree}
    with open(wpath + 'degrees.pkl', 'wb') as f:
        pickle.dump(degrees, f)
    with open(wpath + 'node2time.pkl', 'wb') as f:
        pickle.dump(node2time, f)
    print("Num of nodes:{}".format(G.number_of_nodes()))
    print("Num of times:{}".format(len(node2time)))
get_graph(dataname='weibo')