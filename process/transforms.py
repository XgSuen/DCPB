import networkx as nx
import numpy as np
import scipy as sp
import pickle
from global_params import get_params
from itertools import groupby
import time

dataname = 'weibo'
observation, unit_time, span, window_size, unit_size, _, _, wpath = get_params(dataname)
# 读取re-id，node-to-time和node-to-degree的三个字典文件，node2t为双重字典，级联id对应该级联中所有节点所对应的时间
with open(wpath + 'node2ind.pkl', 'rb') as f:
    node2ind = pickle.load(f)
with open(wpath + 'node2time.pkl', 'rb') as f:
    node2t = pickle.load(f)
with open(wpath + 'degrees.pkl', 'rb') as f:
    node2deg = pickle.load(f)

# hash度数
def hash_deg():
    degs = sorted(set(node2deg.values()))
    hash_degs = {k:v for v,k in enumerate(degs)}
    return hash_degs
hash_degs = hash_deg()

def transform(type = 'train'):
    transformed = {"cid":[], "pubt":[], "graph_lst":[], "pop_lst":[], "t_pos":[], "classes":[], "pop_label":[]}
    with open(wpath + type + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            cid, rid, pub_time, pop, cascade, pops, classes, pop_label = line.split('\t')
            transformed["cid"].append(cid)
            transformed["pubt"].append(pub_time)
            transformed["classes"].append(int(classes))
            transformed["pop_label"].append([float(p) for p in pop_label.split(' ')])
            # 将观测流行度转换为多段 window_size // unit_size 大小的子序列所构成的序列
            pops_arr = np.array([float(p) for p in pops.split(' ')])
            pops_arr = pops_arr.reshape((-1, window_size // unit_size))
            # 暂时放弃子序列
            transformed["pop_lst"].append(pops_arr.sum(1).tolist())
            # 生成时间位置
            timestamps = [int(pub_time) + window_size * (i + 1) for i in range(int(span * unit_time / window_size))]
            t_positions = []
            for stamp in timestamps:
                realtime = time.localtime(stamp)
                d, h, m, s = realtime.tm_mday - 1, realtime.tm_hour, realtime.tm_min, realtime.tm_sec
                t_positions.append([d, h, m, s])
            transformed['t_pos'].append(t_positions)
            # 按照时间切分级联图
            g = nx.DiGraph()
            edges = []
            g_list = []
            for i, path in enumerate(cascade.split(" ")):
                nodes = path.split(":")[0].split("/")
                for j, node in enumerate(nodes):
                    # 处理root节点因自转发导致时间不为0的情况
                    if j == 0:
                        spant = 0
                    else:
                        spant = int(node2t[cid][node])
                    realtime = time.localtime(int(pub_time) + spant)
                    d, h, m, s = realtime.tm_mday-1, realtime.tm_hour, realtime.tm_min, realtime.tm_sec
                    # 先将级联所有节点添加到图中，包括其影响时间和全局影响力（度数）
                    g.add_node(node2ind[node], time=[d, h, m, s], deg=hash_degs[node2deg[node]], nid=node2ind[node])
                # 统计所有的边，并按照从小到大的顺序排列
                if i == 0:
                    # 为根节点添加自环边
                    edges.append((node2ind[nodes[0]],node2ind[nodes[0]],0))
                else:
                    for j in range(1, len(nodes)):
                        edges.append((node2ind[nodes[j-1]],node2ind[nodes[j]],node2t[cid][nodes[j]]))
            edges = sorted(edges, key=lambda x:int(x[-1]))
            edges_dict = dict()
            # 按照窗口大小划分边
            for k, v in groupby(edges, lambda x:int(x[-1]) // window_size):
                edges_dict[k] = list(v)
            # 遍历所有组的边，生成对应子级联图
            for i in range(int(observation // window_size)):
                if i in edges_dict.keys():
                    tempg = g.copy()
                    # 将相对时间归一化
                    wt_edges = [(edge[0], edge[1], float(edge[2])/observation) for edge in edges_dict[i]]
                    tempg.add_weighted_edges_from(wt_edges)
                    g = tempg.copy()
                else:
                    tempg = g.copy()
                g_list.append(tempg)
            transformed["graph_lst"].append(g_list)
    with open(wpath + type + '.pkl', 'wb') as f:
        pickle.dump(transformed, f)

print("generate train val test...")
transform(type = 'train')
print("generate train finish!")
transform(type = 'val')
print("generate val finish!")
transform(type = 'test')
print("generate test finish!")
print("done")