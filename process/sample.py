import networkx as nx
import numpy as np
from global_params import get_params
import pickle
import random

# 设置seed
seed = 512
random.seed(seed)
np.random.seed(seed)

def sample(path, degrees):
    path_deg = []
    random.shuffle(path)
    # 获取每条路路径的累积度数
    for p in path:
        nodes = p.split(':')[0].split('/')
        deg = 0
        for node in nodes:
            deg += degrees[node] + 1
        path_deg.append(deg)
    # 找到最大度数的512条路径
    index = np.argsort(path_deg)[::-1][:512]
    new_path = np.array(path)[index]
    # 按时间排序并返回
    return list(sorted(new_path, key=lambda x:int(x.split(':')[-1])))

def sample_cascade(dataname):
    observation, _, _, _, _, _, rppath, wpath = get_params(dataname)
    count = 0
    with open(rppath, 'r') as f1, open(wpath + 'degrees.pkl', 'rb') as f2, open(wpath + 'weibo_sample.txt', 'w') as f3:
        degrees = pickle.load(f2)
        for line in f1:
            opop = 0
            cid, rid, pub_time, pop, cascade, pops, label, pop_label = line.split('\t')
            observation_path = []
            all_nodes = []
            for c in cascade.split(' '):
                flag2 = 0
                time = int(c.split(':')[-1])
                nodes = c.split(':')[0].split('/')
                if time < observation:
                    # 去除重复转发和多次转发
                    flag1 = sum(map(lambda x: nodes[-1] + ":" in x, observation_path))
                    if flag1 > 0: continue
                    # 去除之前未出现的中间转发节点路径
                    if len(nodes) >= 3:
                        for node in nodes[1:-1]:
                            if node not in all_nodes:
                                flag2 = 1
                                break
                    if flag2: continue
                    observation_path.append(c)
                    opop += 1
                else:
                    break
                all_nodes.extend(nodes)
            # 过滤掉观测长度小于10的级联
            if opop < 5:
                continue
            elif opop > 512:
                new_path = sample(observation_path, degrees)
                f3.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(new_path) +
                         '\t' + pops + '\t' + str(label) + '\t' + pop_label)
                count += 1
            else:
                f3.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(observation_path) +
                     '\t' + pops + '\t' + str(label) + '\t' + pop_label)
                count += 1
    print("Num of cascades: {}".format(count))

sample_cascade(dataname='weibo')