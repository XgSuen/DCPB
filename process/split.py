import pickle
import numpy as np
from global_params import get_params
import random
from collections import Counter

seed = 512
random.seed(seed)

def write_lines(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)

dataname = 'weibo'
observation, _, _, _, _, _, _, wpath = get_params(dataname)
train_txt_path = wpath + 'train.txt'
val_txt_path = wpath + 'val.txt'
test_txt_path = wpath + 'test.txt'
nodes_path = wpath + 'nodes.pkl'
data_path = wpath + '{}_sample.txt'.format(dataname)

val_test_ratio = 0.3

print("split train test val......")
with open(data_path, 'r') as f1:
    lines = f1.readlines()
    samples = len(lines)
    val_samples = []
    test_samples = []
    train_samples = []
    nodes = []
    sort_lines = sorted(lines, key=lambda x:int(x.split('\t')[6]))
    labels = [l.split('\t')[6] for l in sort_lines]
    class_count = list(Counter(labels).values())
    i = 0
    k = 0
    for line in sort_lines:
        cascades = line.split('\t')[4].split(' ')
        for c in cascades:
            cnode = c.split(':')[0].split('/')
            nodes.extend(cnode)
        # 切分数据集 7:1.5:1.5
        cnum = class_count[i]
        labels = line.split('\t')[6]
        # 如果类别数量大于3，直接切分
        if int(labels) > 3:
            # 判断切分值是否为偶数，若为奇数，则加1，保证平等切分
            split_n = int(cnum * val_test_ratio)
            split_n = split_n + 1 if split_n % 2 == 1 else split_n
            if k < split_n // 2:
                test_samples.append(line)
            elif k < split_n:
                val_samples.append(line)
            else:
                train_samples.append(line)
        # 如果类别数量小于等于3，即为2，3，则50%切给训练集，剩下的复制一份分别切给测试集和验证集
        else:
            if k < cnum * 0.5:
                train_samples.append(line)
            else:
                # 当样本量小于3时，复制一个样本给验证集和测试集
                val_samples.append(line)
                test_samples.append(line)
        k += 1
        if k == cnum:
            i += 1
            k = 0
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    random.shuffle(val_samples)
    write_lines(train_txt_path, train_samples)
    write_lines(val_txt_path, val_samples)
    write_lines(test_txt_path, test_samples)
    with open(wpath + "nodes.pkl", 'wb') as f:
        pickle.dump(set(nodes), f)
    node2ind = dict()
    for i, node in enumerate(set(nodes)):
        node2ind[node] = i+1
    with open(wpath + "node2ind.pkl", 'wb') as f:
        pickle.dump(node2ind, f)
    print("finish!")
    print("train_samples: ", len(train_samples))
    print("val_samples: ", len(val_samples))
    print("test_samples: ", len(test_samples))
    print("Num of nodes: ", len(set(nodes)))

