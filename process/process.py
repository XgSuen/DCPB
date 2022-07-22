import math
import numpy as np
from global_params import get_params
from itertools import groupby
import pickle as pkl

def process_data(dataname):
    observation, unit_time, span, window_size, unit_size, rpath,_, wpath = get_params(dataname)
    # all pops
    data_pops = []
    # 窗口总数
    windows = int((unit_time / window_size) * span)
    # 每个窗口包含的单位窗口的数量
    wu = int(window_size // unit_size)
    # 观测窗口数量
    ow = int(observation // window_size)
    with open(rpath, 'r') as f1, open(wpath + '{}.txt'.format(dataname), 'w') as f2:
        for line in f1:
            # cid, rid, pub_time, pop, cascade,
            cid, rid, pub_time, pop, cascade = line.strip().split('\t')
            cascade_l = cascade.split(' ')
            sort_cascade_l = sorted(cascade_l, key=lambda x:int(x.split(':')[-1]))
            # 转发时间列表
            rep_time_list = [float(cas.split(':')[-1]) for cas in sort_cascade_l]
            # 单位窗口内的流行度列表
            pops = [0] * (wu * windows + 1)
            # 按设定单位大小统计流行度
            for k, g in groupby(rep_time_list, key=lambda x:x//unit_size):
                pops[int(k)] = len(list(g))
            # 处理转发时间边界情况
            pops[-2] = pops[-2] + pops[-1]
            pops = pops[:-1]
            # 按窗口内单位窗口数量reshape
            pop_arr = np.array(pops).reshape((-1, wu))
            # 求和得到窗口内流行度
            each_wd_pop = np.sum(pop_arr, -1)
            data_pops.extend(list(each_wd_pop))
            # 过滤掉观测时间内爆发的级联
            label = np.argmax(each_wd_pop)
            if label < ow:
                continue
            pop_list = list(map(str, pops))[:ow * wu]
            pop_label = list(map(str, each_wd_pop))[ow:]
            f2.write(cid + '\t' + rid + '\t' + pub_time + '\t' + pop + '\t' + ' '.join(sort_cascade_l) +
                     '\t' + ' '.join(pop_list) + '\t' + str(label) + '\t' + ' '.join(pop_label) + '\n')
    with open(wpath + 'all_pops.pkl', 'wb') as f3:
        pkl.dump([np.log2(p+1) for p in data_pops], f3)
process_data(dataname='weibo')