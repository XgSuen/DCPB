# 存储处理全局数据集的参数
def get_params(data_name='weibo'):
    if data_name == 'weibo':
        # 观测时间
        observation = 3600*0.5
        # 单位时间
        unit_time = 3600
        # 整个时间跨度
        span = 24
        # 划分窗口大小
        window_size = 300
        # 单位窗口大小
        unit_size = 60
        # 读入原文件路径
        rpath = '../dataset/weibo/dataset.txt'
        # 读入已处理文件路径
        rppath = '../dataset/weibo/weibo.txt'
        # 写入文件路径
        wpath = '../dataset/weibo/'
    return observation, unit_time, span, window_size, unit_size, rpath, rppath, wpath