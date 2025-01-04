import logging
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import Utils

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def SIR_betas(G, a_list, root_path, n_jobs=-1):
    """
    不同 beta 情况下的 SIR 模型仿真
    参数:
        G: 网络图
        a_list: 存放传播概率是传播阈值多少倍的列表
        root_path: 保存结果的路径
        n_jobs: 并行任务的数量，-1 表示使用所有可用的 CPU 核心

    返回:
        sir_list: 包含每个 beta 下 SIR 结果的列表
    """
    sir_list = []
    for inx, a in enumerate(a_list):
        logging.info(f"开始仿真 a = {a}")
        start_time = time.time()
        sir_dict = SIR_dict(G, real_beta=True, a=a, n_jobs=n_jobs)
        sir_list.append(sir_dict)
        path = f"{root_path}{inx}.csv"
        save_sir_dict(sir_dict, path)
        end_time = time.time()
        logging.info(f"仿真 a = {a} 完成，耗时 {end_time - start_time:.2f} 秒")
    return sir_list


def SIR_dict(G, beta=0.1, miu=1, real_beta=None, a=1.5, n_jobs=-1):
    """
    获取网络中所有节点的 SIR 结果
    参数:
        G: 网络图
        beta: 传播概率
        miu: 恢复概率
        real_beta: 如果为 True，按公式计算传播概率
        a: 传播概率阈值的倍数
        n_jobs: 并行任务的数量，-1 表示使用所有可用的 CPU 核心

    返回:
        SIR_dic: 记录所有节点传播能力的字典
    """
    node_list = list(G.nodes())
    SIR_dic = {}

    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
        beta = a * (dc_list.mean() / ((dc_list ** 2).mean() - dc_list.mean()))

    logging.info(f"计算出的 beta: {beta}")

    results = Parallel(n_jobs=n_jobs)(delayed(SIR)(G, infected=[node], beta=beta, miu=miu) for node in tqdm(node_list))

    SIR_dic = {node: result for node, result in zip(node_list, results)}

    return SIR_dic


def SIR(G, infected, beta=0.1, miu=1):
    """
    SIR 模型仿真
    参数:
        G: 网络图
        infected: 初始感染节点
        beta: 传播概率
        miu: 恢复概率

    返回:
        re: N 次仿真后的平均感染规模
    """
    N = 1000
    re = 0
    neighbors_cache = {node: list(G.neighbors(node)) for node in G.nodes()}

    for _ in range(N):
        inf = set(infected)
        R = set()

        while inf:
            newInf = set()
            for i in inf:
                for j in neighbors_cache[i]:
                    if j not in inf and j not in R and np.random.rand() < beta:
                        newInf.add(j)
                if np.random.rand() > miu:
                    newInf.add(i)
                else:
                    R.add(i)
            inf = newInf

        re += len(R) + len(inf)

    return re / N


def save_sir_dict(dic, path):
    """存放SIR的结果
    Parameters:
        dic:sir结果(dict)
        path:目标存放路径
    """
    node = list(dic.keys())
    sir = list(dic.values())
    Sir = pd.DataFrame({'Node': node, 'SIR': sir})
    Sir.to_csv(path, index=False)


def setup_seed(seed):
    """固定种子"""
    np.random.seed(seed)


# 示例用法:
# G = nx.erdos_renyi_graph(100, 0.1)
# a_list = [1, 1.5, 2]
# root_path = './results/'
# SIR_betas(G, a_list, root_path, n_jobs=4)  # 使用 4 个 CPU 核心
np.random.seed(42)
a_list = np.arange(1.0, 2.0, 0.1)
ESS_294_2489_test3 = Utils.load_graph_excel('Networks/trainData/mixsimitimdong5000.xlsx')

ESS_294_2489_test_SIR = SIR_betas(ESS_294_2489_test3, a_list, 'SIR/mixsimitimdong5000/mixsimitimdong5000_')

ESS_294_2489_test3 = Utils.load_graph_excel('Networks/trainData/mixsimitimdong7000.xlsx')

ESS_294_2489_test_SIR = SIR_betas(ESS_294_2489_test3, a_list, 'SIR/mixsimitimdong7000/mixsimitimdong7000_')