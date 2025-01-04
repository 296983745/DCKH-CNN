#!/usr/bin/env python
# coding: utf-8


import os
import random
import time
import warnings

import LCNN

warnings.filterwarnings('ignore')

import keras
from keras.layers import Input
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import networkx as nx

from scipy import stats
from ast import literal_eval
from keras import models

import tensorflow as tf


# # Metrics based on node degree

# In[26]:


def dic_D_1_weights_all_nodes(G):
    dic_D_1_Nodes = {}
    for u in G:
        dic_D_1_Nodes[u] = nx.degree(G, u)
    return dic_D_1_Nodes


def dic_D_2_weights_all_nodes(G, dic_D_1_Nodes):
    dic_D_2_Nodes = {}
    for u in G:
        Tv = [n for n in G.neighbors(u)]  # neighbors of v
        D2 = dic_D_1_Nodes[(u)]
        for v in Tv:
            D2 += dic_D_1_Nodes[(v)]
        dic_D_2_Nodes[u] = D2
    return dic_D_2_Nodes


def dic_D_3_weights_all_nodes(G, dic_D_2_Nodes):
    dic_D_3_Nodes = {}
    for u in G:
        Tv = [n for n in G.neighbors(u)]  # neighbors of v
        D3 = dic_D_2_Nodes[(u)]
        for v in Tv:
            D3 += dic_D_2_Nodes[(v)]
        dic_D_3_Nodes[u] = D3
    return dic_D_3_Nodes


# # Metrics based on H-index

# In[27]:
def r_hop_closeness_centrality(G, r):
    """
    计算 r-hop 限制的接近中心性。

    参数：
        G (networkx.Graph): 网络图对象
        r (int): hop 限制距离

    返回：
        dict: 每个节点的 r-hop 接近中心性
    """
    # 初始化结果字典
    r_hop_cc = {}

    for node in G.nodes:
        # 获取 r-hop 邻域内的节点和距离
        r_hop_nodes = nx.single_source_shortest_path_length(G, source=node, cutoff=r)

        # 计算 r-hop 内的接近中心性
        total_distance = sum(r_hop_nodes.values())  # 距离总和
        reachable_nodes = len(r_hop_nodes) - 1  # 可达节点数（减去自身）

        # 如果没有 r-hop 内的其他节点，则接近中心性为 0
        if reachable_nodes > 0:
            r_hop_cc[node] = reachable_nodes / total_distance
        else:
            r_hop_cc[node] = 0.0

    return r_hop_cc


def H_index(G, node):
    Tv = [n for n in G.neighbors(node)]  # neighbors of v.
    # sorting in ascending order
    citations = [nx.degree(G, v) for v in Tv]
    citations.sort()

    # iterating over the list
    for i, cited in enumerate(citations):

        # finding current result
        result = len(citations) - i
        # if result is less than or equal
        # to cited then return result
        if result <= cited:
            return result
    return 0


# 这段代码用于计算图中每个节点的H指数权重，H指数权重是指一个节点的H指数与其邻居节点的H指数之和。下面是加上注释后的代码：

def H_index_of_All_nodes(G):
    h_index_1_Nodes = {}  # 存储每个节点的H指数权重（H指数为1）

    # 遍历图中的所有节点
    for u in G:
        # 使用H_index函数计算节点u的H指数
        H = H_index(G, u)
        # 将节点u的H指数权重（H指数为1）存储在h_index_1_Nodes中
        h_index_1_Nodes[u] = H
    return h_index_1_Nodes


def H_index_weights_of_All_nodes(G):
    h_index_1_Nodes = {}  # 存储每个节点的H指数权重（H指数为1）
    h_index_2_Nodes = {}  # 存储每个节点的H指数权重（H指数为2）
    h_index_3_Nodes = {}

    # 遍历图中的所有节点
    for u in G:
        # 使用H_index函数计算节点u的H指数
        H = H_index(G, u)
        # 将节点u的H指数权重（H指数为1）存储在h_index_1_Nodes中
        h_index_1_Nodes[u] = H

    # 遍历图中的所有节点
    for u in G:
        # 获取节点u的邻居节点
        Tv = [n for n in G.neighbors(u)]  # neighbors of v.
        # 计算节点u的H指数权重（H指数为2）
        h_index_2 = h_index_1_Nodes[(u)]
        for n in Tv:
            h_index_2 += h_index_1_Nodes[(n)]
        # 将节点u的H指数权重（H指数为2）存储在h_index_2_Nodes中
        h_index_2_Nodes[u] = h_index_2

    # 遍历图中的所有节点
    for u in G:
        # 获取节点u的邻居节点
        Tv = [n for n in G.neighbors(u)]  # neighbors of v.
        # 计算节点u的H指数权重（H指数为3）
        h_index_3 = h_index_2_Nodes[(u)]
        for n in Tv:
            h_index_3 += h_index_2_Nodes[(n)]
        # 将节点u的H指数权重（H指数为3）存储在h_index_3_Nodes中
        h_index_3_Nodes[u] = h_index_3

    return h_index_1_Nodes, h_index_2_Nodes, h_index_3_Nodes


# # Structural channel sets of node representations

# In[28]:


def metrics_one_two_hop_Adj_mat_of_node(G, L, node, dic_D1, dic_D2, dic_D3, K_shell, CC, HI):
    # 获取给定节点的直接邻居（一跳邻居）
    one_hop = list(G.adj[node])

    # 计算每个一跳邻居的权重（这里假设权重来自dic_D1）
    one_hop_weight_a = {u: dic_D1[u] for u in one_hop}

    # 按权重降序排序一跳邻居，并取前L个
    sorted_list = sorted(one_hop_weight_a.items(), key=lambda x: x[1], reverse=True)

    # 选中的邻居包括自身加上排序后的前L个一跳邻居
    selected_nei = [node] + [i for i, j in sorted_list[:L]]

    # 如果一跳邻居不足L个，使用二跳邻居进行填充
    if len(one_hop) < L:
        # 获取二跳邻居
        two_hop = set()
        for n in one_hop:
            two_hop.update(set(G.adj[n]) - set(one_hop) - {node})
        # 计算每个二跳邻居的权重
        two_hop_weight_a = {u: dic_D1[u] for u in two_hop}
        # 按权重降序排序二跳邻居，并取前L-len(one_hop)个
        sorted_two_hop_list = sorted(two_hop_weight_a.items(), key=lambda x: x[1], reverse=True)
        # 添加排序后的前L-len(one_hop)个二跳邻居到结果中
        selected_nei += [i for i, j in sorted_two_hop_list[:(L - len(one_hop))]]

    # 如果一跳和二跳邻居仍不足L个，则用-1填充
    if len(selected_nei) - 1 < L:
        selected_nei += [-1 for _ in range(L - (len(selected_nei) - 1))]

    # 初始化邻接矩阵的各个部分
    arr_D1, arr_D2, arr_D3, arr_H1, arr_H2, arr_H3 = [], [], [], [], [], []

    # 构建邻接矩阵
    for i in selected_nei:
        # 对于每一列
        col_D1, col_D2, col_D3 = [], [], []
        col_H1, col_H2, col_H3 = [], [], []

        for j in selected_nei:
            # 自身与自身的连接
            if i == j:
                col_D1.append(dic_D1[node])
                col_D2.append(dic_D2[node])
                col_D3.append(dic_D3[node])

                col_H1.append(K_shell[node])
                col_H2.append(CC[node])
                col_H3.append(HI[node])

            # 节点间存在边的情况处理
            elif G.has_edge(i, j):
                if i == 0:  # 特殊处理首节点
                    col_D1.append(dic_D1[j])
                    col_D2.append(dic_D2[j])
                    col_D3.append(dic_D3[j])

                    col_H1.append(K_shell[j])
                    col_H2.append(CC[j])
                    col_H3.append(HI[j])
                elif j == 0:  # 当前节点非首节点，首节点处理方式相同
                    col_D1.append(dic_D1[i])
                    col_D2.append(dic_D2[i])
                    col_D3.append(dic_D3[i])

                    col_H1.append(K_shell[i])
                    col_H2.append(CC[i])
                    col_H3.append(HI[i])
                else:  # 其他节点间默认为1
                    col_D1.append(1)
                    col_D2.append(1)
                    col_D3.append(1)

                    col_H1.append(1)
                    col_H2.append(1)
                    col_H3.append(1)

            # 无边则为0
            else:
                col_D1.append(0)
                col_D2.append(0)
                col_D3.append(0)

                col_H1.append(0)
                col_H2.append(0)
                col_H3.append(0)

        # 添加当前列到邻接矩阵的数组
        arr_D1.append(col_D1)
        arr_D2.append(col_D2)
        arr_D3.append(col_D3)

        arr_H1.append(col_H1)
        arr_H2.append(col_H2)
        arr_H3.append(col_H3)

    # 将列表转换为NumPy数组并返回
    return np.array(arr_D1), np.array(arr_D2), np.array(arr_D3), np.array(arr_H1), np.array(arr_H2), np.array(arr_H3)


def metrics_one__hop_Adj_mat_of_all_nodes(G, L):
    dic_local_embedding = {}
    dic_semi_embedding = {}

    # 计算图中所有节点的第 1 层邻接矩阵的度量
    dic_D1 = dic_D_1_weights_all_nodes(G)
    dic_D2 = dic_D_2_weights_all_nodes(G, dic_D1)
    dic_D3 = dic_D_3_weights_all_nodes(G, dic_D2)

    # 计算图中所有节点的第 1 层邻接矩阵的度量
    k_shell = nx.core_number(G)
    CC = r_hop_closeness_centrality(G, L)
    HI = H_index_of_All_nodes(G)

    # 遍历图中的所有节点
    for node in G:
        # 计算节点的第 1 层邻接矩阵的度量
        hop_1_arr_D1, hop_1_arr_D2, hop_1_arr_D3, hop_1_arr_k_shell, hop_1_arr_cc, hop_1_arr_HI = metrics_one_two_hop_Adj_mat_of_node(
            G, L, node, dic_D1, dic_D2, dic_D3, k_shell, CC, HI)

        # 存储第 1 层邻接矩阵的度量
        dic_local_embedding[node] = hop_1_arr_D1, hop_1_arr_D2, hop_1_arr_D3
        # 存储第 1 层邻接矩阵的度量
        dic_semi_embedding[node] = hop_1_arr_k_shell, hop_1_arr_cc, hop_1_arr_HI

    return dic_local_embedding, dic_semi_embedding


# In[29]:


from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, concatenate, \
    Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import pandas as pd


def create_shared_convolution_layers(input_model: keras.layers.Layer, kernel_size: int,
                                     pool_size: int) -> keras.layers.Layer:
    x = Conv2D(filters=18, kernel_size=kernel_size, strides=1, padding='same')(input_model)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((pool_size, pool_size), padding='same')(x)

    x = Conv2D(filters=48, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((pool_size, pool_size), padding='same')(x)

    return x


from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class WeightedAttention(Layer):
    def __init__(self, **kwargs):
        super(WeightedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_local = self.add_weight(shape=(input_shape[0][-1], 1),
                                       initializer="random_normal",
                                       trainable=True,
                                       name="w_local")
        self.w_semi = self.add_weight(shape=(input_shape[1][-1], 1),
                                      initializer="random_normal",
                                      trainable=True,
                                      name="w_semi")
        super(WeightedAttention, self).build(input_shape)

    def call(self, inputs):
        local, semi = inputs
        local_score = K.dot(local, self.w_local)  # 计算局部特征的权重
        semi_score = K.dot(semi, self.w_semi)  # 计算全局特征的权重

        # 通过 softmax 计算注意力权重
        attention_weights = K.softmax(K.concatenate([local_score, semi_score], axis=1), axis=1)

        # 加权求和
        local_attention = attention_weights[:, 0:1] * local
        semi_attention = attention_weights[:, 1:2] * semi
        return K.concatenate([local_attention, semi_attention], axis=1)


def create_Model_new(path_Train_Data: str, path_SIR_Train_Data: str, L: int, Kernel_size: int, MaxPooling: int,
                     dense: int, learning_rate: float, epochN: int) -> Tuple[Model, keras.callbacks.History]:
    in_channel_L = 3
    in_channel_S = 3

    data_G = loadData(path_Train_Data)
    data_G_sir = pd.read_csv(path_SIR_Train_Data)
    data_G_label = dict(zip(np.array(data_G_sir['Node'], dtype=int), data_G_sir['SIR']))

    dic_local_embedding, dic_semi_embedding = metrics_one__hop_Adj_mat_of_all_nodes(data_G, L)

    x1_train, x2_train, y_train = [], [], []

    for node in data_G:
        x1_train.append(dic_local_embedding[(node)])
        x2_train.append(dic_semi_embedding[(node)])
        y_train.append(data_G_label[(node)])

    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)
    y_train = np.array(y_train)

    x1_train = normalize_data(x1_train.reshape(-1, (L + 1) * (L + 1) * in_channel_L)).reshape(-1, L + 1, L + 1,
                                                                                              in_channel_L)
    x2_train = normalize_data(x2_train.reshape(-1, (L + 1) * (L + 1) * in_channel_S)).reshape(-1, L + 1, L + 1,
                                                                                              in_channel_S)

    input_shape_L = (L + 1, L + 1, in_channel_L)
    input_shape_S = (L + 1, L + 1, in_channel_S)

    local_input = Input(shape=input_shape_L)
    semi_input = Input(shape=input_shape_S)

    shared_conv_local = create_shared_convolution_layers(local_input, Kernel_size, MaxPooling)
    shared_conv_semi = create_shared_convolution_layers(semi_input, Kernel_size, MaxPooling)

    shared_conv_local = Flatten()(shared_conv_local)
    shared_conv_semi = Flatten()(shared_conv_semi)

    # 使用自定义 WeightedAttention 层
    attention_output = WeightedAttention()([shared_conv_local, shared_conv_semi])

    convAall = concatenate([shared_conv_local, shared_conv_semi, attention_output])

    dense_layer = Dense(dense)(convAall)
    dense_layer = LeakyReLU(alpha=0.1)(dense_layer)
    dense_layer = Dropout(0.5)(dense_layer)

    dense_layer = Dense(1)(dense_layer)
    output = LeakyReLU(alpha=0.1)(dense_layer)

    model = Model(inputs=[local_input, semi_input], outputs=[output])
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=opt)
    model.summary()

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    history = model.fit([x1_train, x2_train], y_train, epochs=epochN, shuffle=True, batch_size=4,
                        callbacks=[lr_reduction, early_stopping])

    loss = history.history['loss']
    epochs_range = range(len(loss))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss, label='训练损失')
    plt.legend(loc='upper right')
    plt.title('训练和验证损失')
    plt.show()

    return model, history


# In[33]:


def loadData(nameDataset, sep=","):
    # 读取数据时跳过第一行
    df = pd.read_csv(nameDataset, sep=sep, skiprows=1, names=['FromNodeId', 'ToNodeId'], encoding='gbk')

    # 尝试将数据转换为整数类型，无法转换的设置为NaN
    df['FromNodeId'] = pd.to_numeric(df['FromNodeId'], errors='coerce')
    df['ToNodeId'] = pd.to_numeric(df['ToNodeId'], errors='coerce')

    # 删除包含NaN值的行
    df.dropna(inplace=True)

    # 将列转换为整数类型
    df = df.astype({'FromNodeId': 'int', 'ToNodeId': 'int'})

    G = nx.from_pandas_edgelist(df, source="FromNodeId", target="ToNodeId")

    G.remove_edges_from(nx.selfloop_edges(G))

    print(len(G.nodes))
    return G


def load_graph_txt(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    print(len(G.nodes))
    return G


def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def get_data_to_model(G, L):
    dic_local_embedding, dic_semi_embedding = metrics_one__hop_Adj_mat_of_all_nodes(G, L)

    x1_train = []
    x2_train = []

    in_channel_L = 3
    in_channel_S = 3

    for node in G:
        x1_train.append(dic_local_embedding[(node)])
        x2_train.append(dic_semi_embedding[(node)])

    x1_train = np.array(x1_train)
    x2_train = np.array(x2_train)

    x1_train = x1_train.reshape(-1, L + 1, L + 1, in_channel_L)
    x2_train = x2_train.reshape(-1, L + 1, L + 1, in_channel_S)

    # print(x1_train.shape)
    # print(x2_train.shape)
    return x1_train, x2_train


def get_sir_list(pathDataset, nameDataset, sir_rang_list):
    sir_list = []
    for i in range(10):
        sir = pd.read_csv(pathDataset + nameDataset + '/' + nameDataset + '_' + str(i) + '.csv')

        sir_list.append(dict(zip(np.array(sir['Node'], dtype=str), sir['SIR'])))
    return sir_list


def nodesRank(rank):
    SR = sorted(rank)
    re = []
    for i in SR:
        re.append(rank.index(i))
    return re


def get_algo_list(pathDataset, dataName, algoName):
    algo_list = []
    df = pd.read_csv(pathDataset)
    df = df[df['Dataset'] == dataName]
    df = df[df['Algo'] == algoName]
    algo_list = literal_eval(df['Seed'].iloc[0])
    algo_list = algo_list
    return algo_list


def compare_tau(sir_list, alg_list):
    alg_tau_list = []
    for sir in sir_list:
        sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
        sir_rank = np.array(nodesRank(sir_sort), dtype=float)
        alg_rank = np.array(nodesRank(alg_list), dtype=float)
        tau3, _ = stats.kendalltau(sir_rank, alg_rank)
        alg_tau_list.append(tau3)
    return alg_tau_list


# 这段代码用于使用DCKP-CNN算法对数据集进行排序。下面是加上注释后的代码：
def rank_dataset_using_DCKH_CNN(model, model_name, input_Datasets_to_pred, path_input_Datasets, path_SIR_input_Datasets,
                                sir_rang_list, path_saved_ranked_node, L,
                                name_Train_Data, sir_a_value_Train_Data, Kernel_size, MaxPooling, Dense, learning_rate):
    # 遍历要预测的数据集
    for dataName in input_Datasets_to_pred:
        start_time = time.time()  # 记录开始时间

        # 加载数据集
        G = loadData(path_input_Datasets + dataName + '.csv')
        G = load_graph_txt(path_input_Datasets + dataName + '.txt')
        nodes = list(G.nodes())

        data_predictions_start_time = time.time()
        # 将数据集转换为模型可以接受的输入格式
        # 使用模型预测数据集中的节点的排序
        x1_train, x2_train = get_data_to_model(G, L)
        data_predictions = model.predict([x1_train, x2_train])
        data_predictions_time = time.time() - data_predictions_start_time
        my_pred = [i for i, j in sorted(dict(zip(nodes, data_predictions)).items(), key=lambda x: x[1], reverse=True)]

        # 使用LCNN模型进行预测
        lcnn_model = models.load_model("Models/LCNN_Ker_2_Max_2_dense_1024lear_0.0005.h5", compile=False)
        lcnn_predictions_start_time = time.time()
        x1_train_lcnn, x2_train_lcnn = LCNN.get_data_to_model(G, 40)
        data_predictions_LCNN = lcnn_model.predict([x1_train_lcnn, x2_train_lcnn])
        lcnn_predictions_time = time.time() - lcnn_predictions_start_time
        my_pred_lcnn = [i for i, j in
                        sorted(dict(zip(nodes, data_predictions_LCNN)).items(), key=lambda x: x[1], reverse=True)]

        # 计算图的特性
        degree_centrality_start_time = time.time()
        degree_centrality = nx.degree_centrality(G)
        degree_centrality_time = time.time() - degree_centrality_start_time

        k_shell_start_time = time.time()
        k_shell = nx.core_number(G)
        k_shell_time = time.time() - k_shell_start_time

        clustering_coefficient_start_time = time.time()
        clustering_coefficient = nx.clustering(G)
        clustering_coefficient_time = time.time() - clustering_coefficient_start_time

        betweenness_centrality_start_time = time.time()
        betweenness_centrality = nx.betweenness_centrality(G)
        betweenness_centrality_time = time.time() - betweenness_centrality_start_time

        closeness_centrality_start_time = time.time()
        closeness_centrality = nx.closeness_centrality(G)
        closeness_centrality_time = time.time() - closeness_centrality_start_time
        H_index_start_time = time.time()
        H_index = H_index_of_All_nodes(G)
        H_index_time = time.time() - H_index_start_time

        # 对中心性指标进行排序
        degree_centrality_sorted = [node for node, _ in
                                    sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)]
        k_shell_sorted = [node for node, _ in sorted(k_shell.items(), key=lambda x: x[1], reverse=True)]
        clustering_coefficient_sorted = [node for node, _ in
                                         sorted(clustering_coefficient.items(), key=lambda x: x[1], reverse=True)]
        closeness_centrality_sorted = [node for node, _ in
                                       sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)]
        betweenness_centrality_sorted = [node for node, _ in
                                         sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)]
        H_index_sorted = [node for node, _ in sorted(H_index.items(), key=lambda x: x[1], reverse=True)]

        # 将结果保存到Excel
        results_df = pd.DataFrame({
            'DCKP-CNN': my_pred,
            'LCNN': my_pred_lcnn,
            'Degree Centrality': degree_centrality_sorted,
            'K-shell': k_shell_sorted,
            'Clustering Coefficient': clustering_coefficient_sorted,
            'Closeness Centrality': closeness_centrality_sorted,
            'Betweenness Centrality': betweenness_centrality_sorted,
            'H-index': H_index_sorted
        })
        results_df.to_excel(f'Result_New/sort1/{dataName}_sort_Results.xlsx', index=False)

        # 加载SIR列表
        G_SIR = get_sir_list(path_SIR_input_Datasets, dataName, sir_rang_list)

        # 计算Tau值
        DCKP_CNN_tau = compare_tau(G_SIR, my_pred)
        DC_tau = compare_tau(G_SIR, degree_centrality_sorted)
        ks_tau = compare_tau(G_SIR, k_shell_sorted)
        cc_tau = compare_tau(G_SIR, closeness_centrality_sorted)
        clc_tau = compare_tau(G_SIR, clustering_coefficient_sorted)
        bc_tau = compare_tau(G_SIR, betweenness_centrality_sorted)
        Hi_tau = compare_tau(G_SIR, H_index_sorted)
        lcnn_tau = compare_tau(G_SIR, my_pred_lcnn)
        tau_values = {
            'Degree Centrality': DC_tau,
            'K-shell': ks_tau,
            'Clustering Coefficient': clc_tau,
            'Betweenness Centrality': bc_tau,
            'Closeness Centrality': cc_tau,
            'H-idex': Hi_tau,
            'LCNN': lcnn_tau,
            'DCKP_CNN_TAU': DCKP_CNN_tau
        }
        tau_df = pd.DataFrame(tau_values)
        tau_df.to_csv(f'Result_New/tau1/{dataName}_tau_Results.csv', index=False)

        # 生成对比图表
        beta_values = [1.0 + 0.1 * i for i in range(len(G_SIR))]
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
        plt.figure(figsize=(14, 10))
        for i, (feature, tau_list) in enumerate(tau_values.items()):
            plt.plot(beta_values, tau_list, marker=markers[i % len(markers)],
                     linestyle=line_styles[i % len(line_styles)], label=feature)

        plt.xlabel('Beta', fontsize=14)
        plt.ylabel('Tau', fontsize=14)
        plt.title(f'{dataName}_Tau Comparison for Different Features', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # 显示图表
        plt.show()

        # 记录每个方法的时间
        method_times = {
            'Dataset': [dataName],
            'DCKP-CNN': [data_predictions_time],
            'LCNN': [lcnn_predictions_time],
            'Degree Centrality': [degree_centrality_time],
            'K-shell': [k_shell_time],
            'Clustering Coefficient': [clustering_coefficient_time],
            'Betweenness Centrality': [betweenness_centrality_time],
            'Closeness Centrality': [closeness_centrality_time],
            'H-Index': [H_index_time],
        }

        # 将时间记录保存到 DataFrame
        df_seed_DCKP_CNN = pd.DataFrame(method_times)
        df_seed_DCKP_CNN.to_csv(f'Result_New/time/{dataName}_times_Results.csv', index=False)

        print('-------------------------------------------------------------')
        print('done', model_name, ' in  ', dataName)
        print('-------------------------------------------------------------')

        # 打印 DCKP-CNN 的 Tau 值
        print('tau=', DCKP_CNN_tau)
        # df3 = {'Dataset': dataName,
        #        'sir_a_value_Train_Data': sir_a_value_Train_Data, 'Algo': model_name, 'Tau': tau,
        #        'Dense': Dense, 'MaxPooling': MaxPooling, 'Kernel_size': Kernel_size, 'learning_rate': learning_rate}
        # df_tau_result = df_tau_result._append(df3, ignore_index=True)
        #
        # # df_seed_LCNN.to_csv(model_name+'__Seed.csv')
        # df_tau_result.to_csv(path_saved_ranked_node + '/' + model_name + '__Tau.csv')


# In[34]:


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(2)
    tf.random.set_seed(2)
    np.random.seed(2)
    random.seed(2)


if __name__ == '__main__':
    reset_random_seeds()

    # 这段代码用于训练和预测（DCKH_CNN）模型，用于文件共享领域的节点排序。下面是加上注释后的代码：
    L = 8  # 隐层节点数
    epochN = 200  # 训练轮数
    dense = 1024  # 全连接层节点数
    # learning_ra = 0.0005  # 学习率
    learning_ra = 0.0001  # 学习率
    MaxPooling = 2  # 最大池化层大小
    Kernel_size = 2

    name_Train_Data = 'mixsimitimdong2001.csv'  # 训练数据集名称
    sir_a_value_Train_Data = '1.9'  # SIR参数a值
    path_saved_ranked_node = 'Result_New'  # 排序结果保存路径
    sir_rang_list = np.arange(1.0, 2.0, 0.1)

    input_Datasets_to_pred = ['NS', 'Email', 'Faa', 'Figeys', 'Facebook', 'jazz', 'LastFM', 'powergrid',
                              'Sex']  # 预测数据集名称
    sir_rang_list = np.arange(1.0, 2.0, 0.1)

    path_Train_Data = 'Data/' + name_Train_Data
    path_SIR_Train_Data = 'SIR/' + name_Train_Data[:-4] + '/' + name_Train_Data[
                                                                :-4] + '_a[' + sir_a_value_Train_Data + ']_.csv'

    model_name = str(L) + '_DCKH_LCNN' + '_Ker_' + str(Kernel_size) + '_Max_' + str(MaxPooling) + '_dense_' + str(
        dense) + 'lear_' + str(
        learning_ra)
    PATH_saved_model = "DCKH_CNN_Models/" + model_name + ".h5"

    # model, _ = create_Model_new(path_Train_Data, path_SIR_Train_Data, L, Kernel_size, MaxPooling, dense, learning_ra,
    #                             epochN)
    # model.save(PATH_saved_model)

    custom_objects = {
        'WeightedAttention': WeightedAttention
    }
    model = models.load_model(PATH_saved_model, custom_objects=custom_objects, compile=False)
    path_input_Datasets = 'Data/'
    path_SIR_input_Datasets = '../new_ESS_FOUR_RCNN/SIR/'

    rank_dataset_using_DCKH_CNN(model, model_name, input_Datasets_to_pred, path_input_Datasets, path_SIR_input_Datasets,
                                sir_rang_list, path_saved_ranked_node, L,
                                name_Train_Data, sir_a_value_Train_Data, Kernel_size, MaxPooling, dense, learning_ra)
