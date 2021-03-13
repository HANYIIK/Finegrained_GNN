#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/8 11:51
# @Author   : Hanyiik
# @File     : models.py
# @Function : Model 部分
import scipy
import scipy.sparse as sp
import numpy as np
import copy
import pdb

import torch
from torch import nn
from torch.nn import functional as F

from functions import get_config
from grad_cam import GradCam
from utils import graph_utils
from utils import model_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineGrainedGNN(nn.Module):
    def __init__(self, args, adj):
        super(FineGrainedGNN, self).__init__()
        """
        :: 功能: main 网络初始化部分
        :: 输入: args - 参数
                adj - 邻接矩阵(稀疏矩阵)
        :: 输出: 初始化好的一个 FineGrainedGNN 类
        :: 用法: model = FineGrainedGNN(args, adj_matrix)
        """
        self.batch_size = args.batch_size
        self.K = args.K
        self.filter_num = args.filter_num
        self.feature_num = args.feature_len
        self.classes_num = args.classes_num
        self.node_num = args.node_num

        self.adjs_1 = [adj.toarray() for j in range(self.batch_size)]
        laplacian = graph_utils.laplacian(adj, normalized=True)
        laplacian = laplacian_to_sparse(laplacian)
        self.laplacians_1 = [laplacian for i in range(self.batch_size)]

        # --- Gating Notwork
        self.gc = ChebshevGCNN(
            in_channels=self.feature_num,   # 5
            filter_num=self.filter_num,     # 32
            K=self.K,                       # 3
            laplacians=self.laplacians_1
        )
        self.fc = nn.Linear(
            in_features=self.node_num * self.filter_num * self.feature_num,
            out_features=self.classes_num
        )

        # --- Expert 1
        self.gc_expert_1 = copy.deepcopy(self.gc)
        self.fc_expert_1 = copy.deepcopy(self.fc)

        # --- Expert 2
        self.gc_expert_2 = copy.deepcopy(self.gc)
        self.fc_expert_2 = copy.deepcopy(self.fc)

        # --- Expert 3
        self.gc_expert_3 = copy.deepcopy(self.gc)
        self.fc_expert_3 = copy.deepcopy(self.fc)


    def forward(self, x, y):
        """
        :: 功能: main 网络执行部分
        :: 输入: x - (batch_size, 62, 5) 数据
                y - (batch_size,) 标签
        :: 输出: (batch_size, 62, 5 * filter_num)
        :: 用法: logits = model(data, labels)
        """
        # ------------ Expert 1
        gc_output_1 = self.gc_expert_1(x)       # (100, 62, 160)
        batch_size, node_num, feature_len = gc_output_1.size()      # 100  62  160
        gc_output_1_re = torch.reshape(gc_output_1, [batch_size, node_num * feature_len])  # (100, 9920)
        logits_1 = self.fc_expert_1(gc_output_1_re)      # (100, 7)

        with torch.enable_grad():
            grad_cam = GradCam(model=self, feature_extractor=self.gc_expert_1, fc=self.fc_expert_1, rate=0.3)
            mask_1, nodes_cam_1 = grad_cam(x.detach(), y)

        input_box_1, laplacians_list_2, adjs_2 = get_bbox(x=x, adjs=self.adjs_1, indices=mask_1)

        # ------------ Expert 2
        self.gc_expert_2 = ChebshevGCNN(
            in_channels=self.feature_num,   # 5
            filter_num=self.filter_num,     # 32
            K=self.K,    # 3
            laplacians=laplacians_list_2
        ).to(DEVICE)

        gc_output_2 = self.gc_expert_2(input_box_1)  # (100, 62, 160)
        batch_size, node_num, feature_len = gc_output_2.size()
        gc_output_2_re = torch.reshape(gc_output_2, [batch_size, node_num * feature_len])
        logits_expert_2 = self.fc_expert_2(gc_output_2_re)  # (100, 7)

        with torch.enable_grad():
            grad_cam = GradCam(model=self, feature_extractor=self.gc_expert_2, fc=self.fc_expert_2, rate=0.3)
            mask_2, nodes_cam_2 = grad_cam(input_box_1.detach(), y)

        input_box_2, laplacians_list_3, adjs_3 = get_bbox(x=input_box_1, adjs=adjs_2, indices=mask_2)

        # ------------ Expert 3
        self.gc_expert_3 = ChebshevGCNN(
            in_channels=self.feature_num,   # 5
            filter_num=self.filter_num,     # 32
            K=self.K,  # 3
            laplacians=laplacians_list_3
        ).to(DEVICE)
        gc_output_3 = self.gc_expert_3(input_box_2)  # (100, 62, 160)
        batch_size, node_num, feature_len = gc_output_3.size()
        gc_output_3_re = torch.reshape(gc_output_3, [batch_size, node_num * feature_len])  # (100, 9920)
        logits_expert_3 = self.fc_expert_3(gc_output_3_re)  # (100, 7)

        # ------------ Gating Network
        my_gate = self.gc(x)
        batch_size, node_num, feature_len = my_gate.size()
        my_gate = torch.reshape(my_gate, [batch_size, node_num * feature_len])  # (100, 9920)
        my_gate = self.fc(my_gate)
        pr_gate = F.softmax(my_gate, dim=1)  # (100, 7)

        logits_gate = torch.stack([logits_1, logits_expert_2, logits_expert_3], dim=-1)  # (100, 7, 3)
        logits_gate = logits_gate * pr_gate.view(pr_gate.size(0), pr_gate.size(1), 1)
        logits_gate = logits_gate.sum(-1)

        return logits_gate, nodes_cam_1, nodes_cam_2

class ChebshevGCNN(nn.Module):
    """
    :: 功能: 契比雪夫图卷积网络
    :: 输入: in_channels - 输入节点的特征长度
            filter_num - 需要几个卷积核
            K - 看距离本节点多少条边的邻居
            laplacians - 一个 batch 图的拉普拉斯矩阵 list
    :: 输出: (batch_size, node_num, feature_len * kernel_num)
    :: 用法: self.gc = ChebshevGCNN(in_channels=self.feature_len,
                                    filter_num=self.filter_num,
                                    K=self.K,
                                    laplacians=laplacians)
    """
    def __init__(self, in_channels, filter_num, K, laplacians):
        super(ChebshevGCNN, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(K + 1, filter_num))     # in_channels = 5
        self.bias = nn.Parameter(torch.Tensor(1, 1, filter_num * in_channels))
        self.K = K
        self.filter_num = filter_num
        self.laplacians = laplacians

        self.reset_parameters()

    def reset_parameters(self):
        """
        :: 功能: 把 weight 和 bias 重置一下
        """
        model_utils.truncated_normal_(self.weight, mean=0.0, std=0.1)
        model_utils.truncated_normal_(self.bias, mean=0.0, std=0.1)

    def chebyshev(self, x):
        """
        :: 功能: 契比雪夫多项式卷积部分
        :: 输入: (batch_size, node_num, feature_len)类型的 Tensor
        :: 输出: (batch_size, node_num, feature_len * kernel_num)类型的 Tensor
        :: 用法: h1 = chebyshev(x)
        """
        batch_size, node_num, feature_len = x.size()    # (100, 62, 5)

        x_split = []
        for i, a_graph in enumerate(x):     # a_graph.shape = (62, 5)
            '''
            切比雪夫多项式:
                x0 = x
                x1 = L × x
                x2 = 2 × L × x1 - x0
                    ...
                x(k) = 2 × L × x(k-1) - x(k-2)
            '''
            x0 = a_graph
            x_list = [x0]
            x1 = torch.sparse.mm(self.laplacians[i].to(DEVICE), x0)  # (62, 5)
            x_list.append(x1)
            if self.K > 1:
                for k in range(2, self.K+1):
                    x2 = 2 * torch.sparse.mm(self.laplacians[i].to(DEVICE), x1) - x0  # (62, 5)
                    x_list.append(x2)
                    x0, x1 = x1, x2

            # [x0, x1, x2, ..., xK] 横着拼接起来
            a_graph = torch.stack(x_list, dim=0).permute(1, 2, 0)  # (62, 5, K+1=4)
            x_split.append(a_graph)

        x = torch.stack(x_split, dim=0)     # (100, 62, 5, K+1=4)
        x = torch.reshape(x, [batch_size * node_num * feature_len, self.K+1])   # (31000, K+1=4)

        '''
            x.shape = (batch_size * node_num * feature_len, K+1=4)
            weight.shape = (K+1=4, 卷积核个数)
        '''
        x = torch.matmul(x, self.weight)  # (31000, 卷积核个数)
        x = torch.reshape(x, [batch_size, node_num, feature_len, self.filter_num])  # (100, 62, 5, 32)
        x = torch.reshape(x, [batch_size, node_num, feature_len * self.filter_num])  # (100, 62, 160)
        return x    # (100, 62, 160)

    def brelu(self, x):
        """Bias and ReLU. One bias per filter."""
        return F.relu(x + self.bias)

    def forward(self, x):
        x = self.chebyshev(x)
        x = self.brelu(x)
        return x


def get_bbox(x, adjs, indices):
    """
    :: 功能: 为下一个 ChebNet 产生输入和邻接矩阵集合
    :: 输入: x - 上一个 ChebNet 的输入数据 (batch_size, 62, 5)
            adjs - 上一个ChebNet 输入数据的邻接矩阵组 [batch_size 个 (62, 62)]
            indices - grad_cam 选出的 nodes 的索引 (nodes_num,)
    :: 输出: 下一个 ChebNet 的输入(cuda), 下一个 ChebNet 初始化所用的 laplacians_list, 下一个 get_bbox 所用的 adjs
    :: 用法: input_box_1, laplacians_list_2, adjs_2 = get_bbox(x, self.adjs_1, mask_1)
    """
    batch_size = len(indices)

    input_box = []
    adj_input_box = []
    adj_input_box_sparse = []

    for k in range(batch_size):
        new_adj = adj_set_zero(adjs[k], indices[k].cpu().numpy())
        adj_input_box.append(new_adj)
        adj_input_box_sparse.append(sp.csr_matrix(new_adj))

        tmp = x.cpu().numpy()[k, :, :]
        input_box.append(set_zero(tmp, indices[k].cpu().numpy()))

    input_box = torch.stack(input_box, dim=0).to(DEVICE)
    return input_box, get_laplacians(adj_input_box_sparse), adj_input_box

# 邻接矩阵置零
def adj_set_zero(adj_matrix, indices):
    """
    :: 功能:
    :: 输入:
    :: 输出:
    :: 用法:
    """
    input_box = np.zeros_like(adj_matrix)   # (62, 62)
    input_box_2 = np.zeros_like(adj_matrix)
    for i in range(62):
        for item in indices:
            if i == item:
                input_box[i] = np.copy(adj_matrix[i])
    for j in range(62):
        for item in indices:
            if j == item:
                input_box_2[:, j] = np.copy(input_box[:, j])
    return torch.from_numpy(input_box_2)

# 矩阵置零
def set_zero(matrix, indices):
    """
    :: 功能: 将 matrix 中 indices 指定的行置零
    :: 输入:
    :: 输出:
    :: 用法:
    """
    input_box = np.zeros_like(matrix)   # (62, 5)
    for i in range(62):
        for item in indices:
            if i == item:
                input_box[i] = np.copy(matrix[i])
    return torch.from_numpy(input_box)

def get_laplacians(adj_list_sparse):
    """
    :: 功能: 将一个稀疏邻接矩阵(numpy 类型) list 转换为一个稀疏拉普拉斯 Tensor 的 list
    :: 输入: 稀疏邻接矩阵(numpy 类型) list
    :: 输出: 稀疏拉普拉斯 Tensor 的 list
    :: 用法: get_laplacians(adj_input_box_sparse)
    """
    laplacians_list = []
    for adj_item in adj_list_sparse:
        laplacian = graph_utils.laplacian(adj_item, normalized=True)
        laplacian = laplacian_to_sparse(laplacian)
        laplacians_list.append(laplacian)
    return laplacians_list

def laplacian_to_sparse(laplacian):
    """
    :: 功能:
    :: 输入: 一个 numpy 类型的初始 Laplacian 矩阵
    :: 输出: 一个 Tensor 类型的 rescale 过的 Laplacian 稀疏张量( L_rescale = L - I )
    :: 用法:
    """
    laplacian = scipy.sparse.csr_matrix(laplacian)
    laplacian = graph_utils.rescale_L(laplacian, lamda_max=2)
    laplacian = laplacian.tocoo()  # 转换为坐标格式矩阵

    indices = torch.LongTensor(np.row_stack((laplacian.row, laplacian.col)))
    data = torch.FloatTensor(laplacian.data)
    shape = torch.Size(laplacian.shape)

    sparse_lapla = torch.sparse.FloatTensor(indices, data, shape)
    return sparse_lapla