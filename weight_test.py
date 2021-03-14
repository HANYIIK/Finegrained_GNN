#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/10 22:32
# @Author   : Hanyiik
# @File     : weight_test.py
# @Function : 测试代码
import torch
import numpy as np


# ---------------- test 1
feature_map = np.array([[[1, 2, 3, 4],
                         [0, 1, 5, 1]],

                        [[2, 2, 3, 1],
                         [3, 1, 1, 1]],

                        [[0, 2, 0, 2],
                         [1, 6, 0, 1]]])
# (3, 2, 4)
gradient_map = np.array([[[4, 7, 8, 9],
                         [0, 8, 4, 2]],

                        [[9, 0, 2, 1],
                         [3, 0, 5, 1]],

                        [[5, 0, 0, 2],
                         [1, 0, 0, 6]]])

print('feature maps:\n', feature_map)
weight = np.mean(gradient_map, axis=1)   # (3, 4)
print('weights:\n',weight)

node_cam = []
for i, a_graph in enumerate(feature_map):
    node_cam.append(a_graph.dot(weight[i]))

node_cam = np.stack(node_cam)

print(node_cam)
print(node_cam.shape)

# ---------------- test 2
# 邻接矩阵置零
def adj_set_zero(adj_matrix, indices):
    """
    :: 功能: 把 adj_matrix 按照 indices 中指定的'行'与'列'置零
    :: 输入: numpy 类型的 adj_matrix
    :: 输出: 置零后的 numpy 类型的 adj_matrix
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
    :: 功能: 把 matrix 按照 indices 中指定的'行'置零
    :: 输入: numpy 类型的 matrix
    :: 输出: 置零后的 numpy 类型的 matrix
    :: 用法:
    """
    input_box = np.zeros_like(matrix)   # (62, 5)
    for i in range(62):
        for item in indices:
            if i == item:
                input_box[i] = np.copy(matrix[i])
    return torch.from_numpy(input_box)

x = np.array([[1, 2, 4, 7],
              [3, 4, 5, 6],
              [5, 6, 8, 9],
              [1, 2, 3, 4]])

adj = np.array([[0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0]])

indices = [1, 2, 3]

print(set_zero(x, indices))
print(adj_set_zero(adj, indices))