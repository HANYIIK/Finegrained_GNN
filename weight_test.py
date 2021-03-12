#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/10 22:32
# @Author   : Hanyiik
# @File     : weight_test.py
# @Function : 测试代码
import torch
import numpy as np

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