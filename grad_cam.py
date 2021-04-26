#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/12 4:06 下午
# @Author   : Hanyiik
# @File     : grad_cam.py
# @Function : 视觉解释模块
import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.seterr(divide='ignore',invalid='ignore')


class FeatureExtractor:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor  # Chebshev

    def save_gradient(self, grad):
        self.gradients.append(grad)     # (100, 62, 160)

    def __call__(self, x):  # input size: (100, 62, 5)
        self.gradients = []
        x = self.feature_extractor(x)  # output size: (100, 62, 160)
        x.register_hook(self.save_gradient)
        return x    # (100, 62, 160)


class ModelOutputs:
    def __init__(self, feature_extractor, fc):
        self.fc = fc
        self.feature_extractor = FeatureExtractor(feature_extractor)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        features = self.feature_extractor(x)  # features size : (100, 62, 160)
        batch_size, node_num, feature_num = features.size()     # 100  62  160
        output = torch.reshape(features, [batch_size, node_num * feature_num])  # (100, 9920)
        output = self.fc(output)  # 全连接层...emmmmmm output size: (100, 7)
        return features, output
        # [features]经过一层 ChebshevGCNN 的输出 (100, 62, 160)
        # [output]经过一层 ChebshevGCNN 后再经过一层 fc 的最终输出 (100, 7)


class GradCam:
    def __init__(self, model, feature_extractor, fc, rate=0.3):
        self.flag = model.training
        self.model = model.to(DEVICE)
        # Sets the module in evaluation mode, dropout and batchnorm are disabled in the evaluation mode
        self.model.eval()
        self.rate = rate

        self.extractor = ModelOutputs(feature_extractor, fc)

    def forward(self, input_x):
        return self.model(input_x)

    def __call__(self, input_x, index):
        features, output = self.extractor(input_x.to(DEVICE))
        # features.shape = (100, 62, 160)
        # output.shape = (100, 7)

        if output.dim() == 1:
            output = output.unsqueeze(0)

        if index is None:
            index = torch.argmax(output, dim=-1)

        index = index.type(torch.long)
        one_hot = torch.zeros_like(output)  # (100, 7)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot.to(DEVICE) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]  # (100, 62, 160)

        weight = grads_val.cpu().detach().numpy()   # (100, 62, 160)
        weight = np.mean(weight, axis=1)       # (100, 160)
        target = features.cpu().detach().numpy()    # (100, 62, 160)

        nodes_cam = []          # (100, 62) nodes heat CAM
        node_heat_mask = []     # (100, 62) node heat processed MASK

        indices_list = []

        for i, a_graph in enumerate(target):
            nodes_cam.append(np.maximum(a_graph.dot(weight[i]), 0))     # Relu nodes heat

        for j, a_node_heat in enumerate(nodes_cam):
            heat_max = np.max(a_node_heat)
            heat_min = np.min(a_node_heat)
            heat_mask = (a_node_heat - heat_min)/(heat_max - heat_min)
            node_heat_mask.append(heat_mask)

        node_heat_mask = np.stack(node_heat_mask)   # (100, 62)

        cam = np.sign(np.sign(node_heat_mask - self.rate) + 1)

        for my_cam in cam:
            indices_list.append(np.squeeze(np.nonzero(my_cam), axis=0))

        if self.flag:
            self.model.train()

        return indices_list, nodes_cam
        # indices_list 是一个由 batch_size 个 numpy 类型的 [选中点的索引]组成的 list, 长度不定
        # nodes_cam 是一个由 batch_size 个 (62,) 的 numpy 矩阵组成的 list, 每个 numpy 矩阵表示一张 graph 62 个点的"热力图"


class GradCam_filter:
    def __init__(self, model, feature_extractor, fc, rate=0.3):
        self.flag = model.training
        self.model = model.to(DEVICE)
        # Sets the module in evaluation mode, dropout and batchnorm are disabled in the evaluation mode
        self.model.eval()
        self.rate = rate

        self.extractor = ModelOutputs(feature_extractor, fc)

    def forward(self, input_x):
        return self.model(input_x)

    def __call__(self, input_x, index):
        features, output = self.extractor(input_x.to(DEVICE))

        if output.dim() == 1:
            output = output.unsqueeze(0)

        if index is None:
            index = torch.argmax(output, dim=-1)

        index = index.type(torch.long)
        one_hot = torch.zeros_like(output)  # (100, 7)
        one_hot[[torch.arange(output.size(0)), index]] = 1
        one_hot = torch.sum(one_hot.to(DEVICE) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # 得到梯度图 (100, 62, 160)
        grads_val = self.extractor.get_gradients()[-1]

        # 得到每个特征的 62 个点的梯度均值 (100, 160, 1)
        weight = torch.mean(grads_val, dim=1).unsqueeze(-1)

        # 得到 CAM (100, 62), 代表这 100 张 graph 中每张 graph 62 个点的热力值
        nodes_cam = torch.relu(torch.matmul(features, weight)).squeeze(dim=2)  # which represent 62 nodes' heat.

        # 制作 MASK (100, 62), 分布在 0~1
        heat_max = torch.max(nodes_cam, dim=1).values.unsqueeze(dim=-1)
        heat_min = torch.min(nodes_cam, dim=1).values.unsqueeze(dim=-1)
        nodes_mask = (nodes_cam - heat_min) / (heat_max - heat_min)

        # 凡是小于 rate 的地方，均置零
        cam = torch.sign(torch.sign(nodes_mask - self.rate) + 1)

        # 得到选中点的 indices
        indices_list = []
        for my_cam in cam:
            indices_list.append(torch.nonzero(my_cam).squeeze(1).cpu().detach().numpy())

        if self.flag:
            self.model.train()

        return indices_list, nodes_cam
        # indices_list 是一个由 batch_size 个[选中点的索引]的 numpy 组成的 list, 长度不定
        # nodes_cam 是一个 (batch_size, 62) 的 Tensor, 每行表示每一张图 62 个点的"热力图"


if __name__ == '__main__':
    fake_feature_map = torch.from_numpy(np.random.random((100, 62, 160))).to(DEVICE)
    fake_gradient = torch.from_numpy(np.random.random((100, 62, 160))).to(DEVICE)

    def grad_cam_filter(grad, features):
        # 得到每个特征的 62 个点的梯度均值 (100, 160, 1)
        weight = torch.mean(grad, dim=1).unsqueeze(-1)

        # 得到 CAM (100, 62), 代表这 100 张 graph 中每张 graph 62 个点的热力值
        nodes_cam = torch.relu(torch.matmul(features, weight)).squeeze(dim=2)  # which represent 62 nodes' heat.

        # 制作 MASK (100, 62), 分布在 0~1
        heat_max = torch.max(nodes_cam, dim=1).values.unsqueeze(dim=-1)
        heat_min = torch.min(nodes_cam, dim=1).values.unsqueeze(dim=-1)
        nodes_mask = (nodes_cam - heat_min) / (heat_max - heat_min)

        # 凡是小于 rate 的地方，均置零
        cam = torch.sign(torch.sign(nodes_mask - 0.5) + 1)

        # 得到选中点的 indices
        indices_list = []
        for my_cam in cam:
            indices_list.append(torch.nonzero(my_cam).squeeze(1).cpu().detach().numpy())

        return indices_list, nodes_cam

    def grad_cam_origin(grad, features):
        weight = grad.cpu().detach().numpy()  # (100, 62, 160)
        weight = np.mean(weight, axis=1)  # (100, 160)
        target = features.cpu().detach().numpy()  # (100, 62, 160)

        nodes_cam = []  # (100, 62) nodes heat CAM
        node_heat_mask = []  # (100, 62) node heat processed MASK

        indices_list = []

        for i, a_graph in enumerate(target):
            nodes_cam.append(np.maximum(a_graph.dot(weight[i]), 0))  # Relu nodes heat

        for j, a_node_heat in enumerate(nodes_cam):
            heat_max = np.max(a_node_heat)
            heat_min = np.min(a_node_heat)
            heat_mask = (a_node_heat - heat_min) / (heat_max - heat_min)
            node_heat_mask.append(heat_mask)

        node_heat_mask = np.stack(node_heat_mask)  # (100, 62)

        cam = np.sign(np.sign(node_heat_mask - 0.5) + 1)

        for my_cam in cam:
            indices_list.append(np.squeeze(np.nonzero(my_cam), axis=0))


        return indices_list, nodes_cam

    mask_list_1, cam_1 = grad_cam_filter(fake_gradient, fake_feature_map)
    mask_list_2, cam_2 = grad_cam_origin(fake_gradient, fake_feature_map)

    print(mask_list_1[0])
    print(mask_list_2[0])