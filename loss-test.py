#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/12/13 22:59
# @Author   : Hanyiik
# @File     : loss-test.py
# @Function : nn.CrossEntropyLoss() 详解
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 5
CLASS_NUM = 3


def exclude_gt(logit, target, is_log=False):
    """
    :param logit: 全连接层后得到的向量 ---> (batch_size, class_num)
    :param target: 标签 ---> (batch_size, )
    :param is_log: F.kl_div(torch.log(P), Q) ---> 注意 P 需要过一层 log函数，Q 则不需要!!!
    :return: 经过 target 反向 mask 后的 Softmax(logit) / Log_Softmax(logit) ---> (batch_size, class_num) ---> 正确的类别那一行的概率被置零了!!!
    """
    logit = F.log_softmax(logit, dim=-1) if is_log else F.softmax(logit, dim=-1)    # (batch_size, class_num)
    mask = torch.ones_like(logit)   # (batch_size, class_num)
    for i in range(logit.size(0)):
        mask[i, target[i]] = 0

    return mask * logit     # (batch_size, class_num)


def kl_loss(x_pred, x_gt, target):
    # KL Divergence for branch 1 and branch 2
    kl_pred = exclude_gt(x_pred, target, is_log=True)
    kl_gt = exclude_gt(x_gt, target, is_log=False)
    tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')
    tmp_loss = torch.exp(-tmp_loss).mean()
    return tmp_loss


if __name__ == '__main__':
    x_input = torch.randn(BATCH_SIZE, CLASS_NUM)  # 随机生成输入 logits (5, 3)
    print('x_input:\n', x_input, end='\n\n')

    y_input = torch.randn(BATCH_SIZE, CLASS_NUM)  # 随机生成输入 logits (5, 3)
    print('y_input:\n', y_input, end='\n\n')

    target = torch.tensor([1, 2, 0, 1, 1])  # 设置输出具体值
    print('target\n', target, end='\n\n')

    # 计算输入 softmax，此时可以看到每一行加到一起结果都是 1
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(x_input)
    print('softmax_output:\n', soft_output, end='\n\n')

    # 在 softmax 的基础上取 log (默认以 10 为底的 ln)
    log_output = torch.log(soft_output)
    print('log_output:\n', log_output, end='\n\n')

    # 对比 softmax 与 log 的结合与 nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func = nn.LogSoftmax(dim=1)
    logsoftmax_output = logsoftmax_func(x_input)
    print('logsoftmax_output:\n', logsoftmax_output, end='\n\n')

    # pytorch中关于 NLLLoss 的默认参数配置为：reducetion=True、size_average=True
    nllloss_func = nn.NLLLoss()
    nlloss_output = nllloss_func(logsoftmax_output, target)
    print('nlloss_output:\n', nlloss_output, end='\n\n')

    # 直接使用 pytorch 中的 loss_func = nn.CrossEntropyLoss() 看与经过 NLLLoss 的计算是不是一样
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(x_input, target)
    print('crossentropyloss_output:\n', crossentropyloss_output, end='\n\n')

    # KL散度
    kl = kl_loss(x_pred=x_input, x_gt=y_input, target=target)
    print('x_input || y_input = ', kl)

    """
    得出结论：
        torch.nn.CrossEntropyLoss(logits, labels) 这个函数分为三个步骤：
            ① Softmax() ---> 将 (batch, class_num) 的数据映射到 (0, 1) 上，且和为 1，概率化；
            ② ln() ---> 继而映射到 (-infinite, 0)；
            ③ torch.nn.NLLLoss() ---> nn.NLLLoss() 的结果就是把上一步的输出与 label 对应的那个值拿出来，再去掉负号，再求【均值】。
    """