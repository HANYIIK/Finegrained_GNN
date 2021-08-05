#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/29 3:50 下午
# @Author   : Hanyiik
# @File     : get_final_confu.py
# @Function : 得到最终结果的 confusion matrix
import numpy as np
import os

DATASET_NAME = 'SEED'
EXPERTS = 2

def get_final_confu_matrix(dataset_name):
    """
    :: 功能: 得到最终的 confusion matrix
    :: 输入: dataset_name - 数据集名称('SEED'/'MPED'/'SEED_IV')
    :: 输出: 最终的 confusion matrix
    :: 用法: xxxx_confu = get_final_confu_matrix(XXXX)
    """
    file_path = f'../{EXPERTS}_experts_res/{dataset_name}/confusion_matrix/'
    confu_file_name = []

    if dataset_name == 'SEED':
        # SEED 是个 3 分类数据集
        out = np.zeros((3, 3))
    elif dataset_name == 'MPED':
        # MPED 是个 7 分类数据集
        out = np.zeros((7, 7))
    elif dataset_name == 'SEED_IV':
        # SEED_IV 是个 4 分类数据集
        out = np.zeros((4, 4))
    else:
        # 错误数据集名称报错
        raise RuntimeError('错误数据集名称输入！')

    for file_name in os.listdir(file_path):
        if '_confusion.npy' in file_name:
            confu_file_name.append(file_name)
    for confu_file in confu_file_name:
        a_confu = np.load(file_path + confu_file)
        out += a_confu
    row_sum = np.transpose(out.sum(axis=1))
    out = out / row_sum
    return out

if __name__ == '__main__':
    final_confu = get_final_confu_matrix(dataset_name=DATASET_NAME)
    print(final_confu)