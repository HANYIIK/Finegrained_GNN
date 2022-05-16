#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/7 22:29
# @Author   : Hanyiik
# @File     : dataset.py
# @Function : 用于 ChebNet 的 EEG 数据集
import pdb

from torch.utils.data import Dataset

from scipy.sparse import csr_matrix
import numpy as np

from functions import load_one_people_npy


class EEGDataset(Dataset):
    """
      :: 输入1: istrain - 是否为训练集?
      :: 输入2: people - 要跑第几个人的数据?
      :: 返回值: 规定某个人的 TRAIN or TEST 数据
      :: 用法:
            args = get_config()
            train_dataset = EEGDataset(args, istrain=True, people=30)
            adj = train_dataset.build_graph()

          ① SEED_small:
              train_data: (2010, 62, 5)
              train_label: (2010,)
              test_data: (1384, 62, 5)
              test_label: (1384,)
              people: 45

          ② MPED_small:
              train_data: (2520, 62, 5)
              train_label: (2520,)
              test_data: (840, 62, 5)
              test_label: (840,)
              people: 30

          ③ MPED_large:
              train_data: (97440, 62, 5)
              train_label: (97440,)
              test_data: (3360, 62, 5)
              test_label: (3360,)
              people: 30
      """
    def __init__(self, args, istrain=True, people=1):
        super().__init__()
        train_data, train_label, test_data, test_label = load_one_people_npy(args, people=people)

        if istrain:
            # 加载训练集
            data = train_data  # (K1, 62, 5)
            labels = train_label  # (K1,)
        else:
            # 加载测试集
            data = test_data  # (K2, 62, 5)
            labels = test_label  # (K2,)

        self.eeg_data = data
        self.eeg_labels = labels

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.eeg_labels[idx]

    def __len__(self):
        return self.eeg_data.shape[0]

    @staticmethod
    def build_graph():
        row = np.array(
            [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
             13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25, 25, 26, 26,
             27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40,
             41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54,
             54, 55, 55, 56, 57, 58, 59,
             60, 1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12,
             20, 13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26,
             34, 27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47,
             40, 48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58,
             54, 59, 55, 60, 56, 61, 61, 58, 59, 60, 61])

        col = np.array(
            [1, 3, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 6, 14, 7, 15, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20,
             13, 21, 22, 15, 23, 16, 24, 17, 25, 18, 26, 19, 27, 20, 28, 21, 29, 22, 30, 31, 24, 32, 25, 33, 26, 34,
             27, 35, 28, 36, 29, 37, 30, 38, 31, 39, 40, 33, 41, 34, 42, 35, 43, 36, 44, 37, 45, 38, 46, 39, 47, 40,
             48, 49, 42, 50, 43, 51, 44, 52, 45, 52, 46, 53, 47, 54, 48, 54, 49, 55, 56, 51, 57, 52, 57, 53, 58, 54,
             59, 55, 60, 56, 61, 61, 58,
             59, 60, 61, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
             11, 12, 12, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 23, 23, 24, 24, 25,
             25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38,
             39, 39, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 50, 50, 51, 51, 52, 52,
             53, 53, 54, 54, 55, 55, 56, 57, 58, 59, 60])
        weight = np.ones(236).astype('float32')
        A = csr_matrix((weight, (row, col)), shape=(62, 62))
        return A
