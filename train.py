#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/8 11:50
# @Author   : Hanyiik
# @File     : train.py
# @Function : 训练模型
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import FineGrained2GNN
from dataset import EEGDataset
from functions import get_config, get_folders
from utils import train_utils, model_utils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):

    def __init__(self, args, people_index):
        super(Trainer, self).__init__()
        self.args = args
        self.people_index = people_index
        self.batch_size = self.args.batch_size

        self.adj_matrix = EEGDataset.build_graph()

        # 制作 DataLoader
        self.train_dataset = EEGDataset(self.args, istrain=True, people=self.people_index)
        self.test_dataset = EEGDataset(self.args, istrain=False, people=self.people_index)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 加载模型
        self.model = FineGrained2GNN(self.args, adj=self.adj_matrix).to(DEVICE)
        self.model.apply(model_utils.weight_init)

        # 加载 Optimizer 与 Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

        # Accuracy 与 Loss
        self.mean_accuracy = train_utils.MeanAccuracy(self.args.classes_num)
        self.mean_loss = train_utils.MeanLoss(self.batch_size)

    def run(self):
        max_acc = 0

        for epoch in range(1, self.args.max_epochs+1):
            self.mean_loss.reset()

            for step, (data, labels) in enumerate(self.train_loader):
                self.model.train()
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                logits, cam_1 = self.model(data, labels)
                loss = self.criterion(logits, labels.long())
                self.mean_loss.update(loss.cpu().detach().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 5 == 0:
                    acc, confusion = self.test()
                    print(f'Test Results - Epoch: {epoch} Accuracy: {acc * 100:.2f}%')
                    if acc > max_acc:
                        max_acc = acc
                        np.save(f'./confusion_matrix/{self.people_index}_confusion.npy', confusion)
                        torch.save(self.model.state_dict(), f'./state_dict/{self.people_index}_params.pkl')

            mloss = self.mean_loss.compute()
            self.lr_scheduler.step(mloss)

        # self.model.load_state_dict(state_dict)
        str_write = f'第 {self.people_index} 个人的 Max Accuracy: {max_acc * 100:.2f}%'
        print('***********************************' + str_write + '***********************************\n\n\n')
        self.write_result(str_write)


    def test(self):
        self.model.eval()
        self.mean_accuracy.reset()
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1 = self.model(data, None)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
        acc = self.mean_accuracy.compute()
        confusion = self.mean_accuracy.confusion()
        return acc, confusion

    def write_result(self, wtr):
        file_name = 'rate={}、batch={}.txt'.format(self.args.rate_1, self.batch_size)
        file_path = './res/'
        f = open(file_path + file_name, 'a')
        f.write(wtr)
        f.write('\n')
        f.close()


if __name__ == '__main__':
    get_folders()
    my_args = get_config()

    bad = [10, 11, 12, 14, 21, 25, 26, 27, 30]
    middle = [3, 7, 13, 15, 19, 22]

    # ① 暴力全跑
    # for people in range(1, my_args.people_num+1):
    #     trainer = Trainer(my_args, people_index=people)
    #     trainer.run()

    # ② 跑烂
    for people in bad:
        trainer = Trainer(my_args, people_index=people)
        trainer.run()

    # ③ 跑中等
    # for people in middle:
    #     trainer = Trainer(my_args, people_index=people)
    #     trainer.run()