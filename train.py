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

from models import FineGrainedGNN
from dataset import EEGDataset
from functions import get_config
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
        self.model = FineGrainedGNN(self.args, adj=self.adj_matrix).to(DEVICE)
        self.model.apply(model_utils.weight_init)

        # 加载 Optimizer 与 Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

        # Accuracy 与 Loss
        self.mean_accuracy = train_utils.MeanAccuracy(self.args.classes_num)
        self.mean_loss = train_utils.MeanLoss(self.batch_size)

    def train(self):
        self.model.train()
        self.mean_loss.reset()
        desc = "TRAINING - loss: {:.4f}"
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=desc.format(0))
        for step, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            logits, cam_1, cam_2 = self.model(data, labels)
            loss = self.criterion(logits, labels)
            self.mean_loss.update(loss.cpu().detach().numpy())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.desc = desc.format(loss)
            pbar.update(1)
        pbar.close()
        return self.mean_loss.compute()

    def test(self, epoch):
        self.model.eval()
        self.mean_accuracy.reset()
        pbar = tqdm(total=len(self.test_loader), leave=False, desc="TESTING")
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1, cam_2 = self.model(data, None)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
                pbar.update(1)
        pbar.close()
        acc = self.mean_accuracy.compute()
        tqdm.write(f"Test Results - Epoch: {epoch} Accuracy: {acc * 100:.2f}%")
        return acc

    def run(self):
        max_acc = 0
        for epoch in range(1, self.args.max_epochs+1):
            mloss = self.train()
            acc = self.test(epoch)
            if acc > max_acc:
                max_acc = acc
                torch.save(self.model.state_dict(), f'{self.people_index}_params.pkl')
            self.lr_scheduler.step(mloss)

        # self.model.load_state_dict(state_dict)
        str_write = f'第 {self.people_index} 个人的 Max Accuracy: {max_acc * 100:.2f}%'
        print('***********************************' + str_write + '***********************************\n\n\n')
        self.write_result(str_write)
        return max_acc

    def write_result(self, wtr):
        file_name = 'k={}、kernel={}、epoch={}.txt'.format(self.args.K, self.args.filter_num, self.args.max_epochs)
        file_path = './res/'
        f = open(file_path + file_name, 'a')
        f.write(wtr)
        f.write('\n')
        f.close()


if __name__ == '__main__':
    my_args = get_config()
    trainer = Trainer(my_args, people_index=1)
    trainer.run()