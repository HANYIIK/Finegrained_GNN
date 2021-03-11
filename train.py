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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

    def train(self):
        self.model.train()
        total_loss = 0
        for step, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            logits, cam_1, cam_2 = self.model(data, labels)
            loss = self.criterion(logits, labels.long())
            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item() * self.batch_size
            self.optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    def test(self):
        self.model.eval()
        correct = 0
        total_loss = 0
        for step, (data, labels) in enumerate(self.test_loader):
            # data = data.to(DEVICE)
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                logits, cam_1, cam_2 = self.model(data, None)
                loss = self.criterion(logits, labels.long())
                total_loss += loss.item() * self.batch_size
                pred_y = torch.max(logits, 1)[1].float()
            correct += torch.sum(labels == pred_y).item()
        test_acc = correct / len(self.test_loader.dataset)
        test_loss = total_loss / len(self.test_loader.dataset)
        return test_loss, test_acc

    def run(self):
        train_losses = []
        test_losses = []
        test_accs = []

        for epoch in range(1, self.args.max_epochs+1):
            loss = self.train()
            train_losses.append(loss)
            test_loss, test_acc = self.test()
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

            if epoch % 10 == 0:
                plt.plot(train_losses)
                plt.title('Loss vs Epoch')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.savefig('loss_vs_epoch')
                plt.close()

                plt.plot(train_losses, label='Train Loss')
                plt.plot(test_losses, label='Test Loss')
                plt.title('Loss vs Epoch')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig('train_test_loss_vs_epoch')
                plt.close()

                plt.plot(test_accs)
                plt.title('Test Accuracy vs Epoch')
                plt.ylabel('Test Accuracy')
                plt.xlabel('Epoch')
                plt.savefig('test_acc_vs_epoch')
                plt.close()

if __name__ == '__main__':
    my_args = get_config()
    trainer = Trainer(my_args, people_index=1)
    trainer.run()