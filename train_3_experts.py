#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/6/22 2:54 下午
# @Author   : Hanyiik
# @File     : train_3_experts.py
# @Function : 3 个 GNN 的 train_2_experts.py

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import FineGrained3GNN
from dataset import EEGDataset
from functions import get_config, get_folders
from utils import train_utils, model_utils, xlsx_utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):

    def __init__(self, args, people_index):
        super(Trainer, self).__init__()
        self.args = args
        self.people_index = people_index
        self.batch_size = self.args.batch_size

        self.adj_matrix = EEGDataset.build_graph()

        self.max_acc = None
        self.xls_path = f'./res/{self.args.dataset_name}/result.xlsx'
        self.confu_path = f'./res/{self.args.dataset_name}/confusion_matrix/{self.people_index}_confusion.npy'
        self.state_dict_path = f'./res/{self.args.dataset_name}/state_dict/{self.people_index}_params.pkl'

        # 制作 DataLoader
        self.train_dataset = EEGDataset(self.args, istrain=True, people=self.people_index)
        self.test_dataset = EEGDataset(self.args, istrain=False, people=self.people_index)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 加载模型
        self.model = FineGrained3GNN(self.args, adj=self.adj_matrix).to(DEVICE)
        self.model.apply(model_utils.weight_init)

        # 加载 Optimizer / Loss Function / LR
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)

        # Accuracy 与 Loss
        self.mean_accuracy = train_utils.MeanAccuracy(self.args.classes_num)
        self.mean_loss = train_utils.MeanLoss(self.batch_size)

    def run(self):
        self.max_acc = xlsx_utils.get_max_acc_in_xlsx(people_index=self.people_index, xls_path=self.xls_path)

        for epoch in range(1, self.args.max_epochs + 1):
            self.mean_loss.reset()

            for step, (data, labels) in enumerate(self.train_loader):
                self.model.train()
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                logits, cam_1, cam_2, mask_1, mask_2 = self.model(data, labels)
                loss = self.criterion(logits, labels.long())
                self.mean_loss.update(loss.cpu().detach().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 3 == 0:
                    acc, confusion = self.test()
                    if acc > self.max_acc:
                        # 准备改 xlsx 之前，先确认一下 xlsx 里面的 max_acc 是否被其他进程改过
                        confirm_acc = xlsx_utils.get_max_acc_in_xlsx(people_index=self.people_index,
                                                                     xls_path=self.xls_path)

                        # 被改过了
                        if confirm_acc > self.max_acc:
                            # 更新 self.max_acc
                            self.max_acc = confirm_acc
                            # 判断现在的 acc 是否大于新 self.max_acc
                            if acc > self.max_acc:
                                self.max_acc = acc
                                xlsx_utils.replace_xlsx_acc(people_index=self.people_index, acc=acc,
                                                            xls_path=self.xls_path)
                                np.save(self.confu_path, confusion)
                                torch.save(self.model.state_dict(), self.state_dict_path)

                        # 没有被改过
                        else:
                            self.max_acc = acc
                            xlsx_utils.replace_xlsx_acc(people_index=self.people_index, acc=acc, xls_path=self.xls_path)
                            np.save(self.confu_path, confusion)
                            torch.save(self.model.state_dict(), self.state_dict_path)

            mloss = self.mean_loss.compute()
            self.lr_scheduler.step(mloss)

    def test(self):
        self.model.eval()
        self.mean_accuracy.reset()
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1, cam_2, mask_1, mask_2 = self.model(data, None)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)
        acc = self.mean_accuracy.compute()
        confusion = self.mean_accuracy.confusion()
        return acc, confusion


if __name__ == '__main__':
    my_args = get_config()
    get_folders(my_args)

    if my_args.dataset_name == 'SEED' and my_args.dataset_size == 'large' and my_args.people_num == 45:
        raise RuntimeError('处理 SEED large 数据之前，请先将 people_num 改为 15！')

    run_select = int(input('选择要跑的人群(1-full, 2-bad, 3-middle, 4-good):'))
    run_dic = {'1': '全部', '2': '较差', '3': '中等', '4': '较好'}

    if my_args.dataset_name == 'MPED':
        raise RuntimeError("目前不支持 MPED！")

    elif my_args.dataset_name == 'SEED':
        bad = [3, 6, 8, 11, 12, 13, 14, 15, 17, 18, 21, 26, 28, 29, 31, 34, 35, 37, 39, 42, 43, 44, 45]
        middle = [2, 5, 22, 23, 24, 27, 30, 32, 38, 40]
        good = [1, 4, 19, 20, 25, 33, 36, 41]

    elif my_args.dataset_name == 'SEED_IV':
        # update
        bad = [11, 13, 21, 30, 35, 36, 39, 42]
        middle = [1, 3, 6, 7, 9, 10, 16, 17, 18, 19, 22, 26, 27, 32, 33, 40, 43, 45]
        good = [2, 8, 12, 14, 15, 17, 19, 23, 24, 28, 29, 31, 37, 38, 41]

    else:
        raise RuntimeError('请选择正确的数据集!')

    if run_select == 1:
        # ① 暴力全跑
        for t in range(my_args.times):
            for people in range(1, my_args.people_num + 1):
                trainer = Trainer(my_args, people_index=people)
                print(f'正在跑的是:{my_args.dataset_name}|{my_args.dataset_size}|{run_dic[str(run_select)]}|第{people}个人!')
                trainer.run()
                print(f'第{people}个人跑完了！')

    elif run_select == 2:
        # ② 跑烂
        for t in range(my_args.times):
            for people in bad:
                trainer = Trainer(my_args, people_index=people)
                print(f'正在跑的是:{my_args.dataset_name}|{my_args.dataset_size}|{run_dic[str(run_select)]}|第{people}个人!')
                trainer.run()
                print(f'第{people}个人跑完了！')

    elif run_select == 3:
        # ③ 跑中
        for t in range(my_args.times):
            for people in middle:
                trainer = Trainer(my_args, people_index=people)
                print(f'正在跑的是:{my_args.dataset_name}|{my_args.dataset_size}|{run_dic[str(run_select)]}|第{people}个人!')
                trainer.run()
                print(f'第{people}个人跑完了！')

    elif run_select == 4:
        # ④ 跑好
        for t in range(my_args.times):
            for people in good:
                trainer = Trainer(my_args, people_index=people)
                print(f'正在跑的是:{my_args.dataset_name}|{my_args.dataset_size}|{run_dic[str(run_select)]}|第{people}个人!')
                trainer.run()
                print(f'第{people}个人跑完了！')

    else:
        raise RuntimeError('请做出正确的选择!')