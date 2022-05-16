#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2022/2/18 21:04
# @Author   : Hanyiik
# @File     : train_3_experts_with_kl.py
# @Function : 新的损失函数

import pdb

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models_kl import FineGrained3GNN
from dataset import EEGDataset
from functions import get_config, get_folders
from utils import train_utils, model_utils, xlsx_utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# KL Loss
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
    # KL-Divergence for branch 1 and branch 2
    kl_pred = exclude_gt(x_pred, target, is_log=True)
    kl_gt = exclude_gt(x_gt, target, is_log=False)
    tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')   # P = 0.7414    log(Q) = -1.4167    KL(P||Q) = 0.7414 * (ln(0.7414) + 1.4167) = 0.8285
    tmp_loss = torch.exp(-tmp_loss).mean()
    return tmp_loss


def mask_filter(logit, target):
    mask = torch.ones_like(logit)
    for i in range(logit.size(0)):
        mask[i, target[i]] = 0

    return mask * logit

def js_loss(z1, z2, target):
    # JS-Divergence for branch 1 and branch 2
    z1 = F.softmax(z1, dim=-1)
    z2 = F.softmax(z2, dim=-1)
    log_mean = torch.log((z1 + z2) / 2)

    z1 = mask_filter(z1, target)
    z2 = mask_filter(z2, target)
    log_mean = mask_filter(log_mean, target)

    tmp_loss = (F.kl_div(log_mean, z1, reduction='none') + F.kl_div(log_mean, z2, reduction='none')) / 2
    tmp_loss = torch.exp(-tmp_loss).mean()
    return tmp_loss


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

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

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
                logits_list, cam_1, cam_2, mask_1, mask_2 = self.model(data, labels)

                # ----- the new Loss
                my_loss = [self.criterion(logits_list[3], labels.long()),   # Gate Loss
                           js_loss(logits_list[1], logits_list[0], labels), # JS(E1 || E2)
                           js_loss(logits_list[2], logits_list[1], labels), # JS(E2 || E3)
                           js_loss(logits_list[0], logits_list[2], labels)] # JS(E3 || E1)
                final_loss = sum(my_loss)
                my_loss.append(final_loss)

                self.mean_loss.update(my_loss[-1].cpu().detach().numpy())

                self.optimizer.zero_grad()
                my_loss[-1].backward()
                self.optimizer.step()

                # print(f'---------- EPOCH: {epoch} | STEP: {step} | LOSS Gate: {my_loss[0]} | LOSS E1 and E2: {my_loss[1]} | LOSS E2 and E3: {my_loss[2]} | LOSS E3 and E1: {my_loss[3]} ----------')

                if step % 3 == 0:
                    # print('start test!')
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
                logits_list, cam_1, cam_2, mask_1, mask_2 = self.model(data, None)
                probs = F.softmax(logits_list[-1], dim=-1).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                self.mean_accuracy.update(probs, labels)
        acc = self.mean_accuracy.compute()
        confusion = self.mean_accuracy.confusion()
        return acc, confusion


if __name__ == '__main__':
    my_args = get_config()
    get_folders(my_args)

    if my_args.dataset_name == 'SEED' and my_args.dataset_size == 'large' and my_args.people_num == 45:
        raise RuntimeError('处理 SEED large 数据之前，请先将 people_num 改为 15！')

    run_select = int(input('选择要跑的人群(1-up, 2-bad, 3-middle, 4-good, 5-all):'))
    run_dic = {'1': '有提升空间的', '2': '较差', '3': '中等', '4': '较好', '5': '全部'}

    if my_args.dataset_name == 'MPED':
        up = []
        bad = [3, 7, 11, 12, 13, 14, 15, 16, 19, 25, 26, 27, 30]
        middle = []
        good = [1, 2, 4, 5, 6, 8, 9, 10, 17, 18, 20, 21, 22, 23, 24, 28, 29]

    # SEED
    elif my_args.dataset_name == 'SEED':
        up = []
        bad = [2, 5, 11, 13, 14, 18, 27, 28, 30, 32, 34, 37, 38, 40]
        middle = []
        good = [1, 4, 19, 20, 23, 24, 41]

    # SEED_IV
    elif my_args.dataset_name == 'SEED_IV':
        up = [9, 10, 12, 21, 26, 33, 36, 43, 45]
        bad = []
        middle = []
        good = []

    else:
        raise RuntimeError('请选择正确的数据集!')

    if run_select == 1:
        # ① 有提升空间的
        for t in range(my_args.times):
            for people in up:
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

    elif run_select == 5:
        # ④ 暴力全跑
        for t in range(my_args.times):
            for people in range(1, my_args.people_num + 1):
                trainer = Trainer(my_args, people_index=people)
                print(f'正在跑的是:{my_args.dataset_name}|{my_args.dataset_size}|{run_dic[str(run_select)]}|第{people}个人!')
                trainer.run()
                print(f'第{people}个人跑完了！')

    else:
        raise RuntimeError('请做出正确的选择!')