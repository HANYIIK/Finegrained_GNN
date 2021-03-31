#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/30 5:30 下午
# @Author   : Hanyiik
# @File     : get_node_cam.py
# @Function :
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/30 5:09 下午
# @Author   : Hanyiik
# @File     : get_final_cam.py
# @Function : 得到 node heat
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
from utils import train_utils, model_utils, xlsx_utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Node_heater(object):

    def __init__(self, args, people_index):
        super(Node_heater, self).__init__()
        self.args = args
        self.people_index = people_index
        self.batch_size = self.args.batch_size

        self.adj_matrix = EEGDataset.build_graph()

        self.max_acc = None
        self.state_dict_path = f'./res/{self.args.dataset_name}/state_dict/{self.people_index}_params.pkl'

        # 制作 DataLoader
        self.test_dataset = EEGDataset(self.args, istrain=False, people=self.people_index)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # 加载模型, load 模型参数
        self.model = FineGrained2GNN(self.args, adj=self.adj_matrix).to(DEVICE)
        self.model.load_state_dict(torch.load(self.state_dict_path, map_location=torch.device('cpu')))

    def test(self):
        self.model.eval()
        node_heats = np.zeros((1, 62))
        node_heats_list = []
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1 = self.model(data, None)
                pred_y = np.argmax(F.softmax(logits, dim=-1).cpu().detach().numpy(), axis=1)    # (batch_size,)
                labels = labels.numpy()     # (batch_size,)
                for pd_y, gt_y, my_cam in zip(pred_y, labels, cam_1):
                    if pd_y == gt_y:
                        node_heats_list.append(my_cam)
                        node_heats += my_cam
        return node_heats, node_heats_list

if __name__ == '__main__':
    args = get_config()
    heater = Node_heater(args, 5)
    cam_map, cam_map_list = heater.test()
    print(cam_map)