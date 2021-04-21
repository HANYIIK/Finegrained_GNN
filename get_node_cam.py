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
        indices_list = []
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1, indices = self.model(data, None)
                pred_y = np.argmax(F.softmax(logits, dim=-1).cpu().detach().numpy(), axis=1)    # (batch_size,)
                labels = labels.numpy()     # (batch_size,)
                for pd_y, gt_y, my_cam, mask in zip(pred_y, labels, cam_1, indices):
                    if pd_y == gt_y:
                        indices_list.append(mask)
                        node_heats_list.append(my_cam)
                        node_heats += my_cam.cpu().numpy()
        return node_heats, node_heats_list, indices_list

if __name__ == '__main__':
    my_args = get_config()
    final_cam = np.zeros((1, 62))
    for i in range(1, my_args.people_num+1):
        print(f'在跑第{i}个人！')
        heater = Node_heater(my_args, i)
        cam_map, cam_map_list, mask_list = heater.test()
        final_cam += cam_map
    my_max = final_cam.max()
    my_min = final_cam.min()
    # 归一化
    final_cam = (final_cam - my_min) / (my_max - my_min)
    print(final_cam)
    '''
    【SEED_IV】
    0.12556422 0.54960047 0.14936516 0.94210945 1.         0.28011019
    0.40882462 0.32901019 0.2471076  0.44972775 0.30218521 0.41574843
    0.53031228 0.3401455  0.25874884 0.46229481 0.41052598 0.32599669
    0.2872062  0.35592298 0.44084707 0.52831766 0.39626741 0.22691073
    0.49013698 0.50247076 0.42588565 0.22141499 0.37821943 0.43291951
    0.45484825 0.2717623  0.18171915 0.45634424 0.41585634 0.36583692
    0.10065312 0.31605177 0.35086819 0.32742598 0.05747448 0.13096717
    0.38128851 0.27946078 0.22205847 0.20619003 0.25293256 0.30411015
    0.27959918 0.         0.08956352 0.38046731 0.6532613  0.25776518
    0.70518351 0.37502821 0.00422292 0.08399571 0.04865675 0.05235251
    0.03989542 0.02236623
    
    【MPED】
    0.0656388  0.55128085 0.         1.         0.95556437 0.4917683
    0.56956665 0.49938408 0.56942387 0.6488746  0.56424797 0.63079456
    0.70697612 0.448845   0.46955378 0.66878459 0.57206704 0.5707026
    0.58466177 0.62911174 0.63386984 0.74040552 0.56209513 0.39309128
    0.71782563 0.57531175 0.57730759 0.54028319 0.57013237 0.58235887
    0.75160245 0.45788525 0.56265577 0.70541006 0.58726254 0.53583829
    0.48138222 0.40902671 0.45770187 0.73761606 0.58295151 0.51562543
    0.6974304  0.64933333 0.49438997 0.37117451 0.52776821 0.62713085
    0.74982691 0.62415322 0.60658784 0.76368357 0.66581669 0.49360521
    0.76361589 0.78146747 0.57601974 0.33195182 0.49725869 0.43282113
    0.36009646 0.41467111
    '''