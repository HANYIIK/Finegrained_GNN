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
    final_cam = np.array([[358.02830975, 397.62693249, 360.04074799, 433.3194194, 437.47416775,
                       369.11361896, 380.12013738, 373.18390831, 363.13570242, 383.63839349,
                       369.50463355, 381.45057664, 392.46794108, 374.15212427, 366.6902356,
                       385.14821399, 379.32164557, 371.12723593, 366.72370139, 373.46293456,
                       382.2605545, 390.6600531, 376.80254443, 363.65894461, 388.35032884,
                       388.1036715, 379.77121568, 356.18921454, 374.74546209, 379.96486744,
                       381.23051649, 362.09430574, 357.61257095, 383.89602231, 378.90529241,
                       373.06315298, 343.33013481, 365.80006309, 369.67654443, 366.41895447,
                       340.52308749, 350.58804548, 374.49928067, 363.48451422, 357.09976166,
                       353.92510354, 357.89035524, 362.82189092, 360.25744053, 332.74040154,
                       345.27173625, 372.99097317, 399.51515315, 358.21777594, 402.45792496,
                       368.94185056, 332.69301865, 344.16478798, 339.22311612, 338.59097952,
                       336.7045543, 335.18970799]])
    '''