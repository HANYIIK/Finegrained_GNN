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

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from three_cam_model import FineGrained3GNN
from dataset import EEGDataset
from functions import get_config


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
        self.model = FineGrained3GNN(self.args, adj=self.adj_matrix).to(DEVICE)

        self.model.load_state_dict(torch.load(self.state_dict_path, map_location=torch.device('cpu')))


    def test_3_experts(self):
        self.model.eval()
        node_heats_1 = np.zeros((1, 62))
        node_heats_2 = np.zeros((1, 62))
        node_heats_3 = np.zeros((1, 62))

        node_heats_positive_1 = np.zeros((1, 62))
        node_heats_neal_1 = np.zeros((1, 62))
        node_heats_negative_1 = np.zeros((1, 62))

        node_heats_positive_2 = np.zeros((1, 62))
        node_heats_neal_2 = np.zeros((1, 62))
        node_heats_negative_2 = np.zeros((1, 62))

        node_heats_positive_3 = np.zeros((1, 62))
        node_heats_neal_3 = np.zeros((1, 62))
        node_heats_negative_3 = np.zeros((1, 62))

        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].to(DEVICE), batch[1]
                logits, cam_1, cam_2, cam_3, indices_1, indices_2, indices_3 = self.model(data, None)
                pred_y = np.argmax(F.softmax(logits, dim=-1).cpu().detach().numpy(), axis=1)    # (batch_size,)
                labels = labels.numpy()     # (batch_size,)
                for pd_y, gt_y, my_cam_1, my_cam_2, my_cam_3, mask_1, mask_2, mask_3 in zip(pred_y, labels, cam_1, cam_2, cam_3, indices_1, indices_2, indices_3):
                    if pd_y == gt_y:
                        node_heats_1 += my_cam_1
                        node_heats_2 += my_cam_2
                        node_heats_3 += my_cam_3
                        if pd_y == 0:
                            node_heats_positive_1 += my_cam_1
                            node_heats_positive_2 += my_cam_2
                            node_heats_positive_3 += my_cam_3
                        elif pd_y == 1:
                            node_heats_neal_1 += my_cam_1
                            node_heats_neal_2 += my_cam_2
                            node_heats_neal_3 += my_cam_3
                        elif pd_y == 2:
                            node_heats_negative_1 += my_cam_1
                            node_heats_negative_2 += my_cam_2
                            node_heats_negative_3 += my_cam_3
        return node_heats_1, node_heats_positive_1, node_heats_neal_1, node_heats_negative_1, \
               node_heats_2, node_heats_positive_2, node_heats_neal_2, node_heats_negative_2, \
               node_heats_3, node_heats_positive_3, node_heats_neal_3, node_heats_negative_3


def get_map(origin_cam):
        the_max = origin_cam.max()
        the_min = origin_cam.min()
        return (origin_cam - the_min) / (the_max - the_min)

if __name__ == '__main__':
    my_args = get_config()
    people_list = []

    final_cam_1 = np.zeros((1, 62))
    final_cam_positive_1 = np.zeros((1, 62))
    final_cam_neal_1 = np.zeros((1, 62))
    final_cam_negative_1 = np.zeros((1, 62))

    final_cam_2 = np.zeros((1, 62))
    final_cam_positive_2 = np.zeros((1, 62))
    final_cam_neal_2 = np.zeros((1, 62))
    final_cam_negative_2 = np.zeros((1, 62))

    final_cam_3 = np.zeros((1, 62))
    final_cam_positive_3 = np.zeros((1, 62))
    final_cam_neal_3 = np.zeros((1, 62))
    final_cam_negative_3 = np.zeros((1, 62))

    for file_name in os.listdir(f'./res/{my_args.dataset_name}/state_dict/'):
        if '_params.pkl' in file_name:
            people_list.append(int(file_name.split('_params.pkl')[0]))


    for i in sorted(people_list):
        print(f'在跑第{i}个人！')
        heater = Node_heater(my_args, i)
        cam_map_1, cam_map_positive_1, cam_map_neal_1, cam_map_negative_1, \
        cam_map_2, cam_map_positive_2, cam_map_neal_2, cam_map_negative_2, \
        cam_map_3, cam_map_positive_3, cam_map_neal_3, cam_map_negative_3 = heater.test_3_experts()

        final_cam_1 += cam_map_1
        final_cam_positive_1 += cam_map_positive_1
        final_cam_neal_1 += cam_map_neal_1
        final_cam_negative_1 += cam_map_negative_1

        final_cam_2 += cam_map_2
        final_cam_positive_2 += cam_map_positive_2
        final_cam_neal_2 += cam_map_neal_2
        final_cam_negative_2 += cam_map_negative_2

        final_cam_3 += cam_map_3
        final_cam_positive_3 += cam_map_positive_3
        final_cam_neal_3 += cam_map_neal_3
        final_cam_negative_3 += cam_map_negative_3

    # 归一化
    final_cam_1 = str(get_map(final_cam_1)).replace('\n', '\t').strip()
    final_cam_positive_1 = str(get_map(final_cam_positive_1)).replace('\n', '\t').strip()
    final_cam_neal_1 = str(get_map(final_cam_neal_1)).replace('\n', '\t').strip()
    final_cam_negative_1 = str(get_map(final_cam_negative_1)).replace('\n', '\t').strip()

    final_cam_2 = str(get_map(final_cam_2)).replace('\n', '\t').strip()
    final_cam_positive_2 = str(get_map(final_cam_positive_2)).replace('\n', '\t').strip()
    final_cam_neal_2 = str(get_map(final_cam_neal_2)).replace('\n', '\t').strip()
    final_cam_negative_2 = str(get_map(final_cam_negative_2)).replace('\n', '\t').strip()

    final_cam_3 = str(get_map(final_cam_3)).replace('\n', '\t').strip()
    final_cam_positive_3 = str(get_map(final_cam_positive_3)).replace('\n', '\t').strip()
    final_cam_neal_3 = str(get_map(final_cam_neal_3)).replace('\n', '\t').strip()
    final_cam_negative_3 = str(get_map(final_cam_negative_3)).replace('\n', '\t').strip()

    print(
        f'全部：\nCAM_1\n{final_cam_1}\nCAM_2\n{final_cam_2}\nCAM_3\n{final_cam_3}\n'
        f'积极：\nCAM_1\n{final_cam_positive_1}\nCAM_2\n{final_cam_positive_2}\nCAM_3\n{final_cam_positive_3}\n'
        f'中性：\nCAM_1\n{final_cam_neal_1}\nCAM_2\n{final_cam_neal_2}\nCAM_3\n{final_cam_neal_3}\n'
        f'消极：\nCAM_1\n{final_cam_negative_1}\nCAM_2\n{final_cam_negative_2}\nCAM_3\n{final_cam_negative_3}\n'
    )