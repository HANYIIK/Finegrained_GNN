#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/29 3:50 下午
# @Author   : Hanyiik
# @File     : get_final_confu.py
# @Function : 得到最终结果的 confusion matrix，并画出混淆矩阵图。
import types

import numpy as np
import os
import pdb
import seaborn as sns
import matplotlib.pyplot as plt


COLOR = 'Blues'
DATASET_NAME = 'SEED_IV'    # 数据集名称
FONT_STYLE = 'DejaVu Sans'    # 字体
LABEL_FONT_SIZE = 30
FONT_SIZE = 60

"""
可选颜色：
    'Accent', 'Accent_r', 
    'Blues', 'Blues_r', 
    'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 
    'BuPu', 'BuPu_r', 
    'CMRmap', 'CMRmap_r', 
    'Dark2', 'Dark2_r', 
    'GnBu', 'GnBu_r', 
    'Greens', 'Greens_r', 
    'Greys', 'Greys_r', 
    'OrRd', 'OrRd_r', 
    'Oranges', 'Oranges_r', 
    'PRGn', 'PRGn_r', 
    'Paired', 'Paired_r', 
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
    'PiYG', 'PiYG_r', 
    'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 
    'PuOr', 'PuOr_r', 
    'PuRd', 'PuRd_r', 
    'Purples', 'Purples_r', 
    'RdBu', 'RdBu_r', 
    'RdGy', 'RdGy_r', 
    'RdPu', 'RdPu_r', 
    'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 
    'Reds', 'Reds_r', 
    'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
    'Spectral', 'Spectral_r', 
    'Wistia', 'Wistia_r', 
    'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
    'afmhot', 'afmhot_r', 
    'autumn', 'autumn_r', 
    'binary', 'binary_r', 
    'bone', 'bone_r', 
    'brg', 'brg_r', 
    'bwr', 'bwr_r', 
    'cividis', 'cividis_r', 
    'cool', 'cool_r', 
    'coolwarm', 'coolwarm_r', 
    'copper', 'copper_r', 
    'crest', 'crest_r', 
    'cubehelix', 'cubehelix_r', 
    'flag', 'flag_r', 
    'flare', 'flare_r', 
    'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
    'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 
    'gray', 'gray_r', 
    'hot', 'hot_r', 
    'hsv', 'hsv_r', 
    'icefire', 'icefire_r', 
    'inferno', 'inferno_r', 
    'jet', 'jet_r', 
    'magma', 'magma_r', 
    'mako', 'mako_r', 
    'nipy_spectral', 'nipy_spectral_r', 
    'ocean', 'ocean_r', 
    'pink', 'pink_r', 
    'plasma', 'plasma_r', 
    'prism', 'prism_r', 
    'rainbow', 'rainbow_r', 
    'rocket', 'rocket_r', 
    'seismic', 'seismic_r', 
    'spring', 'spring_r', 
    'summer', 'summer_r', 
    'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 
    'terrain', 'terrain_r', 
    'turbo', 'turbo_r', 
    'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 
    'viridis', 'viridis_r', 
    'vlag', 'vlag_r', 
    'winter', 'winter_r'
"""


class ConfusionMatrix(object):
    def __init__(self, datasetName):
        super(ConfusionMatrix, self).__init__()
        self.datasetName = datasetName
        self.countMatrix = None
        self.finalConfuMatrix = None

    def getConfuFromFile(self):
        if self.datasetName == 'SEED':
            self.countMatrix = np.zeros((3, 3))
        elif self.datasetName == 'MPED':
            self.countMatrix = np.zeros((7, 7))
        elif self.datasetName == 'SEED_IV':
            self.countMatrix = np.zeros((4, 4))
        else:
            raise RuntimeError('错误数据集名称输入！')

        # --- read file
        file_path = f'../res/{self.datasetName}/confusion_matrix/'
        confu_file_name = []
        for file_name in os.listdir(file_path):
            if '_confusion.npy' in file_name:
                confu_file_name.append(file_name)
        for confu_file in confu_file_name:
            a_confu = np.load(file_path + confu_file)
            self.countMatrix += a_confu
        row_sum = self.countMatrix.sum(axis=1)[:,None]
        self.finalConfuMatrix = self.countMatrix / row_sum

    def draw(self):
        labels = {0: ['happy', 'neutral', 'sad'],
                  1: ['happy', 'neutral', 'sad', 'fear'],
                  2: ['joy', 'funny', 'neutral', 'sad', 'fear', 'disgust', 'anger']}

        if 'MPED' in self.datasetName:
            label = labels[2]
        elif 'SEED_IV' in self.datasetName:
            label = labels[1]
        else:
            label = labels[0]

        sns.set_theme(font=FONT_STYLE, font_scale=2)
        f, ax = plt.subplots()
        Confu = self.finalConfuMatrix * 100
        sns.heatmap(Confu, annot=True, ax=ax, cmap=COLOR, xticklabels=label, yticklabels=label, annot_kws={'fontsize': FONT_SIZE}, fmt='.2f', cbar=False)

        plt.xticks(fontsize=LABEL_FONT_SIZE, rotation=0)  # x 轴刻度的字体大小（文本包含在 pd_data 中了）
        plt.yticks(fontsize=LABEL_FONT_SIZE, rotation=0)  # y 轴刻度的字体大小（文本包含在 pd_data 中了）
        # plt.savefig(f'../res/{self.datasetName}/confu.png', dpi=300)
        plt.show()

    def generate(self):
        self.getConfuFromFile()
        self.draw()


if __name__ == '__main__':
    c = ConfusionMatrix(DATASET_NAME)
    c.generate()