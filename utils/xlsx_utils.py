#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/15 22:27
# @Author   : Hanyiik
# @File     : xlsx_utils.py
# @Function : 保持 .xlsx 文件里的 acc 始终最大
import numpy as np
import pandas as pd
import sys
import pdb
import os


# 更新哪个数据集
DATASET_NAME = 'MPED'

# 跑的 model 的版本
EXPERTS = '2 Experts'

# txt 文件路径
TXT_PATH=f'../res/{DATASET_NAME}/1.txt'

# xls 文件路径
XLS_PATH=f'../res/{DATASET_NAME}/result.xlsx'

# 用于更新的 xls 文件路径
UPDATE_XLS_PATH = f'C:/Users/HANYIIK/Desktop/liyang_res/{DATASET_NAME}/result.xlsx'
# UPDATE_XLS_PATH = f'C:/Users/HANYIIK/Desktop/hyk_res/{DATASET_NAME}/result.xlsx'
# UPDATE_XLS_PATH = f'F:/FB-relu-2GNN-run/res/{DATASET_NAME}/result.xlsx'

# 用于更新的 state_dict 文件路径
STATE_DICT_PATH = f'C:/Users/HANYIIK/Desktop/liyang_res/{DATASET_NAME}/state_dict/'
# STATE_DICT_PATH = f'C:/Users/HANYIIK/Desktop/hyk_res/{DATASET_NAME}/state_dict/'
# STATE_DICT_PATH = f'F:/FB-relu-2GNN-run/res/{DATASET_NAME}/state_dict/'

# 最终结果的 xls 文件路径
FINAL_XLS_PATH = f'F:/final_res/{DATASET_NAME}/result.xlsx'


def use_txt_update_xlsx(txt_path, xls_path):
    """
    :: 功能: 用 txt 文件里的大 acc 替换 xlsx 文件里的小 acc
    :: 输入: txt_path - .txt 文件路径
            xls_path - .xlsx 文件路径
    :: 输出: 是否替换
    :: 用法: update_max_acc(txt_path='../res/k=2、kernel=32、rate=0.5、epoch=100.txt',
                            xls_path='../res/result.xlsx')
    """
    result_list = extract_txt_accs(txt_path)
    for res in result_list:
        replace_xlsx(res[0], res[1], xls_path)


def replace_xlsx(people_index, acc, xls_path):
    """
    :: 功能: 判断 acc 是否大于 xlsx 里的 max_acc，如果大于，则更改 xlsx 里对应的 max_acc 为 acc 的值。
    :: 输入: people_index - 第几个人？
            acc - 每次训练结束得到的某个人的最大 acc
    :: 输出: 是否替换
    :: 用法: replace_xlsx(people_index=6, acc=0.3875456121455654212)
    """
    acc = round(acc, 4)
    acc = np.float64(acc)
    df = pd.read_excel(xls_path, engine='openpyxl',
                       sheet_name='Sheet1', usecols=['people', EXPERTS],
                       dtype={'people': int, EXPERTS: float}).fillna(0)
    df = pd.DataFrame(df)
    max_acc = df[EXPERTS][df['people'] == people_index].values[0]     # numpy 类型的 (1,) 数值, float64
    if acc > max_acc:
        df.loc[df['people']==people_index, EXPERTS] = acc
        print(f'更新第{people_index}个人的数据为：{acc * 100:.2f}%')
        df.to_excel(xls_path, engine='openpyxl', sheet_name='Sheet1')


def extract_txt_accs(txt_path):
    """
    :: 功能: 得到 .txt 文件里的 people_index 与 对应的 acc
    :: 输入: txt_path - .txt 文件路径
    :: 输出: 由 (people_index, acc) 组成的 n 个人的 list
    :: 用法: result_list = extract_accs(txt_path)
    """

    res_list = []
    with open(txt_path, 'rb') as f:
        for line in f.readlines():
            non1, people_index, non2, strmax, stracc, acc = line.split()
            b, people_index, non = str(people_index).split("'", 3)
            b, acc, non = str(acc).split("'", 3)
            people_index = int(people_index)
            acc = round(float(acc.strip("%"))/100.0, 4)
            res_list.append((people_index, acc))
    return res_list


# 工具人函数
def get_max_acc_in_xlsx(people_index, xls_path):
    """
    :: 功能: 得到 .xlsx 文件中 people_index 个人的 max_acc 数值
    :: 输入: people_index - 第几个人?
            xls_path - .xlsx文件路径
    :: 输出: 第 people_index 个人的 max_acc (numpy 类型的 (1,) 数值, float64)
    :: 用法: max_acc = get_max_acc_in_xlsx(people_index=1, xls_path='../res/result.xlsx')
    """
    df = pd.read_excel(xls_path, engine='openpyxl',
                       sheet_name='Sheet1', usecols=['people', EXPERTS],
                       dtype={'people': int, EXPERTS: float}).fillna(0)
    df = pd.DataFrame(df)
    max_acc = df[EXPERTS][df['people'] == people_index].values[0]
    return max_acc

# 工具人函数
def replace_xlsx_acc(people_index, acc, xls_path):
    """
    :: 功能: 更改 xlsx 里对应的 max_acc 为 acc 的值
    :: 输入: people_index - 第几个人？
            acc - 每次训练结束得到的某个人的最大 acc
    :: 输出:
    :: 用法: replace_xlsx(people_index=6, acc=0.3875456121455654212)
    """
    acc = round(acc, 4)
    acc = np.float64(acc)
    df = pd.read_excel(xls_path, engine='openpyxl',
                       sheet_name='Sheet1', usecols=['people', EXPERTS],
                       dtype={'people': int, EXPERTS: float}).fillna(0)
    df = pd.DataFrame(df)
    df.loc[df['people']==people_index, EXPERTS] = acc
    print(f'【更新第{people_index}个人的数据为：{acc * 100:.2f}%】')
    df.to_excel(xls_path, engine='openpyxl', sheet_name='Sheet1')


def use_xlsx_update_xlsx(final_xls_path, update_xls_path, state_dict_path):
    """
    :: 功能: 两个 xlsx 文件取最大，保持 final_xls_path 文件的结果最大
    :: 输入: final_xls_path - 最终结果 xlsx
            update_xls_path - 用于更新 final_xls_path 的 xlsx
    :: 输出: final_xls_path 文件中没有被改动人组成的 list
    :: 用法: nochange_people = use_xlsx_update_xlsx(final_xls_path=FINAL_XLS_PATH, update_xls_path=UPDATE_XLS_PATH)
    """
    # 先得到 update_xls_path 中被改过的人
    changed_people_list = get_changed_people(state_dict_path)
    nochange = []
    for changed_people in changed_people_list:
        # 先得到各自的 max_acc
        update_acc = get_max_acc_in_xlsx(people_index=changed_people, xls_path=update_xls_path)
        final_acc = get_max_acc_in_xlsx(people_index=changed_people, xls_path=final_xls_path)
        # 如果 update_acc 更大, 则替换 final_xls_path 中的 xlsx 文件中对应人的数值
        if update_acc > final_acc:
            replace_xlsx_acc(people_index=changed_people, acc=update_acc, xls_path=final_xls_path)
        else:
            nochange.append(changed_people)
    return nochange

def get_changed_people(state_dict_path):
    """
    :: 功能: 从 state_dict 文件夹得到被改的人有哪些？
    :: 输入: state_dict_path - state_dict 文件路径
    :: 输出: 被改过人(int)组成的 list
    :: 用法: changed_people_list = get_changed_people(state_dict_path)
    """
    num_list = []
    for file_name in os.listdir(state_dict_path):
        if '_params.pkl' in file_name:
            num = int(file_name.split('_params.pkl')[0])
            num_list.append(num)
    num_list = sorted(num_list)
    return num_list


if __name__ == '__main__':
    nochange_people = use_xlsx_update_xlsx(final_xls_path=FINAL_XLS_PATH, update_xls_path=UPDATE_XLS_PATH, state_dict_path=STATE_DICT_PATH)
    print(nochange_people)