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


EXPERTS = '2 Experts'

def update_max_acc(txt_path, xls_path):
    """
    :: 功能: 用 txt 文件里的大 acc 替换 xlsx 文件里的小 acc
    :: 输入: txt_path - .txt 文件路径
            xls_path - .xlsx 文件路径
    :: 输出: 是否替换
    :: 用法: update_max_acc(txt_path='../res/k=2、kernel=32、rate=0.5、epoch=100.txt',
                            xls_path='../res/result.xlsx')
    """
    result_list = extract_accs(txt_path)
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


def extract_accs(txt_path):
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
    print(f'更新第{people_index}个人的数据为：{acc * 100:.2f}%')
    df.to_excel(xls_path, engine='openpyxl', sheet_name='Sheet1')


if __name__ == '__main__':
    update_max_acc(txt_path='../res/MPED/1.txt',
                   xls_path='../res/MPED/result.xlsx')