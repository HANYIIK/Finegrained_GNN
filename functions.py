#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2021/3/8 17:58
# @Author   : Hanyiik
# @File     : functions.py
# @Function : 一些工具人函数
import json
import argparse
from argparse import Namespace
import h5py
import numpy as np
import random
import os


def create_npy(args):
    """
    :: 功能: 新环境中生成 .npy 文件
    :: 输入:
    :: 输出:
    :: 用法:
            args = get_config()
            create_npy(args)
    """
    train_dataset_list, train_labelset_list, test_dataset_list, test_labelset_list = read_mat(args)

    if args.dataset_name == 'MPED' or 'SEED_IV':
        for r in range(1, args.people_num+1):
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/train_dataset_{}.npy'.format(r),
                    train_dataset_list[r - 1])
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/train_labelset_{}.npy'.format(r),
                    train_labelset_list[r - 1])
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/test_dataset_{}.npy'.format(r),
                    test_dataset_list[r - 1])
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/test_labelset_{}.npy'.format(r),
                    test_labelset_list[r - 1])
            print('成功保存第{}个人的数据！'.format(r))

    elif args.dataset_name == 'SEED':
        for r in range(1, args.people_num+1):
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/train_dataset_{}.npy'.format(r),
                    train_dataset_list[r - 1])
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/train_labelset_{}.npy'.format(r),
                    train_labelset_list[r - 1]+1)
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/test_dataset_{}.npy'.format(r),
                    test_dataset_list[r - 1])
            np.save(args.npy + args.dataset_name + '/' + 'data_' + args.dataset_size + '/test_labelset_{}.npy'.format(r),
                    test_labelset_list[r - 1]+1)
            print('成功保存第{}个人的数据！'.format(r))

    else:
        raise RuntimeError("请输入正确的数据集名称: MPED/SEED/SEED_IV")

def read_mat(args):
    """
    :: 功能: 读取 .mat 文件中的 traindata、trainlabel、testdata、testlabel
    :: 输入:
    :: 输出: list 格式的 30 个人的 traindata(numpy),
            list 格式的 30 个人的 trainlabel(numpy),
            list 格式的 30 个人的 testdata(numpy),
            list 格式的 30 个人的 testlabel(numpy)
    :: 用法:
            train_dataset_list, train_labelset_list, test_dataset_list, test_labelset_list = get_data()
    """
    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    def load_one_mat_file(filename, flag=0):
        """
            :: 功能: 读取一个 .mat 的训练集或者测试集
            :: 输入: filename = 要读取的 .mat 文件名
                    LORS: 0 ---> MPED
                          1 ---> SEED
                          2 ---> SEED_IV
            :: 输出: numpy 格式的 traindata, trainlabel, testdata, testlabel
            :: 用法:
        """
        arrays = {}
        f = h5py.File(args.mat + args.dataset_name + '/' + filename, mode='r+')
        for k, v in f.items():
            arrays[k] = np.array(v)

        def resort_mped(arr):
            """
            :: 功能:
            :: 输入: arr.shape = (5, 个数, 62)
            :: 输出: arr.shape = (个数, 62, 5)
            :: 用法:
            """
            return np.array([np.transpose(arr[:, item, :]) for item in range(arr.shape[1])])

        def resort_seed(arr):
            """
            :: 功能:
            :: 输入: arr.shape = (5, 62, 个数)
            :: 输出: arr.shape = (个数, 62, 5)
            :: 用法:
            """
            return np.array([np.transpose(arr[:, :, item]) for item in range(arr.shape[2])])

        if flag == 0 or flag == 2:
            get_traindata = resort_mped(arrays['traindata']).astype(np.float32)
            get_testdata = resort_mped(arrays['testdata']).astype(np.float32)
            get_trainlabel = np.squeeze(arrays['trainlabel']).astype(np.int)
            get_testlabel = np.squeeze(arrays['testlabel']).astype(np.int)
        elif flag == 1:
            get_traindata = resort_seed(arrays['traindata']).astype(np.float32)
            get_testdata = resort_seed(arrays['testdata']).astype(np.float32)
            get_trainlabel = np.squeeze(arrays['trainlabel']).astype(np.int)
            get_testlabel = np.squeeze(arrays['testlabel']).astype(np.int)
        else:
            raise RuntimeError("请输入正确的数据集名称: MPED/SEED/SEED_IV")

        return get_traindata, get_trainlabel, get_testdata, get_testlabel


    def normalization(data):
        """
            :: 功能: 对特征进行 MaxMin 归一化处理, 默认需要归一化
            :: 输入: numpy 类型的 traindata, testdata(shape=[个数, 62, 5])
            :: 输出: 归一化后的 numpy 类型的 traindata, testdata(shape 不变)
            :: 用法:
        """
        minda = np.tile(np.min(data, axis=2).reshape((data.shape[0], data.shape[1], 1)),
                        (1, 1, data.shape[2]))
        maxda = np.tile(np.max(data, axis=2).reshape((data.shape[0], data.shape[1], 1)),
                        (1, 1, data.shape[2]))
        return (data - minda) / (maxda - minda)


    for j in range(1, args.people_num+1):
        if args.dataset_name == 'MPED':
            if args.dataset_size == 'small':
                traindata, trainlabel, testdata, testlabel = \
                    load_one_mat_file('MPED7forGNN{}.mat'.format(j), flag=0)
                if args.normalize:
                    traindata = normalization(traindata)
                    testdata = normalization(testdata)
                print('load MPED7forGNN{}.mat finished!'.format(j))

            elif args.dataset_size == 'large':
                traindata, trainlabel, testdata, testlabel = \
                    load_one_mat_file('MPED7forGNN{}_transfer_subject.mat'.format(j), flag=0)
                if args.normalize:
                    traindata = normalization(traindata)
                    testdata = normalization(testdata)
                print('load MPED7forGNN{}_transfer_subject.mat finished!'.format(j))

            else:
                raise RuntimeError("请输入正确的数据集大小: small/large")

        elif args.dataset_name == 'SEED':
            traindata, trainlabel, testdata, testlabel = \
                load_one_mat_file('SEEDforGNN{}.mat'.format(j), flag=1)
            if args.normalize:
                traindata = normalization(traindata)
                testdata = normalization(testdata)
            print('load SEEDforGNN{}.mat finished!'.format(j))

        elif args.dataset_name == 'SEED_IV':
            if args.dataset_size == 'small':
                traindata, trainlabel, testdata, testlabel = \
                    load_one_mat_file('SEED4forGNN{}.mat'.format(j), flag=2)
                if args.normalize:
                    traindata = normalization(traindata)
                    testdata = normalization(testdata)
                print('load SEED4forGNN{}.mat finished!'.format(j))

            elif args.dataset_size == 'large':
                traindata, trainlabel, testdata, testlabel = \
                    load_one_mat_file('SEED4forGNN{}_transfer_subject.mat'.format(j), flag=2)
                if args.normalize:
                    traindata = normalization(traindata)
                    testdata = normalization(testdata)
                print('load SEED4forGNN{}_transfer_subject.mat finished!'.format(j))

            else:
                raise RuntimeError("请输入正确的数据集大小: small/large")

        else:
            raise RuntimeError("请输入正确的数据集名称: MPED/SEED/SEED_IV")

        traindata_list.append(traindata)
        trainlabel_list.append(trainlabel)
        testdata_list.append(testdata)
        testlabel_list.append(testlabel)

    return traindata_list, trainlabel_list, testdata_list, testlabel_list

# 工具人函数
def load_one_people_npy(args, people=5):
    """
    :: 功能: 读【一个人】 .npy 文件中的 traindata、trainlabel、testdata、testlabel
    :: 输入: args, people: 你要 load 第几个人的数据呢?
    :: 输出: traindata(numpy),
            trainlabel(numpy),
            testdata(numpy),
            testlabel(numpy)
    :: 用法:
            args = get_config()
            train_data, train_label, test_data, test_label = load_one_people_npy(args, people=循环数，请务必注意从 1 开始)
    """
    if people == 0:
        raise RuntimeError("改代码去吧，people 这个参数要从 1 开始循环(提示：for people in range(1, args.people_num+1):)")
    else:
        if args.dataset_name == 'MPED' or args.dataset_name == 'SEED' or args.dataset_name == 'SEED_IV':
            traindata = np.load(
                args.npy + args.dataset_name + '/data_' + args.dataset_size + '/train_dataset_{}.npy'.format(people))
            trainlabel = np.load(
                args.npy + args.dataset_name + '/data_' + args.dataset_size + '/train_labelset_{}.npy'.format(people))
            testdata = np.load(
                args.npy + args.dataset_name + '/data_' + args.dataset_size + '/test_dataset_{}.npy'.format(people))
            testlabel = np.load(
                args.npy + args.dataset_name + '/data_' + args.dataset_size + '/test_labelset_{}.npy'.format(people))
        else:
            raise RuntimeError("请输入正确的数据集名称: MPED/SEED/SEED_IV")

    return traindata, trainlabel, testdata, testlabel

def test_load_npy_shape(args):
    """
    :: 功能: 测试读取的 .npy 文件的 shape 对不对，无视
    :: 输入:
    :: 输出:
    :: 用法:
            args = get_config()
            test_load_npy_shape(args)
    """
    train_data_list, train_label_list, test_data_list, test_label_list = load_npy(args)
    for i in range(10):
        x = random.randint(1, args.people_num)

        print('shuffle people:', x)
        print('traindata.shape =', train_data_list[x-1].shape)
        print('trainlabel.shape =', train_label_list[x-1].shape)
        print('trainlabel:\n', train_label_list[x - 1])
        print('testdata.shape =', test_data_list[x-1].shape)
        print('testlabel.shape =', test_label_list[x-1].shape)
        print('testlabel:\n', test_label_list[x - 1])
        print('\n')

def load_npy(args):
    """
    :: 功能: 读【所有人】 .npy 文件中的 traindata、trainlabel、testdata、testlabel(已弃用)
    :: 输入:
    :: 输出: list 格式的 30 个人的 traindata(numpy),
            list 格式的 30 个人的 trainlabel(numpy),
            list 格式的 30 个人的 testdata(numpy),
            list 格式的 30 个人的 testlabel(numpy)
    :: 用法:
            args = get_config()
            train_data_list, train_label_list, test_data_list, test_label_list = load_npy(args)
    """
    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    if args.dataset_name == 'MPED' or args.dataset_name == 'SEED' or args.dataset_name == 'SEED_IV':
        for k in range(1, args.people_num+1):
            traindata = np.load(args.npy + args.dataset_name + '/data_' + args.dataset_size + '/train_dataset_{}.npy'.format(k))
            trainlabel = np.load(args.npy + args.dataset_name + '/data_' + args.dataset_size + '/train_labelset_{}.npy'.format(k))
            testdata = np.load(args.npy + args.dataset_name + '/data_' + args.dataset_size + '/test_dataset_{}.npy'.format(k))
            testlabel = np.load(args.npy + args.dataset_name + '/data_' + args.dataset_size + '/test_labelset_{}.npy'.format(k))

            traindata_list.append(traindata)
            trainlabel_list.append(trainlabel)
            testdata_list.append(testdata)
            testlabel_list.append(testlabel)
    else:
        raise RuntimeError("请输入正确的数据集名称: MPED/SEED/SEED_IV")

    return traindata_list, trainlabel_list, testdata_list, testlabel_list

# 工具人函数
def get_config():
    """
    :: 功能: 读取配置信息
    :: 输入:
    :: 输出: 一个 Namespace, 长这样:
        Namespace(
            platform='linux'
            dataset_name='SEED'
            dataset_size='small'
            normalize=1

            mat='E:/DATASET/GNN_DATASETS/'
            npy='E:/DATASET/GNN_NPY_DATASETS/'


            batch_size=100
            max_epochs=100
            dropout=0.3
            learning_rate=0.001
            classes_num=3
            people_num=45

            model='GCN'
            H0=25
            H1=50
            H2=100
        )

        或者

        Namespace(
            platform='windows'
            dataset_name='MPED'
            dataset_size='small'
            normalize=1

            mat='E:/DATASET/GNN_DATASETS/'
            npy='E:/DATASET/GNN_NPY_DATASETS/'


            batch_size=100
            max_epochs=100
            dropout=0.3
            learning_rate=0.001
            classes_num=7
            people_num=30

            model='ChebNet'
            k=2
            filter_num=32
        )
    :: 用法:
    """
    parser = argparse.ArgumentParser()
    # 跑一个 Attention 用的 rate
    parser.add_argument('--rate', type=float, default=0.5, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # 跑两个 Attention 用的 rate_1, rate_2
    parser.add_argument('--rate_1', type=float, default=0.5, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--rate_2', type=float, default=0.4, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    arguments= parser.parse_args()
    arguments = vars(arguments)

    # 读取 Dataset 的配置
    data_cfg = json.load(open('./config.json'))['load_data']
    path_cfg = data_cfg['path'][data_cfg['platform']]
    people_cfg = data_cfg['other'][data_cfg['dataset_name']]
    arguments.update(data_cfg)
    arguments.update(path_cfg)
    arguments.update(people_cfg)

    # 读取 Model 的配置
    model_cfg = json.load(open('./config.json'))['run_para']
    gnn_cfg = model_cfg[model_cfg['model']]
    arguments.update(model_cfg)
    arguments.update(gnn_cfg)

    arguments = Namespace(**arguments)

    return arguments

# 工具人函数
def get_folders(args):
    dataset = args.dataset_name
    path_list = [f"./res/{dataset}/confusion_matrix", f"./res/{dataset}/state_dict", ]

    for path in path_list:
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)


if __name__ == '__main__':
    my_args = get_config()
    create_npy(my_args)
    test_load_npy_shape(my_args)
