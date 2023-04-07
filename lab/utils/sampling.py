#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch

def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb = False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # dict_user是一个字典，key是索引，value是一个一维的arr，里面放着他所拥有的数据的索引
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # idxs_dict是一个字典，key就是从0-9，value是一个列表，里面存放着数据的索引
    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        # dataset.targets会返回所有的标签，一共60000个
        # 这里一通操作属实没看懂是在干什么
        label = torch.tensor(dataset.targets[i]).item()
        # 记录每个标签的数据点
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    # shard_per_user就是每个客户端能分配到几个label，就是病态的数据分布嘛
    # shard_per_class每个类要分成几个shard
    shard_per_class = int(shard_per_user * num_users / num_classes)
    # 这个是每个客户端能拿到多少数据
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        # 这个变量是看有没有会被剩下的，一般是没有
        # 这个标签的数据够不够预设的数量
        num_leftover = len(x) % shard_per_class
        # 把剩下的标签记录一下，看来是没有，返回一个空的
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        # 转变成array方便reshape
        x = x.reshape((shard_per_class, -1))
        # 对一个二维的array list一下的话，就会变成一个list，里面全是arr
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        # 把纯粹的索引改变成[arr,arr]的形式
        # 其实这里的x里面的是没有随机过的，都是按顺序来的
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        # rand_set_all里面有20个0，20个1
        rand_set_all = list(range(num_classes)) * shard_per_class
        # 这样处理会存在一个客户端只有一种标签的情况啊
        # 而且如果只看这里的shuffle的话，随机得根本不够额
        random.shuffle(rand_set_all)
        # 最终rand_set_all变成一个二维的array，有100行，代表着每个客户端能分配到的两个标签的类型
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            # 取出来的两个label
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            # 选取长度之内的一个数字
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    # 这个test是干嘛的，又不返回甚么
    test = []
    # dict.items()返回dict_items类型，里面包括很多元组(key,value)...
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all
