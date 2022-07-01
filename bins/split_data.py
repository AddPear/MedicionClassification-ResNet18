# -*- coding: utf-8 -*-
"""
# @file name  : split_data.py
# @author     :
# @date       : 2022-03-23
# @brief      : 中药数据随机划分为训练集和测试集
"""

import os
import random
import shutil

def split_data(root_dir,ratio):
    data = os.path.join(root_dir, "dai_huafen_data")
    n_imgs = os.listdir(data)
    names = tuple([i for i in n_imgs])
    for i in names:
        # 随机打乱数据
        data_cls = data + "\\"+ i  # 获取到每一类的文件夹
        meiyilei_cls = os.listdir(data_cls)
        meiyilei_names = []
        for j in meiyilei_cls:
            meiyilei_names.append(os.path.join(data_cls, j))
        random.shuffle(meiyilei_names)  # 得到每一类别随机打乱的数据

        # 开始划分数据:一个点分两段
        train_breakpoints = int(len(meiyilei_names) * ratio)
        meiyilei_names_train = meiyilei_names[:train_breakpoints]
        meiyilei_names_val = meiyilei_names[train_breakpoints:]

        # # 训练集移动数据
        # train_path = os.path.join(root_dir, "train")
        # if not os.path.exists(os.path.join(train_path,i)):
        #     os.mkdir(os.path.join(train_path,i))
        #
        # for j in meiyilei_names_train:
        #     # 处理new_name
        #     z = j
        #     img_path_name = os.path.split(j)
        #     img_name = img_path_name[1]
        #     new_name = os.path.join(train_path,i,img_name)
        #     shutil.copyfile(z, new_name)


        # 验证集移动数据
        val_path = os.path.join(root_dir, "val")
        if not os.path.exists(os.path.join(val_path, i)):
            os.mkdir(os.path.join(val_path, i))

        for j in meiyilei_names_val:
            # 处理new_name
            z = j
            img_path_name = os.path.split(j)
            img_name = img_path_name[1]
            new_name = os.path.join(val_path, i, img_name)
            shutil.copyfile(z, new_name)



if __name__ == "__main__":
    data = r'D:\jinhua\xiangmu\zhongyao\bins\zhongyao_train_val'
    split_data(data,ratio=0.9)