# -*- coding: utf-8 -*-
"""
# @file name  : evaluate.py
# @author     : https://github.com/TingsongYu
# @date       : 2021-03-07
# @brief      : 模型在test上进行指标计算
"""
import os

import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt

from datasets.zhongyao import ZYDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tools.common_tools import setup_seed, show_confMat, plot_line, Logger, check_data_dir

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import pickle
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tools.model_trainer import ModelTrainer
from tools.common_tools import setup_seed, show_confMat, plot_line, Logger, check_data_dir
from tools.common_tools import *
from config.zhongyao_config import cfg
from datetime import datetime
from datasets.zhongyao import ZYDataset
from tools.my_loss import LabelSmoothLoss

setup_seed(12345)  # 先固定随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_comfMat(confusion_mat, classes, out_dir, perc=False):
    per_sum = confusion_mat.sum(axis=1)  # 计算每行的和，用于百分比计算
    for i in range(len(classes)):
        confusion_mat[i] = (confusion_mat[i] / per_sum[i])  # 百分比

    # 设置中文负号问题
    plt.rcParams['axes.unicode_minus'] = False
    # 步骤一（替换sans-serif字体）
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(16, 14))

    # cmap = "Greens"
    cmap = 'OrRd'
    plt.imshow(confusion_mat, cmap=cmap)  # 仅画出颜色格子，没有值
    plt.title("Normalized confusion matrix", fontsize=30)  # title
    plt.xlabel("Predict label", fontsize=25)
    plt.ylabel("Truth label", fontsize=25)

    plt.yticks(range(len(classes)), classes, fontsize=17)  # y轴标签
    plt.xticks(range(len(classes)), classes, rotation=45, fontsize=17)  # x轴标签
    # 在图中标注数量/概率信息
    thresh = confusion_mat.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(len(classes)):
        for y in range(len(classes)):
            info = int(confusion_mat[y, x])
            value = float(format('%.2f' % confusion_mat[y, x]))  # 数值处理
            plt.text(x, y, value, fontsize=15, verticalalignment='center', horizontalalignment='center',
                     color="white" if info > thresh else "black")  # 写值

    # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    plt.colorbar()  # 色条
    plt.savefig(out_dir + '/ConfusionMatrix.png', bbox_inches='tight', dpi=300)  # bbox_inches='tight'可确保标签信息显示全
    # plt.show()


if __name__ == '__main__':
    # config
    test_dir = r'../data\zhongyao-test'
    model_path = r'../zhongyao_best.pkl'
    confMat_path = ''

    # 数据预处理
    norm_mean = [0.4914, 0.4822, 0.4465]  # cifar10 from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    norm_std = [0.2023, 0.1994, 0.2010]
    normTransform = transforms.Normalize(norm_mean, norm_std)
    transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform,
    ])
    valid_bs = 24
    workers = 0

    # 创建logger
    res_dir = os.path.join(BASE_DIR, "..", "predict")
    logger, log_dir = make_logger(res_dir)

    # step1: dataset
    test_data = ZYDataset(root_dir=test_dir, transform=transforms_valid)
    test_loader = DataLoader(dataset=test_data, batch_size=valid_bs, num_workers=workers)

    # step2: model
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.cls_num)  # 102
    # load pretrain model
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # step3: inference
    class_num = test_loader.dataset.cls_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        inputs, labels, path_imgs = data
        # inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        # 统计混淆矩阵
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.

    # 打印测试精度
    acc_avg = conf_mat.trace() / conf_mat.sum()
    print("test acc: {:.2%}".format(acc_avg))

    # # 保存混淆矩阵图
    save_comfMat(conf_mat, test_data.names, log_dir, confMat_path)
