# -*- coding: utf-8 -*-
"""
# @file name  : flower_train.py
# @author     : YiHeng Zhao
# @date       : 2022-2-10
# @brief      : 模型训练主代码
"""

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

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--bs', default=None, type=int, help='training batch size')
parser.add_argument('--max_epoch', type=int, default=None)
parser.add_argument('--data_root_dir', default=r"../data/zhongyao-data", type=str,
                    help="path to your dataset")
args = parser.parse_args()

cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.bs if args.bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

if __name__ == "__main__":

    # step0: config
    train_dir = os.path.join(args.data_root_dir, "train")
    valid_dir = os.path.join(args.data_root_dir, "val")
    check_data_dir(train_dir), check_data_dir(valid_dir)

    # 创建logger
    res_dir = os.path.join(BASE_DIR, "..", "results")
    logger, log_dir = make_logger(res_dir)

    # step1： 数据集
    # 构建MyDataset实例， 构建DataLoder
    train_data = ZYDataset(root_dir=train_dir, transform=cfg.transforms_train)
    valid_data = ZYDataset(root_dir=valid_dir, transform=cfg.transforms_valid)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
    # print("111111")
    # step2: 模型
    model = get_model(cfg, train_data.cls_num, logger)
    model.to(device)  # to device， cpu or gpu

    # step3: 损失函数、优化器
    if cfg.label_smooth:
        loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    else:
        loss_f = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # step4: 迭代训练
    # 记录训练所采用的模型、损失函数、优化器、配置参数cfg
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, loss_f, scheduler, optimizer, model))

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.max_epoch):

        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger)

        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device)

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
                    format(epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))
        scheduler.step()

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == cfg.max_epoch - 1)
        # 保存loss曲线， acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 模型保存
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

            # # 保存错误图片的路径
            # err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "error_imgs_best.pkl"
            # path_err_imgs = os.path.join(log_dir, err_ims_name)
            # error_info = {}
            # error_info["train"] = path_error_train
            # error_info["valid"] = path_error_valid
            # pickle.dump(error_info, open(path_err_imgs, 'wb'))

    logger.info("{} done, best acc: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))