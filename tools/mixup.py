# -*- coding: utf-8 -*-
"""
# @file name  : mixup.py
# @author     : YiHeng Zhao
# @date       : 2022-2-10
# @brief      : mixup 实现
"""
import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, device=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    # 通过beta分布获得lambda，beta分布的参数alpha == beta，因此都是alpha
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # 获取需要混叠的图片的标号
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # mixup
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    path_1 = r"F:\cv_paper\lesson\Data\train\cat.4093.jpg"
    path_2 = r"F:\cv_paper\lesson\Data\train\dog.10770.jpg"

    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    img_1 = cv2.resize(img_1, (224, 224))
    img_2 = cv2.resize(img_2, (224, 224))

    alpha = 1.
    figsize = 15
    plt.figure(figsize=(int(figsize), int(figsize)))
    for i in range(1, 10):
        # lam = i * 0.1
        lam = np.random.beta(alpha, alpha)
        im_mixup = (img_1 * lam + img_2 * (1 - lam)).astype(np.uint8)
        im_mixup = cv2.cvtColor(im_mixup, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, i)
        plt.title("lambda_{:.2f}".format(lam))
        plt.imshow(im_mixup)
    plt.show()








