# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @author     : YiHeng Zhao
# @date       : 2022-2-10
# @brief      : flower_102datasets制作（数据集读取）
"""

import os
from PIL import Image  # 处理图片
from torch.utils.data import Dataset

# 括号里面传入参数Dataset说明我们写的这个类是继承了Dataset
class ZYDataset(Dataset):
    # 需要找到目录，需要数据预处理
    cls_num = 12
    names_imgs = os.listdir(r'../data\zhongyao-data\train')
    names = tuple([i for i in names_imgs])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []
        self.label_array = None  # 这个属性用来存放标签

        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        # 随机读，index的生成是有讲究的
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img
        # return img

    def __len__(self):
        '''
        让FlowerDataset这个类变成可迭代
        这个函数里面有return就够了，但是为了检查到时候出问题是不是数据没读取进来，这里抛出了一个异常
        :return:
        '''
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir)
            )

        return len(self.img_info)

    def _get_img_info(self):
        '''
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        得到数据及其对应的标签（而且是一一对应的）
        [(path, label)...]:所有数据放入一个列表，列表中的元素全为元组，每一个元组由对应的数据路径和其对应的标签组成
        :return:
        '''
        # 读取数据路径:将所有数据的路径都放入这个列表中
        name_imgs = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                    name_imgs.append(os.path.join(root, file))
        # 标签读取:并进行了标签转数字
        n_imgs = os.listdir(self.root_dir)
        names = tuple([i for i in n_imgs])
        # names = [str(i) for i in n_imgs]
        label_imgs = []
        for i in name_imgs:
            label_img_shang_path = os.path.split(i)
            label_img = os.path.split(label_img_shang_path[0])
            if label_img[1] == names[0]:
                label_imgs.append(0)
            elif label_img[1] == names[1]:
                label_imgs.append(1)
            elif label_img[1] == names[2]:
                label_imgs.append(2)
            elif label_img[1] == names[3]:
                label_imgs.append(3)
            elif label_img[1] == names[4]:
                label_imgs.append(4)
            elif label_img[1] == names[5]:
                label_imgs.append(5)
            elif label_img[1] == names[6]:
                label_imgs.append(6)
            elif label_img[1] == names[7]:
                label_imgs.append(7)
            elif label_img[1] == names[8]:
                label_imgs.append(8)
            elif label_img[1] == names[9]:
                label_imgs.append(9)
            elif label_img[1] == names[10]:
                label_imgs.append(10)
            elif label_img[1] == names[11]:
                label_imgs.append(11)
        # 匹配label
        self.img_info = [(p, idx) for p, idx in zip(name_imgs, label_imgs)]


# 测试一下，写完一个模块一定要测试一下这个模块能不能使用，能使用了才算完成，不要全写完可再测试
if __name__ == "__main__":
    root_dir = r'../data\zhongyao\train'

    test_dateset = ZYDataset(root_dir)
    # print(len(test_dateset))  # dataset最终是把你的数据都制作好，都返回出来
    # print(type(test_dateset))  # <class '__main__.FlowerDataset'>
    # print(next(iter(test_dateset)))  # iter:迭代器（将这个对象变成可迭代），next这个迭代器的第一个元素

    print(test_dateset[0])

    # root_dir = r'D:\jinhua\xiangmu\zhongyao\data\zhongyao\val'
    #
    # test_dateset = ZYDataset(root_dir)
    # print(len(test_dateset))  # dataset最终是把你的数据都制作好，都返回出来
    # print(type(test_dateset))  # <class '__main__.FlowerDataset'>
    # print(next(iter(test_dateset)))  # iter:迭代器（将这个对象变成可迭代），next这个迭代器的第一个元素


