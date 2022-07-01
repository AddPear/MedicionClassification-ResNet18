#split_data.py
# 划分数据集flower_data，数据集划分到flower_datas中，训练验证比例为8：2
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取data文件夹下所有文件夹名（即需要分类的类名）
#划分数据集flower_data，数据集划分到flower_datas中
file_path = 'F:\数据\中药数据备份\第一阶段\数据备份-勿删\新旧样本混合\zhongyao-data'
new_file_path = '../data/zhongyao-data'

# 划分比例，训练集 : 验证集 = 8 : 1
split_rate = 0.1

data_class = [cla for cla in os.listdir(file_path)]

train_path = new_file_path + '/train/'
val_path = new_file_path + '/val/'
test_path = new_file_path + '/test/'

# 创建 训练集train 文件夹，并由类名在其目录下创建子目录
mkfile(new_file_path)
for cla in data_class:
    mkfile(train_path + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile(new_file_path)
for cla in data_class:
    mkfile(val_path + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile(new_file_path)
for cla in data_class:
    mkfile(test_path + cla)

# 遍历所有类别的全部图像并按比例分成训练集和验证集
for cla in data_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = val_path + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径
            images.remove(image)  # 将选中的删除

        # # 其余的图像保存在训练集train中
        # else:
        #     image_path = cla_path + image
        #     new_path = train_path + cla
        #     copy(image_path, new_path)
        # print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar

    # 划分比例，训练集 : 测试集 = 8 : 2
    split_rate = 0.2
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):
        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = test_path + cla
            copy(image_path, new_path)  # 将选中的图像复制到新路径
            # images.remove(image)

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = train_path + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar

    print()

print("processing done!")