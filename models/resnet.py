'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''


import torch
import torch.nn as nn  # 用于卷积等搭建模块
import torch.nn.functional as F  # 激活函数
import torch.nn.init as init  # 权重初始化


# 写入这么模块可以往外接的接口
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
"""
__all__ 是针对模块公开接口的一种约定，以提供了”白名单“的形式暴露接口
使用from xxx import *导入该文件时，只会导入 __all__ 列出的成员
"""


class ResNet(nn.Module):
    '''
     构建残差类
     这个构建主要是根据你的网络结构建图来的，像这种backdone的构建一定要进行模块化，然后传入的是你不同模块的数量，
     以及你最后的分类数
    '''
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__() # 对nn.model的初始化

        self.in_planes = 16
        # 模块化之前的升维
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 开始第一个模块
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 原版16
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 原版32
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 原版64
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)  # 将随机初始化的参数分别对应的放入每一层：apply：pytorch自带的

    # block其实相当于构建了一个前向传播格式
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # 这里其实构建完同一个模块的构建次数
        layers = []
        for stride in strides:
            # 这里是基础类的实现：block(self.in_planes, planes, stride)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print('0',out.shape)
        out = self.layer1(out)
        # print('1', out.shape)
        out = self.layer2(out)
        # print('2', out.shape)
        out = self.layer3(out)
        # print('3', out.shape)
        out = F.avg_pool2d(out, out.size()[3])
        # print('4', out.shape)
        out = out.view(out.size(0), -1)
        # print('5', out.shape)
        out = self.linear(out)
        # print('6', out.shape)
        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def _weights_init(m):
    classname = m.__class__.__name__
    # print('classname',classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
# 写主函数主要是为了模型的测试
if __name__ == "__main__":
    fake_img = torch.randn((1, 3, 32, 32))  # 输入数据的张量
    # 遍历这个模块提供的所有接口
    for net_name in __all__:
        # 依旧是给一个判断
        if net_name.startswith('resnet'):
            model = globals()[net_name]()  # globals()以字典形式返回全局变量
            output = model(fake_img)
            # print(net_name, output.shape)