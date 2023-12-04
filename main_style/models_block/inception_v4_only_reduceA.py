import torch, os
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from torchsummaryX import summary
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

class BN_Conv2d(nn.Module):
    """
    基礎conv block
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class Stem_v4_Res2(nn.Module):
    """
    原先一開始的卷積,在19*19因尺度問題直接不使用 stem block(Inception-v4 and Inception-RestNet-v2)
    """

    def __init__(self,in_channels):
        super(Stem_v4_Res2, self).__init__()
        self.step1 = nn.Sequential(
            BN_Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            BN_Conv2d(32, 32, 3, 1, 0, bias=False),
            BN_Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.step2_pool = nn.MaxPool2d(3, 2, 0)
        self.step2_conv = BN_Conv2d(64, 96, 3, 2, 0, bias=False)
        self.step3_1 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step3_2 = nn.Sequential(
            BN_Conv2d(160, 64, 1, 1, 0, bias=False),
            BN_Conv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.step4_pool = nn.MaxPool2d(3, 2, 0)
        self.step4_conv = BN_Conv2d(192, 192, 3, 2, 0, bias=False)

    def forward(self, x):
        out = self.step1(x)
        tmp1 = self.step2_pool(out)
        tmp2 = self.step2_conv(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step3_1(out)
        tmp2 = self.step3_2(out)
        out = torch.cat((tmp1, tmp2), 1)
        tmp1 = self.step4_pool(out)
        tmp2 = self.step4_conv(out)
        print(tmp1.shape)
        print(tmp2.shape)
        out = torch.cat((tmp1, tmp2), 1)
        return out

class Stem_Res1(nn.Module):
    """
    原先一開始的卷積,在19*19因尺度問題直接不使用 stem block (Inception-ResNet-v1)
    """

    def __init__(self,in_channels):
        super(Stem_Res1, self).__init__()
        self.stem = nn.Sequential(
            BN_Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            BN_Conv2d(32, 32, 3, 1, 0, bias=False),
            BN_Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.MaxPool2d(3, 2, 0),
            BN_Conv2d(64, 80, 1, 1, 0, bias=False),
            BN_Conv2d(80, 192, 3, 1, 0, bias=False),
            BN_Conv2d(192, 256, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        return self.stem(x)


class Inception_A(nn.Module):
    """
    Inception-A 架構 (Inception-v4)
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)

class Inception_B(nn.Module):
    """
    Inception-B 架構 (Inception-v4)
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x7, b3_n7x1, b4_n1, b4_n1x7_1,
                 b4_n7x1_1, b4_n1x7_2, b4_n7x1_2):
        super(Inception_B, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x7_1, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_1, b4_n7x1_1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b4_n7x1_1, b4_n1x7_2, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b4_n1x7_2, b4_n7x1_2, (7, 1), (1, 1), (3, 0), bias=False)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)

class Inception_C(nn.Module):
    """
    Inception-C 架構 (Inception-v4)
    """

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n1x3_3x1, b4_n1,
                 b4_n1x3, b4_n3x1, b4_n1x3_3x1):
        super(Inception_C, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        )
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3_1 = BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False)
        self.branch3_1x3 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_3x1 = BN_Conv2d(b3_n1, b3_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b4_n1x3, b4_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.branch4_1x3 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_3x1 = BN_Conv2d(b4_n3x1, b4_n1x3_3x1, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        tmp = self.branch3_1(x)
        out3_1 = self.branch3_1x3(tmp)
        out3_2 = self.branch3_3x1(tmp)
        tmp = self.branch4_1(x)
        out4_1 = self.branch4_1x3(tmp)
        out4_2 = self.branch4_3x1(tmp)
        return torch.cat((out1, out2, out3_1, out3_2, out4_1, out4_2), 1)

class Reduction_A(nn.Module):
    """
    Reduction-A架構 (Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2)
    """

    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch2 = BN_Conv2d(in_channels, n, 3, 2, 0, bias=False)
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, k, 1, 1, 0, bias=False),
            BN_Conv2d(k, l, 3, 1, 1, bias=False),
            BN_Conv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)

class Reduction_B_v4(nn.Module):
    """
    Reduction-B架構 (Inception-v4)
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n1x7, b3_n7x1, b3_n3):
        super(Reduction_B_v4, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b3_n1x7, b3_n7x1, (7, 1), (1, 1), (3, 0), bias=False),
            BN_Conv2d(b3_n7x1, b3_n3, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat((out1, out2, out3), 1)

class Reduction_B_Res(nn.Module):
    """
    Reduction-B 架構 for Inception-ResNet-v1 \Inception-ResNet-v1  net
    """

    def __init__(self, in_channels, b2_n1, b2_n3, b3_n1, b3_n3, b4_n1, b4_n3_1, b4_n3_2):
        super(Reduction_B_Res, self).__init__()
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 2, 0, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3, 3, 2, 0, bias=False)
        )
        self.branch4 = nn.Sequential(
            BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b4_n1, b4_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b4_n3_1, b4_n3_2, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        out1 = F.max_pool2d(x, 3, 2, 0)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)

class Inception_A_res(nn.Module):
    """
    Inception-A 架構 for (Inception-ResNet-v1 and Inception-ResNet-v2)
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n3, b3_n1, b3_n3_1, b3_n3_2, n1_linear):
        super(Inception_A_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n3, 3, 1, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b3_n1, b3_n3_1, 3, 1, 1, bias=False),
            BN_Conv2d(b3_n3_1, b3_n3_2, 3, 1, 1, bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3 + b3_n3_2, n1_linear, 1, 1, 0, bias=True)

        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat((out1, out2, out3), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

class Inception_B_res(nn.Module):
    """
    Inception-A 架構 (Inception-ResNet-v1 and Inception-ResNet-v2)
    """

    def __init__(self, in_channels, b1, b2_n1, b2_n1x7, b2_n7x1, n1_linear):
        super(Inception_B_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x7, (1, 7), (1, 1), (0, 3), bias=False),
            BN_Conv2d(b2_n1x7, b2_n7x1, (7, 1), (1, 1), (3, 0), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n7x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

class Inception_C_res(nn.Module):
    """
    Inception-C 架構 (Inception-ResNet-v1 and Inception-ResNet-v2)
    """


    def __init__(self, in_channels, b1, b2_n1, b2_n1x3, b2_n3x1, n1_linear):
        super(Inception_C_res, self).__init__()
        self.branch1 = BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False)
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_n1, 1, 1, 0, bias=False),
            BN_Conv2d(b2_n1, b2_n1x3, (1, 3), (1, 1), (0, 1), bias=False),
            BN_Conv2d(b2_n1x3, b2_n3x1, (3, 1), (1, 1), (1, 0), bias=False)
        )
        self.conv_linear = nn.Conv2d(b1 + b2_n3x1, n1_linear, 1, 1, 0, bias=False)
        self.short_cut = nn.Sequential()
        if in_channels != n1_linear:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, n1_linear, 1, 1, 0, bias=False),
                nn.BatchNorm2d(n1_linear)
            )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat((out1, out2), 1)
        out = self.conv_linear(out)
        out += self.short_cut(x)
        return F.relu(out)

class Inception_Go(nn.Module):
    """
    Go style(Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2)
    """

    def __init__(self, version, in_channels ,num_classes, is_se=False):
        super(Inception_Go, self).__init__()
        self.version = version
        self.stem = Stem_Res1(in_channels) if self.version == "res1" else Stem_v4_Res2(in_channels)
        self.inception_A = self.__make_inception_A(in_channels)
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        # self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.version == "v4":
            self.fc = nn.Sequential(
                nn.Linear(1536, 768),
                nn.Dropout(0.25),
                nn.Linear(768, num_classes)
            )
        elif self.version == "res1":
            self.fc = nn.Sequential(
                nn.Linear(1792, 892),
                nn.Dropout(0.25),
                nn.Linear(892, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2144, 1072),
                nn.Dropout(0.25),
                nn.Linear(1072, num_classes)
            )

    def __make_inception_A(self,in_channels):
        layers = []
        if self.version == "v4":
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_A(in_channels, 96, 96, 64, 96, 64, 96))
                else:
                    layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 32, 32, 256))
                else:
                    layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 48, 64, 384))
                else:
                    layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384)  # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384)  # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384)  # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(4):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))  # 1024
        elif self.version == "res1":
            for _ in range(5):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(5):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_C(1024, 256, 256, 384, 256, 384, 448, 512, 256))
                else:
                    layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_C_res(896, 192, 192, 192, 192, 1792))
                else:
                    layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_C_res(1152, 192, 192, 224, 256, 2144))
                else:
                    layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.stem(x)
        out = self.inception_A(x)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.inception_C(out)
        out = self.avg_pool(out)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Inception_Go_V2(nn.Module):
    """
    Go style(Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2)
    """

    def __init__(self, version, in_channels ,num_classes, is_se=False):
        super(Inception_Go_V2, self).__init__()
        self.version = version
        self.stem = Stem_Res1(in_channels) if self.version == "res1" else Stem_v4_Res2(in_channels)
        self.inception_A = self.__make_inception_A(in_channels)
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        # self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.version == "v4":
            self.fc = nn.Sequential(
                nn.Linear(1536, 768),
                nn.Dropout(0.25),
                nn.Linear(768, num_classes)
            )
        elif self.version == "res1":
            self.fc = nn.Sequential(
                nn.Linear(1792, 892),
                nn.Dropout(0.25),
                nn.Linear(892, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2144, 1072),
                nn.Dropout(0.25),
                nn.Linear(1072, num_classes)
            )

    def __make_inception_A(self,in_channels):
        layers = []
        if self.version == "v4":
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_A(in_channels, 96, 96, 64, 96, 64, 96))
                else:
                    layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 32, 32, 256))
                else:
                    layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 48, 64, 384))
                else:
                    layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384)  # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384)  # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384)  # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(2):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))  # 1024
        elif self.version == "res1":
            for _ in range(3):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(3):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C(1024, 256, 256, 384, 256, 384, 448, 512, 256))
                else:
                    layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_C_res(896, 192, 192, 192, 192, 1792))
                else:
                    layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(2):
                if _ == 0:
                    layers.append(Inception_C_res(1152, 192, 192, 224, 256, 2144))
                else:
                    layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.stem(x)
        out = self.inception_A(x)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.inception_C(out)
        out = self.avg_pool(out)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Inception_Go_V3(nn.Module):
    """
    Go style(Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2)
    """

    def __init__(self, version, in_channels ,num_classes, is_se=False):
        super(Inception_Go_V3, self).__init__()
        self.version = version
        self.stem = Stem_Res1(in_channels) if self.version == "res1" else Stem_v4_Res2(in_channels)
        self.inception_A = self.__make_inception_A(in_channels)
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        # self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.version == "v4":
            self.fc = nn.Sequential(
                nn.Linear(1536, 768),
                nn.Dropout(0.25),
                nn.Linear(768, num_classes)
            )
        elif self.version == "res1":
            self.fc = nn.Sequential(
                nn.Linear(1792, 892),
                nn.Dropout(0.25),
                nn.Linear(892, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2144, 1072),
                nn.Dropout(0.25),
                nn.Linear(1072, num_classes)
            )

    def __make_inception_A(self,in_channels):
        layers = []
        if self.version == "v4":
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_A(in_channels, 96, 96, 64, 96, 64, 96))
                else:
                    layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 32, 32, 256))
                else:
                    layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(3):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 48, 64, 384))
                else:
                    layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384)  # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384)  # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384)  # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))  # 1024
        elif self.version == "res1":
            for _ in range(1):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(1):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C(1024, 256, 256, 384, 256, 384, 448, 512, 256))
                else:
                    layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C_res(896, 192, 192, 192, 192, 1792))
                else:
                    layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C_res(1152, 192, 192, 224, 256, 2144))
                else:
                    layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.stem(x)
        out = self.inception_A(x)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.inception_C(out)
        out = self.avg_pool(out)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Inception_Go_V4(nn.Module):
    """
    Go style(Inception-v4, Inception-ResNet-v1, Inception-ResNet-v2)
    """

    def __init__(self, version, in_channels ,num_classes, is_se=False):
        super(Inception_Go_V4, self).__init__()
        self.version = version
        self.stem = Stem_Res1(in_channels) if self.version == "res1" else Stem_v4_Res2(in_channels)
        self.inception_A = self.__make_inception_A(in_channels)
        self.Reduction_A = self.__make_reduction_A()
        self.inception_B = self.__make_inception_B()
        # self.Reduction_B = self.__make_reduction_B()
        self.inception_C = self.__make_inception_C()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.version == "v4":
            self.fc = nn.Sequential(
                nn.Linear(1536, 768),
                nn.Dropout(0.25),
                nn.Linear(768, num_classes)
            )
        elif self.version == "res1":
            self.fc = nn.Sequential(
                nn.Linear(1792, 892),
                nn.Dropout(0.25),
                nn.Linear(892, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2144, 1072),
                nn.Dropout(0.25),
                nn.Linear(1072, num_classes)
            )

    def __make_inception_A(self,in_channels):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_A(in_channels, 96, 96, 64, 96, 64, 96))
                else:
                    layers.append(Inception_A(384, 96, 96, 64, 96, 64, 96))
        elif self.version == "res1":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 32, 32, 256))
                else:
                    layers.append(Inception_A_res(256, 32, 32, 32, 32, 32, 32, 256))
        else:
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_A_res(in_channels, 32, 32, 32, 32, 48, 64, 384))
                else:
                    layers.append(Inception_A_res(384, 32, 32, 32, 32, 48, 64, 384))
        return nn.Sequential(*layers)

    def __make_reduction_A(self):
        if self.version == "v4":
            return Reduction_A(384, 192, 224, 256, 384)  # 1024
        elif self.version == "res1":
            return Reduction_A(256, 192, 192, 256, 384)  # 896
        else:
            return Reduction_A(384, 256, 256, 384, 384)  # 1152

    def __make_inception_B(self):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                layers.append(Inception_B(1024, 128, 384, 192, 224, 256,
                                          192, 192, 224, 224, 256))  # 1024
        elif self.version == "res1":
            for _ in range(1):
                layers.append(Inception_B_res(896, 128, 128, 128, 128, 896))  # 896
        else:
            for _ in range(1):
                layers.append(Inception_B_res(1152, 192, 128, 160, 192, 1152))  # 1152
        return nn.Sequential(*layers)

    def __make_inception_C(self):
        layers = []
        if self.version == "v4":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C(1024, 256, 256, 384, 256, 384, 448, 512, 256))
                else:
                    layers.append(Inception_C(1536, 256, 256, 384, 256, 384, 448, 512, 256))
        elif self.version == "res1":
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C_res(896, 192, 192, 192, 192, 1792))
                else:
                    layers.append(Inception_C_res(1792, 192, 192, 192, 192, 1792))
        else:
            for _ in range(1):
                if _ == 0:
                    layers.append(Inception_C_res(1152, 192, 192, 224, 256, 2144))
                else:
                    layers.append(Inception_C_res(2144, 192, 192, 224, 256, 2144))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.stem(x)
        out = self.inception_A(x)
        out = self.Reduction_A(out)
        out = self.inception_B(out)
        out = self.inception_C(out)
        out = self.avg_pool(out)
        out = F.dropout(out, 0.2, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def inception_v4_GO(in_channels, num_classes=361):
    return Inception_Go("v4",in_channels, num_classes)

def inception_resnet_v1_GO(in_channels, num_classes=361):
    return Inception_Go("res1",in_channels, num_classes)

def inception_resnet_v2_GO(in_channels, num_classes=361):
    return Inception_Go("res2",in_channels, num_classes)

def inception_v4_GO_V2(in_channels, num_classes=361):
    return Inception_Go_V2("v4",in_channels, num_classes)

def inception_resnet_v1_GO_V2(in_channels, num_classes=361):
    return Inception_Go_V2("res1",in_channels, num_classes)

def inception_resnet_v2_GO_V2(in_channels, num_classes=361):
    return Inception_Go_V2("res2",in_channels, num_classes)

def inception_v4_GO_V3(in_channels, num_classes=361):
    return Inception_Go_V3("v4",in_channels, num_classes)

def inception_resnet_v1_GO_V3(in_channels, num_classes=361):
    return Inception_Go_V3("res1",in_channels, num_classes)

def inception_resnet_v2_GO_V3(in_channels, num_classes=361):
    return Inception_Go_V3("res2",in_channels, num_classes)

def inception_v4_GO_V4(in_channels, num_classes=361):
    return Inception_Go_V4("v4",in_channels, num_classes)

def inception_resnet_v1_GO_V4(in_channels, num_classes=361):
    return Inception_Go_V4("res1",in_channels, num_classes)

def inception_resnet_v2_GO_V4(in_channels, num_classes=361):
    return Inception_Go_V4("res2",in_channels, num_classes)

if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 361
    CHANNELS = 4
    CHANNELS_1 = 4
    CHANNELS_2 = 8
    CROP_SIZE = 17

    model = inception_resnet_v2_GO_V2(CHANNELS_1, NUM_CLASSES)
    model = model.to(DEVICE)
    summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE))
    