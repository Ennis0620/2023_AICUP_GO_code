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

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class InceptionA(nn.Module):
    '''
    64 + 64 + 96(=224) + pool_features
    '''
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):
    '''
    384 + 96(=480) + in_channels
    '''
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    '''
    192 + 192 + 192 + 192 (=768)
    '''
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):
    '''
    320+192(=512)+in_channels
    '''
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)  

class InceptionE(nn.Module):
    '''
    320+384+384+384+192(=2048)
    '''
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionChessClassifier(nn.Module):
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        x = self.out(x)
        return x

class InceptionChessClassifier_with_BWratio(nn.Module):
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier_with_BWratio, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.out = nn.Sequential(
            nn.Linear(2048+1, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x,ratio):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        x = torch.cat((x, ratio),dim=1)
        x = self.out(x)
        return x


class InceptionChessClassifier_2path(nn.Module):
    '''
    seq:conv3d
    '''
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier_2path, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels[0], 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )
        self.down_x2 = nn.Sequential(
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            BasicConv2d(192, 256, kernel_size=3, padding=1),
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.out = nn.Sequential(
            nn.Linear(2048+256, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x, x2):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = x2.permute(0,2,1,3,4)
        x2 = x2.view(-1, x2.size(2),x2.size(3),x2.size(4))
        x2 = self.down_x2(x2)
        x2 = self.GAP(x2)
        x2 = x2.view(x2.size(0), -1)#flatten

        x = torch.cat((x,x2),dim=1)
        x = self.out(x)

        return x

class InceptionChessClassifier_2path_V2(nn.Module):
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier_2path_V2, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels[0], 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )
        self.down_x2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=5),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,361,kernel_size=3),
            nn.InstanceNorm2d(361),
            nn.ReLU(inplace=True),
        )
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.out = nn.Sequential(
            nn.Linear(2048+361, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x, x2):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = x2.permute(0,2,1,3,4)
        x2 = x2.view(-1, x2.size(2),x2.size(3),x2.size(4))
        x2 = self.down_x2(x2)
        x2 = self.GAP(x2)
        x2 = x2.view(x.size(0), -1)#flatten

        x = torch.cat((x,x2),dim=1)
        x = self.out(x)

        return x

class InceptionChessClassifier_2path_V3(nn.Module):
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier_2path_V3, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels[0], 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)

        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1),bias=False),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.seq_MLP = nn.Sequential(
            nn.Linear(64,32,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,64,bias=False),
            nn.Sigmoid()
        )
        

        self.GAP_weight = SELayer('avg_pool',64,reduction=2)
        self.GMP_weight = SELayer('max_pool',64,reduction=2)

        self.out = nn.Sequential(
            nn.Linear(2048 + 64*2 + 64*2, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x, x2):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten

        #sequence處理
        x2_split = torch.split(x2, 8, dim=2)

        x2 = self.sequence(x2_split[0])#input:(B,1,8,19,19)=>output:(B,channel,1,19,19)
        x2 = x2.permute(0,2,1,3,4)
        x2 = x2.view(-1, x2.size(2),x2.size(3),x2.size(4))
        x2_GAP = self.GAP(self.GAP_weight(x2))
        x2_GMP = self.GMP(self.GMP_weight(x2))
        x2 = torch.cat((x2_GMP,x2_GAP),dim=1)
        x2 = x2.view(x2.size(0), -1)#flatten

        x3 = self.sequence(x2_split[1])#input:(B,1,8,19,19)=>output:(B,channel,1,19,19)
        x3 = x3.permute(0,2,1,3,4)
        x3 = x3.view(-1, x3.size(2),x3.size(3),x3.size(4))
        x3_GAP = self.GAP(self.GAP_weight(x3))
        x3_GMP = self.GMP(self.GMP_weight(x3))
        x3 = torch.cat((x3_GMP,x3_GAP),dim=1)
        x3 = x3.view(x3.size(0), -1)#flatten
        
        x = torch.cat((x,x2,x3),dim=1)
        x = self.out(x)

        return x

class InceptionChessClassifier_V3(nn.Module):
    def __init__(self,n_channels,num_classes):
        super(InceptionChessClassifier_V3, self).__init__()
        
        self.down = nn.Sequential(
            BasicConv2d(n_channels, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionB(256),
            InceptionC(736, channels_7x7=128),
            InceptionD(768),
            InceptionE(1280),
        )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1) 


        self.out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self,x):
        x = self.down(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        x = self.out(x)

        return x


#########################################################################

class SELayer(nn.Module):
    '''
        SE Block做chaeenl attention\n
        `mode`:有 'avg_pool'、'max_pool' 2種\n
        `channel`: 一開始進入MLP的通道數量\n
        `reduction`:MLP中間的過度filter size ratio, 預設16\n
        '''
    def __init__(self, mode, channel, reduction=16):
        super(SELayer, self).__init__()
        self.mode = mode
        #兩種初始化pooling方式
        #變成1x1xC 不同Channel做pooling
        if self.mode == 'avg_pool':
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.mode == 'max_pool':
            self.global_avg_pool = nn.AdaptiveMaxPool2d(1)
        #取得每個channel的權重
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel,
                      out_features=channel//reduction,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel//reduction,
                      out_features=channel,
                      bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        B, C, H, W = x.size()
        #[B, C, H, W] => global_avg_pool:[B, C, 1, 1] => view:[B, C] 
        global_avg_pool = self.global_avg_pool(x).view(B,C)
        #經過mlp得到每個channel的權重後要變回[B, C, 1, 1]
        mlp_channel_attention = self.fc(global_avg_pool).view(B, C, 1, 1)
        #return 原來的input 乘上 channel_attention權重
        return x * mlp_channel_attention.expand_as(x) 

class ResidualBlock(nn.Module): 
  '''
  `Resnet基本架構`
  ResidualBlock 由2組Conv組成\n
    [Resnet基本架構(Conv+BN+Relu+Conv+BN) + shortcout] + Relu ,\n
    kernel都是3*3\n
  `in_channel:`傳進來的featuremap數量\n
  `out_channel:`傳出的featuremap數量\n
  `shortcut`:給過此基本block後要給予shortcut結構(conv1*1)把feature map傳入前數量變成和傳出後數量相同才能相加\n
  '''
  def __init__(self, in_channel, out_channel, stride=1, shortcut=None):         
    super(ResidualBlock, self).__init__() 
    self.down = nn.Sequential(   
      nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False), # bias=False是因為bias再BN中已經有了，如果stride=2則shape會變成一半 
      nn.BatchNorm2d(out_channel), 
      nn.ReLU(inplace=True), 
      nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False), # shape前後仍然相同 
      nn.BatchNorm2d(out_channel), 
    ) 
    self.shortcut = shortcut # 根據情況是否做出增維或是縮小shape 

  def forward(self, x): 
    out = self.down(x) 
    #是否要shortcut  如果是None就不會
    residual = x if self.shortcut is None else self.shortcut(x)
    out = out + residual 
    out = F.relu(out, inplace=True) 
    return out

class ChessStyleClassifier(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels, feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.channel_att = SELayer(mode = 'avg_pool',
                                   channel= feature_map[2])
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x):
        x = self.conv_x1(x)
        x = self.channel_att(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)
        return x

class ChessStyleClassifier_V2(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_V2, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels, feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(feature_map[2], 2*feature_map[2], kernel_size=3),
            nn.BatchNorm2d(2*feature_map[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*feature_map[2], feature_map[2], kernel_size=3),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x):
        x = self.conv_x1(x)
        x = self.conv_cat(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)
        return x

class ChessStyleClassifier_two_path(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_two_path, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_x2 = nn.Sequential(
            nn.Conv2d(n_channels[1], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.channel_att = SELayer(mode = 'avg_pool',
                                   channel= 2*feature_map[2])
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)

        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2*feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x, x2):

        x1 = self.conv_x1(x)
        x2 = self.conv_x2(x2)
        
        x = torch.cat((x1,x2),dim=1)
        x = self.channel_att(x)
        x = self.GAP(x)

        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)

        return x
    
class ChessStyleClassifier_two_path_V2(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_two_path_V2, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_x2 = nn.Sequential(
            nn.Conv2d(n_channels[1], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )
        
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)

        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2*feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x, x2):

        x1 = self.conv_x1(x)
        x2 = self.conv_x2(x2)
        
        x = torch.cat((x1,x2),dim=1)
        x = self.GAP(x)

        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)

        return x

class ChessStyleClassifier_two_path_V3(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_two_path_V3, self).__init__()

        feature_map = [192, 384, 576]
        cat_feature_map = [feature_map[-1]*2, feature_map[-2]*2, feature_map[-3]*2]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_x2 = nn.Sequential(
            nn.Conv2d(n_channels[1], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(cat_feature_map[0], cat_feature_map[1], kernel_size=3),
            nn.BatchNorm2d(cat_feature_map[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(cat_feature_map[1], cat_feature_map[2], kernel_size=3),
            nn.BatchNorm2d(cat_feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)

        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(cat_feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x, x2):

        x1 = self.conv_x1(x)
        x2 = self.conv_x2(x2)
        
        x = torch.cat((x1,x2),dim=1)
        x = self.conv_cat(x)
        x = self.GAP(x)

        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)

        return x

class ChessStyleClassifier_two_path_V4(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_two_path_V4, self).__init__()

        feature_map = [192, 384, 576]
        cat_feature_map = [feature_map[-1]*2, feature_map[-2]*2, feature_map[-3]*2]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_x2 = nn.Sequential(
            nn.Conv2d(n_channels[1], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(cat_feature_map[0], cat_feature_map[1], kernel_size=3),
            nn.BatchNorm2d(cat_feature_map[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(cat_feature_map[1], cat_feature_map[2], kernel_size=3),
            nn.BatchNorm2d(cat_feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.channel_att = SELayer(mode = 'avg_pool',
                                   channel= cat_feature_map[2])
        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)

        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(cat_feature_map[2] , feature_map[1])
        self.out = nn.Linear(feature_map[1] , num_classes)

    def forward(self, x, x2):

        x1 = self.conv_x1(x)
        x2 = self.conv_x2(x2)
        
        x = torch.cat((x1,x2),dim=1)
        x = self.conv_cat(x)
        x = self.channel_att(x)
        x = self.GAP(x)

        x = x.view(x.size(0), -1)#flatten
        x = self.drop(self.fc1(x))
        x = self.out(x)

        return x

class ChessStyleClassifier_2path(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_2path, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(feature_map[2], 2*feature_map[2], kernel_size=3),
            nn.BatchNorm2d(2*feature_map[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(2*feature_map[2], feature_map[2], kernel_size=3),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )
        self.down_x2 = nn.Sequential(
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            BasicConv2d(192, 256, kernel_size=3, padding=1),
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Sequential(
            nn.Linear(feature_map[2] + 256 , feature_map[1]),
            nn.Dropout(0.25),
            nn.Linear(feature_map[1] , num_classes)
        )

    def forward(self, x, x2):
        x = self.conv_x1(x)
        x = self.conv_cat(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten

        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = x2.permute(0,2,1,3,4)
        x2 = x2.view(-1, x2.size(2),x2.size(3),x2.size(4))
        x2 = self.down_x2(x2)
        x2 = self.GAP(x2)
        x2 = x2.view(x2.size(0), -1)#flatten
        
        x = torch.cat((x,x2),dim=1)
        x = self.out(x)
        return x

class ChessStyleClassifier_2path_V2(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_2path_V2, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Sequential(
            nn.Linear(feature_map[2] + 64 , feature_map[1]),
            nn.Dropout(0.25),
            nn.Linear(feature_map[1] , num_classes)
        )

    def forward(self, x, x2):
        x = self.conv_x1(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten

        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = self.GAP(x2)
        x2 = x2.view(x2.size(0), -1)#flatten
        
        x = torch.cat((x,x2),dim=1)
        x = self.out(x)
        return x

class ChessStyleClassifier_2path_V3(nn.Module):
    def __init__(self,n_channels, num_classes, CROP_SIZE):
        super(ChessStyleClassifier_2path_V3, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.sequence = nn.Sequential(
            nn.Conv3d(1,1,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Sequential(
            nn.Linear(feature_map[2] + CROP_SIZE*CROP_SIZE , feature_map[1]),
            nn.Dropout(0.25),
            nn.Linear(feature_map[1] , num_classes)
        )

    def forward(self, x, x2):
        x = self.conv_x1(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten

        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = x2.view(x2.size(0), -1)#flatten
        
        x = torch.cat((x,x2),dim=1)
        x = self.out(x)
        return x

class ChessStyleClassifier_2path_V4(nn.Module):
    def __init__(self,n_channels, num_classes):
        super(ChessStyleClassifier_2path_V4, self).__init__()

        feature_map = [192, 384, 576]

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(n_channels[0], feature_map[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(feature_map[0], feature_map[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[1]),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_map[1], feature_map[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_map[2]),
            nn.ReLU(inplace=True),
        )

        self.sequence = nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(8,1,1)),#input處理成(B,1,channel=8,19,19) conv3d後 變成 (B,channel,1,19,19)
        )
        self.sequence_SE = SELayer(mode='avg_pool',channel=64)

        self.GAP = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Sequential(
            nn.Linear(feature_map[2] + 64 , feature_map[1]),
            nn.Dropout(0.25),
            nn.Linear(feature_map[1] , num_classes)
        )

    def forward(self, x, x2):
        x = self.conv_x1(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)#flatten

        #sequence處理
        x2 = self.sequence(x2)#input:(B,1,1,8,19,19)=>output:(B,1,channel,1,19,19)
        x2 = x2.permute(0,2,1,3,4)
        x2 = x2.view(-1, x2.size(2),x2.size(3),x2.size(4))
        x2 = self.sequence_SE(x2)
        x2 = self.GAP(x2)
        x2 = x2.view(x2.size(0), -1)#flatten
        
        x = torch.cat((x,x2),dim=1)
        x = self.out(x)
        return x


if __name__ == '__main__':

    NUM_CLASSES = 3
    CHANNELS = 4
    CHANNELS_1 = 4
    CHANNELS_2 = 16
    CROP_SIZE = 19
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #two path
    # model = ChessStyleClassifier_two_path_V4(n_channels=[CHANNELS_1,CHANNELS_2], num_classes=NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))

    # model = ChessStyleClassifier_V2(CHANNELS,NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS, CROP_SIZE, CROP_SIZE).to(DEVICE))

    # model = InceptionChessClassifier(CHANNELS,NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS, CROP_SIZE, CROP_SIZE).to(DEVICE))
    
    # model = InceptionChessClassifier_2path([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))


    # model = ChessStyleClassifier_2path([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))

    # model = ChessStyleClassifier_2path_V2([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))

    # model = ChessStyleClassifier_2path_V4([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))

    # model = InceptionChessClassifier_2path_V2([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, torch.randn(1, CHANNELS, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))
    
    # model = InceptionChessClassifier_2path_V3([CHANNELS_1,CHANNELS_2],NUM_CLASSES)
    # model = model.to(DEVICE)
    # summary(model, 
    #         torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), 
    #         torch.randn(1, 1, CHANNELS_2, CROP_SIZE, CROP_SIZE).to(DEVICE))

    model = InceptionChessClassifier_with_BWratio(CHANNELS_1,NUM_CLASSES).to(DEVICE)
    summary(model, 
            torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE), torch.randn(1,1).to(DEVICE))