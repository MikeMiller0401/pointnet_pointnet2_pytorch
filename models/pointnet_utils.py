import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# PointNet网络组件

# 定义T-Net 3*3子网络，类似一个mini-pointnet，用于样本的旋转对齐
class STN3d(nn.Module):
    # 定义网络结构
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # torch.nn.Conv1d 一维卷积函数
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # nn.Linear 产生全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 9=3*3
        self.relu = nn.ReLU()
        # nn.BatchNorma1d 归一化
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # 前向传播forward，只需要在实例化一个对象中传入对应的参数就可以自动调用forward函数
    def forward(self, x):
        # 获取batch-size
        batchsize = x.size()[0]
        # 对输入的channel进行三次卷积并归一化，升到1024维
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 对x进行最大池化，并单独取出最大值
        x = torch.max(x, 2, keepdim=True)[0]
        # 使用view函数重构张量的维度为a*1024维，a为估算结果
        x = x.view(-1, 1024)
        # 令x经过三个全连接层
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden:生成3*3的对角矩阵np.array([1,0,0,0,1,0,0,0,1])->view成为1*9的二维数组并重复batch-size次数
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        # 仿射变换，一个全连接后的结果+一个batch-size的对角矩阵
        x = x + iden
        # 转换为batch-size*3*3的矩阵
        x = x.view(-1, 3, 3)
        return x


# 定义STNkd子网络，与STN3d类似，用于特征变换
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        # 传入的是k，k默认为64
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# pointnet编码器
class PointNetEncoder(nn.Module):
    # 网络定义
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        # 定义
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # 定义全局特征global_feat与特征转换feature_transform
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        # feature_transform=ture时调用STNkd
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    # 前向传播
    def forward(self, x):
        # B=batch-size；D=维度，3或者6；N=一个点云所取的点的数目，如1024
        B, D, N = x.size()
        # 调用STN3d T-Net，得到B*3*3的仿射变换矩阵trans
        trans = self.stn(x)
        # 交换x的两个维度，x变成B，N，D
        x = x.transpose(2, 1)
        # D>3说明有法向量，
        if D > 3:
            feature = x[:, :, 3:]  # 若是6个特征（有法向量）的点，则认为法向量为特征
            x = x[:, :, :3]  # xyz仍为特征点
        # 对输入的点云进行转换，计算两个tensor的矩阵乘法，bmm是两个三维张量相乘，两个输入张量大小为(b,n,m)和(b,m,p)
        # 输出(b,n,p)
        x = torch.bmm(x, trans)
        if D > 3:
            # 合并向量
            x = torch.cat([x, feature], dim=2)
        # 交换x的两个维度，x变回B，D，N
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # 如果需要特征转换则转换为64*1024个点
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # 提取局部特征
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # 如果需要全局特征则返回全局特征x
        if self.global_feat:
            return x, trans, trans_feat
        # 否则返回全局特征与局部特征的拼接
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# 对特征矩阵进行正则化
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
