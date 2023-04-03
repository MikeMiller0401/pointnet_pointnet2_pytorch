import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


# 对点云数据进行归一化处理，以centor为中心，球半径为1
def pc_normalize(pc):
    # pc维度为[n,3]
    l = pc.shape[0]
    # 对pc数组的每一列求平均值，得到[x_mean,y_mean,z_mean]，求出中心centroid，
    centroid = np.mean(pc, axis=0)
    # 求这个点集里面的点到中心点的相对坐标
    pc = pc - centroid
    # 将同一行的元素求平方再相加，再开方求最大。x^2+y^2+z^2，得到最大标准差
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # 进行归一化，这里使用的是Z-score标准化方法
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    主要用来在ball query过程中确定每一个点距离采样点的距离
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        输入两组点N和M，C为输入点的通道数目，B为batch
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        返回的是两组点之间的欧氏距离，N*M矩阵dist
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # torch.matmul也是一种矩阵相乘操作，但是它具有广播机制，可以进行维度不同的张量相乘
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # [B,N,M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # [B,N,M]+[B,N,1]dist每一列都加上后面的列值
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # [B,N,M]+[B,1,M]dist每一行都加上后面的行值
    return dist


# 按照输入的点云数据和索引返回索引的点云数据
def index_points(points, idx):
    """

    Input:
        # point为点云数据，C为通道，idx为需要索引的点的索引
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)  # view_shape=[B,S]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # 去掉第零个数，其余变为1,[B,1]

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # [1,S]
    # arrange生成[0,...,B-1],view后变为列向量[B,1],repeat后[B,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]  #从points中取出每个batch_indices对应索引的数据点
    return new_points

# 最远点采样算法
def farthest_point_sample(xyz, npoint):
    """
    最远点采样算法（FPS）的流程如下：
    （1）随机选择一个点作为初始点作为已选择采样点。
    （2）计算未选择采样点集中每个点与已选择采样点集之间的距离distance，将距离最大的那个点加入已选择采样点集。
    （3）更新distance，一直循环迭代下去，直至获得了目标数量的采样点。
    Input:
        # 输入的是点云数据[B,N,3]和需要的采样个数npoint
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        # 输出的是npoint个采样点在原始点云和中的索引
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape

    # 初始化一个中心点矩阵centroids，用于存储采样点的索引位置，大小为B * npoint
    # 返回一个全为0的张量：
    # torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)

    # distance矩阵用于记录某个batch中所有点到某个采样点的距离，初始值很大，后面会迭代
    # 返回一个全为1的张量：
    # torch.ones(*sizes, out=None) → Tensor
    distance = torch.ones(B, N).to(device) * 1e10

    # farthest表示当前最远的点，也是随机初始化，范围0-N，初始化B个
    # 返回一个填充了随机整数的张量，这些整数在low(inclusive) 和high(exclusive) 之间均匀生成。张量的shape由变量参数size定义：
    # torch.randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 初始化0-B-1的数组
    # 返回指定范围内指定步长的张量
    # torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # 迭代，直到采样点数量达到npoint
    for i in range(npoint):
        # 设置当前的centroids采样点为当前的最远点farthest
        centroids[:, i] = farthest

        # 取出centroid坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)

        # 求出所有点到该centroid的欧氏距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # 建立一个mask，如果dist中记录的距离小于distance里的，则更新distance的值，这样distance里保留的就是每个点距离所有已采样的点的最小距离
        # distance矩阵的值会随着迭代逐渐变小
        # 其相当于记录着某个batch中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]

        # 得到最大距离的下标作为下一次的选择点
        farthest = torch.max(distance, -1)[1]
    return centroids

# query_ball_point函数用于寻找球形邻域中的点
# 输入中radius为球形邻域的半径，nsample为每个邻域中要采样的点，xyz为所有的点云数据，new_xyz为centroids点的数据
# 输出为每个样本的每个球形邻域的nsample个采样点集的索引group_idx[B, S, nsample]
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    区域特征提取器
    （1）预设搜索区域半径R和子区域的点数K；
    （2）将使用FPS提取出来的N个点作为球心（Centriods）画半径为R的球体（即Ball query区域）。
    （3）按照距离从小到大排序在每个Ball query区域内搜索离球心最近的K个点，如果某个点的Ball query区域的点数量大于规模K，那么就以这K个点作为子区域，反之则对该点重采样，凑够规模K。
    （4）获取所有点的子区域，每个子区域有K点。
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # 初始化group_idx
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # 调用square_distance函数，返回的是两组点的欧氏距离，用sqrdists记录
    sqrdists = square_distance(new_xyz, xyz)

    # 将所有距离大于radius^2（半径的平方）的点的group_idx置为N，其余的保留原值
    group_idx[sqrdists > radius ** 2] = N

    # 升序排列，前面大于radius^2的都是N，是最大值，所以直接在剩下的点中取出前nsmaple个点
    # 排列函数torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形邻域中不足nample个点的情况），这些点需要舍弃，用第一个点进行代替
    # group_first将group_idx中的第一个点复制
    # mask存储group_idx中赋值为N的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] #替换
    return group_idx

# sampling 和 grouping主要用于将整个点云分散为局部的group，对每个单独的group都可以用pointnet提取特征
# 包含了sample_and_group和sample_and_group_all两个函数，区别在于_all直接将全体点作为group
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原始点云中通过farthest_point_sample函数提取出采样点的索引
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)  # 挑出来作为new_xyz

    # 根据new_xyz调用query_ball_point函数
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # grouped_xyz减去采样点（中心值）
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    # 若每个点上有新的特征维度，则进行拼接，否则直接返回
    # 用来拼接点的特征数据与坐标数据
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# PointNetSetAbstraction类实现普通的Set Abstraction
# 首先通过sample_and_group形成局部group
# 然后对局部group中的每个点进行MLP，最后最大池化后的得到局部的全局特征
class PointNetSetAbstraction(nn.Module):
    # 例如:npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128,128,256],group_all=False
    # npoint：farthest_point_sample中centrois的个数
    # nsample：每个局部group的采样点数
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        # 构造mlp
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            输入点的位置数据xyz: input points position data, [B, C, N]
            输入点的数据points: input points data, [B, D, N]
        Return:
            采样点的位置数据new_xyz: sampled points position data, [B, C, S]
            采样点的特征数据new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # 局部group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        # 进行pointnet操作
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 最大池化，得到局部的全局特征
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# PointNetSetAbstractionMSG类实现MSG的Set Abstraction——针对非均匀点云
# radius不再是固定的，而是一个list
# 进行不同半径的ball query并将不同半径的点云特征拼接在一起
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        # 不同半径下的点云特征保存在new_points_list中
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)

        # 拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

# 上采样，用于分割
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

