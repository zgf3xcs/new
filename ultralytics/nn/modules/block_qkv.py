import torch
from torch import nn

from ultralytics.nn.modules import DWConv, Conv


class TranQKVConcat(nn.Module):
    def __init__(self, dim, dimension=1, eps=1e-5):  # dim：特征图的通道数。dimension：拼接维度，默认为1，按照一维进行拼接实际上就是按照通道数进行拼接
        super(TranQKVConcat, self).__init__()
        self.d = dimension
        self.q = DWConv(dim, dim, k=3, s=1)
        # 初始化一个深度可分离卷积层DWConv，用于处理查询（Query）部分：
        # 输入通道数为dim。
        # 输出通道数也为dim。
        # 卷积核大小为3。
        # 步长为1。

        self.k = nn.Sequential(*(DWConv(dim, dim, k=3, s=1) for _ in range(2)))
        # 初始化一个顺序容器nn.Sequential，包含两个DWConv层，用于处理键（Key）部分：
        # 每个DWConv的输入通道数为dim。
        # 输出通道数也为dim。
        # 卷积核大小为3。
        # 步长为1。

        self.v = nn.Identity()
        # 初始化一个恒等映射层nn.Identity，用于处理值（Value）部分，不改变输入。

        self.linear = Conv(dim, dim, k=1, s=1)
        # 初始化一个1x1卷积层Conv，用于调整通道数：，这里为什么可以用于调整通道数是因为我的进行卷积的时候可以自己定义输入输出通道数，而且由于是1x1卷积，所以不会改变输入的形状。
        # 输入通道数为dim。
        # 输出通道数也为dim。
        # 卷积核大小为1。
        # 步长为1。

        # self.gn = nn.GroupNorm(num_groups=dim // 32, num_channels=dim)    # ×
        # self.gn = nn.GroupNorm(num_groups=dim // 16, num_channels=dim)    # √
        self.gn = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)  # √ √
        # 初始化一个组归一化层nn.GroupNorm：
        # 组数为dim // 8。
        # 通道数为dim。组归一化层能够在每个小批量内对特征图进行归一化处理，
        # 从而稳定训练过程并减少内部协变量偏移。这对于深层神经网络的训练非常有益

        # self.gn = nn.GroupNorm(num_groups=dim // 4, num_channels=dim)
        # self.gn = nn.GroupNorm(num_groups=dim // 2, num_channels=dim)
        # self.innorm = nn.InstanceNorm2d(dim)
        # self.bn = nn.BatchNorm2d(dim)
        self.eps = eps

    def forward(self, x):
        x = torch.cat(x, self.d)
        # 将输入张量或张量列表x沿指定维度self.d进行拼接
        # 这里直接按照通道数进行拼接

        # qkv = (self.innorm(self.k(x) * self.q(x)) * self.v(x)).permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # qkv = self.ln_1(qkv).permute(0, 3, 1, 2)

        # return self.linear(
        #     self.innorm(self.k(x) * self.q(x)) * self.v(x)
        # )

        # 80*80*512
        return self.linear(
            self.gn(self.k(x) * self.q(x)) * self.v(x)
        )
        # 前向传播的主要计算步骤：
        # 计算self.k(x)和self.q(x)的逐元素乘积。
        # 将乘积结果通过组归一化层self.gn。
        # 再与self.v(x)进行逐元素乘积。
        # 最后通过1x1卷积层self.linear进行线性变换并返回结果

        # return self.linear(
        #     self.gn((self.k(x) + self.eps) * (self.q(x) + self.eps)) * (self.v(x) + self.eps)
        # )

        # return x + self.linear(self.bn(
        #     (self.q(x) + self.eps) * (self.k(x) + self.eps)
        # ) * (self.v(x) + self.eps))

        # y = self.linear(self.bn(
        #     (self.q(x) + self.eps) * (self.k(x) + self.eps)
        # ) * (self.v(x) + self.eps))

        # return self.linear(self.act(self.bn(
        #     (self.q(x) + self.eps) * (self.k(x) + self.eps)
        # )) * (self.v(x) + self.eps))
