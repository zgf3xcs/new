import torch
from torch import nn

from ultralytics.nn.modules import DWConv, Conv



class TranQKVConcat(nn.Module):
    def __init__(self, dim, dimension=1, eps=1e-5):  # dim：特征图的通道数。dimension：拼接维度，默认为1，按照一维进行拼接实际上就是按照通道数进行拼接
        super(TranQKVConcat, self).__init__()
        self.d = dimension
        self.q = DWConv(dim, dim, k=3, s=1)


        self.k = nn.Sequential(*(DWConv(dim, dim, k=3, s=1) for _ in range(2)))


        self.v = nn.Identity()


        self.linear = Conv(dim, dim, k=1, s=1)



        self.gn = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)  # √ √



        self.eps = eps

    def forward(self, x):
        x = torch.cat(x, self.d)

        # 80*80*512
        return self.linear(
            self.gn(self.k(x) * self.q(x)) * self.v(x)
        )
