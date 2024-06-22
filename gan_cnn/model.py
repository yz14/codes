
from blocks import ConvBlock, Upsample, NConvBlock, Downsample, Conv, ResConvBlock
import torch 
from torch import nn 
from einops.layers.torch import Rearrange


# ======================== CNN version ========================
class Up(nn.Module):
    def __init__(self, c_in, c_out, mode='bilinear'):
        super().__init__()
        self.u = Upsample(mode)
        # self.c = NConvBlock(c_in, c_out)
        self.c = ResConvBlock(c_in, c_out)

    def forward(self, x):
        x = self.u(x)
        x = self.c(x)
        return x


class Down(nn.Module):
    def __init__(self, c_in, c_out, mode='pool'):
        super().__init__()
        self.d = Downsample(mode)
        # self.c = NConvBlock(c_in, c_out)
        self.c = ResConvBlock(c_in, c_out)

    def forward(self, x):
        x = self.d(x)
        x = self.c(x)
        return x


class G(nn.Module):
    def __init__(self, d_init, chs, c_last):
        super().__init__()
        in_out = list(zip(chs[:-1], chs[1:]))

        self.inc = nn.Sequential(
            nn.Linear(d_init, 256),
            Rearrange('b (c h w) -> b c h w', c=4, h=8, w=8),
            nn.BatchNorm2d(4),
            ConvBlock(4, chs[0]))
        self.g = []
        for c_in, c_out in in_out:
            self.g.append(Up(c_in, c_out))
        self.g = nn.Sequential(*self.g,
                               Conv(c_out, c_last, 3, 1, 1),
                               nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x = self.g(x)
        return x


class D(nn.Module):
    def __init__(self, c_init, chs, d_last):
        super().__init__()
        in_out = list(zip(chs[:-1], chs[1:]))
        
        self.d = []
        self.inc = ConvBlock(c_init, chs[0])
        for _, (c_in, c_out) in enumerate(in_out):
            self.d.append(Down(c_in, c_out))
        self.d = nn.Sequential(*self.d,
                               nn.AdaptiveAvgPool2d(4),
                               Rearrange('c h w d -> c (h w d)'),
                               nn.Linear(16*c_out, d_last),
                               nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x = self.d(x)
        return x