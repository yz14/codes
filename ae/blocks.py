
from torch import nn
import torch
from einops.layers.torch import Rearrange


# ============== Global Setting ==============
N_DIM = 2 # data dimension
Conv  = getattr(nn, 'Conv%dd' % N_DIM)
Norm  = getattr(nn, 'BatchNorm%dd' % N_DIM)
Pool  = getattr(nn, 'MaxPool%dd' % N_DIM)
Act   = getattr(nn, 'ReLU')
# ============================================

class ConvBlock(nn.Module):
    """ [conv => norm => act] """
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=1,
                 norm=True, act=True):
        super().__init__()
        p = k // 2 # keep size unchange
        self.conv = Conv(c_in, c_out, k, s, p, d, g)
        self.norm = Norm(c_out) if norm else None
        self.act  = Act(inplace=True) if act else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act:  x = self.act(x)
        return x


class NConvBlock(nn.Module):
    """ [conv => norm => act] * N """
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=1,
                 n_conv=2):
        super().__init__()
        self.conv = nn.ModuleList([
            ConvBlock(c_in, c_out, k,s,d,g)])
        for _ in range(1, n_conv):
            self.conv.append(ConvBlock(c_out, c_out))
    
    def forward(self, x):
        for c in self.conv:
            x = c(x)
        return x


class ResConvBlock(nn.Module):
    """ x + conv(x) """
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=1,
                 n_res=2):
        super().__init__()
        # res
        if (c_in != c_out) or (s != 1):
            kd, sd = k, s # TODO: 1x1
            self.res = ConvBlock(c_in, c_out, kd, sd, norm=False, act=False)
        else:
            self.res = nn.Identity()
        # conv
        self.conv = [ConvBlock(c_in, c_out, k,s,d,g)]
        for i in range(1, n_res):
            not_last = (i+1) != n_res
            self.conv.append(
                ConvBlock(c_out, c_out, norm=not_last, act=not_last))
        self.conv = nn.Sequential(*self.conv)
        self.norm = Norm(c_out)
        self.act  = Act(inplace=True)

    def forward(self, x):
        x = self.res(x) + self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """ pool / conv / resize """
    def __init__(self, mode, c_in=None, k=3, s=2, d=1, g=1,
                 n_res=2, norm=False, act=False):
        super().__init__()
        factor = 2
        if mode == 'pool':
            self.d = Pool(factor)
        elif mode == 'conv':
            self.d = ConvBlock(c_in, c_in, k,s,d,g, norm=norm, act=act)
        elif mode == 'resize':
            self.d = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
                ConvBlock(4*c_in, c_in, k,s,d,g, norm=norm, act=act))
        else:
            print("==== Method not implemented ====")

    def forward(self, x):
        x = self.d(x)
        return x


class Upsample(nn.Module):
    """ upscale using [upsample, conv] """
    def __init__(self, mode, c_in=None):
        super().__init__()
        opts = ['nearest', 'bilinear', 'trilinear']
        if mode in opts:
            self.u = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else:
            k,s = 2, 2
            self.u = nn.ConvTranspose2d(c_in, c_in, k,s)

    def forward(self, x):
        x = self.u(x)
        return x