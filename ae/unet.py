
import torch 
from torch import nn
from blocks import (ConvBlock, ResConvBlock, Downsample,
                    Upsample, NConvBlock)

Convx2 = NConvBlock

class Down(nn.Module):
    """ downsample: [down => conv] """
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=1):
        super().__init__()
        mode = 'pool'
        self.down = Downsample(mode)
        self.conv = Convx2(c_in, c_out)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """ upscale: [up => conv] """
    def __init__(self, c_in, c_out, mode='bilinear'):
        super().__init__()
        self.up   = Upsample(mode)
        self.conv = Convx2(c_in, c_out)

    def forward(self, x, x1):
        x = self.up(x)
        x = self.conv(torch.cat([x, x1], dim=1))
        return x


class UNet(nn.Module):
    def __init__(self, c_init, c_last, channels=[64, 128, 128, 256, 256]):
        super().__init__()

        in_out = list(zip(channels[:-1], channels[1:]))
        # downsample
        self.enc = nn.ModuleList([Convx2(c_init, channels[0])])
        for c_in, c_out in in_out:
            self.enc.append(Down(c_in, c_out))
        # upsample
        self.dec = nn.ModuleList([])
        for c_in, c_out in reversed(in_out):
            self.dec.append(Up(c_in+c_out, c_in))
        # final conv
        self.outc = nn.Sequential(
            ConvBlock(c_in, c_last, norm=False, act=False),
            nn.Sigmoid())

    def forward(self, x):
        hs = []
        for d in self.enc:
            x = d(x)
            hs.append(x)
        x = hs.pop()
        for u in self.dec:
            x = u(x, hs.pop())
        x = self.outc(x)
        return x


if __name__ == "__main__":
    m = UNet(3, 3, channels=[64, 128, 128, 256, 256])
    x  = torch.randn(2,3,128,128)
    y = m(x)
    print(y.shape)
    print(m)