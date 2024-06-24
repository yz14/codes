""" Diffusion Model """

import torch 
from torch import nn, einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from functools import partial

# ============== Global Setting ==============
N_DIM = 2 # data dimension
Conv  = getattr(nn, 'Conv%dd' % N_DIM)
Norm  = getattr(nn, 'GroupNorm')
Pool  = getattr(nn, 'MaxPool%dd' % N_DIM)
Act   = getattr(nn, 'SiLU')
# ============================================


class Residual(nn.Module):
    """ x + f(x) """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """ f(norm(x)) """
    def __init__(self, c_in, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, c_in)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ============== Convs ==============
class WeightStandardizedConv2d(Conv):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()
        x = F.conv2d(x, normalized_weight, self.bias, 
                     self.stride, self.padding, self.dilation, self.groups)
        return x


class ConvBlock(nn.Module):
    """ [conv => norm => act]"""
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=8,
                 norm=True, act=True):
        super().__init__()
        p = k // 2 # keep size unchange
        self.conv = WeightStandardizedConv2d(c_in, c_out, k,s,p,d)
        self.norm = Norm(g, c_out) if norm else None
        self.act  = Act() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x


class ResConvBlock(nn.Module):
    """ res(x) + conv(...conv(x)) """
    def __init__(self, c_in, c_out, n_res=2, k=3, s=1, d=1, g=8):
        super().__init__()
        self.res = Conv(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        self.conv = [ConvBlock(c_in, c_out, k,s,d,g)]
        for _ in range(1, n_res):
            self.conv.append(ConvBlock(c_out, c_out, g=g))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = self.res(x) + self.conv(x)
        return x


class Down(nn.Module):
    """ resize => conv """
    def __init__(self, c_in, c_out, k=3, s=2, d=1, g=1):
        super().__init__()
        p = k // 2
        self.d = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.c = Conv(4*c_in, c_out, k,s,p,d,g)

    def forward(self, x):
        x = self.d(x)
        x = self.c(x)
        return x


class Up(nn.Module):
    """ up => conv """
    def __init__(self, c_in, c_out, k=3, s=1, d=1, g=1):
        super().__init__()
        p = k // 2
        self.u = nn.Upsample(scale_factor=2, mode='nearest')
        self.c = Conv(c_in, c_out, k,s,p,d,g)

    def forward(self, x):
        x = self.u(x)
        x = self.c(x)
        return x


# ============== Attentions ==============
class Attention(nn.Module):
    def __init__(self, c_in, heads=4, c_head=32):
        super().__init__()
        self.scale = c_head**-0.5
        self.heads = heads
        c_hid = c_head * heads
        self.to_qkv = Conv(c_in, 3*c_hid, 1, bias=False)
        self.to_out = Conv(c_hid, c_in, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        
        q = q * self.scale
        sim = einsum("b h c i, b h c j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h c j -> b h i c", attn, v)
        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, c_in, heads=4, c_head=32):
        super().__init__()
        self.scale = c_head**-0.5
        self.heads = heads
        c_hid = c_head * heads

        self.to_qkv = Conv(c_in, 3*c_hid, 1, bias=False)
        self.to_out = nn.Sequential(Conv(c_hid, c_in, 1), 
                                    nn.GroupNorm(1, c_in))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class EncBlock(nn.Module):
    def __init__(self, c_in, c_out, *, down=True, groups=8):
        super().__init__()
        self.c1 = ResConvBlock(c_in, c_in, g=groups)
        self.c2 = nn.Sequential(
            ResConvBlock(c_in, c_in, g=groups),
            Residual(PreNorm(c_in, LinearAttention(c_in))))
        self.d  = (Down(c_in, c_out, 1, 1) if down 
                   else Conv(c_in, c_out, 3,1,1))

    def forward(self, x, h):
        x = self.c1(x)
        h.append(x)
        x = self.c2(x)
        h.append(x)
        x = self.d(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, c_in, c_out, *, up=True, groups=8):
        super().__init__()
        self.c1 = ResConvBlock(c_in+c_out, c_out, g=groups)
        self.c2 = nn.Sequential(
            ResConvBlock(c_in+c_out, c_out, g=groups),
            Residual(PreNorm(c_out, LinearAttention(c_out))))
        self.u  = Up(c_out, c_in, 1, 1) if up \
            else Conv(c_out, c_in, 3,1,1)

    def forward(self, x, h):
        x = self.c1(torch.cat([x, h.pop()], dim=1))
        x = self.c2(torch.cat([x, h.pop()], dim=1))
        x = self.u(x)
        return x


class Unet(nn.Module):
    """ UNet with self-attention """
    def __init__(self, c_init, c_last, channels=(64, 128, 128, 256), 
                 condition=False, groups=4):
        super().__init__()

        # config
        self.condition = condition
        if condition: c_init *= 2
        in_out = list(zip(channels[:-1], channels[1:]))
        n_layer = len(in_out)
        c_mid  = channels[-1]
        # encoder & decoder
        self.inc = Conv(c_init, channels[0], 1,1,0) # initial conv

        self.enc, self.dec = nn.ModuleList([]), nn.ModuleList([])
        for i, (c_in, c_out) in enumerate(in_out):
            not_last = (i+1) < n_layer
            self.enc.append(
                EncBlock(c_in, c_out, down=not_last, groups=groups))

        self.mid_conv = nn.Sequential(
            ResConvBlock(c_mid, c_mid, g=groups),
            Residual(PreNorm(c_mid, Attention(c_mid))),
            ResConvBlock(c_mid, c_mid, g=groups))

        for i, (c_in, c_out) in enumerate(reversed(in_out)):
            not_last = (i+1) < n_layer
            self.dec.append(
                DecBlock(c_in, c_out, up=not_last, groups=groups))

        self.outc = nn.Sequential(
            ResConvBlock(2*channels[0], channels[0], g=groups),
            Conv(channels[0], c_last, 1),
            nn.Sigmoid())

    def forward(self, x, x_cond=None):
        if self.condition:
            x = torch.cat((x_cond, x), dim=1)
        
        h = []
        x = self.inc(x)
        r = x.clone()

        for enc in self.enc:
            x = enc(x, h)
        x = self.mid_conv(x)
        for dec in self.dec:
            x = dec(x, h)
        x = self.outc(torch.cat([x, r], dim=1))
        return x