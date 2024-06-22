
from utils import *
from torch import nn 
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from functools import partial


class MLP(nn.Module):
    """ [lin => norm => act => drop] 
    see https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout """
    def __init__(self, d_in, d_out, norm=True, act=True, drop_p=False):
        super().__init__()
        self.lin = [nn.Linear(d_in, d_out)]
        if norm  : self.lin.append(nn.BatchNorm1d(d_out))
        if act   : self.lin.append(nn.ReLU(inplace=True))
        if drop_p: self.lin.append(nn.Dropout(drop_p))
        self.lin = nn.Sequential(*self.lin)
    
    def forward(self, x):
        x = self.lin(x)
        return x


class NMLP(nn.Module):
    """ [lin => norm => act => drop] * N """
    def __init__(self, d_in, d_out, n_mlp=2):
        super().__init__()
        # n mlp
        self.lin = [MLP(d_in, d_out)]
        for i in range(1, n_mlp):
            self.lin.append(MLP(d_out, d_out))
        self.lin = nn.Sequential(*self.lin)
    
    def forward(self, x):
        x = self.lin(x)
        return x


class ResMLP(nn.Module):
    """ [lin => norm => act => drop] * N """
    def __init__(self, d_in, d_out, n_res=2, norm=True, act=True, drop_p=0.5):
        super().__init__()
        # for res
        if d_in != d_out:
            norm_, act_, drop_p_ = False, False, False
            self.res = MLP(d_in, d_out, norm_, act_, drop_p_)
        else:
            self.res = nn.Identity()
        # n mlp
        self.lin = [MLP(d_in, d_out, norm, act, drop_p)]
        for i in range(1, n_res):
            not_last = (i+1) != n_res
            self.lin.append(MLP(d_out, d_out, not_last, not_last, not_last*drop_p))
        self.lin  = nn.Sequential(*self.lin)
        self.out  = nn.Sequential(
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
            nn.Dropout(inplace=True))
    
    def forward(self, x):
        x = self.res(x) + self.lin(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    m = ResMLP(3,4)
    x = torch.randn(2,3)
    y = m(x)
    print(m)
    print(y.shape)