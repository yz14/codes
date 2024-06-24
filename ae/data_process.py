""" load dataset """

import os
import os.path as osp
from io import open
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def norm_win(x, wl=40, ww=400):
    """ 窗宽窗位归一化 """
    vmin, vmax = wl - ww/2.0, wl + ww/2.0
    x = torch.clamp(x, vmin, vmax)
    x -= vmin
    x /= (x.max())
    return x

def norm_val(x, vmin, vmax):
    """ 阈值归一化 """
    x = torch.clamp(x, vmin, vmax)
    x -= vmin
    x /= (x.max())
    return x

def add_noise(x, scale=0.1):
    noise = torch.randn_like(x)
    x += noise*scale
    return x

def aug(x):
    x = add_noise(x.clone())
    return x
    


def load_FashionMNIST(root, tsf, bsz):
    """ 读取fashionMNIST """
    xs = [0, 1]
    for i in range(2):
        xs[i] = datasets.FashionMNIST(root=root, train=i,
                                      transform=tsf)
        xs[i] = DataLoader(xs[i], batch_size=bsz, shuffle=i)
    return xs


def load_Cifar10(root, tsf, bsz):
    """读取Cifar-10"""
    xs = [0, 1]
    for i in range(2):
        xs[i] = datasets.CIFAR10(root=root, train=True, 
                                 transform=tsf)
        
        xs[i] = DataLoader(xs[i], batch_size=bsz, 
                           shuffle=i, num_workers=4, 
                           drop_last=True)
    return xs