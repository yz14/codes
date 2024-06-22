""" load dataset """

import os
import os.path as osp
from io import open
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


def load_FashionMNIST(root, tsf, bsz):
    """ 读取fashionMNIST """
    xs = [0, 1]
    for i in range(2):
        xs[i] = datasets.FashionMNIST(root=root, train=i,
                                      transform=tsf)
        xs[i] = DataLoader(xs[i], batch_size=bsz, shuffle=i)
    return xs

def aug(x):
    tsf = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()])
    x = tsf(x)
    return x


if __name__ == "__main__":
    root = "D:/codes/data"
    tsf  = transforms.ToTensor()
    bsz  = 64
    
    test_loader, train_loader = load_FashionMNIST(root, tsf, bsz)
    for (x,y) in train_loader:
        plt.imshow(x[0,0])
        plt.show()
        break