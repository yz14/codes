""" load, process, augmentation, etc. """

import os
import os.path as osp
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def load_MNIST(root, tsf, bsz):
    xs = [0, 1]
    for i in range(2):
        xs[i] = datasets.MNIST(root=root, train=i,
                               transform=tsf)
        xs[i] = DataLoader(xs[i], batch_size=bsz, shuffle=i)
    return xs


def preprocess(x):
    pass


def aug(x):
    tsf = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()])
    x_tsf = tsf(x)
    return x, x_tsf




if __name__ == "__main__":
    # root = "D:/codes/data"
    # tsf  = ToTensor()
    # bsz  = 64
    # test_loader, train_loader = load_FashionMNIST(root, tsf, bsz)

    root = "D:/codes/data/lung_4dcm/train.txt"
    # tsf  = None
    # bsz  = 16
    # train_loader = dcm_from_text(root, tsf, bsz)
    # for x in train_loader:
    #     img = x[0,0]
    #     break


    # plt.imshow(img, cmap='gray')
    # plt.show()