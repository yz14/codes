""" Registration for MNIST """

from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from data_process import load_MNIST, aug
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model import VoxelModel
import os
from einops import rearrange


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.load_dataset()
        self.set_model()
        # self.load_model(self.args.model_fn) # pre-train or eval
        self.fit(self.args.epochs)
        # self.evaluate()

    def load_dataset(self):
        tsf = Compose([Resize(64), ToTensor()])
        self.valid_loader, self.train_loader = load_MNIST(
            self.args.root, tsf, self.args.bsz)

    def set_model(self):
        self.model = VoxelModel(
            self.args.img_size, self.args.c_init, self.args.c_out)
        self.model.to(self.args.dev)

    def fit(self, epochs):
        """ train and valid """
        losses = []
        optim  = Adam(self.model.parameters(), lr=self.args.lr)
        for e in tqdm(range(epochs), desc="Epochs: ", leave=True):
            self.training(optim, losses, e)
            self.save_model(f"{self.args.save_root}/ckps", f"{e}")
            self.evaluate(e)

    def training(self, optim, losses, e):
        self.model.train()
        for i, (x,y) in enumerate(self.train_loader):
            x = x.to(self.args.dev)
            x, xt = aug(x)
            optim.zero_grad()
            x_rec, flow = self.model(x, xt)
            loss = self.compute_loss(x_rec, xt)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            # login
            if i%100 == 0:
                print("[Epoch {:4d}] [Item {:4d}] [Loss: {:.4f}]".format(
                    e, i, loss.item()))
                img = torch.cat([x[:4,], x_rec[:4], xt[:4]], dim=0).detach().cpu()
                save_image(img, f"{self.args.save_root}/imgs/{e}-{i}.png", normalize=False, nrow=4)

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        loss = 0
        for i,(x,y) in enumerate(self.valid_loader):
            x = x.to(self.args.dev)
            x, xt = aug(x)
            x_rec, flow = self.model(x, xt)
            loss += self.compute_loss(x_rec, xt)
            break
        loss /= (i+1)
        print(f"valid loss: {loss.item()}")
        img = torch.cat([x[:4], x_rec[:4], xt[:4]], dim=0).detach().cpu()
        save_image(img, f"{self.args.save_root}/imgs/valid{epoch}.png", normalize=False, nrow=4)
        
    def compute_loss(self, y_pred, y):
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        return loss

    def save_model(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        fn = f"{path}/{epoch}.pth"
        torch.save(self.model.state_dict(), fn)
        print(f"Saving epoch: {fn}.")

    def load_model(self, model_fn):
        self.model.load_state_dict(torch.load(model_fn))


def get_config():
    parser = argparse.ArgumentParser(description='cofig')
    parser.add_argument('--root', type=str, default='D:/codes/data', help='数据路径')
    parser.add_argument('--save-root', type=str, default='D:/codes/work-projects/voxelmorph', help='结果路径')
    parser.add_argument('--model-fn', type=str, default='D:/codes/work-projects/voxelmorph/ckps/0.pth', help='模型路径')
    parser.add_argument('--dev', type=str, default='cuda')
    parser.add_argument('--c_init', type=int, default=2, help='输入通道')
    parser.add_argument('--c_out', type=int, default=16, help='输出通道')
    parser.add_argument('--channels', type=int, default=[64, 128, 128, 256, 256], help='输出通道')
    parser.add_argument('--img_size', type=list, default=[64, 64], help='图像尺寸')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_config()
    trainer = Trainer(args)