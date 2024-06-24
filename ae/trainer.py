""" Autoencoder """

from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from data_process import load_FashionMNIST, aug
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt
from diffusion import Unet as UNet
# from unet import UNet


class Trainer(object):
    def __init__(self, args):
        start_epoch = 0
        self.args = args
        self.load_dataset()
        self.set_model()
        # self.load_save_model(f"{self.args.save_root}/ckps", start_epoch, mode='load')
        self.fit(start_epoch)
        # self.evaluate()

    def load_dataset(self):
        tsf = Compose([Resize(32), ToTensor()])
        self.valid_loader, self.train_loader = load_FashionMNIST(
            self.args.root, tsf, self.args.bsz)

    def set_model(self):
        self.model = UNet(self.args.c_init, self.args.c_last)
        self.model.to(self.args.dev)

    def fit(self, start_epoch=0):
        """ train and valid """
        st, ed = start_epoch+1, start_epoch+1+self.args.epochs
        losses = []
        optim = Adam(self.model.parameters(), lr=self.args.lr)
        for e in tqdm(range(st, ed), desc="Epochs: ", leave=True):
            self.training(optim, losses, e)
            self.load_save_model(f"{self.args.save_root}/ckps", e)
            self.evaluate(e)

    def training(self, optim, losses, e):
        self.model.train()
        for i, (x,y) in enumerate(self.train_loader):
            x = x.to(self.args.dev)
            x_in = aug(x)
            optim.zero_grad()
            x_rec = self.model(x_in)
            loss = self.compute_loss(x_rec, x)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            # login
            if i%100 == 0:
                print("[Epoch {:4d}] [Item {:4d}] [Loss: {:.4f}]".format(
                    e, i, loss.item()))
                img = torch.cat((x[:4], x_rec[:4]), dim=0).detach().cpu()
                self.save_img(img, f"{self.args.save_root}/imgs/{e}-{i}.png")

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        for i, (x,y) in enumerate(self.valid_loader):
            x = x.to(self.args.dev)
            x_in = aug(x)
            x_rec = self.model(x_in)
            break
        img = torch.cat((x_in[:4], x_rec[:4]), dim=0).detach().cpu()
        self.save_img(img, f"{self.args.save_root}/imgs/valid{epoch}.png")
    
    def compute_loss(self, y_pred, y):
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        return loss

    def load_save_model(self, root, epoch, mode="save"):
        fn = f"{root}/{epoch}.pth"
        if mode == "save":
            torch.save(self.model.state_dict(), fn)
            print(f"Saving epoch: {epoch}.")
        else:
            self.model.load_state_dict(torch.load(fn))
            print(f"Loading epoch: {epoch}.")

    def save_img(self, imgs, out_fn):
        save_image(imgs, out_fn, normalize=False, nrow=4)


def get_config():
    parser = argparse.ArgumentParser(description='cofig')
    parser.add_argument('--root', type=str, default='D:/codes/data')
    parser.add_argument('--save_root', type=str, default='D:/codes/work-projects/ae')
    parser.add_argument('--dev', type=str, default='cuda')
    parser.add_argument('--c_init', type=int, default=1, help='输入通道')
    parser.add_argument('--c_last', type=int, default=1, help='输出通道')
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_config()
    trainer = Trainer(args)