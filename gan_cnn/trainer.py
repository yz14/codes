
from tqdm import tqdm
import torch
from torch import nn
from data_process import load_FashionMNIST
from torchvision.transforms import Resize, ToTensor, Compose
import argparse
from torchvision.utils import save_image
from model import G, D


class Trainer(object):
    def __init__(self, args):
        start_epoch = 0
        self.args = args
        self.load_data()
        self.set_model()
        # self.load_save_model(f"{self.args.save_root}/ckps", start_epoch, mode='load')
        self.fit(start_epoch)
        # self.evaluate()

    def load_data(self):
        tsf = Compose([Resize(32), ToTensor()])
        self.valid_loader, self.train_loader = load_FashionMNIST(
            self.args.root, tsf, self.args.bsz)

    def set_model(self):
        self.G = G(self.args.d_latent, self.args.chs_G, self.args.c_init)
        self.D = D(self.args.c_init, self.args.chs_D, self.args.d_last)
        self.G.to(self.args.dev)
        self.D.to(self.args.dev)

    def fit(self, start_epoch):
        st, ed = start_epoch+1, start_epoch+1+self.args.epochs
        losses = []
        optimD = torch.optim.Adam(self.D.parameters(), lr=self.args.lr)
        optimG = torch.optim.Adam(self.G.parameters(), lr=self.args.lr)
        for e in tqdm(range(st, ed), desc="Epochs: ", leave=True):
            self.training(optimD, optimG, losses, e)
            self.load_save_model(f"{self.args.save_root}/ckps", e)
            self.evaluate(e)

    def training(self, optimD, optimG, losses, epoch):
        self.D.train()
        self.G.train()
        for i, (x,_) in enumerate(self.train_loader):
            bsz = x.size(0)
            noise = torch.randn(bsz, self.args.d_latent)
            x = x.to(self.args.dev)
            x_fake = self.G(noise.to(self.args.dev))
            real = torch.ones(bsz, 1).to(self.args.dev).float()
            # train G
            optimG.zero_grad()
            logits = self.D(x_fake)
            loss_G = self.compute_loss(logits, real) # fool D
            loss_G.backward()
            optimG.step()
            # train D
            optimD.zero_grad()
            logits = torch.cat([self.D(x), self.D(x_fake.detach())], dim=0)
            loss_D = self.compute_loss(logits, torch.cat([real, real*0], dim=0))
            loss_D.backward()
            optimD.step()
            # info
            losses.append([loss_G.item(), loss_D.item()])
            if i%100 == 0:
                print("[Epoch {:4d}] [Item {:4d}] [LossD: {:.4f}] [lossG: {:.4f}]".format(
                    epoch, i, loss_D.item(), loss_G.item()))
                img = x_fake[:16].detach().cpu().reshape(16, 1, 32, 32)
                self.save_img(img, f"{self.args.save_root}/imgs/{epoch}-{i}.png")

    @torch.no_grad()
    def evaluate(self, epoch):
        self.G.eval()
        noise = torch.randn(16, self.args.d_latent)
        x_pred = self.G(noise.to(self.args.dev))
        img = x_pred[:16].detach().cpu().reshape(16, 1, 32, 32)
        self.save_img(img, f"{self.args.save_root}/imgs/valid{epoch}.png")
    
    def compute_loss(self, y_pred, y):
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        return loss

    def load_save_model(self, root, epoch, mode="save"):
        fn_g = f"{root}/G{epoch}.pth"
        fn_d = f"{root}/D{epoch}.pth"
        if mode == "save":
            torch.save(self.G.state_dict(), fn_g)
            torch.save(self.D.state_dict(), fn_d)
            print(f"Saving epoch: {epoch}.")
        else:
            self.G.load_state_dict(torch.load(fn_g))
            self.D.load_state_dict(torch.load(fn_d))
            print(f"Loading epoch: {epoch}.")

    def save_img(self, imgs, out_fn):
        save_image(imgs, out_fn, normalize=False, nrow=4)


def get_config():
    parser = argparse.ArgumentParser(description='cofig')
    parser.add_argument('--root', type=str, default='D:/codes/data', help='数据路径')
    parser.add_argument('--save_root', type=str, default='D:/codes/self-projects/gan_cnn')
    parser.add_argument('--dev', type=str, default='cuda')
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    # CNN version
    parser.add_argument('--d_latent', type=int, default=256, help='输入通道')
    parser.add_argument('--d_last', type=int, default=1, help='输入通道')
    parser.add_argument('--c_init', type=int, default=1, help='输入通道')
    parser.add_argument('--chs_D', type=list, default=[64, 128, 128], help='判别器通道')
    parser.add_argument('--chs_G', type=list, default=[64, 128, 128], help='生成器通道')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_config()
    trainer = Trainer(args)