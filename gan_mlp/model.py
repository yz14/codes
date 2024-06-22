
from blocks import MLP, ResMLP, NMLP
import torch 
from torch import nn 
from einops.layers.torch import Rearrange
from attentions import Attention

# ======================== MLP version ========================
class G(nn.Module):
    def __init__(self, d_init, dims, d_last):
        super().__init__()
        in_out = list(zip(dims[:-1], dims[1:]))

        self.inc = MLP(d_init, dims[0], False, False)
        self.g = []
        for d_in, d_out in in_out:
            self.g.append(NMLP(d_in, d_out))
        self.g = nn.Sequential(
            *self.g,
            MLP(d_out, d_last, False, False, False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x = self.g(x)
        return x


class D(nn.Module):
    def __init__(self, d_init, dims, d_last):
        super().__init__()
        in_out = list(zip(dims[:-1], dims[1:]))

        self.inc = MLP(d_init, dims[0])
        self.d = []
        for _, (d_in, d_out) in enumerate(in_out):
            self.d.append(NMLP(d_in, d_out))
        self.d = nn.Sequential(
            *self.d,
            MLP(d_out, d_last, False, False, False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.inc(x)
        x = self.d(x)
        return x
        

if __name__ == "__main__":
    netG = G(256, [256,256,512,512], 1024)
    netD = D(1024, [512,512,256,256], 1)

    x = torch.randn(2, 256)
    g = netG(x)
    y = netD(g)
    print(g.shape, y.shape)
    