
import torch
from torch import nn 
from unet import UNet
from trans import SpatialTransformer
from torch.distributions.normal import Normal
from einops import rearrange

class VoxelModel(nn.Module):
    def __init__(self, img_size, c_init, c_out):
        super().__init__()
        n_dim = len(img_size)
        self.unet = UNet(c_init, c_out)
        # unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % n_dim)
        self.flow = Conv(c_out, n_dim, kernel_size=3, padding=1)

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # spatial transformer
        self.transformer = SpatialTransformer(img_size)

    def forward(self, source, target, test=False):
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet(x)
        flow_field = self.flow(x)
        flow_field = rearrange(flow_field, 'n c h w -> n h w c')
        y_source = self.transformer(source, flow_field)

        return y_source, flow_field