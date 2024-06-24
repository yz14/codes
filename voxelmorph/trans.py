
import torch
from torch import nn 
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, dim=-1) # x,y position
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # model parameters but not update during training
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        shape = flow.shape[1:-1]                  # [n, hwd, zyx]
        to_zyx = list(range(len(shape)))[::-1]    # zyx grid
        grid = self.grid + flow
        for i in range(len(shape)):
            grid[...,i] /= (shape[i] - 1)         # [0, 1]
            grid[...,i] = 2 * (grid[...,i] - 0.5) # [-1, 1]
        grid = grid[..., to_zyx]                  # xyz => zyx

        return F.grid_sample(src, grid, align_corners=True, mode=self.mode)
