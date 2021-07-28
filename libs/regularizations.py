import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_path_rate=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_path_rate

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random.floor_()
        return x.div(keep_prob) * random
