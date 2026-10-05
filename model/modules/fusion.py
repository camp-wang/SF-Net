import torch
from torch import nn

from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class ConcatFusion(nn.Module):
    """Concatenate two skip features and project them back to dim channels."""

    def __init__(self, dim):
        super(ConcatFusion, self).__init__()
        self.conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=True)

    def forward(self, x, y):
        return self.conv(torch.cat([x, y], dim=1))
