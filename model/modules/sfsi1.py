import torch
import torch.nn as nn
import torch.nn.functional as F


from .deconv import DEConv
from .cga import SpatialAttention, ChannelAttention, PixelAttention


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpaBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, use_deconv=True):
        super(SpaBlock, self).__init__()
        if use_deconv:
            self.conv1 = DEConv(dim)
        else:
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        return x_out


class SFDA(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8,
                 use_spatial=True, use_frequency=True, use_deconv=True):
        super(SFDA, self).__init__()
        if not (use_spatial or use_frequency):
            raise ValueError('At least one SFDA branch must be enabled')
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        if use_spatial:
            self.spatial_process = SpaBlock(
                conv, dim, kernel_size, use_deconv=use_deconv
            )
        if use_frequency:
            self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(dim)
        branch_count = int(use_spatial) + int(use_frequency)
        self.cat = nn.Conv2d(branch_count * dim, dim, 1, 1, 0)

    def forward(self, x):
        identity = x
        _, _, H, W = x.shape
        features = []
        x_freq = None
        if self.use_frequency:
            with torch.cuda.amp.autocast(enabled=False):
                x_freq = torch.fft.rfft2(x.float(), norm='backward')
                x_freq = self.frequency_process(x_freq)
                x_freq_spatial = torch.fft.irfft2(
                    x_freq, s=(H, W), norm='backward'
                )
        if self.use_spatial:
            features.append(self.spatial_process(x))
        if self.use_frequency:
            features.append(x_freq_spatial.to(dtype=x.dtype))

        xcat = torch.cat(features, 1)
        res = self.cat(xcat)
        res = res + identity
        return res
