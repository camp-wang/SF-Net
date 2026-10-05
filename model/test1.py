import torch.nn as nn
import torch.nn.functional as F

from .modules import *

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Net1(nn.Module):
    def __init__(self, base_dim=32, use_spatial=True, use_frequency=True,
                 use_ffcm=True, use_pa=True, use_deconv=True,
                 fusion_variant='cga'):
        super(Net1, self).__init__()
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_ffcm = use_ffcm
        self.use_pa = use_pa
        self.use_deconv = use_deconv
        self.fusion_variant = fusion_variant

        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        # level1
        self.down_level1_block1 = self._sfda(base_dim)
        self.down_level1_block2 = self._sfda(base_dim)
        self.down_level1_block3 = self._sfda(base_dim)
        self.down_level1_block4 = self._sfda(base_dim)
        self.up_level1_block1 = self._sfda(base_dim)
        self.up_level1_block2 = self._sfda(base_dim)
        self.up_level1_block3 = self._sfda(base_dim)
        self.up_level1_block4 = self._sfda(base_dim)
        # level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = self._sfda(base_dim * 2)
        self.down_level2_block2 = self._sfda(base_dim * 2)
        self.down_level2_block3 = self._sfda(base_dim * 2)
        self.down_level2_block4 = self._sfda(base_dim * 2)
        self.up_level2_block1 = self._sfda(base_dim * 2)
        self.up_level2_block2 = self._sfda(base_dim * 2)
        self.up_level2_block3 = self._sfda(base_dim * 2)
        self.up_level2_block4 = self._sfda(base_dim * 2)
        # Level 3 bottleneck
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = self._dsrm(base_dim * 4)
        self.level3_block2 = self._dsrm(base_dim * 4)
        self.level3_block3 = self._dsrm(base_dim * 4)
        self.level3_block4 = self._dsrm(base_dim * 4)
        self.level3_block5 = self._dsrm(base_dim * 4)
        self.level3_block6 = self._dsrm(base_dim * 4)
        self.level3_block7 = self._dsrm(base_dim * 4)
        self.level3_block8 = self._dsrm(base_dim * 4)

        # up-sample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))
        # feature fusion
        if fusion_variant == 'cga':
            self.mix1 = CGAFusion(base_dim * 4, reduction=8)
            self.mix2 = CGAFusion(base_dim * 2, reduction=4)
        elif fusion_variant == 'concat':
            self.mix1 = ConcatFusion(base_dim * 4)
            self.mix2 = ConcatFusion(base_dim * 2)
        else:
            raise ValueError('Unsupported fusion_variant: {}'.format(fusion_variant))

    def _sfda(self, dim):
        return SFDA(
            default_conv,
            dim,
            3,
            use_spatial=self.use_spatial,
            use_frequency=self.use_frequency,
            use_deconv=self.use_deconv,
        )

    def _dsrm(self, dim):
        return DSRM(
            default_conv,
            dim,
            3,
            use_ffcm=self.use_ffcm,
            use_pa=self.use_pa,
            use_deconv=self.use_deconv,
        )

    def forward(self, x):
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out
