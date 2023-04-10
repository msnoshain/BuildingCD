import torch
import torch.nn as nn
from models.UNet import UNet


class SiamNet(nn.Module):
    def __init__(self, in_ch1=3, out_ch1=1, in_ch2=3, out_ch2=1):
        super(SiamNet, self).__init__()

        self.Unet1 = UNet(in_ch=in_ch1, out_ch=out_ch1)
        self.Unet2 = UNet(in_ch=in_ch2, out_ch=out_ch2)

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        return torch.abs(self.Unet1(x1) - self.Unet2(x2))
