import torch
import torch.nn as nn
from models.UNet import UNetDecoder, UNetEncoder


class SiameseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SiameseUNet, self).__init__()
        
        self.encoder = UNetEncoder(in_channels, out_channels)
        self.decoder1 = UNetDecoder(out_channels*2, out_channels)
        self.decoder2 = UNetDecoder(out_channels*2, out_channels)
        self.activate = nn.Sigmoid()

    def forward(self, x1, x2):
        enc1 = self.encoder(x1)
        enc2 = self.encoder(x2)

        dec1 = self.decoder1(enc1, enc2)
        dec2 = self.decoder2(enc2, enc1)

        return self.activate(torch.abs(dec1 - dec2))