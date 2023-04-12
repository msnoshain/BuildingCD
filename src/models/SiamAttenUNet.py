import torch
import torch.nn as nn
import models.Common as Common


class SiamAttenUNet(nn.Module):
    def __init__(self, in_ch: int=3, out_ch: int=1):
        super(SiamAttenUNet, self).__init__()
        self.encoder1 = Common.UNetEncoder(in_ch)
        self.encoder2 = Common.UNetEncoder(in_ch)
        self.conv = Common.DoubleConv(1024, 2048)
        self.decoder = Common.UNetDecoderWithAttentionGate(2048, out_ch)
        self.active = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        
        x1_feature064, x1_feature128, x1_feature256, x1_feature512, x1 = self.encoder1(
            x1)
        x2_feature064, x2_feature128, x2_feature256, x2_feature512, x2 = self.encoder2(
            x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.decoder(torch.cat([x1_feature064, x2_feature064], dim=1), 
                         torch.cat([x1_feature128, x2_feature128], dim=1), 
                         torch.cat([x1_feature256, x2_feature256], dim=1), 
                         torch.cat([x1_feature512, x2_feature512], dim=1), 
                         x)
        
        return self.active(x)