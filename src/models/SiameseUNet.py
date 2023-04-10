import torch
import torch.nn as nn

from models.UNet import DoubleConv


class UNetEncoder(nn.Module):
    def __init__(self, in_ch):
        super(UNetEncoder, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feature064 = self.conv1(x)
        feature128 = self.conv2(self.pool(feature064))
        feature256 = self.conv3(self.pool(feature128))
        feature512 = self.conv4(self.pool(feature256))

        return feature064, feature128, feature256, feature512, self.pool(feature512)


class UNetDecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetDecoder, self).__init__()
        self.conv1 = DoubleConv(in_ch, 512)
        self.conv2 = DoubleConv(1024, 256)
        self.conv3 = DoubleConv(512, 128)
        self.conv4 = DoubleConv(256, 64)
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, feature064, feature128, feature256, feature512, x):
        x = self.up(x)
        x = torch.cat([x, feature512], dim=1)
        x = self.conv1(x)

        x = self.up(x)
        x = torch.cat([x, feature256], dim=1)
        x = self.conv2(x)

        x = self.up(x)
        x = torch.cat([x, feature128], dim=1)
        x = self.conv3(x)

        x = self.up(x)
        x = torch.cat([x, feature064], dim=1)
        x = self.conv4(x)

        return self.conv(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(SiameseUNet, self).__init__()
        self.encoder1 = UNetEncoder(in_ch)
        self.encoder2 = UNetEncoder(in_ch)
        self.conv = DoubleConv(1024, 2048)
        self.decoder = UNetDecoder(3072, out_ch)
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