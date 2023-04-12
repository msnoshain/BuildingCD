import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


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
        self.conv1 = DoubleConv(in_ch+1024, 512)
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


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, f_ch, out_ch):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_ch, out_ch, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_ch, 1, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi  # 与low-level feature相乘，将权重矩阵赋值进去


class UNetDecoderWithAttentionGate(UNetDecoder):
    def __init__(self, in_ch, out_ch):
        super(UNetDecoderWithAttentionGate, self).__init__(in_ch, out_ch)
        self.atten1 = AttentionBlock(in_ch, 1024, 1024)
        self.atten2 = AttentionBlock(512, 512, 512)
        self.atten3 = AttentionBlock(256, 256, 256)
        self.atten4 = AttentionBlock(128, 128, 128)

    def forward(self, feature064, feature128, feature256, feature512, x):
        x = self.up(x)
        feature512 = self.atten1(x, feature512)
        x = torch.cat([x, feature512], dim=1)
        x = self.conv1(x)

        x = self.up(x)
        feature256 = self.atten2(x, feature256)
        x = torch.cat([x, feature256], dim=1)
        x = self.conv2(x)

        x = self.up(x)
        feature128 = self.atten3(x, feature128)
        x = torch.cat([x, feature128], dim=1)
        x = self.conv3(x)

        x = self.up(x)
        feature064 = self.atten4(x, feature064)
        x = torch.cat([x, feature064], dim=1)
        x = self.conv4(x)

        return self.conv(x)
