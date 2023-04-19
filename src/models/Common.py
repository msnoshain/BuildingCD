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
        self.atten1_1 = AttentionBlock(in_ch, 512, 1024)
        self.atten1_2 = AttentionBlock(in_ch, 512, 1024)
        self.atten2_1 = AttentionBlock(512, 256, 512)
        self.atten2_2 = AttentionBlock(512, 256, 512)
        self.atten3_1 = AttentionBlock(256, 128, 256)
        self.atten3_2 = AttentionBlock(256, 128, 256)
        self.atten4_1 = AttentionBlock(128, 64, 128)
        self.atten4_2 = AttentionBlock(128, 64, 128)

    def forward(self, feature128, feature256, feature512, feature1024, x):
        x = self.up(x)
        x1_f512, x2_f512=torch.chunk(feature1024, 2, 1)
        x1_f512 = self.atten1_1(x, x1_f512)
        x2_f512 = self.atten1_2(x, x2_f512)
        x = torch.cat([x, x1_f512, x2_f512], dim=1)
        x = self.conv1(x)

        x = self.up(x)
        x1_f256, x2_f256=torch.chunk(feature512, 2, 1)
        x1_f256 = self.atten2_1(x, x1_f256)
        x2_f256 = self.atten2_2(x, x2_f256)
        x = torch.cat([x, x1_f256, x2_f256], dim=1)
        x = self.conv2(x)

        x = self.up(x)
        x1_f128, x2_f128=torch.chunk(feature256, 2, 1)
        x1_f128 = self.atten3_1(x, x1_f128)
        x2_f128 = self.atten3_2(x, x2_f128)
        x = torch.cat([x, x1_f128, x2_f128], dim=1)
        x = self.conv3(x)

        x = self.up(x)
        x1_f064, x2_f064=torch.chunk(feature128, 2, 1)
        x1_f064 = self.atten4_1(x, x1_f064)
        x2_f064 = self.atten4_2(x, x2_f064)
        x = torch.cat([x, x1_f064, x2_f064], dim=1)
        x = self.conv4(x)

        return self.conv(x)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(
            self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
    

class UNetDecoderWithCBAM(UNetDecoder):
    def __init__(self, in_ch, out_ch):
        super(UNetDecoderWithCBAM, self).__init__(in_ch, out_ch)
        self.CBAM1 = CBAMLayer(3072)
        self.CBAM2 = CBAMLayer(1024)
        self.CBAM3 = CBAMLayer(512)
        self.CBAM4 = CBAMLayer(256)

    def forward(self, feature128, feature256, feature512, feature1024, x):
        x = self.up(x)
        x = self.CBAM1(torch.cat([x, feature1024], dim=1))
        x = self.conv1(x)

        x = self.up(x)
        x = self.CBAM2(torch.cat([x, feature512], dim=1))
        x = self.conv2(x)

        x = self.up(x)
        x = self.CBAM3(torch.cat([x, feature256], dim=1))
        x = self.conv3(x)

        x = self.up(x)
        x = self.CBAM4(torch.cat([x, feature128], dim=1))
        x = self.conv4(x)

        return self.conv(x)
