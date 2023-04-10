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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        init_ch = 64
        ch_count = [init_ch, init_ch * 2, init_ch * 4, init_ch * 8, init_ch * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_ch, ch_count[0])
        self.Conv2 = DoubleConv(ch_count[0], ch_count[1])
        self.Conv3 = DoubleConv(ch_count[1], ch_count[2])
        self.Conv4 = DoubleConv(ch_count[2], ch_count[3])
        self.Conv5 = DoubleConv(ch_count[3], ch_count[4])

        self.Up5 = UpConv(ch_count[4], ch_count[3])
        self.Up_conv5 = DoubleConv(ch_count[4], ch_count[3])

        self.Up4 = UpConv(ch_count[3], ch_count[2])
        self.Up_conv4 = DoubleConv(ch_count[3], ch_count[2])

        self.Up3 = UpConv(ch_count[2], ch_count[1])
        self.Up_conv3 = DoubleConv(ch_count[2], ch_count[1])

        self.Up2 = UpConv(ch_count[1], ch_count[0])
        self.Up_conv2 = DoubleConv(ch_count[1], ch_count[0])

        self.Conv = nn.Conv2d(ch_count[0], out_ch,
                              kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d1 = self.active(out)

        return d1