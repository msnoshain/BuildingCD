import torch.nn as nn
import torch

class Conv2dTwice(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv2dTwice, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.init_channel_count=16
        
        self.conv1 = Conv2dTwice(in_channels, self.init_channel_count)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = Conv2dTwice(self.init_channel_count, self.init_channel_count*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = Conv2dTwice(self.init_channel_count*2, self.init_channel_count*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = Conv2dTwice(self.init_channel_count*4, self.init_channel_count*8)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = Conv2dTwice(self.init_channel_count*8, self.init_channel_count*16)
        
        self.up6 = nn.ConvTranspose2d(self.init_channel_count*16, self.init_channel_count*8, 2, stride=2)
        self.conv6 = Conv2dTwice(self.init_channel_count*16, self.init_channel_count*8)
        
        self.up7 = nn.ConvTranspose2d(self.init_channel_count*8, self.init_channel_count*4, 2, stride=2)
        self.conv7 = Conv2dTwice(self.init_channel_count*8, self.init_channel_count*4)
        
        self.up8 = nn.ConvTranspose2d(self.init_channel_count*4, self.init_channel_count*2, 2, stride=2)
        self.conv8 = Conv2dTwice(self.init_channel_count*4, self.init_channel_count*2)
        
        self.up9 = nn.ConvTranspose2d(self.init_channel_count*2, self.init_channel_count, 2, stride=2)
        self.conv9 = Conv2dTwice(self.init_channel_count*2, self.init_channel_count)
        
        self.conv10 = nn.Conv2d(self.init_channel_count, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out
