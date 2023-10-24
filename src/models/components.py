import torch.nn as nn                

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        # print("residual",residual.shape)
        out = self.cv1(x)
        # print("cv1",out.shape)
        out = self.bn1(out)
        # print("bn1",out.shape)
        out = self.lr1(out)
        # print("lr1",out.shape)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.maxp(out)
        return out


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, middle_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.tcv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        # print("residual",residual.shape)
        out = self.cv1(x)
        # print("cv1",out.shape)
        out = self.bn1(out)
        # print("bn1",out.shape)
        out = self.lr1(out)
        # print("lr1",out.shape)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.tcv(out)
        return out


class _CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_CenterBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.tcv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.lr1(out)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.tcv(out)
        return out

