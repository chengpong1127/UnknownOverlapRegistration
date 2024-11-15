from torch import nn
from torchsparse import nn as spnn
from torchsparse.nn import functional as spf

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = spnn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, bias=False)
        self.bn1 = spnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.conv2 = spnn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation, bias=False)
        self.bn2 = spnn.BatchNorm(out_channels, momentum=bn_momentum)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = spf.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = spf.relu(out)
        return out