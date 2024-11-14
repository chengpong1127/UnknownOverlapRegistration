import torch
from torch import nn
import torchsparse
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.nn import functional as spf
from .residual_block import ResidualBlock


class ResUNet(nn.Module):
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]
    
    def __init__(self, 
        in_channels=3, 
        out_channels=32, 
        bn_momentum=0.1,
        normalize_feature=False,
        conv1_kernel_size=3,
    ):
        super(ResUNet, self).__init__()
        
        self.normalize_feature = normalize_feature
        self.conv1 = spnn.Conv3d(in_channels, self.CHANNELS[1], kernel_size=conv1_kernel_size, stride=1, dilation=1, bias=False)
        self.norm1 = spnn.BatchNorm(self.CHANNELS[1], momentum=bn_momentum)
        self.block1 = ResidualBlock(self.CHANNELS[1], self.CHANNELS[1], bn_momentum=bn_momentum)
    
        self.conv2 = spnn.Conv3d(self.CHANNELS[1], self.CHANNELS[2], kernel_size=3, stride=1, dilation=1, bias=False)
        self.norm2 = spnn.BatchNorm(self.CHANNELS[2], momentum=bn_momentum)
        self.block2 = ResidualBlock(self.CHANNELS[2], self.CHANNELS[2], bn_momentum=bn_momentum)
        
        self.conv3 = spnn.Conv3d(self.CHANNELS[2], self.CHANNELS[3], kernel_size=3, stride=1, dilation=1, bias=False)
        self.norm3 = spnn.BatchNorm(self.CHANNELS[3], momentum=bn_momentum)
        self.block3 = ResidualBlock(self.CHANNELS[3], self.CHANNELS[3], bn_momentum=bn_momentum)
        
        self.conv4 = spnn.Conv3d(self.CHANNELS[3], self.CHANNELS[4], kernel_size=3, stride=1, dilation=1, bias=False)
        self.norm4 = spnn.BatchNorm(self.CHANNELS[4], momentum=bn_momentum)
        self.block4 = ResidualBlock(self.CHANNELS[4], self.CHANNELS[4], bn_momentum=bn_momentum)
        
        self.conv4_tr = spnn.Conv3d(self.CHANNELS[4], self.TR_CHANNELS[4], kernel_size=3, stride=1, dilation=1, bias=False, transposed=True)
        self.norm4_tr = spnn.BatchNorm(self.TR_CHANNELS[4], momentum=bn_momentum)
        self.block4_tr = ResidualBlock(self.TR_CHANNELS[4], self.TR_CHANNELS[4], bn_momentum=bn_momentum)
        
        self.conv3_tr = spnn.Conv3d(self.CHANNELS[3] + self.TR_CHANNELS[4], self.TR_CHANNELS[3], kernel_size=3, stride=1, dilation=1, bias=False, transposed=True)
        self.norm3_tr = spnn.BatchNorm(self.TR_CHANNELS[3], momentum=bn_momentum)
        self.block3_tr = ResidualBlock(self.TR_CHANNELS[3], self.TR_CHANNELS[3], bn_momentum=bn_momentum)
        
        self.conv2_tr = spnn.Conv3d(self.CHANNELS[2] + self.TR_CHANNELS[3], self.TR_CHANNELS[2], kernel_size=3, stride=1, dilation=1, bias=False, transposed=True)
        self.norm2_tr = spnn.BatchNorm(self.TR_CHANNELS[2], momentum=bn_momentum)
        self.block2_tr = ResidualBlock(self.TR_CHANNELS[2], self.TR_CHANNELS[2], bn_momentum=bn_momentum)
        
        self.conv1_tr = spnn.Conv3d(self.CHANNELS[1] + self.TR_CHANNELS[2], self.TR_CHANNELS[1], kernel_size=3, stride=1, dilation=1, bias=False, transposed=True)
        self.final = spnn.Conv3d(self.TR_CHANNELS[1], out_channels, kernel_size=1, stride=1, dilation=1, bias=True)

        
    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = spf.relu(out_s1)
        
        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = spf.relu(out_s2)
        
        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = spf.relu(out_s4)
        
        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = spf.relu(out_s8)
        
        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = spf.relu(out)
        
        out = torchsparse.cat([out_s4_tr, out_s4])
        
        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = spf.relu(out)
        
        out = torchsparse.cat([out_s2_tr, out_s2])
        
        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = spf.relu(out)
        
        out = torchsparse.cat([out_s1_tr, out_s1])
        out = self.conv1_tr(out)
        out = spf.relu(out)
        out = self.final(out)
        
        if self.normalize_feature:
            out = SparseTensor(
                coords=out.coords, 
                feats=out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
                stride=out.stride)
        
        return out
        
        