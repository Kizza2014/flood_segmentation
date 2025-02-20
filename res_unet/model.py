import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from .components import OutConv, DoubleConv, Up, UpResNet

class ResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_type="resnet18", bilinear=False, dropout_rate = 0.1):
        super(ResNetUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_type = resnet_type
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        # Define the backbone network
        if self.resnet_type == "resnet18":
            self.backbone_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        elif self.resnet_type == "resnet34":
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        elif self.resnet_type == "resnet50":
            self.channel_distribution = [3, 64, 256, 512, 1024]
            self.backbone_model = torchvision.models.resnet50(weights="DEFAULT")
        else:
            print("Resnet type is not recognized. Loading ResNet 18 as backbone model")
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        
        self.backbone_layers = list(self.backbone_model.children())
        
        # Define the ResNetUNet
        self.inc = DoubleConv(in_channels, 64)
        
        self.block1 = nn.Sequential(*self.backbone_layers[0:3])
        self.block2 = nn.Sequential(*self.backbone_layers[3:5])
        self.block3 = nn.Sequential(*self.backbone_layers[5])
        self.block4 = nn.Sequential(*self.backbone_layers[6])
        
        self.up1 = Up(self.channel_distribution[-1], self.channel_distribution[-2], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up2 = Up(self.channel_distribution[-2], self.channel_distribution[-3], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up3 = UpResNet(self.channel_distribution[-3], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up4 = UpResNet(self.channel_distribution[-4], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        y1 = self.up1(x4, x3)
        y2 = self.up2(x3, x2)
        y3 = self.up3(x2, x1)
        y4 = self.up4(x1, x0)

        logits = self.outc(y4)
        
        return logits