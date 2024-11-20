import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import SegmentationModel

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Atrous Spatial Pyramid Pooling
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        x5 = self.avg_pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class DeepLabV3(SegmentationModel):
    def __init__(self, in_channels=3, out_channels=24):
        super().__init__()
        
        # Save parameters
        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'model_type': 'deeplabv3',
            'description': 'DeepLabV3 with ASPP and ResNet-like backbone'
        }
        
        # Encoder (ResNet-like)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # ASPP
        self.aspp = ASPP(512, 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, out_channels, 1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        if not self.training:
            x = F.softmax(x, dim=1)
            
        return x
        
    def get_parameters(self):
        """Return model parameters for logging"""
        return self.params
