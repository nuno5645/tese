import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import SegmentationModel

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Resize g to match x's spatial dimensions
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual(x)
        out = self.double_conv(x)
        return self.relu(out + identity)

class EnhancedUNet(SegmentationModel):
    def __init__(self, in_channels=3, out_channels=24):
        super().__init__()
        
        # Save parameters
        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'model_type': 'enhanced_unet',
            'description': 'Enhanced U-Net with attention gates, residual connections, and deep supervision'
        }
        
        # Encoder with increased capacity
        self.conv1 = ResidualDoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ResidualDoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ResidualDoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ResidualDoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = ResidualDoubleConv(512, 1024)
        
        # Attention Gates
        self.attention4 = AttentionGate(1024, 512, 256)
        self.attention3 = AttentionGate(512, 256, 128)
        self.attention2 = AttentionGate(256, 128, 64)
        self.attention1 = AttentionGate(128, 64, 32)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = ResidualDoubleConv(1024, 512)  # 512 + 512 from skip connection
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = ResidualDoubleConv(512, 256)   # 256 + 256 from skip connection
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = ResidualDoubleConv(256, 128)   # 128 + 128 from skip connection
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = ResidualDoubleConv(128, 64)    # 64 + 64 from skip connection
        
        # Output layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # Deep supervision outputs
        self.deep_sup4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.deep_sup3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(128, out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        p1 = self.dropout(p1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        p2 = self.dropout(p2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        p3 = self.dropout(p3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        p4 = self.dropout(p4)
        
        c5 = self.conv5(p4)
        
        # Decoder with attention and deep supervision
        u4 = self.up1(c5)
        a4 = self.attention4(c5, c4)
        u4 = torch.cat([u4, a4], dim=1)
        c6 = self.conv6(u4)
        ds4 = self.deep_sup4(c6)
        
        u3 = self.up2(c6)
        a3 = self.attention3(c6, c3)
        u3 = torch.cat([u3, a3], dim=1)
        c7 = self.conv7(u3)
        ds3 = self.deep_sup3(c7)
        
        u2 = self.up3(c7)
        a2 = self.attention2(c7, c2)
        u2 = torch.cat([u2, a2], dim=1)
        c8 = self.conv8(u2)
        ds2 = self.deep_sup2(c8)
        
        u1 = self.up4(c8)
        a1 = self.attention1(c8, c1)
        u1 = torch.cat([u1, a1], dim=1)
        c9 = self.conv9(u1)
        
        outputs = self.final_conv(c9)
        
        if self.training:
            # During training return deep supervision outputs
            return outputs, F.interpolate(ds4, size=outputs.shape[2:]), \
                   F.interpolate(ds3, size=outputs.shape[2:]), \
                   F.interpolate(ds2, size=outputs.shape[2:])
        else:
            # During inference return only the final output
            return F.softmax(outputs, dim=1)
            
    def get_parameters(self):
        """Return model parameters for logging"""
        return self.params
