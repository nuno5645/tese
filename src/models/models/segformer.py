import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import SegmentationModel

class MixFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = x_.reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_).transpose(1, 2).reshape(B, C, H // self.sr_ratio, W // self.sr_ratio)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
            
        k, v = kv.chunk(2, dim=1)
        
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, C, H, W)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, expansion_factor=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, expansion_factor)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SegFormer(SegmentationModel):
    def __init__(self, in_channels=3, out_channels=24):
        super().__init__()
        
        # Encoder stages
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=4, padding=3),
            TransformerBlock(64, 1, 8),
            TransformerBlock(64, 1, 8),
            TransformerBlock(64, 1, 8)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            TransformerBlock(128, 2, 4),
            TransformerBlock(128, 2, 4),
            TransformerBlock(128, 2, 4)
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            TransformerBlock(256, 4, 2),
            TransformerBlock(256, 4, 2),
            TransformerBlock(256, 4, 2)
        )
        
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            TransformerBlock(512, 8, 1),
            TransformerBlock(512, 8, 1),
            TransformerBlock(512, 8, 1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(960, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.stage1(x)      # 1/4
        x2 = self.stage2(x1)     # 1/8
        x3 = self.stage3(x2)     # 1/16
        x4 = self.stage4(x3)     # 1/32
        
        # Multi-scale feature fusion
        x1 = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to original size
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)
        
        if not self.training:
            x = F.softmax(x, dim=1)
            
        return x
