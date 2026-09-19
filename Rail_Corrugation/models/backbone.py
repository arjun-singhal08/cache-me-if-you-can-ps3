"""
Model backbone components for rail corrugation detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleConv(nn.Module):
    """Parallel multi-scale convolution branches."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15, 31]
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ) for k in kernel_sizes
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(branch_outs, dim=1))


class DilatedResidualBlock(nn.Module):
    """Residual block with dilated convolutions for long-range context."""
    
    def __init__(self, channels: int, dilation: int, use_se: bool = True):
        super().__init__()
        padding = dilation  # For kernel=3
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        return self.relu(out)


class AttentionPool(nn.Module):
    """Learnable attention pooling over temporal dimension."""
    
    def __init__(self, channels: int, attn_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attn_dim, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        weights = self.attn(x)  # [B, 1, T]
        weights = F.softmax(weights, dim=2)
        return (x * weights).sum(dim=2)  # [B, C]


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN backbone with shared weights for both sides.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_ch = config.model.in_channels
        base_ch = config.model.base_channels
        kernel_sizes = config.model.kernel_sizes
        use_se = config.model.use_se
        num_blocks = config.model.num_residual_blocks
        attn_dim = config.model.attention_dim
        
        # Initial multi-scale conv
        self.stem = MultiScaleConv(in_ch, base_ch, kernel_sizes)
        
        # Downsampling stages
        self.stage1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        
        # Dilated residual blocks for long-range dependencies
        dilations = [1, 2, 4, 8, 16, 1]  # Cycle through
        self.res_blocks = nn.ModuleList([
            DilatedResidualBlock(base_ch * 2, dilations[i % len(dilations)], use_se)
            for i in range(num_blocks)
        ])
        
        # Second downsampling
        self.stage2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        
        # More residual blocks
        self.res_blocks2 = nn.ModuleList([
            DilatedResidualBlock(base_ch * 4, dilations[i % len(dilations)], use_se)
            for i in range(num_blocks)
        ])
        
        # Attention pooling
        self.attention_pool = AttentionPool(base_ch * 4, attn_dim)
        
        # Final embedding projection
        self.embed_proj = nn.Sequential(
            nn.Linear(base_ch * 4, attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.stem(x)
        x = self.stage1(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.stage2(x)
        
        for block in self.res_blocks2:
            x = block(x)
        
        # Attention pooling
        x = self.attention_pool(x)  # [B, C]
        
        # Project to embedding
        x = self.embed_proj(x)  # [B, attn_dim]
        
        return x