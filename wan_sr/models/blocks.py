import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .sigma_embedding import AdaGroupNorm3D


class SigmaConditionedResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm3D(channels, cond_dim)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm3D(channels, cond_dim)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        return x + h


class SpatialPixelShuffle2x(nn.Module):
    """2x pixel shuffle over H/W for tensors shaped [B, C, T, H, W]."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels, channels * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return rearrange(x, "b (c r1 r2) t h w -> b c t (h r1) (w r2)", r1=2, r2=2)


class ConvInOut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
