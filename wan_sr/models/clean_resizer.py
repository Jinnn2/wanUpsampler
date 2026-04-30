from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PlainResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_valid_groups(channels), channels)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class WanCleanLatentResizer(nn.Module):
    """Clean-latent spatial resizer for LightX2V changing_resolution.

    Inputs and outputs are [B, C, T, H, W]. Only H/W are resized.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 256,
        num_res_blocks: int = 8,
        scale_factor: float = 1.5,
        dropout: float = 0.0,
        residual_skip: bool = True,
    ) -> None:
        super().__init__()
        if num_res_blocks < 2:
            raise ValueError("num_res_blocks must be at least 2")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = float(scale_factor)
        self.residual_skip = residual_skip and in_channels == out_channels

        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        pre_blocks = num_res_blocks // 2
        post_blocks = num_res_blocks - pre_blocks
        self.pre_blocks = nn.ModuleList(
            [PlainResBlock3D(hidden_channels, dropout=dropout) for _ in range(pre_blocks)]
        )
        self.post_blocks = nn.ModuleList(
            [PlainResBlock3D(hidden_channels, dropout=dropout) for _ in range(post_blocks)]
        )
        self.out_norm = nn.GroupNorm(_valid_groups(hidden_channels), hidden_channels)
        self.out = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if z0_lr.ndim != 5:
            raise ValueError(f"z0_lr must be [B, C, T, H, W], got {tuple(z0_lr.shape)}")
        if z0_lr.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {z0_lr.shape[1]}")

        target_h, target_w = self._target_spatial_size(z0_lr, output_size)
        target_size = (z0_lr.shape[2], target_h, target_w)

        h = self.stem(z0_lr)
        for block in self.pre_blocks:
            h = block(h)
        h = F.interpolate(h, size=target_size, mode="trilinear", align_corners=False)
        for block in self.post_blocks:
            h = block(h)
        residual = self.out(F.silu(self.out_norm(h)))

        if not self.residual_skip:
            return residual

        skip = F.interpolate(z0_lr, size=target_size, mode="trilinear", align_corners=False)
        return skip + residual

    def _target_spatial_size(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if output_size is not None:
            if len(output_size) != 2:
                raise ValueError("output_size must be (height, width)")
            return int(output_size[0]), int(output_size[1])
        return (
            int(round(z0_lr.shape[-2] * self.scale_factor)),
            int(round(z0_lr.shape[-1] * self.scale_factor)),
        )


def _valid_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while channels % groups != 0:
        groups -= 1
    return groups
