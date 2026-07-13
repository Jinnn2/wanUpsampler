from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """LTX2-order residual block using Conv3d for Wan video latents."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_valid_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_valid_groups(channels), channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + residual)


class PixelShuffle(nn.Module):
    """Spatial pixel shuffle over H/W for [B, C, T, H, W] tensors."""

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        if upscale_factor < 1:
            raise ValueError("upscale_factor must be positive")
        self.upscale_factor = int(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, T, H, W], got {tuple(x.shape)}")

        r = self.upscale_factor
        batch, channels, frames, height, width = x.shape
        divisor = r * r
        if channels % divisor != 0:
            raise ValueError(f"channel count {channels} is not divisible by {divisor}")

        out_channels = channels // divisor
        x = x.reshape(batch, out_channels, r, r, frames, height, width)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return x.reshape(batch, out_channels, frames, height * r, width * r)


class BlurDownsample(nn.Module):
    """Fixed anti-aliased downsample over H/W only."""

    def __init__(self, stride: int, kernel_size: int = 5) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)

        coeff = torch.tensor([math.comb(kernel_size - 1, i) for i in range(kernel_size)])
        kernel = coeff[:, None] @ coeff[None, :]
        kernel = (kernel / kernel.sum()).float()
        self.register_buffer("kernel", kernel[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        if x.ndim != 5:
            raise ValueError(f"x must be [B, C, T, H, W], got {tuple(x.shape)}")

        batch, channels, frames, height, width = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        weight = self.kernel.expand(channels, 1, self.kernel_size, self.kernel_size)
        x2d = F.conv2d(
            x2d,
            weight=weight,
            bias=None,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=channels,
        )
        out_h, out_w = x2d.shape[-2:]
        return x2d.reshape(batch, frames, channels, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()


class SpatialRationalResampler(nn.Module):
    """Learned 3/2 spatial resampler: Conv3d expansion -> shuffle x3 -> blur /2."""

    def __init__(self, channels: int, scale: float = 1.5) -> None:
        super().__init__()
        if float(scale) != 1.5:
            raise ValueError("SpatialRationalResampler currently supports scale=1.5 only")
        self.channels = int(channels)
        self.scale = float(scale)
        self.num = 3
        self.den = 2
        self.conv = nn.Conv3d(channels, channels * self.num * self.num, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle(self.num)
        self.blur_down = BlurDownsample(stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.blur_down(x)


class SpatialIntegerResampler(nn.Module):
    """Learned integer spatial resampler: Conv3d expansion -> pixel shuffle."""

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        if int(scale) != scale or int(scale) < 1:
            raise ValueError("SpatialIntegerResampler requires a positive integer scale")
        self.channels = int(channels)
        self.scale = int(scale)
        self.conv = nn.Conv3d(
            channels,
            channels * self.scale * self.scale,
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = PixelShuffle(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.conv(x))


class WanCleanLatentResizerStage2(nn.Module):
    """LTX2-style Stage 2 clean latent resizer for Wan changing_resolution."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 256,
        num_res_blocks: int = 8,
        scale_factor: float = 1.5,
        dropout: float = 0.0,
        residual_skip: bool = False,
        resblock_type: str = "ltx2",
        resize_op: str = "rational_conv3d_pixel_shuffle",
    ) -> None:
        super().__init__()
        if num_res_blocks < 2:
            raise ValueError("num_res_blocks must be at least 2")
        if dropout != 0:
            raise ValueError("LTX2-style Stage 2 block does not support dropout")
        if resblock_type != "ltx2":
            raise ValueError("Stage 2 currently supports resblock_type='ltx2' only")

        valid_resize_configs = {
            "rational_conv3d_pixel_shuffle": 1.5,
            "conv3d_pixel_shuffle_crop": 2.0,
        }
        if resize_op not in valid_resize_configs:
            raise ValueError(
                "Stage 2 resize_op must be 'rational_conv3d_pixel_shuffle' or "
                "'conv3d_pixel_shuffle_crop'"
            )
        expected_scale = valid_resize_configs[resize_op]
        if float(scale_factor) != expected_scale:
            raise ValueError(
                f"Stage 2 resize_op={resize_op!r} requires scale_factor={expected_scale}, "
                f"got {scale_factor}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.scale_factor = float(scale_factor)
        self.residual_skip = bool(residual_skip) and in_channels == out_channels
        self.resblock_type = resblock_type
        self.resize_op = resize_op

        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(_valid_groups(hidden_channels), hidden_channels)
        self.initial_activation = nn.SiLU()

        pre_blocks = num_res_blocks // 2
        post_blocks = num_res_blocks - pre_blocks
        self.pre_blocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(pre_blocks)])
        if resize_op == "rational_conv3d_pixel_shuffle":
            self.feature_resizer = SpatialRationalResampler(hidden_channels, scale=scale_factor)
        else:
            self.feature_resizer = SpatialIntegerResampler(hidden_channels, scale=int(scale_factor))
        self.post_blocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(post_blocks)])
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
        h = self.initial_norm(h)
        h = self.initial_activation(h)
        for block in self.pre_blocks:
            h = block(h)

        h = self.feature_resizer(h)
        if self.resize_op == "conv3d_pixel_shuffle_crop":
            h = _center_crop_spatial(h, target_h, target_w)
        elif h.shape[-2:] != (target_h, target_w):
            raise ValueError(
                "Stage 2 rational resizer produced unexpected spatial size: "
                f"{tuple(h.shape[-2:])} vs {(target_h, target_w)}"
            )

        for block in self.post_blocks:
            h = block(h)

        residual = self.out(h)
        if not self.residual_skip:
            return residual

        skip = F.interpolate(z0_lr, size=target_size, mode="trilinear", align_corners=False)
        return skip + residual

    def _target_spatial_size(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | None,
    ) -> tuple[int, int]:
        expected = (
            int(round(z0_lr.shape[-2] * self.scale_factor)),
            int(round(z0_lr.shape[-1] * self.scale_factor)),
        )
        if output_size is None:
            return expected
        if len(output_size) != 2:
            raise ValueError("output_size must be (height, width)")
        target = (int(output_size[0]), int(output_size[1]))
        if self.resize_op == "rational_conv3d_pixel_shuffle" and target != expected:
            raise ValueError(f"Stage 2 rational path expects output_size={expected}, got {target}")
        if self.resize_op == "conv3d_pixel_shuffle_crop":
            if target[0] <= 0 or target[1] <= 0:
                raise ValueError(f"output_size must be positive, got {target}")
            if target[0] > expected[0] or target[1] > expected[1]:
                raise ValueError(
                    "Stage 2 pixel-shuffle crop path cannot enlarge beyond its native output: "
                    f"native={expected}, requested={target}"
                )
        return target


def _center_crop_spatial(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    if target_h > height or target_w > width:
        raise ValueError(
            f"cannot crop spatial size {(height, width)} to larger target {(target_h, target_w)}"
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return x[..., top : top + target_h, left : left + target_w]


def _valid_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while channels % groups != 0:
        groups -= 1
    return groups
