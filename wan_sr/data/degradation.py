from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def degrade_video(
    x_hr: torch.Tensor,
    lr_size: tuple[int, int],
    blur_prob: float = 0.5,
    noise_prob: float = 0.5,
    jpeg_prob: float = 0.5,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Create an LR video from an HR clip.

    Args:
        x_hr: [T, H, W, C] float tensor in [0, 1].
        lr_size: (height, width).
    """

    if x_hr.ndim != 4 or x_hr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"x_hr must be [T,H,W,C], got {tuple(x_hr.shape)}")
    x = x_hr.clamp(0, 1)
    meta: dict[str, object] = {}

    if random.random() < blur_prob:
        sigma = random.uniform(0.2, 1.2)
        x = gaussian_blur_video(x, sigma=sigma)
        meta["blur_sigma"] = sigma
    else:
        meta["blur_sigma"] = 0.0

    kernel = random.choice(["bicubic", "bilinear", "area"])
    x = resize_video(x, lr_size, mode=kernel)
    meta["resize_kernel"] = kernel

    if random.random() < noise_prob:
        noise_std = random.uniform(0.0, 0.02)
        x = (x + torch.randn_like(x) * noise_std).clamp(0, 1)
        meta["noise_std"] = noise_std
    else:
        meta["noise_std"] = 0.0

    if random.random() < jpeg_prob:
        quality = random.randint(70, 95)
        x = jpeg_roundtrip_video(x, quality=quality)
        meta["jpeg_quality"] = quality
    else:
        meta["jpeg_quality"] = None

    return x.clamp(0, 1), meta


def resize_video(x: torch.Tensor, size: tuple[int, int], mode: str = "bicubic") -> torch.Tensor:
    t, h, w, c = x.shape
    frames = x.permute(0, 3, 1, 2)
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    out = F.interpolate(frames, size=size, mode=mode, **kwargs)
    return out.permute(0, 2, 3, 1)


def center_crop_resize_video(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = size
    _, h, w, _ = x.shape
    src_ratio = w / h
    dst_ratio = target_w / target_h
    if src_ratio > dst_ratio:
        crop_w = int(h * dst_ratio)
        left = (w - crop_w) // 2
        x = x[:, :, left : left + crop_w]
    else:
        crop_h = int(w / dst_ratio)
        top = (h - crop_h) // 2
        x = x[:, top : top + crop_h]
    return resize_video(x, size, mode="bicubic")


def gaussian_blur_video(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = max(3, int(sigma * 6) | 1)
    radius = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - radius
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]

    t, h, w, c = x.shape
    frames = x.permute(0, 3, 1, 2)
    weight = kernel_2d.expand(c, 1, kernel_size, kernel_size)
    blurred = F.conv2d(frames, weight, padding=radius, groups=c)
    return blurred.permute(0, 2, 3, 1)


def jpeg_roundtrip_video(x: torch.Tensor, quality: int) -> torch.Tensor:
    frames = []
    for frame in x.detach().cpu():
        array = (frame.clamp(0, 1).numpy() * 255.0).round().astype("uint8")
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        restored = Image.open(buffer).convert("RGB")
        frames.append(torch.from_numpy(np.array(restored)).float() / 255.0)
    return torch.stack(frames, dim=0).to(device=x.device, dtype=x.dtype)
