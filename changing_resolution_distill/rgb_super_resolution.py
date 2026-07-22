from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


class RGBSuperResolver(Protocol):
    def resize(
        self, video: torch.Tensor, *, target_height: int, target_width: int
    ) -> torch.Tensor:
        """Resize [T,H,W,3] RGB float video in [0,1] on CPU."""


def center_crop_video(
    video: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,3], got {tuple(video.shape)}")
    height, width = int(video.shape[1]), int(video.shape[2])
    if height < target_height or width < target_width:
        raise ValueError(
            f"cannot crop {(height, width)} to {(target_height, target_width)}"
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return video[
        :, top : top + target_height, left : left + target_width, :
    ].contiguous()


@dataclass
class BicubicRGBSuperResolver:
    scale: float = 2.0

    def resize(
        self, video: torch.Tensor, *, target_height: int, target_width: int
    ) -> torch.Tensor:
        _validate_video(video)
        source_height, source_width = int(video.shape[1]), int(video.shape[2])
        scaled_height = int(round(source_height * self.scale))
        scaled_width = int(round(source_width * self.scale))
        frames = video.permute(0, 3, 1, 2).to(dtype=torch.float32)
        frames = F.interpolate(
            frames,
            size=(scaled_height, scaled_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        scaled = frames.permute(0, 2, 3, 1).clamp(0, 1)
        return center_crop_video(scaled, target_height, target_width)


class RealESRGANX2SuperResolver:
    """Frame-wise Real-ESRGAN x2, matching the RGB stage used by MrFlow.

    The official Real-ESRGAN Python API consumes BGR uint8 numpy arrays.  We
    deliberately process frames one at a time so the intermediate 81-frame
    736x1280 video does not occupy GPU memory at once.  The result is returned
    as CPU RGB float32 and center-cropped to Wan's patch-aligned 720x1248 target.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = True,
        gpu_id: int | None = 0,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Real-ESRGAN x2 checkpoint not found: {checkpoint_path}"
            )
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError(
                "RGB endpoint with backend=realesrgan requires the official "
                "Real-ESRGAN package and BasicSR on PYTHONPATH."
            ) from exc

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        self.upsampler = RealESRGANer(
            scale=2,
            model_path=str(checkpoint_path),
            model=model,
            tile=int(tile),
            tile_pad=int(tile_pad),
            pre_pad=int(pre_pad),
            half=bool(half),
            gpu_id=gpu_id,
        )

    def resize(
        self, video: torch.Tensor, *, target_height: int, target_width: int
    ) -> torch.Tensor:
        _validate_video(video)
        frames: list[torch.Tensor] = []
        for frame in video:
            rgb = frame.clamp(0, 1).mul(255.0).round().to(torch.uint8).numpy()
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            output_bgr, _ = self.upsampler.enhance(bgr, outscale=2.0)
            output_rgb = np.ascontiguousarray(output_bgr[..., ::-1])
            frames.append(torch.from_numpy(output_rgb).to(torch.float32).div_(255.0))
        scaled = torch.stack(frames, dim=0)
        return center_crop_video(scaled, target_height, target_width)


def build_rgb_super_resolver(config: dict) -> RGBSuperResolver:
    backend = str(config.get("wan_rgb_sr_backend", "realesrgan")).lower()
    if backend == "bicubic":
        return BicubicRGBSuperResolver(scale=float(config.get("wan_rgb_sr_scale", 2.0)))
    if backend != "realesrgan":
        raise ValueError("wan_rgb_sr_backend must be 'realesrgan' or 'bicubic'")
    checkpoint = config.get("wan_rgb_sr_checkpoint")
    if not checkpoint:
        raise ValueError("wan_rgb_sr_checkpoint is required for backend=realesrgan")
    return RealESRGANX2SuperResolver(
        checkpoint,
        tile=int(config.get("wan_rgb_sr_tile", 0)),
        tile_pad=int(config.get("wan_rgb_sr_tile_pad", 10)),
        pre_pad=int(config.get("wan_rgb_sr_pre_pad", 0)),
        half=bool(config.get("wan_rgb_sr_half", True)),
        gpu_id=config.get("wan_rgb_sr_gpu_id", 0),
    )


def _validate_video(video: torch.Tensor) -> None:
    if video.device.type != "cpu":
        raise ValueError("RGB super-resolution expects a CPU video tensor")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,3], got {tuple(video.shape)}")
    if not video.is_floating_point():
        raise ValueError("video must be floating point in [0,1]")
