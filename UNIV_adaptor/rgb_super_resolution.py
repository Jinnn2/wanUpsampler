from __future__ import annotations

from dataclasses import dataclass


def cover_scale(
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> float:
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source and target sizes must be positive")
    return max(target_height / source_height, target_width / source_width)


def _validate_scale(scale: float) -> None:
    if not 1.0 <= scale <= 2.0 + 1e-6:
        raise ValueError(
            "the Real-ESRGAN x2 transition requires a realized scale in [1, 2], "
            f"got {scale}"
        )


@dataclass
class AdaptiveBicubicSuperResolver:
    def resize(self, video, *, target_height: int, target_width: int):
        import torch.nn.functional as functional

        from changing_resolution_distill.rgb_super_resolution import (
            _validate_video,
            center_crop_video,
        )

        _validate_video(video)
        source_height, source_width = int(video.shape[1]), int(video.shape[2])
        scale = cover_scale(
            source_height, source_width, target_height, target_width
        )
        _validate_scale(scale)
        scaled_height = max(target_height, int(round(source_height * scale)))
        scaled_width = max(target_width, int(round(source_width * scale)))
        frames = video.permute(0, 3, 1, 2).to(dtype=video.dtype)
        frames = functional.interpolate(
            frames,
            size=(scaled_height, scaled_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return center_crop_video(
            frames.permute(0, 2, 3, 1).clamp(0, 1),
            target_height,
            target_width,
        )


class AdaptiveRealESRGANX2SuperResolver:
    """Use the x2 network with an action-dependent output scale.

    Real-ESRGAN performs its learned x2 pass internally, then its official API
    resizes to ``outscale``. Choosing the minimum scale that covers the target
    preserves the full field of view for every allowed UNIV spatial ratio.
    """

    def __init__(self, base_resolver) -> None:
        self.base_resolver = base_resolver

    def resize(self, video, *, target_height: int, target_width: int):
        import numpy as np
        import torch

        from changing_resolution_distill.rgb_super_resolution import (
            _validate_video,
            center_crop_video,
        )

        _validate_video(video)
        source_height, source_width = int(video.shape[1]), int(video.shape[2])
        scale = cover_scale(
            source_height, source_width, target_height, target_width
        )
        _validate_scale(scale)
        frames: list[torch.Tensor] = []
        for frame in video:
            rgb = frame.clamp(0, 1).mul(255.0).round().to(torch.uint8).numpy()
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            output_bgr, _ = self.base_resolver.upsampler.enhance(
                bgr,
                outscale=scale,
            )
            output_rgb = np.ascontiguousarray(output_bgr[..., ::-1])
            frames.append(torch.from_numpy(output_rgb).to(torch.float32).div_(255.0))
        scaled = torch.stack(frames, dim=0)
        return center_crop_video(scaled, target_height, target_width)


def build_univ_rgb_super_resolver(config):
    backend = str(config.get("wan_rgb_sr_backend", "realesrgan")).lower()
    if backend == "bicubic":
        return AdaptiveBicubicSuperResolver()
    if backend != "realesrgan":
        raise ValueError("wan_rgb_sr_backend must be 'realesrgan' or 'bicubic'")

    from changing_resolution_distill.rgb_super_resolution import (
        RealESRGANX2SuperResolver,
    )

    checkpoint = config.get("wan_rgb_sr_checkpoint")
    if not checkpoint:
        raise ValueError("wan_rgb_sr_checkpoint is required for backend=realesrgan")
    base = RealESRGANX2SuperResolver(
        checkpoint,
        tile=int(config.get("wan_rgb_sr_tile", 0)),
        tile_pad=int(config.get("wan_rgb_sr_tile_pad", 10)),
        pre_pad=int(config.get("wan_rgb_sr_pre_pad", 0)),
        half=bool(config.get("wan_rgb_sr_half", True)),
        gpu_id=config.get("wan_rgb_sr_gpu_id", 0),
    )
    return AdaptiveRealESRGANX2SuperResolver(base)
