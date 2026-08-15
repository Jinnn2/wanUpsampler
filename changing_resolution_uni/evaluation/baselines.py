from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F


def resize_latent(
    latent: torch.Tensor,
    output_size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    """External spatial interpolation baseline; never used by U-ITU forward."""

    if latent.ndim != 5:
        raise ValueError(f"latent must be [B,C,T,H,W], got {tuple(latent.shape)}")
    target_h, target_w = (int(output_size[0]), int(output_size[1]))
    if mode == "nearest":
        return F.interpolate(
            latent,
            size=(latent.shape[2], target_h, target_w),
            mode="nearest",
        )
    if mode == "trilinear":
        return F.interpolate(
            latent,
            size=(latent.shape[2], target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
    if mode == "bicubic":
        batch, channels, frames, height, width = latent.shape
        flat = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        resized = F.interpolate(
            flat, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        return resized.reshape(batch, frames, channels, target_h, target_w).permute(
            0, 2, 1, 3, 4
        )
    raise ValueError(f"Unsupported interpolation baseline: {mode!r}")


def make_interpolation_runner(
    mode: str,
) -> Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]:
    return lambda latent, output_size: resize_latent(latent, output_size, mode=mode)


def load_specialist(
    checkpoint_path: str,
    *,
    device: torch.device,
    train_config_path: str | None = None,
    use_ema: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load an optional legacy fixed Stage2 specialist with its native factory."""

    from wan_sr.models import build_clean_latent_resizer
    from wan_sr.training.config import load_yaml
    from wan_sr.training.ema import EMA

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    fallback = load_yaml(train_config_path) if train_config_path else {}
    model_config = payload.get("config", {}).get("model", fallback.get("model", {}))
    if not model_config:
        raise ValueError(
            "Specialist checkpoint does not contain model config; pass --specialist_config"
        )
    model = build_clean_latent_resizer(model_config).to(device)
    model.load_state_dict(payload.get("model", payload), strict=True)
    if use_ema:
        if "ema" not in payload:
            raise ValueError(
                "--specialist_use_ema requested but specialist checkpoint has no EMA state"
            )
        ema = EMA(model)
        ema.load_state_dict(payload["ema"])
        ema.copy_to(model)
    model.eval()
    return model, payload
