from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import UniversalCleanLatentUpsampler


def load_universal_upsampler(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> tuple[UniversalCleanLatentUpsampler, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("config", {})
    model_config = dict(config.get("model", payload.get("model_config", {})))
    model = UniversalCleanLatentUpsampler(**model_config).to(device)
    state = payload.get("model", payload)
    if use_ema and isinstance(payload.get("ema"), dict):
        shadow = payload["ema"].get("shadow")
        if isinstance(shadow, dict):
            state = dict(state)
            state.update(shadow)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def upsample_clean_latent(
    model: UniversalCleanLatentUpsampler,
    latent: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Upsample CTHW or BCTHW clean latents without changing time."""

    unbatched = latent.ndim == 4
    if unbatched:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(f"latent must be [C,T,H,W] or [B,C,T,H,W], got {tuple(latent.shape)}")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prediction = model(latent.to(device=device, dtype=dtype), output_size=output_size)
    return prediction.squeeze(0) if unbatched else prediction
