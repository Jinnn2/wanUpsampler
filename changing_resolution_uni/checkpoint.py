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
    # Keep optimizer/EMA tensors on CPU while constructing the inference model.
    # Mapping the whole training checkpoint to CUDA can OOM before evaluation.
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_universal_upsampler_from_payload(
        payload, device=device, use_ema=use_ema
    )
    return model, payload


def build_universal_upsampler_from_payload(
    payload: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> UniversalCleanLatentUpsampler:
    """Build raw or EMA inference weights from an already loaded checkpoint."""

    config = payload.get("config", {})
    model_config = dict(config.get("model", payload.get("model_config", {})))
    model = UniversalCleanLatentUpsampler(**model_config)
    state = payload.get("model", payload)
    if use_ema and isinstance(payload.get("ema"), dict):
        shadow = payload["ema"].get("shadow")
        if isinstance(shadow, dict):
            state = dict(state)
            state.update(shadow)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


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
        raise ValueError(
            f"latent must be [C,T,H,W] or [B,C,T,H,W], got {tuple(latent.shape)}"
        )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prediction = model(latent.to(device=device, dtype=dtype), output_size=output_size)
    return prediction.squeeze(0) if unbatched else prediction
