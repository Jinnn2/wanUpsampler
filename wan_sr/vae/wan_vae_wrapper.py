from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class WanVAEWrapper:
    """Thin Wan VAE adapter.

    This wrapper first tries diffusers' AutoencoderKLWan. If your training
    environment uses the official Wan repository instead, keep the public
    encode/decode methods here and replace only the private backend calls.
    """

    def __init__(
        self,
        model_root: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_root = Path(model_root)
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend = self._load_diffusers_backend()
        self.backend.eval()
        for param in self.backend.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode [T,H,W,C] or [B,T,H,W,C] float video in [0,1] to [B,C,T,H,W]."""

        batched = video.ndim == 5
        if not batched:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"video must be [T,H,W,C] or [B,T,H,W,C], got {tuple(video.shape)}")

        x = video.to(device=self.device, dtype=self.dtype)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x * 2.0 - 1.0

        encoded = self.backend.encode(x)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.mode()
        elif hasattr(encoded, "latents"):
            latents = encoded.latents
        elif torch.is_tensor(encoded):
            latents = encoded
        else:
            raise TypeError(f"Unsupported VAE encode output type: {type(encoded)!r}")

        scaling = getattr(getattr(self.backend, "config", object()), "scaling_factor", None)
        if scaling is not None:
            latents = latents * float(scaling)
        return latents.float().cpu() if not batched else latents.float().cpu()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [C,T,H,W] or [B,C,T,H,W] latents to [B,T,H,W,C] in [0,1]."""

        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"latents must be [C,T,H,W] or [B,C,T,H,W], got {tuple(latents.shape)}")

        z = latents.to(device=self.device, dtype=self.dtype)
        scaling = getattr(getattr(self.backend, "config", object()), "scaling_factor", None)
        if scaling is not None:
            z = z / float(scaling)
        decoded = self.backend.decode(z)
        if hasattr(decoded, "sample"):
            video = decoded.sample
        elif torch.is_tensor(decoded):
            video = decoded
        else:
            raise TypeError(f"Unsupported VAE decode output type: {type(decoded)!r}")
        video = ((video.float() + 1.0) / 2.0).clamp(0, 1)
        return video.permute(0, 2, 3, 4, 1).cpu()

    def _load_diffusers_backend(self) -> Any:
        try:
            from diffusers import AutoencoderKLWan
        except Exception as exc:
            raise ImportError(
                "diffusers with AutoencoderKLWan is required for the default VAE wrapper. "
                "Install requirements.txt or adapt wan_sr/vae/wan_vae_wrapper.py to your Wan checkout."
            ) from exc

        errors: list[str] = []
        for kwargs in ({"subfolder": "vae"}, {}):
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    str(self.model_root),
                    torch_dtype=self.dtype,
                    **kwargs,
                )
                return vae.to(self.device)
            except Exception as exc:
                errors.append(f"kwargs={kwargs}: {exc}")
        joined = "\n".join(errors)
        raise RuntimeError(f"Failed to load AutoencoderKLWan from {self.model_root}:\n{joined}")
