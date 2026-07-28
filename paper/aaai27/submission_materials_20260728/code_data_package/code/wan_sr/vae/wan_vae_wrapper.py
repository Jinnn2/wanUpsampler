from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


class WanVAEWrapper:
    """Thin Wan VAE adapter.

    Supported backends:
    - official: ``wan.modules.vae.WanVAE``
    - lightx2v: ``lightx2v.models.video_encoders.hf.wan.vae.WanVAE``
    - diffusers: ``diffusers.AutoencoderKLWan``

    Official Wan checkpoint directories contain DiT config/weights plus a
    separate ``Wan2.1_VAE.pth`` file. A diffusers-converted checkpoint instead
    has a ``vae/config.json`` subfolder. The wrapper keeps those cases separate
    so the DiT root is never loaded as an AutoencoderKLWan.
    """

    def __init__(
        self,
        model_root: str | Path,
        vae_path: str | Path | None = None,
        wan_repo: str | Path | None = None,
        backend: str = "auto",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_root = Path(model_root)
        self.vae_path = Path(vae_path) if vae_path is not None else None
        self.wan_repo = Path(wan_repo) if wan_repo is not None else None
        self.backend_name = backend
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend, self.backend_kind = self._load_backend()

        module = getattr(self.backend, "model", self.backend)
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "parameters"):
            for param in module.parameters():
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

        if self.backend_kind == "official":
            latents = self.backend.encode([sample for sample in x])
            return torch.stack([latent.float().cpu() for latent in latents], dim=0)
        if self.backend_kind == "lightx2v":
            latents = [self.backend.encode(sample.unsqueeze(0)).float().cpu() for sample in x]
            return torch.stack(latents, dim=0)

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
        if self.backend_kind == "official":
            decoded = self.backend.decode([sample for sample in z])
            video = torch.stack([sample.float().cpu() for sample in decoded], dim=0)
            return ((video.permute(0, 2, 3, 4, 1) + 1.0) / 2.0).clamp(0, 1)
        if self.backend_kind == "lightx2v":
            decoded = [self.backend.decode(sample).squeeze(0).float().cpu() for sample in z]
            video = torch.stack(decoded, dim=0)
            return ((video.permute(0, 2, 3, 4, 1) + 1.0) / 2.0).clamp(0, 1)

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

    def _load_backend(self) -> tuple[Any, str]:
        if self.backend_name not in {"auto", "official", "lightx2v", "diffusers"}:
            raise ValueError("backend must be one of: auto, official, lightx2v, diffusers")

        official_errors: list[str] = []
        vae_path = self._resolve_official_vae_path()
        if self.backend_name in {"auto", "official"} and vae_path is not None:
            try:
                return self._load_official_backend(vae_path), "official"
            except Exception as exc:
                official_errors.append(str(exc))
                if self.backend_name == "official":
                    raise

        if self.backend_name in {"auto", "lightx2v"} and vae_path is not None:
            try:
                return self._load_lightx2v_backend(vae_path), "lightx2v"
            except Exception as exc:
                official_errors.append(str(exc))
                if self.backend_name == "lightx2v":
                    raise

        if self.backend_name in {"auto", "diffusers"}:
            diffusers_target = self._resolve_diffusers_vae_target()
            if diffusers_target is not None:
                try:
                    return self._load_diffusers_backend(diffusers_target), "diffusers"
                except Exception as exc:
                    if self.backend_name == "diffusers":
                        raise
                    official_errors.append(str(exc))

        hints = [
            f"model_root={self.model_root}",
            "No usable Wan VAE backend was found.",
            "For official Wan checkpoints, pass --vae_path /path/to/Wan2.1_VAE.pth "
            "and make sure the Wan repo is importable, or pass --wan_repo /path/to/Wan2.1.",
            "For LightX2V, pass --vae_backend lightx2v and --wan_repo /path/to/LightX2V.",
            "For diffusers checkpoints, pass --model_root to a diffusers-converted model "
            "that contains vae/config.json.",
        ]
        if official_errors:
            hints.append("Backend errors:\n" + "\n".join(official_errors))
        raise RuntimeError("\n".join(hints))

    def _resolve_official_vae_path(self) -> Path | None:
        candidates = []
        if self.vae_path is not None:
            candidates.append(self.vae_path)
        candidates.extend(
            [
                self.model_root / "Wan2.1_VAE.pth",
                self.model_root / "vae" / "Wan2.1_VAE.pth",
            ]
        )
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_official_backend(self, vae_path: Path) -> Any:
        repo = self.wan_repo or os.environ.get("WAN_REPO") or os.environ.get("WAN_CODE_DIR")
        if repo is not None:
            repo_path = Path(repo)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
        try:
            from wan.modules.vae import WanVAE
        except Exception as exc:
            raise ImportError(
                "Found official Wan VAE weights, but could not import wan.modules.vae.WanVAE. "
                "Run from the Wan2.1 repo, install it on PYTHONPATH, set WAN_REPO, "
                "or pass --wan_repo /path/to/Wan2.1."
            ) from exc
        return WanVAE(vae_pth=str(vae_path), dtype=self.dtype, device=str(self.device))

    def _load_lightx2v_backend(self, vae_path: Path) -> Any:
        repo = self.wan_repo or os.environ.get("LIGHTX2V_REPO") or os.environ.get("WAN_REPO")
        if repo is not None:
            repo_path = Path(repo)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
        try:
            from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
        except Exception as exc:
            raise ImportError(
                "Found VAE weights, but could not import LightX2V WanVAE. "
                "Install LightX2V, set LIGHTX2V_REPO, or pass --wan_repo /path/to/LightX2V."
            ) from exc
        return WanVAE(
            vae_path=str(vae_path),
            dtype=self.dtype,
            device=str(self.device),
            parallel=False,
            use_tiling=False,
            cpu_offload=False,
        )

    def _resolve_diffusers_vae_target(self) -> tuple[Path, str | None] | None:
        subfolder_config = self.model_root / "vae" / "config.json"
        if subfolder_config.exists():
            return self.model_root, "vae"

        root_config = self.model_root / "config.json"
        if not root_config.exists():
            return None

        with root_config.open("r", encoding="utf-8") as f:
            config = json.load(f)
        class_name = str(config.get("_class_name", ""))
        has_vae_keys = "AutoencoderKLWan" in class_name or "latent_channels" in config
        is_wan_transformer = config.get("model_type") in {"t2v", "i2v", "ti2v"} or "num_layers" in config
        if has_vae_keys and not is_wan_transformer:
            return self.model_root, None
        return None

    def _load_diffusers_backend(self, target: tuple[Path, str | None]) -> Any:
        try:
            from diffusers import AutoencoderKLWan
        except Exception as exc:
            raise ImportError(
                "diffusers with AutoencoderKLWan is required for diffusers VAE loading."
            ) from exc

        root, subfolder = target
        kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        vae = AutoencoderKLWan.from_pretrained(str(root), **kwargs)
        return vae.to(self.device)
