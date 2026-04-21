from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from wan_sr.schedulers.noise_utils import add_flow_noise
from wan_sr.schedulers.sigma_sampler import SigmaSampler


class LatentPairDataset(Dataset):
    """Dataset of precomputed Wan LR/HR latent pairs.

    Expected sample layout:
      sample_id/z0_lr.safetensors
      sample_id/z0_hr.safetensors
      sample_id/meta.json
    """

    def __init__(
        self,
        data_dir: str | Path,
        sigma_mode: str = "mid",
        force_clean: bool = False,
        dtype: torch.dtype = torch.float32,
        strict_meta: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sigma_sampler = SigmaSampler("clean" if force_clean else sigma_mode)
        self.force_clean = force_clean
        self.dtype = dtype
        self.strict_meta = strict_meta
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(f"No latent samples found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.samples[index]
        z0_lr = _load_latent(sample_dir / "z0_lr.safetensors").to(self.dtype)
        z0_hr = _load_latent(sample_dir / "z0_hr.safetensors").to(self.dtype)
        meta = _load_meta(sample_dir / "meta.json")

        if z0_lr.ndim == 4:
            pass
        elif z0_lr.ndim == 5 and z0_lr.shape[0] == 1:
            z0_lr = z0_lr.squeeze(0)
        else:
            raise ValueError(f"z0_lr must be [C,T,H,W], got {tuple(z0_lr.shape)} at {sample_dir}")

        if z0_hr.ndim == 5 and z0_hr.shape[0] == 1:
            z0_hr = z0_hr.squeeze(0)
        if z0_hr.ndim != 4:
            raise ValueError(f"z0_hr must be [C,T,H,W], got {tuple(z0_hr.shape)} at {sample_dir}")

        _validate_shapes(z0_lr, z0_hr, sample_dir, self.strict_meta)

        sigma = self.sigma_sampler.sample(1, device=z0_lr.device).to(self.dtype)
        x_t_lr, noise = add_flow_noise(z0_lr.unsqueeze(0), sigma)
        return {
            "x_t_lr": x_t_lr.squeeze(0),
            "sigma": sigma.squeeze(0),
            "z0_lr": z0_lr,
            "z0_hr": z0_hr,
            "noise": noise.squeeze(0),
            "sample_id": sample_dir.name,
            "meta_json": json.dumps(meta, ensure_ascii=False),
        }

    def _discover_samples(self) -> list[Path]:
        if not self.data_dir.exists():
            return []
        samples: list[Path] = []
        for path in sorted(self.data_dir.iterdir()):
            if not path.is_dir():
                continue
            if (path / "z0_lr.safetensors").exists() and (path / "z0_hr.safetensors").exists():
                samples.append(path)
        return samples


def _load_latent(path: Path) -> torch.Tensor:
    data = load_file(str(path), device="cpu")
    if "latent" in data:
        return data["latent"]
    if len(data) != 1:
        raise KeyError(f"{path} must contain key 'latent' or exactly one tensor, got {list(data)}")
    return next(iter(data.values()))


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_shapes(
    z0_lr: torch.Tensor,
    z0_hr: torch.Tensor,
    sample_dir: Path,
    strict_meta: bool,
) -> None:
    if z0_lr.shape[0] != z0_hr.shape[0]:
        raise ValueError(f"channel mismatch at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")
    if z0_lr.shape[1] != z0_hr.shape[1]:
        raise ValueError(f"latent time mismatch at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")
    if z0_hr.shape[-2] != z0_lr.shape[-2] * 2 or z0_hr.shape[-1] != z0_lr.shape[-1] * 2:
        raise ValueError(f"expected HR spatial size to be 2x LR at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")
    if strict_meta and z0_lr.shape[0] != 16:
        raise ValueError(f"expected Wan2.1 z_dim=16 at {sample_dir}, got {z0_lr.shape[0]}")
