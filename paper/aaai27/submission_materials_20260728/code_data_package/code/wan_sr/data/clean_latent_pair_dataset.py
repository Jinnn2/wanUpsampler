from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class CleanLatentPairDataset(Dataset):
    """Dataset of clean LR/HR Wan latent pairs."""

    def __init__(
        self,
        data_dir: str | Path,
        dtype: torch.dtype = torch.float32,
        strict_channels: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dtype = dtype
        self.strict_channels = strict_channels
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(f"No clean latent samples found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.samples[index]
        z0_lr = _normalize_latent(_load_latent(sample_dir / "z0_lr.safetensors"), sample_dir, "z0_lr")
        z0_hr = _normalize_latent(_load_latent(sample_dir / "z0_hr.safetensors"), sample_dir, "z0_hr")
        meta = _load_meta(sample_dir / "meta.json")

        if z0_lr.shape[0] != z0_hr.shape[0]:
            raise ValueError(f"channel mismatch at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")
        if z0_lr.shape[1] != z0_hr.shape[1]:
            raise ValueError(f"latent time mismatch at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")
        if self.strict_channels and z0_lr.shape[0] != 16:
            raise ValueError(f"expected Wan z_dim=16 at {sample_dir}, got {z0_lr.shape[0]}")
        if z0_hr.shape[-2] <= z0_lr.shape[-2] or z0_hr.shape[-1] <= z0_lr.shape[-1]:
            raise ValueError(f"expected HR spatial size > LR at {sample_dir}: {z0_lr.shape} vs {z0_hr.shape}")

        return {
            "z0_lr": z0_lr.to(self.dtype),
            "z0_hr": z0_hr.to(self.dtype),
            "sample_id": sample_dir.name,
            "meta_json": json.dumps(meta, ensure_ascii=False),
        }

    def _discover_samples(self) -> list[Path]:
        if not self.data_dir.exists():
            return []
        return [
            path
            for path in sorted(self.data_dir.iterdir())
            if path.is_dir()
            and (path / "z0_lr.safetensors").exists()
            and (path / "z0_hr.safetensors").exists()
        ]


def _load_latent(path: Path) -> torch.Tensor:
    data = load_file(str(path), device="cpu")
    if "latent" in data:
        return data["latent"]
    if len(data) != 1:
        raise KeyError(f"{path} must contain key 'latent' or exactly one tensor, got {list(data)}")
    return next(iter(data.values()))


def _normalize_latent(latent: torch.Tensor, sample_dir: Path, name: str) -> torch.Tensor:
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 4:
        raise ValueError(f"{name} must be [C,T,H,W], got {tuple(latent.shape)} at {sample_dir}")
    return latent


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
