"""Universal clean-latent upsampling for Wan video latents."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import ScaleBucketBatchSampler, UniversalCleanLatentDataset
    from .losses import UniversalCleanUpsampleLoss
    from .model import UniversalCleanLatentUpsampler

__all__ = [
    "ScaleBucketBatchSampler",
    "UniversalCleanLatentDataset",
    "UniversalCleanLatentUpsampler",
    "UniversalCleanUpsampleLoss",
]


_LAZY_EXPORTS = {
    "ScaleBucketBatchSampler": (".data", "ScaleBucketBatchSampler"),
    "UniversalCleanLatentDataset": (".data", "UniversalCleanLatentDataset"),
    "UniversalCleanUpsampleLoss": (".losses", "UniversalCleanUpsampleLoss"),
    "UniversalCleanLatentUpsampler": (".model", "UniversalCleanLatentUpsampler"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
