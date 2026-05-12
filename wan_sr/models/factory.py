from __future__ import annotations

from torch import nn

from .clean_resizer import WanCleanLatentResizer
from .stage2_resizer import WanCleanLatentResizerStage2


def infer_clean_resizer_model_type(model_config: dict) -> str:
    config = dict(model_config)
    model_type = str(config.pop("model_type", config.pop("architecture", ""))).lower()
    if model_type:
        return model_type
    if "resblock_type" in config or "resize_op" in config:
        return "stage2"
    return "stage1"


def build_clean_latent_resizer(model_config: dict) -> nn.Module:
    config = dict(model_config)
    model_type = infer_clean_resizer_model_type(config)
    config.pop("model_type", None)
    config.pop("architecture", None)

    if model_type in {"stage1", "clean_resizer", "wan_clean_latent_resizer"}:
        return WanCleanLatentResizer(**config)
    if model_type in {"stage2", "clean_resizer_stage2", "wan_clean_latent_resizer_stage2"}:
        return WanCleanLatentResizerStage2(**config)

    raise ValueError(f"Unsupported clean resizer model_type={model_type!r}")
