from .clean_resizer import WanCleanLatentResizer
from .factory import build_clean_latent_resizer, infer_clean_resizer_model_type
from .stage2_resizer import WanCleanLatentResizerStage2

__all__ = [
    "WanCleanLatentResizer",
    "WanCleanLatentResizerStage2",
    "WanNoisyLatentUpsampler",
    "build_clean_latent_resizer",
    "infer_clean_resizer_model_type",
]


def __getattr__(name: str):
    if name == "WanNoisyLatentUpsampler":
        from .upsampler import WanNoisyLatentUpsampler

        return WanNoisyLatentUpsampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
