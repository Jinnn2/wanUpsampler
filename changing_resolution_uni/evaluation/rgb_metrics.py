from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class VideoRGBMetrics:
    """Paired RGB metrics aggregated frame-first and then video-first."""

    def __init__(self, names: list[str], *, device: torch.device) -> None:
        self.names = tuple(dict.fromkeys(name.lower() for name in names))
        unsupported = set(self.names) - {"psnr", "ssim", "lpips"}
        if unsupported:
            raise ValueError(f"Unsupported RGB metrics: {sorted(unsupported)}")
        self.device = device
        self.metrics: dict[str, Any] = {}
        if any(name in self.names for name in ("ssim", "lpips")):
            try:
                from torchmetrics.image import (
                    LearnedPerceptualImagePatchSimilarity,
                    StructuralSimilarityIndexMeasure,
                )
            except Exception as exc:
                raise RuntimeError(
                    "SSIM/LPIPS require torchmetrics and lpips: pip install torchmetrics lpips"
                ) from exc
            if "ssim" in self.names:
                self.metrics["ssim"] = StructuralSimilarityIndexMeasure(
                    data_range=1.0,
                    reduction="elementwise_mean",
                ).to(device)
            if "lpips" in self.names:
                self.metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(
                    normalize=True
                ).to(device)

    @torch.no_grad()
    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_size: int = 4,
    ) -> dict[str, float]:
        pred = _video_to_nchw(prediction)
        ref = _video_to_nchw(target)
        if pred.shape != ref.shape:
            raise ValueError(f"RGB video shape mismatch: {pred.shape} vs {ref.shape}")
        difference = pred - ref
        per_frame_mse = difference.square().flatten(1).mean(dim=1)
        result = {
            "pixel_mse": float(per_frame_mse.mean()),
            "rgb_temporal_delta_l1": temporal_delta_l1(prediction, target),
        }
        if "psnr" in self.names:
            per_frame_psnr = torch.where(
                per_frame_mse > 0,
                -10.0 * torch.log10(per_frame_mse),
                torch.full_like(per_frame_mse, float("inf")),
            )
            result["psnr"] = float(per_frame_psnr.mean())
        batch_size = max(1, int(batch_size))
        for name, metric in self.metrics.items():
            metric.reset()
            for start in range(0, pred.shape[0], batch_size):
                metric.update(
                    pred[start : start + batch_size].to(self.device),
                    ref[start : start + batch_size].to(self.device),
                )
            result[name] = float(metric.compute())
        target_hf = high_frequency_energy(target)
        prediction_hf = high_frequency_energy(prediction)
        result.update(
            {
                "target_hf_energy": target_hf,
                "prediction_hf_energy": prediction_hf,
                "hf_energy_error": abs(prediction_hf - target_hf),
            }
        )
        return result


def temporal_delta_l1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape[0] < 2:
        return 0.0
    pred_dt = prediction[1:].float() - prediction[:-1].float()
    target_dt = target[1:].float() - target[:-1].float()
    return float((pred_dt - target_dt).abs().mean())


def high_frequency_energy(video: torch.Tensor) -> float:
    video = video.float()
    if video.shape[1] < 2 or video.shape[2] < 2:
        return 0.0
    horizontal = (video[:, :, 1:] - video[:, :, :-1]).abs().mean()
    vertical = (video[:, 1:, :] - video[:, :-1, :]).abs().mean()
    return float((horizontal + vertical) * 0.5)


def save_keyframes(video: torch.Tensor, directory: str | Path, stem: str) -> list[str]:
    import imageio.v3 as iio

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indices = sorted({0, max(0, video.shape[0] // 2), max(0, video.shape[0] - 1)})
    paths = []
    array = (video.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    for index in indices:
        path = directory / f"{stem}_frame{index:03d}.png"
        iio.imwrite(path, array[index])
        paths.append(str(path))
    return paths


def _video_to_nchw(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must be [T,H,W,C], got {tuple(video.shape)}")
    return video.float().clamp(0, 1).permute(0, 3, 1, 2).contiguous()
