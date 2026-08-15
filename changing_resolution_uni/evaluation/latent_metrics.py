from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from changing_resolution_uni.losses import resize_spatial_area


@torch.no_grad()
def compute_latent_metrics(
    prediction: torch.Tensor,
    target_hr: torch.Tensor,
    source_lr: torch.Tensor,
    *,
    charbonnier_eps: float = 1e-3,
) -> dict[str, float]:
    if prediction.shape != target_hr.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} vs {target_hr.shape}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Prediction contains NaN or Inf")
    difference = prediction.float() - target_hr.float()
    mse = difference.square().mean()
    prediction_lr = resize_spatial_area(prediction.float(), source_lr.shape[-2:])
    content = F.l1_loss(prediction_lr, source_lr.float())
    if prediction.shape[2] > 1:
        pred_dt = prediction.float()[:, :, 1:] - prediction.float()[:, :, :-1]
        target_dt = target_hr.float()[:, :, 1:] - target_hr.float()[:, :, :-1]
        temporal = F.l1_loss(pred_dt, target_dt)
    else:
        temporal = prediction.new_zeros((), dtype=torch.float32)
    charbonnier = torch.sqrt(difference.square() + float(charbonnier_eps) ** 2).mean()
    return {
        "latent_charbonnier": float(charbonnier),
        "latent_mae": float(difference.abs().mean()),
        "latent_mse": float(mse),
        "latent_rmse": math.sqrt(max(0.0, float(mse))),
        "content_l1": float(content),
        "temporal_delta_l1": float(temporal),
    }
