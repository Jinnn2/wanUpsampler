from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from wan_sr.schedulers.noise_utils import spatial_downsample_latent


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


def temporal_difference_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[2] <= 1:
        return pred.new_tensor(0.0)
    pred_dt = pred[:, :, 1:] - pred[:, :, :-1]
    target_dt = target[:, :, 1:] - target[:, :, :-1]
    return F.l1_loss(pred_dt, target_dt)


@dataclass
class LossWeights:
    latent_weight: float = 1.0
    low_freq_weight: float = 0.2
    temporal_weight: float = 0.1
    charbonnier_eps: float = 1e-3


class LatentUpsamplerLoss(nn.Module):
    def __init__(
        self,
        latent_weight: float = 1.0,
        low_freq_weight: float = 0.2,
        temporal_weight: float = 0.1,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.weights = LossWeights(
            latent_weight=latent_weight,
            low_freq_weight=low_freq_weight,
            temporal_weight=temporal_weight,
            charbonnier_eps=charbonnier_eps,
        )

    def forward(
        self,
        pred_z0_hr: torch.Tensor,
        z0_hr: torch.Tensor,
        z0_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = compute_loss(
            pred_z0_hr,
            z0_hr,
            z0_lr,
            latent_weight=self.weights.latent_weight,
            low_freq_weight=self.weights.low_freq_weight,
            temporal_weight=self.weights.temporal_weight,
            charbonnier_eps=self.weights.charbonnier_eps,
        )
        return losses["loss"], losses


def compute_loss(
    pred_z0_hr: torch.Tensor,
    z0_hr: torch.Tensor,
    z0_lr: torch.Tensor,
    latent_weight: float = 1.0,
    low_freq_weight: float = 0.2,
    temporal_weight: float = 0.1,
    charbonnier_eps: float = 1e-3,
) -> dict[str, torch.Tensor]:
    if pred_z0_hr.shape != z0_hr.shape:
        raise ValueError(f"pred/target shape mismatch: {pred_z0_hr.shape} vs {z0_hr.shape}")

    pred_down = spatial_downsample_latent(pred_z0_hr, scale=2)
    if pred_down.shape != z0_lr.shape:
        raise ValueError(f"downsampled pred and z0_lr mismatch: {pred_down.shape} vs {z0_lr.shape}")

    latent = charbonnier_loss(pred_z0_hr, z0_hr, charbonnier_eps)
    low = F.l1_loss(pred_down, z0_lr)
    temporal = temporal_difference_loss(pred_z0_hr, z0_hr)
    total = latent_weight * latent + low_freq_weight * low + temporal_weight * temporal
    return {
        "loss": total,
        "latent_loss": latent.detach(),
        "low_freq_loss": low.detach(),
        "temporal_loss": temporal.detach(),
    }
