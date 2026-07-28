from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .latent_losses import charbonnier_loss, temporal_difference_loss


@dataclass
class CleanLatentLossWeights:
    latent_weight: float = 1.0
    low_freq_weight: float = 0.2
    temporal_weight: float = 0.1
    residual_weight: float = 0.0
    charbonnier_eps: float = 1e-3


class CleanLatentResizeLoss(nn.Module):
    def __init__(
        self,
        latent_weight: float = 1.0,
        low_freq_weight: float = 0.2,
        temporal_weight: float = 0.1,
        residual_weight: float = 0.0,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.weights = CleanLatentLossWeights(
            latent_weight=latent_weight,
            low_freq_weight=low_freq_weight,
            temporal_weight=temporal_weight,
            residual_weight=residual_weight,
            charbonnier_eps=charbonnier_eps,
        )

    def forward(
        self,
        pred_z0_hr: torch.Tensor,
        z0_hr: torch.Tensor,
        z0_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred_z0_hr.shape != z0_hr.shape:
            raise ValueError(f"pred/target shape mismatch: {pred_z0_hr.shape} vs {z0_hr.shape}")

        pred_down = F.interpolate(
            pred_z0_hr,
            size=(z0_lr.shape[2], z0_lr.shape[3], z0_lr.shape[4]),
            mode="trilinear",
            align_corners=False,
        )
        latent = charbonnier_loss(pred_z0_hr, z0_hr, self.weights.charbonnier_eps)
        low = F.l1_loss(pred_down, z0_lr)
        temporal = temporal_difference_loss(pred_z0_hr, z0_hr)
        residual = self._residual_loss(pred_z0_hr, z0_lr)
        total = (
            self.weights.latent_weight * latent
            + self.weights.low_freq_weight * low
            + self.weights.temporal_weight * temporal
            + self.weights.residual_weight * residual
        )
        return total, {
            "loss": total.detach(),
            "latent_loss": latent.detach(),
            "low_freq_loss": low.detach(),
            "temporal_loss": temporal.detach(),
            "residual_loss": residual.detach(),
        }

    def _residual_loss(self, pred_z0_hr: torch.Tensor, z0_lr: torch.Tensor) -> torch.Tensor:
        if self.weights.residual_weight <= 0:
            return pred_z0_hr.new_tensor(0.0)
        up = F.interpolate(
            z0_lr,
            size=(pred_z0_hr.shape[2], pred_z0_hr.shape[3], pred_z0_hr.shape[4]),
            mode="trilinear",
            align_corners=False,
        )
        return (pred_z0_hr - up).abs().mean()
