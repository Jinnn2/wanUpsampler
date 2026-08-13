from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def resize_spatial_area(x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Content-consistency reduction on H/W without interpolation.

    Adaptive average pooling is used only to compare the reconstructed HR
    latent with its LR input; it is not part of the model's upsampling path.
    """

    if x.ndim != 5:
        raise ValueError(f"x must be [B,C,T,H,W], got {tuple(x.shape)}")
    batch, channels, frames, height, width = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat = F.adaptive_avg_pool2d(flat, output_size)
    target_h, target_w = output_size
    return flat.reshape(batch, frames, channels, target_h, target_w).permute(0, 2, 1, 3, 4)


class UniversalCleanUpsampleLoss(nn.Module):
    def __init__(
        self,
        latent_weight: float = 1.0,
        content_weight: float = 0.2,
        temporal_weight: float = 0.1,
        charbonnier_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_weight = float(latent_weight)
        self.content_weight = float(content_weight)
        self.temporal_weight = float(temporal_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        if self.charbonnier_eps <= 0:
            raise ValueError("charbonnier_eps must be positive")

    def forward(
        self,
        prediction: torch.Tensor,
        target_hr: torch.Tensor,
        source_lr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target_hr.shape:
            raise ValueError(
                f"prediction and target_hr must match, got {tuple(prediction.shape)} and {tuple(target_hr.shape)}"
            )
        if source_lr.shape[:3] != target_hr.shape[:3]:
            raise ValueError("source and target must share batch, channel, and temporal dimensions")

        difference = prediction - target_hr
        latent = torch.sqrt(difference.square() + self.charbonnier_eps**2).mean()
        prediction_lr = resize_spatial_area(prediction, source_lr.shape[-2:])
        content = F.l1_loss(prediction_lr, source_lr)
        if prediction.shape[2] > 1:
            prediction_dt = prediction[:, :, 1:] - prediction[:, :, :-1]
            target_dt = target_hr[:, :, 1:] - target_hr[:, :, :-1]
            temporal = F.l1_loss(prediction_dt, target_dt)
        else:
            temporal = prediction.new_zeros(())
        total = (
            self.latent_weight * latent
            + self.content_weight * content
            + self.temporal_weight * temporal
        )
        return total, {
            "loss": total.detach(),
            "latent_loss": latent.detach(),
            "content_loss": content.detach(),
            "temporal_loss": temporal.detach(),
        }
