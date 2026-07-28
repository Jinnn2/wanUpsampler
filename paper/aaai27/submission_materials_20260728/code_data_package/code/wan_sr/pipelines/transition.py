from __future__ import annotations

import torch

from wan_sr.schedulers.noise_utils import add_flow_noise


@torch.no_grad()
def transition_lr_to_hr(
    x_t_lr: torch.Tensor,
    sigma: torch.Tensor,
    upsampler: torch.nn.Module,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert LR noisy latent to HR noisy latent at the same sigma."""

    was_training = upsampler.training
    upsampler.eval()
    pred_z0_hr = upsampler(x_t_lr, sigma)
    x_t_hr, _ = add_flow_noise(pred_z0_hr, sigma, noise=noise)
    if was_training:
        upsampler.train()
    return x_t_hr, pred_z0_hr
