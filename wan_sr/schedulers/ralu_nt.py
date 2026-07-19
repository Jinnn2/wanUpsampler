from __future__ import annotations

import torch
import torch.nn.functional as F


def correlated_projection_noise(
    fine_noise: torch.Tensor,
    low_spatial_size: tuple[int, int],
    covariance_scale: float,
) -> torch.Tensor:
    """Approximate ``N(0, I-c Sigma)`` for nearest-neighbor upsampling.

    For an exact integer 2x transition and ``c=1/4``, this is the blockwise
    projection ``eps - Up(AvgPool(eps))``. Adaptive pooling keeps the same
    construction usable for Wan's slightly non-integer 368x640 -> 720x1248
    latent geometry.
    """

    if fine_noise.ndim != 4:
        raise ValueError(f"fine_noise must have shape [C,T,H,W], got {tuple(fine_noise.shape)}")
    low_h, low_w = (int(low_spatial_size[0]), int(low_spatial_size[1]))
    if low_h <= 0 or low_w <= 0:
        raise ValueError(f"low_spatial_size must be positive, got {(low_h, low_w)}")

    high_h, high_w = fine_noise.shape[-2:]
    max_replication = ((high_h + low_h - 1) // low_h) * ((high_w + low_w - 1) // low_w)
    scale = float(covariance_scale)
    if scale <= 0.0 or scale * max_replication > 1.0 + 1e-8:
        raise ValueError(
            "covariance_scale must satisfy 0 < c <= 1/max_replication; "
            f"got c={scale}, max_replication={max_replication}"
        )

    alpha = 1.0 - (max(0.0, 1.0 - scale * max_replication) ** 0.5)
    coarse = F.adaptive_avg_pool3d(
        fine_noise.unsqueeze(0),
        output_size=(fine_noise.shape[-3], low_h, low_w),
    )
    correlated = F.interpolate(
        coarse,
        size=fine_noise.shape[-3:],
        mode="nearest",
    ).squeeze(0)
    return fine_noise - alpha * correlated
