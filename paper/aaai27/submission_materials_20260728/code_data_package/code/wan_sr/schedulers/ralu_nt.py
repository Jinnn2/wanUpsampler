from __future__ import annotations

import torch


def exact_grouped_projection_noise(
    reference: torch.Tensor,
    *,
    group_size: int = 4,
    covariance_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample exact blockwise ``N(0, I-c 11^T)`` projection noise.

    ``reference`` is shaped ``[num_groups, group_size, ...]``. This routine
    implements the covariance used by RALU for an aligned integer transition.
    """

    if reference.ndim < 2 or reference.shape[1] != int(group_size):
        raise ValueError(
            "reference must have shape [num_groups, group_size, ...], "
            f"got {tuple(reference.shape)} for group_size={group_size}"
        )
    size = int(group_size)
    scale = float(covariance_scale)
    if size < 1:
        raise ValueError(f"group_size must be positive, got {size}")
    if scale <= 0.0 or scale * size > 1.0 + 1e-8:
        raise ValueError(f"covariance_scale must satisfy 0 < c <= 1/{size}, got {scale}")

    eps = torch.randn(
        reference.shape,
        dtype=torch.float32,
        device=reference.device,
        generator=generator,
    )
    # If P=(1/m)11^T, (I-aP)^2 = I-(2a-a^2)P.  Choosing
    # a=1-sqrt(1-mc) therefore gives covariance I-c11^T.
    projection_weight = 1.0 - max(0.0, 1.0 - scale * size) ** 0.5
    projected = eps - projection_weight * eps.mean(dim=1, keepdim=True)
    return projected.to(dtype=reference.dtype)
