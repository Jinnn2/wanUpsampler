import torch
from torch.nn import functional as F


def expand_sigma(sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=target.device, dtype=target.dtype)
    sigma = sigma.to(device=target.device, dtype=target.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(target.shape[0])
    return sigma.view(target.shape[0], *([1] * (target.ndim - 1)))


def add_flow_noise(
    z0: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simplified flow-style noising: x_sigma = (1-sigma) z0 + sigma eps."""

    if noise is None:
        noise = torch.randn_like(z0)
    sigma_view = expand_sigma(sigma, z0)
    x_t = (1.0 - sigma_view) * z0 + sigma_view * noise
    return x_t, noise


def spatial_downsample_latent(x: torch.Tensor, scale: int = 2, mode: str = "area") -> torch.Tensor:
    b, c, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    down = F.interpolate(flat, size=(h // scale, w // scale), mode=mode)
    return down.reshape(b, t, c, h // scale, w // scale).permute(0, 2, 1, 3, 4)


def spatial_upsample_latent(
    x: torch.Tensor,
    scale: int = 2,
    mode: str = "trilinear",
) -> torch.Tensor:
    if mode in {"nearest", "area"}:
        return F.interpolate(x, scale_factor=(1, scale, scale), mode=mode)
    return F.interpolate(x, scale_factor=(1, scale, scale), mode=mode, align_corners=False)
