import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierFeatures(nn.Module):
    """Fixed sinusoidal features for scalar sigma values."""

    def __init__(self, embedding_dim: int = 256, max_period: float = 10000.0) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.float().view(-1, 1)
        args = sigma * self.frequencies.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SigmaEmbedding(nn.Module):
    """Fourier features followed by an MLP."""

    def __init__(self, embedding_dim: int = 256, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 4
        self.features = FourierFeatures(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.features(sigma))


class AdaGroupNorm3D(nn.Module):
    """GroupNorm modulated by a conditioning vector."""

    def __init__(self, channels: int, cond_dim: int, num_groups: int = 32) -> None:
        super().__init__()
        groups = min(num_groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return self.norm(x) * (1.0 + scale) + shift


def timestep_like_sigma(sigma: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=device)
    sigma = sigma.to(device=device)
    if sigma.ndim == 0:
        sigma = sigma.expand(batch_size)
    return sigma.view(batch_size)
