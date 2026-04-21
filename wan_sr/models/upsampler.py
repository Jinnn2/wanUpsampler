import torch
from torch import nn

from .blocks import SigmaConditionedResBlock3D, SpatialPixelShuffle2x
from .sigma_embedding import SigmaEmbedding, timestep_like_sigma


class WanNoisyLatentUpsampler(nn.Module):
    """Sigma-conditioned 2x spatial upsampler for Wan latents.

    Inputs are shaped [B, C, T, H, W]. Only H/W are upsampled.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 256,
        num_res_blocks: int = 8,
        sigma_embed_dim: int = 256,
        scale: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("The first implementation supports scale=2 only")
        if num_res_blocks < 2:
            raise ValueError("num_res_blocks must be at least 2")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.sigma_embedding = SigmaEmbedding(sigma_embed_dim)
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)

        pre_blocks = num_res_blocks // 2
        post_blocks = num_res_blocks - pre_blocks
        self.pre_blocks = nn.ModuleList(
            [
                SigmaConditionedResBlock3D(hidden_channels, sigma_embed_dim, dropout)
                for _ in range(pre_blocks)
            ]
        )
        self.upsample = SpatialPixelShuffle2x(hidden_channels)
        self.post_blocks = nn.ModuleList(
            [
                SigmaConditionedResBlock3D(hidden_channels, sigma_embed_dim, dropout)
                for _ in range(post_blocks)
            ]
        )
        self.out_norm = nn.GroupNorm(_valid_groups(hidden_channels), hidden_channels)
        self.out = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t_lr: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if x_t_lr.ndim != 5:
            raise ValueError(f"x_t_lr must be [B, C, T, H, W], got {tuple(x_t_lr.shape)}")
        if x_t_lr.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_t_lr.shape[1]}")

        sigma = timestep_like_sigma(sigma, x_t_lr.shape[0], x_t_lr.device)
        cond = self.sigma_embedding(sigma).to(dtype=x_t_lr.dtype)

        h = self.stem(x_t_lr)
        for block in self.pre_blocks:
            h = block(h, cond)
        h = self.upsample(h)
        for block in self.post_blocks:
            h = block(h, cond)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))


def _valid_groups(channels: int, preferred: int = 32) -> int:
    groups = min(preferred, channels)
    while channels % groups != 0:
        groups -= 1
    return groups
