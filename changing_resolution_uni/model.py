from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _valid_groups(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ScaleConditionEncoder(nn.Module):
    """Encode actual source/target grid geometry instead of a discrete scale id."""

    input_dim = 8

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source_h, source_w = source_size
        target_h, target_w = target_size
        scale_h = target_h / source_h
        scale_w = target_w / source_w
        # Keep geometry arithmetic in fp32, then match the condition MLP's
        # parameter dtype. This avoids unsupported CPU bf16 tensor creation
        # while remaining safe when the whole model is explicitly cast.
        values = torch.tensor(
            [
                math.log(scale_h),
                math.log(scale_w),
                2.0 / source_h,
                2.0 / source_w,
                2.0 / target_h,
                2.0 / target_w,
                math.log(scale_h / scale_w),
                math.log(scale_h * scale_w),
            ],
            device=device,
            dtype=torch.float32,
        )
        values = values.to(dtype=self.net[0].weight.dtype)
        return self.net(values.unsqueeze(0).expand(batch_size, -1))


class ScaleConditionedResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _valid_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels, affine=False)
        self.norm2 = nn.GroupNorm(groups, channels, affine=False)
        self.cond1 = nn.Linear(cond_dim, channels * 2)
        self.cond2 = nn.Linear(cond_dim, channels * 2)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _modulate(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        scale, shift = params.chunk(2, dim=1)
        return x * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self._modulate(self.norm1(x), self.cond1(cond))
        h = self.conv1(F.silu(h))
        h = self._modulate(self.norm2(h), self.cond2(cond))
        h = self.conv2(self.dropout(F.silu(h)))
        return x + h


def make_target_coordinate_features(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    batch_size: int,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return target coordinates and sub-source-cell sampling phases.

    The four channels are normalized target y/x and the fractional y/x phase
    relative to the nearest source-grid center.  They let the shared decoder
    distinguish, for example, a 1.5x target sample from a 3x target sample
    without selecting a resolution-specific branch.
    """

    source_h, source_w = source_size
    target_h, target_w = target_size
    # Build geometry in fp32 for device portability (some CPU backends do not
    # implement arange directly in bf16), then cast the feature map to the
    # latent activation dtype used by the decoder.
    geometry_dtype = torch.float32
    y_index = torch.arange(target_h, device=device, dtype=geometry_dtype)
    x_index = torch.arange(target_w, device=device, dtype=geometry_dtype)
    y_norm = (y_index + 0.5) * (2.0 / target_h) - 1.0
    x_norm = (x_index + 0.5) * (2.0 / target_w) - 1.0

    source_y = (y_index + 0.5) * (source_h / target_h) - 0.5
    source_x = (x_index + 0.5) * (source_w / target_w) - 0.5
    phase_y = source_y - torch.round(source_y)
    phase_x = source_x - torch.round(source_x)

    yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
    py, px = torch.meshgrid(phase_y * 2.0, phase_x * 2.0, indexing="ij")
    features = torch.stack([yy, xx, py, px], dim=0).to(dtype=dtype)
    return features[None, :, None].expand(batch_size, -1, frames, -1, -1)


class DynamicLearnedSubpixelResampler3D(nn.Module):
    """Learned 3x3x3 local aggregation without interpolation.

    Each target-grid position gathers the nearest source-cell center and its
    full temporal/spatial 27-neighborhood using integer indexing. A shared
    pointwise predictor then produces content- and geometry-dependent logits
    over the 27 neighbors. Softmax mixing directly reconstructs the target
    feature; no ``F.interpolate``, ``grid_sample`` or fixed PixelShuffle is
    used in this path.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        coordinate_channels: int = 4,
        attention_dim: int | None = None,
        chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        self.coordinate_channels = int(coordinate_channels)
        self.attention_dim = int(attention_dim or max(32, channels // 2))
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        query_in = channels + cond_dim + coordinate_channels
        self.query_predictor = nn.Sequential(
            nn.Linear(query_in, channels),
            nn.SiLU(),
            nn.Linear(channels, self.attention_dim),
        )
        self.key_predictor = nn.Linear(channels, self.attention_dim)
        self.neighbor_embedding = nn.Parameter(torch.zeros(27, self.attention_dim))
        self.neighbor_bias = nn.Parameter(torch.zeros(27))
        nn.init.normal_(self.neighbor_embedding, std=0.02)

    @staticmethod
    def _target_indices(
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        frames: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        source_h, source_w = source_size
        target_h, target_w = target_size
        t = torch.arange(frames, device=device, dtype=torch.long)
        y = torch.arange(target_h, device=device, dtype=torch.float32)
        x = torch.arange(target_w, device=device, dtype=torch.float32)
        source_y = torch.round((y + 0.5) * (source_h / target_h) - 0.5).to(torch.long)
        source_x = torch.round((x + 0.5) * (source_w / target_w) - 0.5).to(torch.long)
        offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
        tt = (t[:, None] + offsets[None, :]).clamp(0, frames - 1)
        yy = (source_y[:, None] + offsets[None, :]).clamp(0, source_h - 1)
        xx = (source_x[:, None] + offsets[None, :]).clamp(0, source_w - 1)
        # [T,H,W,27], neighbor order is temporal-major then y then x.
        index = (
            tt[:, None, None, :, None, None] * source_h * source_w
            + yy[None, :, None, None, :, None] * source_w
            + xx[None, None, :, None, None, :]
        )
        index = index.expand(frames, target_h, target_w, 3, 3, 3)
        return index.reshape(frames, target_h, target_w, 27)

    def forward(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        cond: torch.Tensor,
        *,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if features.ndim != 5:
            raise ValueError(f"features must be [B,C,T,H,W], got {tuple(features.shape)}")
        batch, channels, frames, source_h, source_w = features.shape
        if channels != self.channels or (source_h, source_w) != source_size:
            raise ValueError("feature shape does not match resampler metadata")
        target_h, target_w = target_size
        index = self._target_indices(source_size, target_size, frames, device=features.device)
        flat = features.reshape(batch, channels, frames * source_h * source_w)
        point_indices = index.reshape(-1, 27)
        flat_coordinates = coordinates.reshape(batch, self.coordinate_channels, -1)
        flat_cond = cond[:, :, None].expand(-1, -1, flat_coordinates.shape[-1])
        output_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        total = point_indices.shape[0]
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            chunk_index = point_indices[start:end].reshape(1, 1, -1).expand(batch, channels, -1)
            gathered = flat.gather(2, chunk_index).reshape(batch, channels, end - start, 27)
            local_mean = gathered.mean(dim=-1)
            query_input = torch.cat(
                [
                    local_mean,
                    flat_coordinates[:, :, start:end],
                    flat_cond[:, :, start:end],
                ],
                dim=1,
            ).transpose(1, 2)
            query = self.query_predictor(query_input)
            neighbors = gathered.permute(0, 2, 3, 1)
            keys = self.key_predictor(neighbors) + self.neighbor_embedding[None, None]
            logits = (keys * query[:, :, None]).sum(dim=-1) / math.sqrt(self.attention_dim)
            logits = logits + self.neighbor_bias[None, None]
            weights = torch.softmax(logits, dim=-1)
            output_chunks.append((neighbors * weights[:, :, :, None]).sum(dim=2).transpose(1, 2))
            if return_weights:
                weight_chunks.append(weights.transpose(1, 2))
        aggregated = torch.cat(output_chunks, dim=-1).reshape(batch, channels, frames, target_h, target_w)
        all_weights = torch.cat(weight_chunks, dim=-1) if return_weights else None
        return aggregated, all_weights


class UniversalCleanLatentUpsampler(nn.Module):
    """One-weight clean Wan latent upsampler for multiple spatial ratios.

    Contract:
        [B,16,T,h,w] + output_size=(H,W) -> [B,16,T,H,W]

    A shared Conv3D encoder feeds a dynamic learned 3x3x3 subpixel resampler.
    Every target location gathers integer-indexed LR features and predicts its
    own 27-neighbor mixing weights. The model directly reconstructs target-grid
    features and latents; it has no interpolation or output skip path. No
    temporal resampling is performed.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 192,
        cond_dim: int = 256,
        pre_blocks: int = 4,
        post_blocks: int = 4,
        dropout: float = 0.0,
        output_scale: float = 1.0,
        zero_init_output: bool = False,
        resampler_attention_dim: int | None = None,
        resampler_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if pre_blocks < 1 or post_blocks < 1:
            raise ValueError("pre_blocks and post_blocks must both be >= 1")
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        self.output_scale = float(output_scale)
        self.condition = ScaleConditionEncoder(cond_dim)
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.pre = nn.ModuleList(
            [ScaleConditionedResBlock3D(hidden_channels, cond_dim, dropout) for _ in range(pre_blocks)]
        )
        self.resampler = DynamicLearnedSubpixelResampler3D(
            hidden_channels,
            cond_dim,
            attention_dim=resampler_attention_dim,
            chunk_size=resampler_chunk_size,
        )
        self.target_fuse = nn.Conv3d(hidden_channels + 4, hidden_channels, kernel_size=3, padding=1)
        self.post = nn.ModuleList(
            [ScaleConditionedResBlock3D(hidden_channels, cond_dim, dropout) for _ in range(post_blocks)]
        )
        self.output_norm = nn.GroupNorm(_valid_groups(hidden_channels), hidden_channels)
        self.output = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | Sequence[int],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | tuple[int, int]]]:
        if z0_lr.ndim != 5:
            raise ValueError(f"z0_lr must be [B,C,T,H,W], got {tuple(z0_lr.shape)}")
        if z0_lr.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {z0_lr.shape[1]}")
        if len(output_size) != 2:
            raise ValueError(f"output_size must be (height,width), got {output_size}")
        target_size = (int(output_size[0]), int(output_size[1]))
        source_size = (int(z0_lr.shape[-2]), int(z0_lr.shape[-1]))
        if target_size[0] < source_size[0] or target_size[1] < source_size[1]:
            raise ValueError(
                "UniversalCleanLatentUpsampler only performs spatial upsampling: "
                f"source={source_size}, target={target_size}"
            )
        batch, _, frames, _, _ = z0_lr.shape
        cond = self.condition(
            source_size,
            target_size,
            batch_size=batch,
            device=z0_lr.device,
            dtype=z0_lr.dtype,
        )
        features = self.stem(z0_lr)
        for block in self.pre:
            features = block(features, cond)

        coordinates = make_target_coordinate_features(
            source_size,
            target_size,
            batch_size=batch,
            frames=frames,
            device=z0_lr.device,
            dtype=z0_lr.dtype,
        )
        features_hr, weights = self.resampler(
            features,
            coordinates,
            cond,
            source_size=source_size,
            target_size=target_size,
            # Keep the full [B,27,T*H*W] attention map out of the training
            # graph; it is materialized only for explicit debug/inspection.
            return_weights=return_aux,
        )
        decoded = self.target_fuse(torch.cat([features_hr, coordinates], dim=1))
        for block in self.post:
            decoded = block(decoded, cond)
        prediction = self.output(F.silu(self.output_norm(decoded))) * self.output_scale

        if not return_aux:
            return prediction
        return prediction, {
            "direct_output": prediction,
            "subpixel_weights": weights,
            "condition": cond,
            "source_size": source_size,
            "target_size": target_size,
        }
