from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from changing_resolution.ralu_nt_math import ralu_resume_parameters
from wan_sr.schedulers.ralu_nt import exact_grouped_projection_noise

logger = logging.getLogger(__name__)


@dataclass
class RALUPackedState:
    """Wan raw-patch latent state for RALU's mixed-resolution stage."""

    values: torch.Tensor
    coords: torch.Tensor
    coarse_parent_indices: torch.Tensor
    fine_indices: torch.Tensor
    coarse_count: int
    channels: int
    frames: int
    coarse_grid: tuple[int, int]
    fine_grid: tuple[int, int]

    @property
    def coarse_values(self) -> torch.Tensor:
        return self.values[: self.coarse_count]

    @property
    def fine_values(self) -> torch.Tensor:
        return self.values[self.coarse_count :]


def patchify_wan_latent(latent: torch.Tensor) -> torch.Tensor:
    """Pack ``[C,T,H,W]`` into Wan's non-overlapping ``1x2x2`` raw patches."""

    if latent.ndim != 4:
        raise ValueError(f"expected [C,T,H,W], got {tuple(latent.shape)}")
    channels, frames, height, width = latent.shape
    if height % 2 or width % 2:
        raise ValueError(f"Wan latent height/width must be even, got {(height, width)}")
    return (
        latent.reshape(channels, frames, height // 2, 2, width // 2, 2)
        .permute(1, 2, 4, 3, 5, 0)
        .contiguous()
        .reshape(frames * (height // 2) * (width // 2), channels * 4)
    )


def scatter_wan_patches(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    channels: int,
    frames: int,
    grid: tuple[int, int],
) -> torch.Tensor:
    """Scatter packed raw patches into a dense ``[C,T,2H,2W]`` latent."""

    grid_h, grid_w = (int(grid[0]), int(grid[1]))
    expected_dim = int(channels) * 4
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"expected patch values [N,{expected_dim}], got {tuple(values.shape)}")
    if indices.ndim != 1 or indices.numel() != values.shape[0]:
        raise ValueError("indices must be one-dimensional and match the number of patches")
    total = int(frames) * grid_h * grid_w
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= total):
        raise ValueError(f"patch index outside [0,{total})")

    packed = values.new_zeros((total, expected_dim))
    packed[indices.long()] = values
    packed = packed.reshape(frames, grid_h, grid_w, 2, 2, channels)
    return (
        packed.permute(5, 0, 1, 3, 2, 4)
        .contiguous()
        .reshape(channels, frames, grid_h * 2, grid_w * 2)
    )


def _token_coordinates(frames: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    t, y, x = torch.meshgrid(
        torch.arange(frames, device=device),
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    return torch.stack([t, y, x], dim=-1).reshape(-1, 3)


def _expand_parent_patches(
    parent_values: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    coarse_grid: tuple[int, int],
    fine_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand every coarse patch into four aligned fine patches."""

    coarse_h, coarse_w = coarse_grid
    fine_h, fine_w = fine_grid
    if (fine_h, fine_w) != (2 * coarse_h, 2 * coarse_w):
        raise ValueError("fine_grid must be exactly 2x coarse_grid")
    if parent_values.shape[0] != parent_indices.numel():
        raise ValueError("parent values/indices disagree")

    parent_indices = parent_indices.long()
    spatial = coarse_h * coarse_w
    frame = torch.div(parent_indices, spatial, rounding_mode="floor")
    remainder = parent_indices.remainder(spatial)
    parent_y = torch.div(remainder, coarse_w, rounding_mode="floor")
    parent_x = remainder.remainder(coarse_w)
    offsets = torch.tensor(((0, 0), (0, 1), (1, 0), (1, 1)), device=parent_values.device)

    child_y = 2 * parent_y[:, None] + offsets[None, :, 0]
    child_x = 2 * parent_x[:, None] + offsets[None, :, 1]
    child_indices = (frame[:, None] * fine_h * fine_w + child_y * fine_w + child_x).reshape(-1)
    child_coords = torch.stack(
        [
            frame[:, None].expand(-1, 4),
            child_y,
            child_x,
        ],
        dim=-1,
    ).reshape(-1, 3)
    child_values = parent_values[:, None, :].expand(-1, 4, -1).reshape(-1, parent_values.shape[-1])
    return child_values, child_indices, child_coords.to(torch.float32)


def _independent_transition_noise(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the unit Gaussian used for tokens not upsampled at this handoff."""

    return torch.randn(
        reference.shape,
        dtype=torch.float32,
        device=reference.device,
        generator=generator,
    ).to(reference.dtype)


def first_ralu_handoff(
    endpoint_latent: torch.Tensor,
    spatial_edge_mask: torch.Tensor,
    *,
    end_data_time: float,
    z_value: float,
    generator: torch.Generator,
) -> RALUPackedState:
    """Create the aligned coarse/fine token mixture at the first handoff."""

    channels, frames, latent_h, latent_w = endpoint_latent.shape
    if latent_h % 2 or latent_w % 2:
        raise ValueError("endpoint latent must align to Wan's 1x2x2 patch geometry")
    coarse_grid = (latent_h // 2, latent_w // 2)
    fine_grid = (coarse_grid[0] * 2, coarse_grid[1] * 2)
    if tuple(spatial_edge_mask.shape) != coarse_grid:
        raise ValueError(f"edge mask {tuple(spatial_edge_mask.shape)} != coarse grid {coarse_grid}")

    parent_values = patchify_wan_latent(endpoint_latent)
    parent_coords = _token_coordinates(frames, *coarse_grid, endpoint_latent.device)
    selected = spatial_edge_mask.to(device=endpoint_latent.device, dtype=torch.bool)
    selected = selected.reshape(1, -1).expand(frames, -1).reshape(-1)
    selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
    coarse_indices = torch.nonzero(~selected, as_tuple=False).flatten()

    coarse_values = parent_values[coarse_indices]
    coarse_coords = parent_coords[coarse_indices].to(torch.float32)
    fine_values, fine_indices, fine_coords = _expand_parent_patches(
        parent_values[selected_indices],
        selected_indices,
        coarse_grid=coarse_grid,
        fine_grid=fine_grid,
    )
    # Follow the official RALU position-ID convention during the mixed stage:
    # retained tokens keep integer coarse-grid IDs, while four children use
    # offsets {0, 0.5}.  At Stage 3 all IDs become the ordinary 2x integer grid.
    fine_coords[:, 1:] *= 0.5

    resume, upsample_weight, noise_weight = ralu_resume_parameters(end_data_time, z_value)
    covariance_scale = 1.0 / float(z_value) ** 2
    coarse_noise = _independent_transition_noise(
        coarse_values,
        generator=generator,
    )
    fine_noise = exact_grouped_projection_noise(
        fine_values.reshape(-1, 4, fine_values.shape[-1]),
        covariance_scale=covariance_scale,
        generator=generator,
    ).reshape_as(fine_values)
    values = torch.cat(
        [
            upsample_weight * coarse_values + noise_weight * coarse_noise,
            upsample_weight * fine_values + noise_weight * fine_noise,
        ],
        dim=0,
    )
    logger.info(
        "RALU handoff 1: "
        f"selected_spatial={int(spatial_edge_mask.sum())}/{spatial_edge_mask.numel()}, "
        f"coarse_tokens={coarse_values.shape[0]}, fine_tokens={fine_values.shape[0]}, "
        f"resume={resume:.6f}, a={upsample_weight:.6f}, b={noise_weight:.6f}"
    )
    return RALUPackedState(
        values=values,
        coords=torch.cat([coarse_coords, fine_coords], dim=0),
        coarse_parent_indices=coarse_indices,
        fine_indices=fine_indices,
        coarse_count=coarse_values.shape[0],
        channels=channels,
        frames=frames,
        coarse_grid=coarse_grid,
        fine_grid=fine_grid,
    )


def second_ralu_handoff(
    state: RALUPackedState,
    *,
    end_data_time: float,
    z_value: float,
    generator: torch.Generator,
    output_latent_size: tuple[int, int],
) -> torch.Tensor:
    """Expand remaining coarse tokens, inject exact noise, then apply geometry A."""

    new_values, new_indices, _ = _expand_parent_patches(
        state.coarse_values,
        state.coarse_parent_indices,
        coarse_grid=state.coarse_grid,
        fine_grid=state.fine_grid,
    )
    resume, upsample_weight, noise_weight = ralu_resume_parameters(end_data_time, z_value)
    covariance_scale = 1.0 / float(z_value) ** 2
    new_noise = exact_grouped_projection_noise(
        new_values.reshape(-1, 4, new_values.shape[-1]),
        covariance_scale=covariance_scale,
        generator=generator,
    ).reshape_as(new_values)
    old_noise = _independent_transition_noise(
        state.fine_values,
        generator=generator,
    )
    values = torch.cat(
        [
            upsample_weight * new_values + noise_weight * new_noise,
            upsample_weight * state.fine_values + noise_weight * old_noise,
        ],
        dim=0,
    )
    indices = torch.cat([new_indices, state.fine_indices], dim=0)
    expected = state.frames * state.fine_grid[0] * state.fine_grid[1]
    unique_count = torch.unique(indices).numel()
    if indices.numel() != expected or unique_count != expected:
        raise RuntimeError(
            "RALU handoff 2 does not cover the aligned HR grid exactly: "
            f"tokens={indices.numel()}, unique={unique_count}, expected={expected}"
        )
    order = torch.argsort(indices)
    sorted_indices = indices[order]
    if not torch.equal(sorted_indices, torch.arange(expected, device=indices.device)):
        raise RuntimeError("RALU handoff 2 produced a non-contiguous HR token grid")
    aligned = scatter_wan_patches(
        values[order],
        sorted_indices,
        channels=state.channels,
        frames=state.frames,
        grid=state.fine_grid,
    )
    out_h, out_w = (int(output_latent_size[0]), int(output_latent_size[1]))
    if out_h % 2 or out_w % 2 or out_h > aligned.shape[-2] or out_w > aligned.shape[-1]:
        raise ValueError(f"invalid patch-aligned output crop {(out_h, out_w)} for {tuple(aligned.shape[-2:])}")
    cropped = aligned[..., :out_h, :out_w].contiguous()
    logger.info(
        "RALU handoff 2: "
        f"aligned_latent={tuple(aligned.shape)} -> cropped_latent={tuple(cropped.shape)}, "
        f"resume={resume:.6f}, a={upsample_weight:.6f}, b={noise_weight:.6f}"
    )
    return cropped
