from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniversalAction:
    """User-facing low-resolution trajectory action.

    ``spatial_ratio`` is an edge-length ratio, not an area/token ratio.
    ``temporal_ratio`` scales the number of latent temporal intervals while
    preserving both endpoints. ``lr_nfe_ratio`` is the fraction of LR solver
    positions that perform a full DiT recomputation. The remaining positions
    reuse a cached prediction. ``switch_ratio`` is measured against the full
    reference trajectory.
    """

    spatial_ratio: float
    temporal_ratio: float
    lr_nfe_ratio: float
    switch_ratio: float

    def validate(self) -> None:
        if not 0.5 <= self.spatial_ratio <= 1.0:
            raise ValueError(
                "spatial_ratio must be in [0.5, 1.0] for the x2 RGB SR path, "
                f"got {self.spatial_ratio}"
            )
        if not 0.0 < self.temporal_ratio <= 1.0:
            raise ValueError(
                f"temporal_ratio must be in (0, 1], got {self.temporal_ratio}"
            )
        if not 0.0 < self.lr_nfe_ratio <= 1.0:
            raise ValueError(
                f"lr_nfe_ratio must be in (0, 1], got {self.lr_nfe_ratio}"
            )
        if not 0.8 <= self.switch_ratio <= 1.0:
            raise ValueError(
                "switch_ratio must be in [0.8, 1.0], "
                f"got {self.switch_ratio}"
            )


@dataclass(frozen=True)
class ResolvedSchedule:
    reference_nfe: int
    target_latent_shape: tuple[int, int, int, int]
    low_latent_shape: tuple[int, int, int, int]
    switch_step: int
    lr_compute_steps: tuple[int, ...]
    lr_cache_steps: tuple[int, ...]
    hr_compute_steps: tuple[int, ...]
    requested_spatial_ratio: float
    actual_spatial_ratio_h: float
    actual_spatial_ratio_w: float
    requested_temporal_ratio: float
    actual_temporal_ratio: float
    requested_lr_nfe_ratio: float
    actual_lr_nfe_ratio: float

    @property
    def lr_solver_steps(self) -> tuple[int, ...]:
        return tuple(range(self.switch_step))

    @property
    def total_full_dit_evaluations(self) -> int:
        return len(self.lr_compute_steps) + len(self.hr_compute_steps)

    @property
    def target_video_frames(self) -> int:
        return 4 * (self.target_latent_shape[1] - 1) + 1

    @property
    def low_video_frames(self) -> int:
        return 4 * (self.low_latent_shape[1] - 1) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_nfe": self.reference_nfe,
            "target_latent_shape": list(self.target_latent_shape),
            "low_latent_shape": list(self.low_latent_shape),
            "target_video_frames": self.target_video_frames,
            "low_video_frames": self.low_video_frames,
            "switch_step": self.switch_step,
            "lr_compute_steps": list(self.lr_compute_steps),
            "lr_cache_steps": list(self.lr_cache_steps),
            "hr_compute_steps": list(self.hr_compute_steps),
            "requested_spatial_ratio": self.requested_spatial_ratio,
            "actual_spatial_ratio_h": self.actual_spatial_ratio_h,
            "actual_spatial_ratio_w": self.actual_spatial_ratio_w,
            "requested_temporal_ratio": self.requested_temporal_ratio,
            "actual_temporal_ratio": self.actual_temporal_ratio,
            "requested_lr_nfe_ratio": self.requested_lr_nfe_ratio,
            "actual_lr_nfe_ratio": self.actual_lr_nfe_ratio,
            "total_full_dit_evaluations": self.total_full_dit_evaluations,
        }
