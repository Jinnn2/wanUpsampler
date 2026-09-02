from __future__ import annotations

import math
from collections.abc import Mapping

from .core import ResolvedSchedule, UniversalAction


def action_from_config(config: Mapping[str, object]) -> UniversalAction:
    raw = config.get("univ_action")
    if not isinstance(raw, Mapping):
        raise ValueError("config must contain an object-valued 'univ_action'")
    required = (
        "spatial_ratio",
        "temporal_ratio",
        "lr_nfe_ratio",
        "switch_ratio",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"univ_action is missing required keys: {missing}")
    action = UniversalAction(**{key: float(raw[key]) for key in required})
    action.validate()
    return action


def _nearest_even(value: float, *, minimum: int = 2) -> int:
    # Use half-up rounding so an exact 0.5 target such as 90 -> 45 chooses 46
    # instead of Python's banker's-rounding result 44. The x2 SR stage must be
    # able to cover the requested target without scaling beyond its native x2.
    rounded = int(math.floor(value / 2.0 + 0.5)) * 2
    return max(minimum, rounded)


def _scaled_temporal_length(target_length: int, ratio: float) -> int:
    if target_length < 2:
        return target_length
    # Scale intervals rather than samples so the first and last video frames
    # remain shared anchor locations.
    return max(2, min(target_length, int(round((target_length - 1) * ratio)) + 1))


def uniform_topk_steps(prefix_steps: int, full_compute_steps: int) -> tuple[int, ...]:
    """Return an exact, endpoint-preserving uniform top-k schedule.

    Step indices are zero-based. For a multi-step prefix, both the first step
    (cache initialization) and the final LR step (fresh transition estimate)
    are mandatory full computations.
    """

    if prefix_steps <= 0:
        raise ValueError(f"prefix_steps must be positive, got {prefix_steps}")
    if not 1 <= full_compute_steps <= prefix_steps:
        raise ValueError(
            "full_compute_steps must be in [1, prefix_steps], "
            f"got {full_compute_steps} for prefix {prefix_steps}"
        )
    if prefix_steps == 1:
        return (0,)
    if full_compute_steps < 2:
        raise ValueError(
            "a multi-step cache schedule needs at least two full computations "
            "to preserve its first and last steps"
        )
    if full_compute_steps == prefix_steps:
        return tuple(range(prefix_steps))

    # round(i * (n - 1) / (k - 1)) is strictly increasing when k <= n.
    indices = tuple(
        int(round(i * (prefix_steps - 1) / (full_compute_steps - 1)))
        for i in range(full_compute_steps)
    )
    if len(set(indices)) != full_compute_steps:
        raise RuntimeError(f"uniform top-k construction produced duplicates: {indices}")
    return indices


def resolve_schedule(
    action: UniversalAction,
    *,
    reference_nfe: int,
    target_latent_shape: tuple[int, int, int, int],
) -> ResolvedSchedule:
    action.validate()
    if reference_nfe < 2:
        raise ValueError(f"reference_nfe must be at least 2, got {reference_nfe}")
    if len(target_latent_shape) != 4:
        raise ValueError(
            "target_latent_shape must be [C,T,H,W], "
            f"got {target_latent_shape}"
        )
    channels, target_t, target_h, target_w = (int(v) for v in target_latent_shape)
    if min(channels, target_t, target_h, target_w) <= 0:
        raise ValueError(f"target_latent_shape must be positive, got {target_latent_shape}")
    if target_h % 2 or target_w % 2:
        raise ValueError(
            "Wan target latent H/W must be even for patching, "
            f"got {(target_h, target_w)}"
        )

    low_t = _scaled_temporal_length(target_t, action.temporal_ratio)
    low_h = min(target_h, _nearest_even(target_h * action.spatial_ratio))
    low_w = min(target_w, _nearest_even(target_w * action.spatial_ratio))

    switch_step = int(round(reference_nfe * action.switch_ratio))
    switch_step = max(1, min(reference_nfe, switch_step))
    requested_compute = int(round(switch_step * action.lr_nfe_ratio))
    minimum_compute = 1 if switch_step == 1 else 2
    lr_compute_count = max(minimum_compute, min(switch_step, requested_compute))
    lr_compute_steps = uniform_topk_steps(switch_step, lr_compute_count)
    compute_set = set(lr_compute_steps)
    lr_cache_steps = tuple(step for step in range(switch_step) if step not in compute_set)
    hr_compute_steps = tuple(range(switch_step, reference_nfe))

    low_shape = (channels, low_t, low_h, low_w)
    return ResolvedSchedule(
        reference_nfe=reference_nfe,
        target_latent_shape=(channels, target_t, target_h, target_w),
        low_latent_shape=low_shape,
        switch_step=switch_step,
        lr_compute_steps=lr_compute_steps,
        lr_cache_steps=lr_cache_steps,
        hr_compute_steps=hr_compute_steps,
        requested_spatial_ratio=action.spatial_ratio,
        actual_spatial_ratio_h=low_h / target_h,
        actual_spatial_ratio_w=low_w / target_w,
        requested_temporal_ratio=action.temporal_ratio,
        actual_temporal_ratio=(low_t - 1) / max(1, target_t - 1),
        requested_lr_nfe_ratio=action.lr_nfe_ratio,
        actual_lr_nfe_ratio=lr_compute_count / switch_step,
    )
