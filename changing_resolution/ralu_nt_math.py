from __future__ import annotations

import math


def ralu_transition_coefficients(end_data_time: float, covariance_scale: float) -> tuple[float, float, float]:
    """Return the single-transition NT-matching coefficients from RALU Eq. (7)."""

    end = float(end_data_time)
    scale = float(covariance_scale)
    if not 0.0 < end < 1.0:
        raise ValueError(f"end_data_time must be in (0, 1), got {end}")
    if scale <= 0.0:
        raise ValueError(f"covariance_scale must be positive, got {scale}")

    delta = (1.0 - end) / math.sqrt(scale)
    denominator = delta + end
    resume = end / denominator
    upsample_weight = 1.0 / denominator
    noise_weight = delta / denominator
    return resume, upsample_weight, noise_weight


def shifted_sigma_suffix(resume_sigma: float, num_steps: int, shift: float) -> list[float]:
    """Build a truncated shifted-flow sigma suffix including terminal zero."""

    sigma = float(resume_sigma)
    steps = int(num_steps)
    flow_shift = float(shift)
    if not 0.0 < sigma < 1.0:
        raise ValueError(f"resume_sigma must be in (0, 1), got {sigma}")
    if steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {steps}")
    if flow_shift <= 0.0:
        raise ValueError(f"shift must be positive, got {flow_shift}")

    raw_start = sigma / (flow_shift - (flow_shift - 1.0) * sigma)
    values = []
    for index in range(steps + 1):
        raw_sigma = raw_start * (1.0 - index / steps)
        shifted = flow_shift * raw_sigma / (1.0 + (flow_shift - 1.0) * raw_sigma)
        values.append(float(shifted))
    values[-1] = 0.0
    return values
