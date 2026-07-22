from __future__ import annotations


def ralu_resume_parameters(end_data_time: float, z_value: float) -> tuple[float, float, float]:
    """Return official RALU ``(s, a, b)`` handoff parameters.

    ``s`` is the resumed data time, ``a`` multiplies the upsampled state, and
    ``b`` multiplies corrective noise. RALU denotes ``Z = 1/sqrt(c)``.
    """

    end = float(end_data_time)
    z = float(z_value)
    if not 0.0 < end < 1.0:
        raise ValueError(f"end_data_time must be in (0, 1), got {end}")
    if z < 2.0:
        raise ValueError(f"z_value must be at least 2 for a four-child transition, got {z}")
    denominator = z * (1.0 - end) + end
    resume = end / denominator
    upsample_weight = 1.0 / denominator
    noise_weight = 1.0 - resume
    return resume, upsample_weight, noise_weight


def ralu_stage_sigmas(
    *,
    start_data_time: float,
    end_data_time: float,
    num_steps: int,
    shift: float,
) -> list[float]:
    """Build one official truncated shifted-flow stage, including both ends."""

    start = float(start_data_time)
    end = float(end_data_time)
    steps = int(num_steps)
    flow_shift = float(shift)
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(f"expected 0 <= start < end <= 1, got {(start, end)}")
    if steps < 1:
        raise ValueError(f"num_steps must be positive, got {steps}")
    if flow_shift <= 0.0:
        raise ValueError(f"shift must be positive, got {flow_shift}")

    sigma_start = 1.0 - start
    sigma_end = 1.0 - end

    def inverse_shift(sigma: float) -> float:
        return sigma / (flow_shift - (flow_shift - 1.0) * sigma)

    raw_start = inverse_shift(sigma_start)
    raw_end = inverse_shift(sigma_end)
    values: list[float] = []
    for index in range(steps + 1):
        fraction = index / steps
        raw = raw_start + fraction * (raw_end - raw_start)
        shifted = flow_shift * raw / (1.0 + (flow_shift - 1.0) * raw)
        values.append(float(shifted))
    values[0] = sigma_start
    values[-1] = sigma_end
    return values
