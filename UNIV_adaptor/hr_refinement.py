"""Independent HR grids for a fixed, already reconstructed transition state."""
from __future__ import annotations

import math
from collections.abc import Sequence


def resample_hr_sigmas(
    reference_sigmas: Sequence[float], *, boundary_step: int, hr_steps: int
) -> tuple[float, ...]:
    """Interpolate the reference suffix at uniform fractional reference indices.

    This preserves both endpoints, and returns the original suffix exactly when
    its evaluation count is unchanged. It does not apply sample_shift again.
    The terminal zero is a solver endpoint, not a model-evaluation timestep.
    """
    reference = tuple(float(value) for value in reference_sigmas)
    if len(reference) < 2 or any(not math.isfinite(v) for v in reference):
        raise ValueError("reference sigmas must contain finite values")
    if reference[-1] != 0.0 or not 0.0 < reference[0] < 1.0:
        raise ValueError("reference sigmas must start in (0, 1) and end at zero")
    if any(a <= b for a, b in zip(reference, reference[1:])):
        raise ValueError("reference sigmas must be strictly decreasing")
    if type(boundary_step) is not int or not 0 <= boundary_step < len(reference) - 1:
        raise ValueError("boundary_step must leave a nonempty HR suffix")
    if type(hr_steps) is not int or hr_steps < 1:
        raise ValueError("hr_steps must be a positive integer")
    suffix = reference[boundary_step:]
    intervals = len(suffix) - 1
    if hr_steps == intervals:
        return suffix
    result = []
    for index in range(hr_steps):
        position = index * intervals / hr_steps
        left = int(math.floor(position))
        fraction = position - left
        result.append(suffix[left] + fraction * (suffix[left + 1] - suffix[left]))
    return (*result, 0.0)


def direct_hr_sigmas(*, start_sigma: float, hr_steps: int) -> tuple[float, ...]:
    """Build a MrFlow-style linear grid from an explicit sigma to zero."""
    sigma = float(start_sigma)
    if not math.isfinite(sigma) or not 0.0 < sigma < 1.0:
        raise ValueError("start_sigma must be finite and in (0, 1)")
    if type(hr_steps) is not int or hr_steps < 1:
        raise ValueError("hr_steps must be a positive integer")
    return tuple(sigma * (hr_steps - index) / hr_steps for index in range(hr_steps + 1))


def install_hr_grid(scheduler, *, reference_sigmas, boundary_step: int, hr_steps: int):
    """Install actual adjacent solver intervals, with fresh multistep history.

    Keep prefix indices so the full-reference baseline has identical indexing.
    infer_steps is restored by the caller before a later pipeline preparation.
    """
    import torch

    suffix = resample_hr_sigmas(
        reference_sigmas, boundary_step=boundary_step, hr_steps=hr_steps
    )
    prefix = [float(value) for value in reference_sigmas[:boundary_step]]
    sigmas = torch.tensor([*prefix, *suffix], dtype=torch.float32, device="cpu")
    if hr_steps == len(reference_sigmas) - 1 - boundary_step:
        timesteps = scheduler.timesteps.clone()
    else:
        suffix_timesteps = (sigmas[boundary_step:-1] * scheduler.num_train_timesteps).to(
            device=scheduler.timesteps.device, dtype=scheduler.timesteps.dtype
        )
        timesteps = torch.cat((scheduler.timesteps[:boundary_step], suffix_timesteps))
    if bool((timesteps[1:] >= timesteps[:-1]).any()) or int(timesteps[-1]) <= 0:
        raise ValueError("HR grid collapses after model timestep quantization")
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    scheduler.infer_steps = len(timesteps)
    scheduler.reset_solver_history()
    return {
        "grid_policy": "linear_interpolation_in_reference_index",
        "boundary_step": boundary_step,
        "hr_steps": hr_steps,
        "sigmas": sigmas[boundary_step:].tolist(),
        "model_timesteps": timesteps[boundary_step:].cpu().tolist(),
        "compute_indices": list(range(boundary_step, len(timesteps))),
    }


def install_direct_hr_grid(scheduler, *, start_sigma: float, hr_steps: int):
    """Install an independent linear HR grid and clear all LR solver history."""
    import torch

    values = direct_hr_sigmas(start_sigma=start_sigma, hr_steps=hr_steps)
    sigmas = torch.tensor(values, dtype=torch.float32, device="cpu")
    timesteps = (sigmas[:-1] * scheduler.num_train_timesteps).to(
        device=scheduler.timesteps.device, dtype=scheduler.timesteps.dtype
    )
    if bool((timesteps[1:] >= timesteps[:-1]).any()) or int(timesteps[-1]) <= 0:
        raise ValueError("direct HR grid collapses after model timestep quantization")
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    scheduler.infer_steps = hr_steps
    scheduler.reset_solver_history()
    return {
        "grid_policy": "direct_sigma_linear",
        "start_sigma": float(sigmas[0]),
        "hr_steps": hr_steps,
        "sigmas": sigmas.tolist(),
        "model_timesteps": timesteps.cpu().tolist(),
        "compute_indices": list(range(hr_steps)),
    }
