"""Independent HR grids for a fixed, already reconstructed transition state."""
from __future__ import annotations

import math
import struct
from collections.abc import Sequence


def wan_reference_sigmas(*, reference_nfe: int, sample_shift: float) -> tuple[float, ...]:
    """Reproduce Wan's shifted inference sigma grid without importing LightX2V."""
    if type(reference_nfe) is not int or reference_nfe < 1:
        raise ValueError("reference_nfe must be a positive integer")
    shift = float(sample_shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("sample_shift must be finite and positive")
    raw = (0.999 * (1.0 - index / reference_nfe) for index in range(reference_nfe))
    shifted = tuple(shift * value / (1.0 + (shift - 1.0) * value) for value in raw)
    return (*shifted, 0.0)


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


def quantize_float32_timesteps(
    sigmas: Sequence[float], *, num_train_timesteps: int
) -> tuple[int, ...]:
    """Match torch float32 multiplication followed by integer conversion."""
    if type(num_train_timesteps) is not int or num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be a positive integer")

    def float32(value: float) -> float:
        return struct.unpack("=f", struct.pack("=f", float(value)))[0]

    scale = float32(num_train_timesteps)
    return tuple(int(float32(float32(value) * scale)) for value in sigmas)


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


def install_lr_grid(scheduler, *, reference_sigmas, lr_steps: int):
    """Replace the dense reference trajectory with true, fully-computed LR intervals."""
    import torch

    reference = tuple(float(value) for value in reference_sigmas)
    values = resample_hr_sigmas(reference, boundary_step=0, hr_steps=lr_steps)
    sigmas = torch.tensor(values, dtype=torch.float32, device="cpu")
    if lr_steps == len(reference) - 1:
        timesteps = scheduler.timesteps.clone()
    else:
        timesteps = (sigmas[:-1] * scheduler.num_train_timesteps).to(
            device=scheduler.timesteps.device, dtype=scheduler.timesteps.dtype
        )
    if bool((timesteps[1:] >= timesteps[:-1]).any()) or int(timesteps[-1]) <= 0:
        raise ValueError("LR grid collapses after model timestep quantization")
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    scheduler.infer_steps = lr_steps
    scheduler.reset_solver_history()
    return {
        "grid_policy": "linear_interpolation_in_reference_index",
        "reference_nfe": len(reference) - 1,
        "lr_steps": lr_steps,
        "sigmas": sigmas.tolist(),
        "model_timesteps": timesteps.cpu().tolist(),
        "compute_indices": list(range(lr_steps)),
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
