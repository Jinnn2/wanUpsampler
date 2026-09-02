from __future__ import annotations

import math


def _is_torch_tensor(value) -> bool:
    return type(value).__module__.startswith("torch")


def _float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def _validate_state(state, name: str) -> None:
    if int(state.ndim) != 4:
        raise ValueError(f"{name} must be [C,T,H,W], got {tuple(state.shape)}")
    if any(int(length) <= 0 for length in state.shape):
        raise ValueError(f"{name} has a non-positive shape: {tuple(state.shape)}")


def state_moments(state) -> dict[str, object]:
    """Return JSON-safe population statistics for a latent state."""

    _validate_state(state, "state")
    if _is_torch_tensor(state):
        import torch

        values = state.detach().to(dtype=torch.float32)
        return {
            "shape": [int(value) for value in state.shape],
            "mean": _float(values.mean()),
            "std": _float(values.std(unbiased=False)),
            "rms": _float(values.square().mean().sqrt()),
            "min": _float(values.min()),
            "max": _float(values.max()),
        }

    import numpy as np

    values = np.asarray(state, dtype=np.float64)
    return {
        "shape": [int(value) for value in values.shape],
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def temporal_difference_metrics(state) -> dict[str, object]:
    """Measure first-order latent motion along the temporal axis."""

    _validate_state(state, "state")
    if int(state.shape[1]) < 2:
        return {"available": False, "reason": "temporal_length_lt_2"}
    if _is_torch_tensor(state):
        import torch

        values = state.detach().to(dtype=torch.float32)
        difference = values[:, 1:] - values[:, :-1]
        return {
            "available": True,
            "mean_abs": _float(difference.abs().mean()),
            "rms": _float(difference.square().mean().sqrt()),
        }

    import numpy as np

    values = np.asarray(state, dtype=np.float64)
    difference = np.diff(values, axis=1)
    return {
        "available": True,
        "mean_abs": float(np.mean(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
    }


def _spectral_summary(power, normalized_frequency) -> dict[str, float]:
    total = power.sum()
    total_value = _float(total)
    if total_value <= 0.0:
        return {
            "total_power": 0.0,
            "mean_power": 0.0,
            "centroid_nyquist": 0.0,
            "high_frequency_ratio": 0.0,
        }
    centroid = (power * normalized_frequency).sum() / total
    high = power[normalized_frequency >= 0.5].sum() / total
    return {
        "total_power": total_value,
        "mean_power": _float(power.mean()),
        "centroid_nyquist": _float(centroid),
        "high_frequency_ratio": _float(high),
    }


def spectral_metrics(state) -> dict[str, object]:
    """Summarize temporal and spatial spectra of the channel-mean latent.

    Frequencies are normalized to Nyquist. ``high_frequency_ratio`` is the
    fraction of power at normalized frequency/radius >= 0.5.
    """

    _validate_state(state, "state")
    if _is_torch_tensor(state):
        import torch

        signal = state.detach().to(dtype=torch.float32).mean(dim=0)
        temporal_power = (
            torch.fft.fft(signal, dim=0, norm="ortho")
            .abs()
            .square()
            .mean(dim=(1, 2))
        )
        temporal_frequency = torch.fft.fftfreq(
            int(signal.shape[0]), device=signal.device
        ).abs() / 0.5

        spatial_power = (
            torch.fft.fft2(signal, dim=(-2, -1), norm="ortho")
            .abs()
            .square()
            .mean(dim=0)
        )
        height_frequency = torch.fft.fftfreq(
            int(signal.shape[-2]), device=signal.device
        )[:, None]
        width_frequency = torch.fft.fftfreq(
            int(signal.shape[-1]), device=signal.device
        )[None, :]
        spatial_radius = torch.sqrt(
            height_frequency.square() + width_frequency.square()
        ) / (0.5 * math.sqrt(2.0))
        return {
            "channel_reduction": "mean",
            "high_frequency_threshold_nyquist": 0.5,
            "temporal": _spectral_summary(temporal_power, temporal_frequency),
            "spatial": _spectral_summary(spatial_power, spatial_radius),
        }

    import numpy as np

    signal = np.asarray(state, dtype=np.float64).mean(axis=0)
    temporal_power = np.square(
        np.abs(np.fft.fft(signal, axis=0, norm="ortho"))
    ).mean(axis=(1, 2))
    temporal_frequency = np.abs(np.fft.fftfreq(int(signal.shape[0]))) / 0.5

    spatial_power = np.square(
        np.abs(np.fft.fft2(signal, axes=(-2, -1), norm="ortho"))
    ).mean(axis=0)
    height_frequency = np.fft.fftfreq(int(signal.shape[-2]))[:, None]
    width_frequency = np.fft.fftfreq(int(signal.shape[-1]))[None, :]
    spatial_radius = np.sqrt(height_frequency**2 + width_frequency**2) / (
        0.5 * math.sqrt(2.0)
    )
    return {
        "channel_reduction": "mean",
        "high_frequency_threshold_nyquist": 0.5,
        "temporal": _spectral_summary(temporal_power, temporal_frequency),
        "spatial": _spectral_summary(spatial_power, spatial_radius),
    }


def state_diagnostics(state) -> dict[str, object]:
    return {
        "moments": state_moments(state),
        "spectrum": spectral_metrics(state),
        "temporal_difference": temporal_difference_metrics(state),
    }


def state_distance(candidate, reference) -> dict[str, object]:
    """Distance from a transitioned state to a native HR trajectory state."""

    _validate_state(candidate, "candidate")
    _validate_state(reference, "reference")
    if tuple(candidate.shape) != tuple(reference.shape):
        raise ValueError(
            "native HR state shape mismatch: "
            f"candidate={tuple(candidate.shape)}, reference={tuple(reference.shape)}"
        )

    if _is_torch_tensor(candidate) or _is_torch_tensor(reference):
        import torch

        if not _is_torch_tensor(candidate) or not _is_torch_tensor(reference):
            raise TypeError("candidate and reference must use the same array backend")
        left = candidate.detach().to(dtype=torch.float32)
        right = reference.detach().to(device=left.device, dtype=torch.float32)
        difference = left - right
        denominator = right.square().mean().sqrt().clamp_min(1e-12)
        cosine_denominator = left.norm() * right.norm()
        cosine_similarity = torch.where(
            cosine_denominator > 0,
            (left.flatten() @ right.flatten()) / cosine_denominator,
            torch.ones_like(cosine_denominator),
        )
        return {
            "available": True,
            "rmse": _float(difference.square().mean().sqrt()),
            "mae": _float(difference.abs().mean()),
            "relative_l2": _float(difference.square().mean().sqrt() / denominator),
            "cosine_distance": _float(1.0 - cosine_similarity),
        }

    import numpy as np

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    difference = left - right
    reference_rms = max(float(np.sqrt(np.mean(np.square(right)))), 1e-12)
    cosine_denominator = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
    cosine_similarity = (
        float(np.dot(left.ravel(), right.ravel()) / cosine_denominator)
        if cosine_denominator > 0.0
        else 1.0
    )
    return {
        "available": True,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.sqrt(np.mean(np.square(difference))) / reference_rms),
        "cosine_distance": float(1.0 - cosine_similarity),
    }


def transition_state_diagnostics(
    *,
    clean_lr,
    clean_hr,
    renoised_hr,
    native_hr_state=None,
) -> dict[str, object]:
    result = {
        "schema": "univ_transition_diagnostics_v1",
        "states": {
            "clean_lr": state_diagnostics(clean_lr),
            "clean_hr": state_diagnostics(clean_hr),
            "renoised_hr": state_diagnostics(renoised_hr),
        },
    }
    if native_hr_state is None:
        result["native_hr_state_distance"] = {
            "available": False,
            "reason": "native_hr_state_not_provided",
        }
    else:
        result["states"]["native_hr"] = state_diagnostics(native_hr_state)
        result["native_hr_state_distance"] = state_distance(
            renoised_hr,
            native_hr_state,
        )
    return result
