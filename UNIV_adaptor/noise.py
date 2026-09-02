from __future__ import annotations

import math

from .transition import dvg_rounded_anchors


_MASK64 = (1 << 64) - 1


def _splitmix64(values):
    import numpy as np

    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def _anchor_indices(source_length: int, target_length: int):
    import numpy as np

    return np.asarray(
        dvg_rounded_anchors(source_length, target_length),
        dtype=np.uint64,
    )


def coordinate_gaussian_numpy(
    shape: tuple[int, int, int, int],
    *,
    seed: int,
    reference_shape: tuple[int, int, int, int] | None = None,
):
    """Generate deterministic coordinate-hash Gaussian noise.

    For a low-resolution shape, coordinates are mapped to endpoint-preserving
    anchors in ``reference_shape`` before hashing. Shared anchors therefore
    receive exactly the same noise as the target HR tensor.
    """

    import numpy as np

    if len(shape) != 4 or any(int(v) <= 0 for v in shape):
        raise ValueError(f"shape must be positive [C,T,H,W], got {shape}")
    shape = tuple(int(v) for v in shape)
    reference = shape if reference_shape is None else tuple(int(v) for v in reference_shape)
    if len(reference) != 4 or any(v <= 0 for v in reference):
        raise ValueError(
            f"reference_shape must be positive [C,T,H,W], got {reference_shape}"
        )
    if shape[0] != reference[0]:
        raise ValueError(
            f"channel count must match reference: shape={shape}, reference={reference}"
        )
    if any(source > target for source, target in zip(shape[1:], reference[1:])):
        raise ValueError(
            f"source spatial/temporal axes cannot exceed reference: {shape} vs {reference}"
        )

    channels, _, _, _ = shape
    _, ref_t, ref_h, ref_w = reference
    c = np.arange(channels, dtype=np.uint64)[:, None, None, None]
    t = _anchor_indices(shape[1], ref_t)[None, :, None, None]
    h = _anchor_indices(shape[2], ref_h)[None, None, :, None]
    w = _anchor_indices(shape[3], ref_w)[None, None, None, :]
    with np.errstate(over="ignore"):
        keys = (((c * np.uint64(ref_t) + t) * np.uint64(ref_h) + h) * np.uint64(ref_w) + w)
        keys = keys ^ np.uint64(seed & _MASK64)
        hash1 = _splitmix64(keys)
        hash2 = _splitmix64(hash1 ^ np.uint64(0xD2B74407B1CE6E93))

    scale = 1.0 / float(1 << 53)
    u1 = ((hash1 >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    u2 = ((hash2 >> np.uint64(11)).astype(np.float64) + 0.5) * scale
    gaussian = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u2)
    return gaussian.astype(np.float32)


def coordinate_gaussian_tensor(
    shape: tuple[int, int, int, int],
    *,
    seed: int,
    device,
    dtype,
    reference_shape: tuple[int, int, int, int] | None = None,
):
    import torch

    array = coordinate_gaussian_numpy(
        shape,
        seed=seed,
        reference_shape=reference_shape,
    )
    return torch.from_numpy(array).to(device=device, dtype=dtype)
