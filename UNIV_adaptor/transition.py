from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any


DVG_LATENT_ANCHOR = "dvg_latent_anchor"
RGB_SR_VAE = "rgb_sr_vae"
TRANSITION_BASELINES = frozenset({DVG_LATENT_ANCHOR, RGB_SR_VAE})


@dataclass(frozen=True)
class DVGAnchorPlan:
    """Discrete form of DVG equations (11) and (12) for one axis."""

    source_length: int
    target_length: int
    anchors: tuple[int, ...]
    lower_source: tuple[int, ...]
    upper_source: tuple[int, ...]
    beta: tuple[float, ...]


def dvg_rounded_anchors(source_length: int, target_length: int) -> tuple[int, ...]:
    """Return DVG source anchors ``round(i * (N - 1) / (K - 1))``.

    Length-changing transitions are reconstruction operations, so ``N >= K``.
    Positive half ties use the conventional mathematical half-up rule. The
    paper does not specify a programming-language tie convention.
    """

    source_length = int(source_length)
    target_length = int(target_length)
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length and target_length must be positive")
    if target_length < source_length:
        raise ValueError(
            "DVG reconstruction cannot shrink an axis: "
            f"source={source_length}, target={target_length}"
        )
    if source_length == 1:
        return (0,)

    denominator = source_length - 1
    anchors = tuple(
        (2 * index * (target_length - 1) + denominator) // (2 * denominator)
        for index in range(source_length)
    )
    if anchors[0] != 0 or anchors[-1] != target_length - 1:
        raise RuntimeError(f"DVG anchor endpoints are invalid: {anchors}")
    if any(left >= right for left, right in zip(anchors, anchors[1:])):
        raise RuntimeError(f"DVG anchors are not strictly increasing: {anchors}")
    return anchors


def dvg_anchor_plan(source_length: int, target_length: int) -> DVGAnchorPlan:
    """Build the exact neighboring-anchor interpolation plan from DVG Eq. 12."""

    anchors = dvg_rounded_anchors(source_length, target_length)
    if source_length == 1:
        repeated = tuple(0 for _ in range(target_length))
        return DVGAnchorPlan(
            source_length=1,
            target_length=target_length,
            anchors=anchors,
            lower_source=repeated,
            upper_source=repeated,
            beta=tuple(0.0 for _ in range(target_length)),
        )

    lower_source: list[int] = []
    upper_source: list[int] = []
    beta: list[float] = []
    for position in range(target_length):
        lower = min(
            source_length - 2,
            max(0, bisect.bisect_right(anchors, position) - 1),
        )
        upper = lower + 1
        denominator = anchors[upper] - anchors[lower]
        lower_source.append(lower)
        upper_source.append(upper)
        beta.append((position - anchors[lower]) / denominator)
    return DVGAnchorPlan(
        source_length=source_length,
        target_length=target_length,
        anchors=anchors,
        lower_source=tuple(lower_source),
        upper_source=tuple(upper_source),
        beta=tuple(beta),
    )


def dvg_resize_axis(tensor, target_length: int, *, axis: int):
    """Apply the DVG rounded-anchor operator to one NumPy or Torch axis."""

    ndim = int(tensor.ndim)
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for a {ndim}D tensor")
    source_length = int(tensor.shape[axis])
    target_length = int(target_length)
    if source_length == target_length:
        return tensor
    plan = dvg_anchor_plan(source_length, target_length)

    if type(tensor).__module__.startswith("torch"):
        import torch

        moved = torch.movedim(tensor, axis, 0)
        if not moved.dtype.is_floating_point:
            raise TypeError(f"DVG interpolation requires a floating tensor, got {moved.dtype}")
        compute_dtype = (
            torch.float32
            if moved.dtype in {torch.float16, torch.bfloat16}
            else moved.dtype
        )
        moved_compute = moved.to(dtype=compute_dtype)
        lower_index = torch.tensor(plan.lower_source, device=tensor.device)
        upper_index = torch.tensor(plan.upper_source, device=tensor.device)
        weight = torch.tensor(plan.beta, device=tensor.device, dtype=compute_dtype).reshape(
            (target_length,) + (1,) * (ndim - 1)
        )
        resized = moved_compute.index_select(0, lower_index) * (1.0 - weight)
        resized = resized + moved_compute.index_select(0, upper_index) * weight
        resized = resized.to(dtype=tensor.dtype)
        return torch.movedim(resized, 0, axis).contiguous()

    import numpy as np

    moved = np.moveaxis(np.asarray(tensor), axis, 0)
    if not np.issubdtype(moved.dtype, np.floating):
        raise TypeError(f"DVG interpolation requires a floating array, got {moved.dtype}")
    weight = np.asarray(plan.beta, dtype=moved.dtype).reshape(
        (target_length,) + (1,) * (ndim - 1)
    )
    resized = moved[np.asarray(plan.lower_source)] * (1.0 - weight)
    resized = resized + moved[np.asarray(plan.upper_source)] * weight
    return np.ascontiguousarray(np.moveaxis(resized, 0, axis))


def dvg_resize_latent(
    clean_lr,
    target_latent_shape: tuple[int, int, int, int],
):
    """Apply DVG Eq. (11)-(12) sequentially along latent T, H, and W."""

    if clean_lr.ndim != 4:
        raise ValueError(f"clean_lr must be [C,T,H,W], got {tuple(clean_lr.shape)}")
    target = tuple(int(value) for value in target_latent_shape)
    if len(target) != 4 or any(value <= 0 for value in target):
        raise ValueError(f"target_latent_shape must be positive [C,T,H,W], got {target}")
    if int(clean_lr.shape[0]) != target[0]:
        raise ValueError(
            f"DVG cannot change latent channels: {clean_lr.shape[0]} -> {target[0]}"
        )

    resized = clean_lr
    for axis in (1, 2, 3):
        resized = dvg_resize_axis(resized, target[axis], axis=axis)
    return resized


def linear_resample_video(video, target_frames: int):
    """Endpoint-aligned pixel-space temporal interpolation for RGB SR baseline."""

    import torch

    if video.ndim != 4:
        raise ValueError(f"video must be [T,H,W,C], got {tuple(video.shape)}")
    source_frames = int(video.shape[0])
    if source_frames == target_frames:
        return video
    if target_frames < source_frames:
        raise ValueError(
            f"temporal reconstruction cannot shrink {source_frames} to {target_frames}"
        )
    if source_frames == 1:
        return video.expand(target_frames, -1, -1, -1).contiguous()
    positions = torch.linspace(
        0,
        source_frames - 1,
        target_frames,
        device=video.device,
        dtype=torch.float32,
    )
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long).clamp(max=source_frames - 1)
    weight = (positions - lower.to(positions.dtype)).to(video.dtype)
    weight = weight.view(target_frames, 1, 1, 1)
    return (video[lower] * (1.0 - weight) + video[upper] * weight).contiguous()


@dataclass(frozen=True)
class TransitionResult:
    baseline: str
    clean_hr: Any
    source_latent_shape: tuple[int, int, int, int]
    target_latent_shape: tuple[int, int, int, int]
    decoded_frames: int | None
    reconstructed_frames: int | None
    source_height: int | None
    source_width: int | None
    target_height: int | None
    target_width: int | None
    spatial_restore_applied: bool
    temporal_restore_applied: bool

    @property
    def spatial_sr_applied(self) -> bool:
        """Compatibility alias for the original RGB-only runtime record."""

        return self.spatial_restore_applied


class WanDVGAnchorTransition:
    """Paper-conformant DVG Eq. (11)-(12) latent T/H/W reconstruction."""

    baseline = DVG_LATENT_ANCHOR

    def lift(self, clean_lr, *, target_latent_shape: tuple[int, int, int, int]):
        source_shape = tuple(int(value) for value in clean_lr.shape)
        target_shape = tuple(int(value) for value in target_latent_shape)
        clean_hr = dvg_resize_latent(clean_lr, target_shape)
        if tuple(clean_hr.shape) != target_shape:
            raise RuntimeError(
                "DVG transition produced the wrong HR latent shape: "
                f"got {tuple(clean_hr.shape)}, expected {target_shape}"
            )
        return TransitionResult(
            baseline=self.baseline,
            clean_hr=clean_hr,
            source_latent_shape=source_shape,
            target_latent_shape=target_shape,
            decoded_frames=None,
            reconstructed_frames=None,
            source_height=None,
            source_width=None,
            target_height=None,
            target_width=None,
            spatial_restore_applied=source_shape[-2:] != target_shape[-2:],
            temporal_restore_applied=source_shape[1] != target_shape[1],
        )


class WanRGBSRTransition:
    """Wan clean latent -> RGB SR/interpolation -> VAE latent baseline."""

    baseline = RGB_SR_VAE

    def __init__(self, *, vae_codec, spatial_resolver, target_height: int, target_width: int):
        self.vae_codec = vae_codec
        self.spatial_resolver = spatial_resolver
        self.target_height = int(target_height)
        self.target_width = int(target_width)

    def _decode(self, clean_lr):
        decoded = self.vae_codec.decode(clean_lr)
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            decoded = decoded[0]
        if decoded.ndim != 4 or int(decoded.shape[0]) != 3:
            raise RuntimeError(
                f"unexpected Wan VAE decode shape: {tuple(decoded.shape)}"
            )
        return decoded

    def _encode(self, video, *, device, dtype):
        vae_input = video.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=dtype)
        encoded = self.vae_codec.encode(vae_input.mul(2.0).sub(1.0))
        if isinstance(encoded, (list, tuple)):
            if len(encoded) != 1:
                raise RuntimeError(
                    f"unexpected Wan VAE encode list length: {len(encoded)}"
                )
            encoded = encoded[0]
        if encoded.ndim == 5 and encoded.shape[0] == 1:
            encoded = encoded[0]
        if encoded.ndim != 4:
            raise RuntimeError(
                f"unexpected Wan VAE encode shape: {tuple(encoded.shape)}"
            )
        return encoded.to(device=device, dtype=dtype)

    def lift(self, clean_lr, *, target_latent_shape: tuple[int, int, int, int]):
        import torch

        source_shape = tuple(int(value) for value in clean_lr.shape)
        target_shape = tuple(int(value) for value in target_latent_shape)
        if not hasattr(self.vae_codec, "encode"):
            raise RuntimeError("RGB transition requires a full Wan VAE with encode()")
        if source_shape == target_shape:
            frames = 4 * (int(clean_lr.shape[1]) - 1) + 1
            return TransitionResult(
                baseline=self.baseline,
                clean_hr=clean_lr,
                source_latent_shape=source_shape,
                target_latent_shape=target_shape,
                decoded_frames=frames,
                reconstructed_frames=frames,
                source_height=self.target_height,
                source_width=self.target_width,
                target_height=self.target_height,
                target_width=self.target_width,
                spatial_restore_applied=False,
                temporal_restore_applied=False,
            )

        device = clean_lr.device
        dtype = clean_lr.dtype
        decoded = self._decode(clean_lr)
        source_height, source_width = int(decoded.shape[2]), int(decoded.shape[3])
        rgb_video = (
            ((decoded.float().clamp(-1, 1) + 1.0) * 0.5)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
        )
        del decoded
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

        spatial_needed = (
            source_height != self.target_height or source_width != self.target_width
        )
        if spatial_needed:
            if self.spatial_resolver is None:
                raise RuntimeError("spatial resize is required but no RGB SR resolver is loaded")
            if source_height * 2 < self.target_height or source_width * 2 < self.target_width:
                raise RuntimeError(
                    "x2 RGB SR cannot reach target size: "
                    f"source={(source_height, source_width)}, "
                    f"target={(self.target_height, self.target_width)}"
                )
            rgb_video = self.spatial_resolver.resize(
                rgb_video,
                target_height=self.target_height,
                target_width=self.target_width,
            )

        target_frames = 4 * (int(target_shape[1]) - 1) + 1
        decoded_frames = int(rgb_video.shape[0])
        temporal_needed = decoded_frames != target_frames
        if temporal_needed:
            rgb_video = linear_resample_video(rgb_video, target_frames)

        clean_hr = self._encode(rgb_video, device=device, dtype=dtype)
        if tuple(clean_hr.shape) != target_shape:
            raise RuntimeError(
                "RGB transition produced the wrong HR latent shape: "
                f"got {tuple(clean_hr.shape)}, expected {target_shape}"
            )
        del rgb_video
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
        return TransitionResult(
            baseline=self.baseline,
            clean_hr=clean_hr,
            source_latent_shape=source_shape,
            target_latent_shape=target_shape,
            decoded_frames=decoded_frames,
            reconstructed_frames=target_frames,
            source_height=source_height,
            source_width=source_width,
            target_height=self.target_height,
            target_width=self.target_width,
            spatial_restore_applied=spatial_needed,
            temporal_restore_applied=temporal_needed,
        )
