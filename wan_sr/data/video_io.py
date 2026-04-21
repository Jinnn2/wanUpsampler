from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
import torch


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def list_videos(video_dir: str | Path) -> list[Path]:
    root = Path(video_dir)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def read_video_frames(path: str | Path, max_frames: int | None = None) -> torch.Tensor:
    frames = []
    for index, frame in enumerate(iio.imiter(path)):
        if max_frames is not None and index >= max_frames:
            break
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(torch.from_numpy(frame).float() / 255.0)
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return torch.stack(frames, dim=0)


def iter_fixed_length_clips(
    frames: torch.Tensor,
    num_frames: int,
    stride: int | None = None,
    max_clips: int | None = None,
) -> Iterable[torch.Tensor]:
    stride = stride or num_frames
    count = 0
    for start in range(0, max(0, frames.shape[0] - num_frames + 1), stride):
        yield frames[start : start + num_frames]
        count += 1
        if max_clips is not None and count >= max_clips:
            return


def write_video(path: str | Path, video: torch.Tensor, fps: int = 16) -> None:
    """Write [T,H,W,C] float video in [0,1]."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (video.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    iio.imwrite(path, array, fps=fps)
