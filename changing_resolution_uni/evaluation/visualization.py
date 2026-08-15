from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


def make_comparison_panel(
    labeled_inputs: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Create a labeled horizontal MP4 panel with ffmpeg."""

    if not labeled_inputs:
        raise ValueError("At least one input is required for a comparison panel")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required when --save_visuals is enabled")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_dir = output_path.parent / ".panels" / output_path.stem
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_paths = []
    for index, (input_path, label) in enumerate(labeled_inputs):
        panel_path = panel_dir / f"{index:02d}.mp4"
        safe_label = _escape_drawtext(label)
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                (
                    f"scale={width}:{height}:flags=bicubic,fps={fps},"
                    "drawbox=x=0:y=0:w=iw:h=42:color=black@0.55:t=fill,"
                    f"drawtext=text='{safe_label}':x=16:y=10:fontsize=22:fontcolor=white"
                ),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(panel_path),
            ],
            check=True,
        )
        panel_paths.append(panel_path)
    inputs = [item for path in panel_paths for item in ("-i", str(path))]
    labels = "".join(f"[{index}:v]" for index in range(len(panel_paths)))
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            f"{labels}hstack=inputs={len(panel_paths)}[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_path),
        ],
        check=True,
    )
    return output_path


def _escape_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
