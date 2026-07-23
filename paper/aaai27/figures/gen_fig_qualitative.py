#!/usr/bin/env python3
"""Generate Supplementary Figure S1 from content-aligned TrajScale video groups.

The spatial panel uses prompt 05 / seed 9705 at frame 41. The temporal
panel uses prompt 07 / seed 9707 around frame 61. All crop coordinates are
defined in the 1248x720 output coordinate system and scaled proportionally for
the 640x368 Native-HR (estimated) reference.

Outputs:
    supplementary/fig_qualitative.pdf
    supplementary/fig_qualitative.png
    supplementary/fig_qualitative_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from _figure_style import COLORS, REPO_ROOT, apply_publication_style, save_figure


try:
    import imageio_ffmpeg
except ModuleNotFoundError:
    local_deps = REPO_ROOT / "outputs" / "aaai27_figure_work" / "python_deps"
    sys.path.insert(0, str(local_deps))
    import imageio_ffmpeg


VIDEO_ROOT = REPO_ROOT / "outputs" / "aaai27_figure_work" / "talh_figure_video_candidates"
REFERENCE_SIZE = (1248, 720)
SPATIAL_FRAME = 40  # zero-based; displayed as Frame 41
TEMPORAL_CENTER = 60  # zero-based; displayed as Frame 61
TEMPORAL_FRAMES = (56, 58, 60, 62, 64)
SPATIAL_CROPS = {
    "A": (180, 220, 680, 720),
    "B": (450, 0, 950, 500),
}
TEMPORAL_CROP = (420, 140, 930, 650)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(directory: Path, token: str) -> Path:
    matches = sorted(path for path in directory.glob("*.mp4") if token.lower() in path.name.lower())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one video matching {token!r} in {directory}, found {matches}")
    return matches[0]


def decode_frames(path: Path, indices: set[int]) -> dict[int, Image.Image]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    frames: dict[int, Image.Image] = {}
    try:
        for frame_index, frame_bytes in enumerate(reader):
            if frame_index in indices:
                frames[frame_index] = Image.frombytes("RGB", (width, height), frame_bytes)
            if frame_index >= max(indices):
                break
    finally:
        reader.close()
    missing = indices - set(frames)
    if missing:
        raise RuntimeError(f"Missing frames {sorted(missing)} in {path}")
    return frames


def scaled_box(box: tuple[int, int, int, int], image: Image.Image) -> tuple[int, int, int, int]:
    sx = image.width / REFERENCE_SIZE[0]
    sy = image.height / REFERENCE_SIZE[1]
    x0, y0, x1, y1 = box
    return tuple(round(value * scale) for value, scale in zip((x0, y0, x1, y1), (sx, sy, sx, sy)))


def show_image(ax: plt.Axes, image: Image.Image) -> None:
    ax.imshow(image, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_crop_box(ax: plt.Axes, image: Image.Image, box: tuple[int, int, int, int], color: str) -> None:
    x0, y0, x1, y1 = scaled_box(box, image)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.15))


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(scaled_box(box, image))


def label_axis(ax: plt.Axes, label: str, color: str, *, estimated: bool = False) -> None:
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.14), 0.035, 0.72, transform=ax.transAxes, facecolor=color, edgecolor="none"))
    ax.text(0.10, 0.56 if estimated else 0.50, label, transform=ax.transAxes, ha="left", va="center", fontsize=7.25, fontweight="bold", color=COLORS["text"], linespacing=1.05)
    if estimated:
        ax.text(0.10, 0.20, "368p aligned", transform=ax.transAxes, ha="left", va="center", fontsize=5.9, color=COLORS["muted"])


def main() -> None:
    apply_publication_style()
    spatial_dir = VIDEO_ROOT / "TALH-Q_spatial_detail" / "prompt_05_seed9705"
    temporal_dir = VIDEO_ROOT / "TALH-E_motion" / "prompt_07_seed9707"

    spatial_specs = [
        ("Native-HR\n(estimated)", "Native-HR", COLORS["native"], True),
        ("Trilinear @ 40", "Trilinear-at-40", COLORS["baseline"], False),
        ("CRLU-only @ 40", "CLL-only-at-40", COLORS["cll"], False),
        ("TrajScale-40", "TALH-Q-at-40", COLORS["talh_q"], False),
    ]
    temporal_specs = [
        ("Native-HR\n(estimated)", "Native-HR", COLORS["native"], True),
        ("Trilinear\n@ 45", "Trilinear-at-45", COLORS["baseline"], False),
        ("CRLU-only\n@ 45", "CLL-only-at-45", COLORS["cll"], False),
        ("TrajScale-45", "TALH-E-at-45", COLORS["talh_e"], False),
    ]

    spatial_data = []
    temporal_data = []
    manifest_videos: list[dict[str, object]] = []
    for label, token, color, estimated in spatial_specs:
        path = find_video(spatial_dir, token)
        frames = decode_frames(path, {SPATIAL_FRAME})
        spatial_data.append((label, color, estimated, path, frames[SPATIAL_FRAME]))
        manifest_videos.append({"panel": "spatial", "method": label.replace("\n", " "), "path": str(path), "sha256": sha256(path)})
    for label, token, color, estimated in temporal_specs:
        path = find_video(temporal_dir, token)
        frames = decode_frames(path, set(TEMPORAL_FRAMES) | {TEMPORAL_CENTER})
        temporal_data.append((label, color, estimated, path, frames))
        manifest_videos.append({"panel": "temporal", "method": label.replace("\n", " "), "path": str(path), "sha256": sha256(path)})

    fig = plt.figure(figsize=(7.0, 4.0), facecolor="white")
    outer = fig.add_gridspec(1, 2, left=0.012, right=0.995, bottom=0.075, top=0.885, wspace=0.055, width_ratios=[1.0, 1.06])
    spatial_grid = outer[0].subgridspec(4, 4, width_ratios=[1.10, 1.62, 1.0, 1.0], hspace=0.035, wspace=0.035)
    temporal_grid = outer[1].subgridspec(4, 7, width_ratios=[1.12, 1.55, 0.72, 0.72, 0.72, 0.72, 0.72], hspace=0.035, wspace=0.028)

    fig.text(0.015, 0.965, "(a)  Spatial detail — prompt 05, seed 9705", ha="left", va="top", fontsize=9.2, fontweight="bold")
    fig.text(0.515, 0.965, "(b)  Temporal behavior — prompt 07, seed 9707", ha="left", va="top", fontsize=9.2, fontweight="bold")
    fig.text(0.202, 0.905, "Full frame", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    fig.text(0.367, 0.905, "Crop A", ha="center", va="bottom", fontsize=7.2, color=COLORS["cll"], fontweight="bold")
    fig.text(0.455, 0.905, "Crop B", ha="center", va="bottom", fontsize=7.2, color=COLORS["taa"], fontweight="bold")
    temporal_headers = [(0.657, "Context"), (0.774, "t-4"), (0.817, "t-2"), (0.860, "t"), (0.903, "t+2"), (0.946, "t+4")]
    for xpos, text in temporal_headers:
        fig.text(xpos, 0.905, text, ha="center", va="bottom", fontsize=7.0, fontweight="bold")

    for row, (label, color, estimated, _path, image) in enumerate(spatial_data):
        label_ax = fig.add_subplot(spatial_grid[row, 0])
        label_axis(label_ax, label, color, estimated=estimated)
        full_ax = fig.add_subplot(spatial_grid[row, 1])
        show_image(full_ax, image)
        add_crop_box(full_ax, image, SPATIAL_CROPS["A"], COLORS["cll"])
        add_crop_box(full_ax, image, SPATIAL_CROPS["B"], COLORS["taa"])
        for column, (crop_name, crop_color) in enumerate([("A", COLORS["cll"]), ("B", COLORS["taa"])], start=2):
            crop_ax = fig.add_subplot(spatial_grid[row, column])
            show_image(crop_ax, crop(image, SPATIAL_CROPS[crop_name]))
            for spine in crop_ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(crop_color)
                spine.set_linewidth(0.8)

    for row, (label, color, estimated, _path, frames) in enumerate(temporal_data):
        label_ax = fig.add_subplot(temporal_grid[row, 0])
        label_axis(label_ax, label, color, estimated=estimated)
        context = frames[TEMPORAL_CENTER]
        context_ax = fig.add_subplot(temporal_grid[row, 1])
        show_image(context_ax, context)
        add_crop_box(context_ax, context, TEMPORAL_CROP, COLORS["talh_e"])
        for column, frame_index in enumerate(TEMPORAL_FRAMES, start=2):
            frame_ax = fig.add_subplot(temporal_grid[row, column])
            show_image(frame_ax, crop(frames[frame_index], TEMPORAL_CROP))
            if estimated:
                for spine in frame_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(COLORS["native"])
                    spine.set_linestyle((0, (2, 2)))
                    spine.set_linewidth(0.55)

    fig.add_artist(plt.Line2D([0.5, 0.5], [0.075, 0.972], transform=fig.transFigure, color="#CFCFCF", linewidth=0.7))
    fig.text(
        0.5,
        0.026,
        "Native-HR (estimated) is a content-aligned 368p reference; all other rows are 720p outputs.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=COLORS["muted"],
    )

    pdf_path, png_path = save_figure(fig, "supplementary/fig_qualitative", png_dpi=350)
    plt.close(fig)

    manifest = {
        "spatial": {
            "prompt_index": 5,
            "seed": 9705,
            "frame_zero_based": SPATIAL_FRAME,
            "crop_boxes_in_1248x720": SPATIAL_CROPS,
        },
        "temporal": {
            "prompt_index": 7,
            "seed": 9707,
            "center_frame_zero_based": TEMPORAL_CENTER,
            "frames_zero_based": list(TEMPORAL_FRAMES),
            "crop_box_in_1248x720": TEMPORAL_CROP,
        },
        "native_hr_estimated_note": (
            "Content-aligned 640x368 full LR trajectory used for qualitative context only; "
            "the quantitative Native-HR Sampling baseline is actual 720p generation."
        ),
        "videos": manifest_videos,
    }
    manifest_path = Path(__file__).resolve().parent / "supplementary" / "fig_qualitative_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
